"""
ICICI Direct Breeze — historical NFO options downloader.

Why Breeze over Truedata:
  - 3 years of free 1-min historical options data (no monthly fee)
  - Comes free with any ICICI Direct trading account
  - WebSocket streaming OHLCV also free (useful later for self-record fallback)
  - Rate limit: 100 req/min, 5000 req/day (generous for backfill)

Caveats:
  - session_token expires daily ~8 AM IST. Manual browser-based refresh needed
    (open https://api.icicidirect.com/apiuser/login?api_key=<URL-encoded-AppKey>,
     log in, copy `apisession=...` query param from the redirect URL).
  - Static IP whitelisting applies to ORDER PLACEMENT only — reads work from any IP.
  - Historical Data v2 returns max 1000 candles per request.
    For 1-min bars: 1 trading day = ~375 candles → fits 1 day per request.

Symbol conventions:
  - Underlying via stock_code: "NIFTY", "CNXBAN" (BankNifty), "ICIBAN" (ICICI Bank), etc.
  - Exchange: "NFO" for index/stock options, "NSE" for cash, "BFO" for BSE F&O
  - Options need: expiry_date (ISO 8601), right ("call"/"put"), strike_price (string)
  - Dates throughout: ISO 8601 with .000Z suffix, e.g. "2026-05-22T09:15:00.000Z"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from nifty_core.data.storage import daily_partition_path, write_parquet

logger = logging.getLogger(__name__)


# -------- Connection -----------------------------------------------------

def connect(api_key: str, api_secret: str, session_token: str):
    """
    Authenticate against Breeze. Returns a BreezeConnect instance.

    session_token here is the `apisession` query-param value the user copies
    from the post-login redirect URL — NOT the long-lived AppKey.
    """
    from breeze_connect import BreezeConnect
    breeze = BreezeConnect(api_key=api_key)
    breeze.generate_session(api_secret=api_secret, session_token=session_token)
    return breeze


def login_url(api_key: str) -> str:
    """Build the URL the user opens in a browser to obtain a fresh apisession."""
    import urllib.parse
    return f"https://api.icicidirect.com/apiuser/login?api_key={urllib.parse.quote_plus(api_key)}"


# -------- Auto-login (Playwright + TOTP) ---------------------------------
# Approach ported (with thanks) from aranjan/icici-mcp (MIT license):
#   https://github.com/aranjan/icici-mcp/blob/main/src/icici_mcp/auth.py
# ICICI Direct does NOT expose a programmatic auth endpoint — the only way
# to get a fresh apisession is to drive the actual web login form. We use
# headless Chromium via Playwright.

import json as _json
from pathlib import Path as _Path

from nifty_core.paths import META_ROOT as _META_ROOT  # noqa: E402

_BREEZE_TOKEN_CACHE = _META_ROOT / "breeze_token.json"


def _load_cached_session_token(api_key: str) -> str | None:
    """Return today's cached session token, or None if missing/expired/stale."""
    if not _BREEZE_TOKEN_CACHE.exists():
        return None
    try:
        data = _json.loads(_BREEZE_TOKEN_CACHE.read_text())
    except Exception:
        return None
    today = datetime.now().strftime("%Y-%m-%d")
    if data.get("date") != today or data.get("api_key") != api_key:
        return None
    return data.get("session_token")


def _save_cached_session_token(api_key: str, session_token: str) -> None:
    _BREEZE_TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _BREEZE_TOKEN_CACHE.write_text(_json.dumps({
        "api_key": api_key,
        "session_token": session_token,
        "date": datetime.now().strftime("%Y-%m-%d"),
    }))


def _automated_login_playwright(
    api_key: str,
    user_id: str,
    password: str,
    totp_secret: str,
    headless: bool = True,
    timeout_ms: int = 30000,
) -> str:
    """
    Drive the ICICI Direct web login form via Chromium and capture the
    apisession token from the redirect URL. Returns the session_token.
    """
    import asyncio
    import pyotp
    from urllib.parse import parse_qs, urlparse

    url = login_url(api_key)

    async def _run() -> str:
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise RuntimeError(
                "Playwright not installed. Run: uv sync && uv run playwright install chromium"
            ) from e

        apisession: str | None = None
        browser = None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=headless)
                page = await browser.new_page()

                # Intercept any request whose URL contains apisession=
                def on_request(request):
                    nonlocal apisession
                    try:
                        if "apisession" in request.url:
                            token = parse_qs(urlparse(request.url).query).get("apisession", [None])[0]
                            if token:
                                apisession = token
                                logger.info("breeze auto-login: captured apisession")
                    except Exception:
                        pass

                page.on("request", on_request)

                # Capture any JS dialog (alert/confirm/prompt) so we don't deadlock
                dialog_msgs: list[str] = []

                def on_dialog(dialog):
                    dialog_msgs.append(f"{dialog.type}: {dialog.message}")
                    # Schedule accept without blocking
                    import asyncio as _a
                    _a.create_task(dialog.accept())

                page.on("dialog", on_dialog)

                # Step 1: navigate to login page
                await page.goto(url, wait_until="networkidle", timeout=timeout_ms)

                # Step 2: fill UCC + password + T&C checkbox.
                # Use press_sequentially so any on-input JS handlers fire naturally.
                uid_field = await page.query_selector("#txtuid")
                if not uid_field:
                    raise RuntimeError(
                        "Login page structure changed: #txtuid not found. "
                        "ICICI Direct may have updated their login page."
                    )
                await page.fill("#txtuid", "")
                await page.locator("#txtuid").press_sequentially(user_id, delay=30)
                await page.fill("#txtPass", "")
                await page.locator("#txtPass").press_sequentially(password, delay=30)
                await page.check("#chkssTnc")

                # Step 3: submit, wait for 2FA page (longer wait + retry)
                await page.click("#btnSubmit")
                await page.wait_for_timeout(6000)

                # Step 4: generate TOTP and fill the OTP boxes
                totp_code = pyotp.TOTP(totp_secret).now()
                otp_inputs = await page.query_selector_all("input[tg-nm=otp]")
                if not otp_inputs:
                    # Dump diagnostics so we can see why
                    debug_dir = _BREEZE_TOKEN_CACHE.parent / "breeze_debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    try:
                        await page.screenshot(path=str(debug_dir / f"otp_missing_{ts}.png"),
                                              full_page=True)
                        html = await page.content()
                        (debug_dir / f"otp_missing_{ts}.html").write_text(html, encoding="utf-8")
                        cur_url = page.url
                    except Exception:
                        cur_url = "<unknown>"
                    err_el = await page.query_selector(".text-danger, #errmsg, .error")
                    msg = ""
                    if err_el:
                        try:
                            msg = (await err_el.inner_text()).strip()
                        except Exception:
                            msg = ""
                    raise RuntimeError(
                        f"OTP input fields not found after credential submit. "
                        f"Current URL: {cur_url}. Page error text: {msg!r}. "
                        f"JS dialogs seen: {dialog_msgs!r}. "
                        f"Screenshot + HTML dumped to {debug_dir}"
                    )
                if len(otp_inputs) == 6:
                    for i, digit in enumerate(totp_code):
                        await otp_inputs[i].fill(digit)
                else:
                    await otp_inputs[0].fill(totp_code)

                # Step 5: submit OTP (page exposes submitotp() JS)
                try:
                    await page.evaluate("submitotp()")
                except Exception:
                    btn = await page.query_selector("#btnOTP, button[type=submit]")
                    if btn:
                        await btn.click()

                # Step 6: wait for the redirect that carries apisession=
                await page.wait_for_timeout(6000)
        finally:
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass

        if not apisession:
            raise RuntimeError(
                "Auto-login completed but no apisession captured. "
                "Possible causes: TOTP code drift, wrong password, "
                "or ICICI Direct redirect flow changed."
            )
        return apisession

    return asyncio.run(_run())


def auto_login(
    api_key: str,
    api_secret: str,
    user_id: str,
    password: str,
    totp_secret: str,
    use_cache: bool = True,
    force_refresh: bool = False,
    headless: bool = True,
):
    """
    Fully automated Breeze login. Returns (BreezeConnect, session_token).

    Order of operations:
      1. If `use_cache` and cache from today exists -> try it (validate with get_customer_details)
      2. Else drive headless Chromium to perform the full login + TOTP, capture apisession
      3. Cache it at data/_meta/breeze_token.json (overwritten each day)
    """
    if use_cache and not force_refresh:
        cached = _load_cached_session_token(api_key)
        if cached:
            try:
                breeze = connect(api_key, api_secret, cached)
                breeze.get_customer_details(api_session=cached)
                logger.info("breeze auto-login: used cached token")
                return breeze, cached
            except Exception as e:
                logger.warning("breeze cached token invalid (%s); re-running auto-login", e)

    token = _automated_login_playwright(
        api_key, user_id, password, totp_secret, headless=headless
    )
    breeze = connect(api_key, api_secret, token)
    # Validate with a cheap call before caching
    breeze.get_customer_details(api_session=token)
    _save_cached_session_token(api_key, token)
    return breeze, token


# -------- Helpers --------------------------------------------------------

def _iso_breeze(dt: datetime) -> str:
    """
    Format an IST datetime for the Breeze API.

    IMPORTANT: despite the docs saying "ISO 8601 with .000Z", the Breeze server
    treats the timestamp as IST regardless of any UTC offset. Empirical probe on
    2026-05-25 confirmed this: sending 09:15 IST converted-to-UTC (03:45Z) yielded
    only 46 bars starting 09:15, because the server read 03:45 as 03:45 IST and
    only returned data from market open (09:15) onward.

    So: format the IST wall-clock time as if it were UTC. Strip any tzinfo.
    """
    if dt.tzinfo is not None:
        # Convert to IST first, then drop tz
        ist = timezone(timedelta(hours=5, minutes=30))
        dt = dt.astimezone(ist).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def atm_strike_grid(spot: float, step: int, n_each_side: int) -> list[int]:
    """Return strikes centred on spot, snapped to `step`."""
    atm = round(spot / step) * step
    return [atm + i * step for i in range(-n_each_side, n_each_side + 1)]


# -------- Historical fetch -----------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
    reraise=True,
)
def _fetch_one(
    breeze,
    *,
    interval: str,
    from_dt: datetime,
    to_dt: datetime,
    stock_code: str,
    exchange_code: str,
    product_type: str,
    expiry_date: datetime,
    right: str,
    strike_price: str,
) -> list[dict]:
    """One get_historical_data_v2 call with retry."""
    resp = breeze.get_historical_data_v2(
        interval=interval,
        from_date=_iso_breeze(from_dt),
        to_date=_iso_breeze(to_dt),
        stock_code=stock_code,
        exchange_code=exchange_code,
        product_type=product_type,
        expiry_date=_iso_breeze(expiry_date),
        right=right,
        strike_price=strike_price,
    )
    if not isinstance(resp, dict):
        raise RuntimeError(f"unexpected response shape: {type(resp).__name__}")
    if resp.get("Status") != 200:
        err = resp.get("Error") or resp.get("error") or "unknown error"
        raise RuntimeError(f"breeze API error: {err}")
    return resp.get("Success") or []


@dataclass
class BreezeResult:
    stock_code: str
    expiry: date
    strike: int
    right: str
    day: date
    bar_size: str
    rows: int
    path: str
    error: str | None = None


def backfill_option_day(
    breeze,
    *,
    stock_code: str,
    expiry: date,
    strike: int,
    right: str,
    day: date,
    interval: str = "1minute",
    overwrite: bool = False,
    exchange_code: str = "NFO",     # "NFO" for NSE F&O, "BFO" for BSE (SENSEX)
    polite_sleep_s: float = 0.65,  # ~92 req/min, just under 100/min limit
    rate_limiter=None,             # optional engine.utils.rate_limiter.TokenBucket
) -> BreezeResult:
    """Backfill one option contract for one trading day.

    If `rate_limiter` is provided, it's acquired BEFORE the API call and
    `polite_sleep_s` is ignored (the limiter alone enforces global rate).
    The skip-if-exists check happens BEFORE acquiring a token (free).
    """
    right_norm = right.lower()
    if right_norm not in ("call", "put"):
        raise ValueError(f"right must be 'call' or 'put', got {right!r}")

    symbol = f"{stock_code}_{expiry:%Y%m%d}_{strike}_{right_norm.upper()}"
    segment = f"breeze/options_{interval.replace(' ', '')}"
    path = daily_partition_path("breeze", f"options_{interval}", day, symbol=symbol)
    if path.exists() and not overwrite:
        return BreezeResult(stock_code, expiry, strike, right_norm, day, interval, 0,
                            str(path), error="skipped (exists)")

    # NSE intraday window (UTC: 09:15 IST = 03:45 UTC, 15:30 IST = 10:00 UTC).
    # Use IST-localised datetimes and let _iso_utc convert.
    from datetime import timezone as _tz
    ist = _tz(timedelta(hours=5, minutes=30))
    from_dt = datetime.combine(day, datetime.strptime("09:15", "%H:%M").time(), tzinfo=ist)
    to_dt = datetime.combine(day, datetime.strptime("15:30", "%H:%M").time(), tzinfo=ist)
    expiry_dt = datetime.combine(expiry, datetime.strptime("12:00", "%H:%M").time(), tzinfo=ist)

    try:
        # Throttle: prefer token bucket (precise, thread-safe) when supplied.
        if rate_limiter is not None:
            rate_limiter.acquire(1.0)
        bars = _fetch_one(
            breeze,
            interval=interval,
            from_dt=from_dt,
            to_dt=to_dt,
            stock_code=stock_code,
            exchange_code=exchange_code,
            product_type="options",
            expiry_date=expiry_dt,
            right=right_norm,
            strike_price=str(strike),
        )
    except Exception as e:
        return BreezeResult(stock_code, expiry, strike, right_norm, day, interval, 0,
                            str(path), error=str(e)[:200])
    finally:
        # Only fall back to crude sleep if no limiter is in play
        if rate_limiter is None:
            time.sleep(polite_sleep_s)

    if not bars:
        return BreezeResult(stock_code, expiry, strike, right_norm, day, interval, 0,
                            str(path), error="empty")
    df = pd.DataFrame(bars)
    df["symbol"] = symbol
    write_parquet(df, path, overwrite=overwrite)
    return BreezeResult(stock_code, expiry, strike, right_norm, day, interval, len(df), str(path))


def backfill_grid(
    breeze,
    *,
    stock_codes: Iterable[str],
    expiries: Iterable[date],
    strikes_by_underlying: dict[str, list[int]] | None = None,
    strikes_by_day: dict[tuple[str, date], list[int]] | None = None,
    days: Iterable[date],
    interval: str = "1minute",
    overwrite: bool = False,
    exchange_code: str = "NFO",
    polite_sleep_s: float = 0.65,
    lookahead_expiry_days: int | None = None,
    calls_per_minute: float | None = None,
    workers: int = 1,
) -> list[BreezeResult]:
    """
    Cross-product backfill: every (stock × expiry × strike × CE/PE) × every day.

    Provide EITHER:
      - strikes_by_underlying: static grid per underlying (legacy mode)
      - strikes_by_day: dynamic grid keyed by (stock_code, trading_day) →
        per-day list of strikes. Use this for ATM-adaptive backfills.

    lookahead_expiry_days: if set, only fetch expiries within this many days AFTER
        the trading day. Avoids requesting future contracts that didn't exist yet
        (NSE lists weeklies ~3 weeks ahead). 21 = current + next ~2 weeklies.

    Throttling — three modes (in order of preference):
      - calls_per_minute=N + workers=M  → token-bucket rate limiter shared by M
        ThreadPoolExecutor workers. Strict global cap of N/min. Recommended.
      - calls_per_minute=N + workers=1  → single-thread token bucket (no parallel
        in-flight calls but precise rate enforcement; replaces polite_sleep).
      - calls_per_minute=None           → legacy per-call time.sleep(polite_sleep_s).
        Effective throughput ≈ 60/polite_sleep_s req/min minus API latency.

    Empirically on Breeze: rate cap is ~100 req/min. Default calls_per_minute=95
    gives a 5% safety margin. Workers=4 smooths over occasional slow calls
    without exceeding the cap.
    """
    if strikes_by_underlying is None and strikes_by_day is None:
        raise ValueError("must provide strikes_by_underlying or strikes_by_day")

    # Build the token bucket if a rate is specified.
    rate_limiter = None
    if calls_per_minute is not None:
        from nifty_core.utils.rate_limiter import TokenBucket
        rate_limiter = TokenBucket.per_minute(calls_per_minute)
        logger.info("backfill_grid: rate-limited to %.1f calls/min, workers=%d",
                    calls_per_minute, workers)

    # Enumerate all work units first (lightweight tuples, no API calls).
    work: list[tuple[str, date, int, str, date]] = []
    days_list = sorted(days)
    for d in days_list:
        if d.weekday() >= 5:
            continue
        for stock in stock_codes:
            if strikes_by_day is not None:
                strikes = strikes_by_day.get((stock, d), [])
            else:
                strikes = (strikes_by_underlying or {}).get(stock, [])
            if not strikes:
                continue
            for exp in expiries:
                if exp < d:
                    continue
                if lookahead_expiry_days is not None and (exp - d).days > lookahead_expiry_days:
                    continue
                for strike in strikes:
                    for r in ("call", "put"):
                        work.append((stock, exp, strike, r, d))

    results: list[BreezeResult] = []
    cap_hit = False

    def _do_one(stock: str, exp: date, strike: int, r: str, d: date) -> BreezeResult:
        return backfill_option_day(
            breeze,
            stock_code=stock, expiry=exp, strike=strike, right=r, day=d,
            interval=interval, overwrite=overwrite, exchange_code=exchange_code,
            polite_sleep_s=polite_sleep_s, rate_limiter=rate_limiter,
        )

    def _is_cap_error(res: BreezeResult) -> bool:
        if not res.error:
            return False
        e = res.error.lower()
        return any(t in e for t in ("5000", "daily limit", "rate limit", "quota"))

    if workers <= 1:
        for stock, exp, strike, r, d in work:
            if cap_hit:
                break
            res = _do_one(stock, exp, strike, r, d)
            results.append(res)
            if res.error and "skipped" not in res.error:
                logger.warning("breeze %s exp=%s K=%s %s %s -> %s",
                               stock, exp, strike, r, d, res.error)
            if _is_cap_error(res):
                logger.error("breeze daily quota hit; stopping early. "
                             "Re-run tomorrow to resume (idempotent).")
                cap_hit = True
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_do_one, stock, exp, strike, r, d):
                    (stock, exp, strike, r, d)
                for stock, exp, strike, r, d in work
            }
            for fut in as_completed(futures):
                if cap_hit:
                    continue
                stock, exp, strike, r, d = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = BreezeResult(stock, exp, strike, r, d, interval, 0, "",
                                       error=str(e)[:200])
                results.append(res)
                if res.error and "skipped" not in res.error:
                    logger.warning("breeze %s exp=%s K=%s %s %s -> %s",
                                   stock, exp, strike, r, d, res.error)
                if _is_cap_error(res):
                    logger.error("breeze daily quota hit; stopping early. "
                                 "Re-run tomorrow to resume (idempotent).")
                    cap_hit = True
    return results

