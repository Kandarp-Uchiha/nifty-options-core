"""
Research helpers — utilities for empirical probes against Phase A backfill data.

Loaders for:
  - Breeze 1-min option bars (per contract per day)
  - yfinance index daily + 1-min
  - India VIX daily
  - nselib F&O bhavcopy (EOD)
Plus small primitives:
  - ATM strike snapping, weekly expiry enumeration
  - Spread mid/credit/MtM computation
  - Common backtest outcome harness for defined-risk multi-leg trades
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from nifty_core.data.storage import RAW_ROOT

logger = logging.getLogger(__name__)

BREEZE_OPTIONS_ROOT = RAW_ROOT / "breeze" / "options_1minute"
YF_INDEX_DAILY_ROOT = RAW_ROOT / "yfinance" / "index_daily"
YF_INDEX_MIN_ROOT = RAW_ROOT / "yfinance" / "index_1m"
NSELIB_BHAV_ROOT = RAW_ROOT / "nselib" / "fno_bhav"

Right = Literal["CALL", "PUT", "CE", "PE"]


def _norm_right(r: str) -> str:
    r = r.upper()
    return {"CE": "CALL", "PE": "PUT", "CALL": "CALL", "PUT": "PUT"}[r]


def _short_right(r: str) -> str:
    return {"CALL": "CE", "PUT": "PE"}[_norm_right(r)]


def symbol_for(stock_code: str, expiry: date, strike: int, right: str) -> str:
    return f"{stock_code}_{expiry:%Y%m%d}_{strike}_{_norm_right(right)}"


# -------- Loaders --------------------------------------------------------

def load_option_minute(
    stock_code: str,
    expiry: date,
    strike: int,
    right: str,
    day: date,
) -> pd.DataFrame | None:
    """
    Load 1-min option bars for one (contract × day). Returns None if missing.
    Columns: datetime (pd.Timestamp, IST naive), open, high, low, close, volume,
             open_interest, symbol, strike_price, right
    """
    sym = symbol_for(stock_code, expiry, strike, right)
    p = BREEZE_OPTIONS_ROOT / sym / f"{day:%Y%m%d}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime").sort_index()


def load_index_daily(symbol: str = "NIFTY_50") -> pd.DataFrame:
    """Load the daily snapshot for an index ('NIFTY_50' / 'NIFTY_BANK' / 'INDIA_VIX').

    Picks the snapshot with the MOST rows (longest history), not the
    newest filename. A short refresh (e.g. `--daily 1y`) writes a fresh
    date-stamped file that would otherwise sort last and silently shadow the
    full-history file, truncating every backtest. Selecting by row-count makes
    the loader robust to that footgun.
    """
    files = sorted((YF_INDEX_DAILY_ROOT / symbol).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No daily file for {symbol}")
    best_path, best_df, best_rows = None, None, -1
    for f in files:
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue
        if len(d) > best_rows:
            best_path, best_df, best_rows = f, d, len(d)
    if best_df is None:
        raise FileNotFoundError(f"No readable daily file for {symbol}")
    df = best_df
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df.set_index("Date").sort_index()


def load_vix_daily() -> pd.Series:
    """India VIX close series."""
    return load_index_daily("INDIA_VIX")["Close"].rename("vix")


CROSS_ASSET_ROOT = RAW_ROOT / "yfinance" / "cross_asset_daily"


def load_cross_asset_close(label: str) -> pd.Series:
    """Load latest daily-close series for a cross-asset symbol (e.g. 'SP500_FUT').

    Returns a Series indexed by tz-naive date (Asia/Kolkata wall date), named `label`.
    Caller is responsible for shifting to avoid lookahead.
    """
    files = sorted((CROSS_ASSET_ROOT / label).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No cross-asset file for {label} under {CROSS_ASSET_ROOT}")
    df = pd.read_parquet(files[-1])
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df.set_index("Date")["Close"].rename(label).sort_index()


def load_bhavcopy(day: date) -> pd.DataFrame | None:
    """Load nselib F&O bhavcopy for a given day. Returns None if missing."""
    p = NSELIB_BHAV_ROOT / f"{day:%Y%m%d}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


# -------- Universe enumeration -------------------------------------------

def available_option_days(
    stock_code: str | None = None,
    expiry: date | None = None,
    strike: int | None = None,
    right: str | None = None,
) -> list[date]:
    """List trading days we have option bars for (across any contract by default)."""
    pat = "*"
    if all([stock_code, expiry, strike, right]):
        sym = symbol_for(stock_code, expiry, strike, right)  # type: ignore
        files = (BREEZE_OPTIONS_ROOT / sym).glob("*.parquet")
    else:
        files = BREEZE_OPTIONS_ROOT.glob(f"{stock_code or pat}*/*.parquet")
    days = {datetime.strptime(p.stem, "%Y%m%d").date() for p in files}
    return sorted(days)


def available_contracts(stock_code: str = "NIFTY") -> list[tuple[date, int, str]]:
    """List (expiry, strike, right) tuples we have at least one day of data for."""
    out: list[tuple[date, int, str]] = []
    for d in BREEZE_OPTIONS_ROOT.iterdir():
        if not d.is_dir() or not d.name.startswith(f"{stock_code}_"):
            continue
        parts = d.name.split("_")
        try:
            exp = datetime.strptime(parts[1], "%Y%m%d").date()
            strike = int(parts[2])
            right = parts[3]
            out.append((exp, strike, right))
        except (ValueError, IndexError):
            continue
    return sorted(out)


def weekly_expiries(start: date, end: date, weekday: int = 1) -> list[date]:
    """Tuesdays in [start, end] for NIFTY weekly (weekday=1)."""
    out, cur = [], start
    while cur <= end:
        if cur.weekday() == weekday:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def atm_strike(spot: float, step: int = 50) -> int:
    return int(round(spot / step) * step)


# -------- Spread building blocks -----------------------------------------

@dataclass
class Leg:
    stock_code: str
    expiry: date
    strike: int
    right: str           # CE/PE or CALL/PUT
    action: str          # "BUY" or "SELL"
    qty: int = 1         # in lots (informational; we work in per-lot premiums)

    def signed_premium(self, price: float) -> float:
        """+ve premium = cash inflow (we received). -ve = cash outflow (we paid)."""
        return price * (1 if self.action == "SELL" else -1)


def _bar_close_at(
    legs_minute: dict[str, pd.DataFrame],
    leg_key: str,
    ts: pd.Timestamp,
    side: Literal["mid", "close"] = "close",
) -> float | None:
    """Find the option close price at-or-before ts for one leg's bar series."""
    df = legs_minute.get(leg_key)
    if df is None or df.empty:
        return None
    # Find last bar with index <= ts
    sub = df.loc[:ts]
    if sub.empty:
        return None
    return float(sub["close"].iloc[-1])


def load_legs_minute(legs: list[Leg], day: date) -> dict[str, pd.DataFrame]:
    """Load minute bars for each leg for the given day. Missing legs → not in dict."""
    out = {}
    for leg in legs:
        df = load_option_minute(leg.stock_code, leg.expiry, leg.strike, leg.right, day)
        if df is not None:
            key = f"{leg.action}_{leg.strike}_{_norm_right(leg.right)}"
            out[key] = df
    return out


def spread_net_premium_at(
    legs: list[Leg],
    day: date,
    minute: time,
) -> float | None:
    """
    Net premium of the spread at the given (day, minute). Per lot.
    Sign convention: positive = net credit received (we collected money).
    Returns None if any leg has no bar at-or-before that minute.
    """
    legs_min = load_legs_minute(legs, day)
    ts = pd.Timestamp(datetime.combine(day, minute))
    total = 0.0
    for leg in legs:
        key = f"{leg.action}_{leg.strike}_{_norm_right(leg.right)}"
        px = _bar_close_at(legs_min, key, ts)
        if px is None:
            return None
        total += leg.signed_premium(px)
    return total


def spread_pnl_series(
    legs: list[Leg],
    day: date,
    entry_minute: time = time(9, 20),
    exit_minute: time = time(15, 15),
    resample: str = "1min",
) -> pd.DataFrame | None:
    """
    Per-minute MtM P&L (per lot) from entry_minute to exit_minute.
    Returns DataFrame indexed by datetime with columns:
        net_premium    = signed net premium at that minute (credit positive)
        pnl            = net_premium - net_premium_at_entry  (credit-spread convention: gain if premium SHRINKS)
                         For a debit spread (entry premium negative), we invert so positive = gain.
    """
    legs_min = load_legs_minute(legs, day)
    if len(legs_min) != len(legs):
        return None  # incomplete

    entry_ts = pd.Timestamp(datetime.combine(day, entry_minute))
    exit_ts = pd.Timestamp(datetime.combine(day, exit_minute))

    # Build per-minute close per leg using forward-fill
    combined = []
    for leg in legs:
        key = f"{leg.action}_{leg.strike}_{_norm_right(leg.right)}"
        df = legs_min[key][["close"]].rename(columns={"close": key})
        combined.append(df)
    wide = pd.concat(combined, axis=1).sort_index()
    wide = wide.loc[(wide.index >= entry_ts) & (wide.index <= exit_ts)].ffill()

    # Build signed premium per row
    sign_map = {f"{leg.action}_{leg.strike}_{_norm_right(leg.right)}":
                (1 if leg.action == "SELL" else -1) for leg in legs}
    nets = sum(wide[k] * v for k, v in sign_map.items())

    entry_premium = nets.iloc[0] if not nets.empty else None
    if entry_premium is None:
        return None
    # If entry_premium > 0: it's a credit spread → gain when premium SHRINKS
    # If entry_premium < 0: it's a debit spread → gain when premium GROWS
    sign = 1 if entry_premium > 0 else -1
    pnl = (entry_premium - nets) * sign  # always positive = profit
    out = pd.DataFrame({"net_premium": nets, "pnl": pnl})
    out.attrs["entry_premium"] = float(entry_premium)
    out.attrs["spread_class"] = "credit" if entry_premium > 0 else "debit"
    return out


# -------- Convenience: regime classifier ---------------------------------

@dataclass
class RegimeSnapshot:
    day: date
    vix_close_prev: float
    vix_chg_pct_5d: float
    rv_5d: float                # realized vol of NIFTY close-to-close 5d annualized
    label: str                  # LOW_VOL | MID_VOL | HIGH_VOL


def classify_regime(
    day: date,
    low_thresh: float = 14.0,
    high_thresh: float = 18.0,
) -> RegimeSnapshot | None:
    """
    Classify a trading day by previous-close VIX + 5d realized vol.
    Uses yfinance daily files (so works for any day with NIFTY + VIX coverage).

    Thresholds calibrated 2026-05-31 against 7-yr VIX distribution
    (2019-02-11 .. 2026-05-29, N=1780): p35=14.0, p65=17.7. Splits roughly
    35/30/35 across LOW/MID/HIGH. Was 12/18 on small-sample calibration.
    """
    nifty = load_index_daily("NIFTY_50")["Close"]
    vix = load_vix_daily()
    cutoff = pd.Timestamp(day)
    nifty_h = nifty.loc[:cutoff].iloc[-21:]
    vix_h = vix.loc[:cutoff].iloc[-6:]
    if len(nifty_h) < 6 or len(vix_h) < 2:
        return None
    rv_5d = (nifty_h.pct_change().iloc[-5:].std() * (252 ** 0.5) * 100.0)
    vix_close_prev = float(vix_h.iloc[-2])  # close before `day`
    vix_chg_pct_5d = float((vix_h.iloc[-1] / vix_h.iloc[-6] - 1) * 100) if len(vix_h) >= 6 else 0.0
    label = ("LOW_VOL" if vix_close_prev < low_thresh
             else "HIGH_VOL" if vix_close_prev > high_thresh
             else "MID_VOL")
    return RegimeSnapshot(day, vix_close_prev, vix_chg_pct_5d, float(rv_5d), label)
