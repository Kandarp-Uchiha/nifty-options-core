"""
nselib — EOD F&O bhavcopy + India VIX downloader (FREE).

Per docs/02-data-sources.md, nselib is our primary free source for:
  - Per-day F&O bhavcopy (every contract: futures + options, OHLC + OI)
  - India VIX daily OHLC
  - Per-expiry option price-volume history

Resilient to:
  - NSE 404 on weekends / holidays
  - nselib upstream format changes (we re-throw with context)
  - 424 transient errors via retry
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from nifty_core.data.storage import daily_partition_path, write_parquet, missing_days


@dataclass
class BhavcopyResult:
    day: date
    rows: int
    path: str
    skipped: bool = False
    holiday: bool = False
    error: str | None = None


# Holiday cache (sourced from observed "empty bhavcopy" responses). Persists across
# runs so we don't keep re-asking NSE for known holidays. NSE official holiday list
# could be loaded from `nselib.trading_holiday_calendar()` in a future iteration.
from nifty_core.paths import META_ROOT as _META_ROOT  # noqa: E402

_HOLIDAY_CACHE_PATH = _META_ROOT / "nse_holidays.txt"


def _load_holiday_cache() -> set[date]:
    if not _HOLIDAY_CACHE_PATH.exists():
        return set()
    out: set[date] = set()
    for line in _HOLIDAY_CACHE_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.add(date.fromisoformat(line))
        except ValueError:
            continue
    return out


def _save_holiday(d: date) -> None:
    _HOLIDAY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_holiday_cache()
    if d in cache:
        return
    cache.add(d)
    _HOLIDAY_CACHE_PATH.write_text("\n".join(sorted(d.isoformat() for d in cache)) + "\n")


# -------- Helpers ----------------------------------------------------------

def _is_market_day(d: date) -> bool:
    """Quick filter: NSE is closed weekends. Holiday calendar handled by 404s upstream."""
    return d.weekday() < 5


def date_range(start: date, end: date) -> list[date]:
    """Inclusive list of market days (Mon–Fri) between start and end."""
    days: list[date] = []
    cur = start
    while cur <= end:
        if _is_market_day(cur):
            days.append(cur)
        cur += timedelta(days=1)
    return days


# -------- Downloaders ------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.25, max=4),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def fetch_fno_bhavcopy(day: date) -> pd.DataFrame:
    """Fetch one day's full F&O bhavcopy via nselib. ~45k rows for a typical day."""
    from nselib import derivatives  # local import: keeps cold-start fast for callers
    trade_date_str = day.strftime("%d-%m-%Y")
    df = derivatives.fno_bhav_copy(trade_date=trade_date_str)
    if df is None or len(df) == 0:
        raise RuntimeError(f"empty F&O bhavcopy for {trade_date_str}")
    return df


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.25, max=4),
    reraise=True,
)
def fetch_india_vix_history(period: str = "1Y") -> pd.DataFrame:
    """Fetch India VIX historical OHLC for a period (e.g. '1W','1M','6M','1Y')."""
    from nselib import capital_market
    df = capital_market.india_vix_data(period=period)
    if df is None or len(df) == 0:
        raise RuntimeError("empty India VIX result")
    return df


# -------- Backfill orchestration ------------------------------------------

def backfill_fno_bhavcopy(
    start: date,
    end: date,
    overwrite: bool = False,
) -> list[BhavcopyResult]:
    """
    Download every missing F&O bhavcopy between [start, end] inclusive.
    Idempotent: skips days already present (unless overwrite=True).
    Empty bhavcopy from NSE is treated as a holiday and cached.
    Today's bhavcopy is treated as 'not yet published' if empty, not a holiday.
    """
    wanted = date_range(start, end)
    holidays = _load_holiday_cache()
    # Drop known holidays so we don't waste calls
    candidates = [d for d in wanted if d not in holidays]
    to_fetch = candidates if overwrite else missing_days("nselib", "fno_bhav", candidates)
    # Days that are present on disk and not a holiday -> count as skipped
    already_on_disk = set(candidates) - set(to_fetch)

    results: list[BhavcopyResult] = []
    # Surface skipped-holidays in the result so the summary is honest
    for d in (wanted_d for wanted_d in wanted if wanted_d in holidays):
        results.append(BhavcopyResult(day=d, rows=0, path="", holiday=True, skipped=True))
    # Surface already-on-disk days too
    for d in sorted(already_on_disk):
        path = daily_partition_path("nselib", "fno_bhav", d)
        results.append(BhavcopyResult(day=d, rows=0, path=str(path), skipped=True))

    today = date.today()
    for d in to_fetch:
        path = daily_partition_path("nselib", "fno_bhav", d)
        if path.exists() and not overwrite:
            results.append(BhavcopyResult(day=d, rows=0, path=str(path), skipped=True))
            continue
        try:
            df = fetch_fno_bhavcopy(d)
            write_parquet(df, path, overwrite=overwrite)
            results.append(BhavcopyResult(day=d, rows=len(df), path=str(path)))
        except RuntimeError as e:
            msg = str(e)
            if "empty F&O bhavcopy" in msg:
                if d == today:
                    # Not yet published; don't cache as holiday.
                    results.append(BhavcopyResult(day=d, rows=0, path=str(path), error="not yet published"))
                else:
                    _save_holiday(d)
                    results.append(BhavcopyResult(day=d, rows=0, path=str(path), holiday=True))
            else:
                results.append(BhavcopyResult(day=d, rows=0, path=str(path), error=msg[:200]))
        except Exception as e:
            results.append(BhavcopyResult(day=d, rows=0, path=str(path), error=str(e)[:200]))
    return results


def filter_to_index_options(
    df: pd.DataFrame, underlyings: Iterable[str] = ("NIFTY", "BANKNIFTY")
) -> pd.DataFrame:
    """Convenience filter for our v1 scope. Column names per new UDiFF format."""
    # The bhavcopy ticker symbol column is typically TckrSymb or similar.
    cols_lower = {c.lower(): c for c in df.columns}
    tkr_col = cols_lower.get("tckrsymb") or cols_lower.get("symbol")
    fininstrtype_col = cols_lower.get("fininstrmtp")
    if not tkr_col:
        return df  # graceful degradation
    underlyings_set = {u.upper() for u in underlyings}
    mask = df[tkr_col].astype(str).str.upper().isin(underlyings_set)
    if fininstrtype_col:
        # Restrict to options (OPTIDX). Futures = STF/STO; index opt = IDO/IDF in some formats.
        mask &= df[fininstrtype_col].astype(str).str.upper().isin({"IDO", "OPTIDX", "OPT", "STO"})
    return df.loc[mask].copy()
