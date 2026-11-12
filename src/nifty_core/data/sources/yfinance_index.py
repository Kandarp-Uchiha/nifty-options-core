"""
yfinance — index intraday + daily OHLC (FREE).

Used for:
  - 1-min OHLC of NIFTY 50, NIFTY BANK, INDIA VIX for feature engineering
  - Daily history for long-window features

Coverage caveat: yfinance only retains 1-min intraday for the last ~7 calendar days
on free tier. For deeper history of intraday underlying, supplement with Truedata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from nifty_core.data.storage import daily_partition_path, write_parquet

INDEX_SYMBOLS: dict[str, str] = {
    "^NSEI": "NIFTY_50",
    "^NSEBANK": "NIFTY_BANK",
    "^INDIAVIX": "INDIA_VIX",
}


@dataclass
class YFResult:
    symbol: str
    label: str
    interval: str
    rows: int
    path: str
    error: str | None = None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.25, max=4), reraise=True)
def _fetch(ticker_obj, period: str, interval: str) -> pd.DataFrame:
    return ticker_obj.history(period=period, interval=interval, auto_adjust=False)


def fetch_index_intraday_1m(yf_symbol: str, days_back: int = 5) -> pd.DataFrame:
    """Fetch last `days_back` calendar days of 1-min bars (capped at 7 by yfinance)."""
    import yfinance as yf
    t = yf.Ticker(yf_symbol)
    df = _fetch(t, period=f"{min(days_back, 7)}d", interval="1m")
    df = df.reset_index()
    df["yf_symbol"] = yf_symbol
    return df


def fetch_index_daily(yf_symbol: str, period: str = "5y") -> pd.DataFrame:
    """Fetch daily OHLC for a long period (default 5 years)."""
    import yfinance as yf
    t = yf.Ticker(yf_symbol)
    df = _fetch(t, period=period, interval="1d")
    df = df.reset_index()
    df["yf_symbol"] = yf_symbol
    return df


def backfill_index_intraday(days_back: int = 5) -> list[YFResult]:
    """Pull 1-min bars for every configured index symbol, split into per-day parquet files."""
    results: list[YFResult] = []
    for yf_symbol, label in INDEX_SYMBOLS.items():
        try:
            df = fetch_index_intraday_1m(yf_symbol, days_back=days_back)
        except Exception as e:
            results.append(YFResult(yf_symbol, label, "1m", 0, "", str(e)[:200]))
            continue
        if df.empty:
            results.append(YFResult(yf_symbol, label, "1m", 0, "", "empty result"))
            continue

        date_col = next((c for c in ("Datetime", "Date", "index") if c in df.columns), df.columns[0])
        df["__day__"] = pd.to_datetime(df[date_col]).dt.date
        for day, day_df in df.groupby("__day__"):
            out = day_df.drop(columns=["__day__"])
            path = daily_partition_path("yfinance", f"index_1m/{label}", day)
            write_parquet(out, path, overwrite=True)
            results.append(YFResult(yf_symbol, label, "1m", len(out), str(path)))
    return results


def backfill_index_daily(period: str = "5y") -> list[YFResult]:
    """Pull daily OHLC for every configured index symbol — one parquet per symbol."""
    results: list[YFResult] = []
    for yf_symbol, label in INDEX_SYMBOLS.items():
        try:
            df = fetch_index_daily(yf_symbol, period=period)
            # Guard 1: a blocked/failed fetch can return an empty frame (e.g. when
            # routed through a proxy that does not allowlist Yahoo). NEVER write a
            # 0-row file -- that would overwrite previously-good daily history.
            if df.empty:
                results.append(YFResult(yf_symbol, label, "1d", 0, "",
                                        "empty result (NOT written; kept existing file)"))
                continue
            # Guard 2: never write a snapshot SHORTER than the longest existing
            # one. A short refresh (e.g. --daily 1y) writes a fresh date-stamped
            # file that would shadow the full-history file and silently truncate
            # every backtest. Refuse unless this fetch has >= the rows we already
            # hold (allows same-length daily updates, blocks accidental truncation).
            seg_dir = (daily_partition_path("yfinance", f"index_daily/{label}",
                                            date.today())).parent
            existing_max = 0
            if seg_dir.exists():
                for f in seg_dir.glob("*.parquet"):
                    try:
                        existing_max = max(existing_max,
                                           len(pd.read_parquet(f, columns=["Date"])))
                    except Exception:
                        pass
            if existing_max and len(df) < existing_max:
                results.append(YFResult(
                    yf_symbol, label, "1d", len(df), "",
                    f"refused: {len(df)} rows < existing {existing_max} "
                    f"(use a longer --daily period to extend history)"))
                continue
            today = date.today()
            path = daily_partition_path("yfinance", f"index_daily/{label}", today)
            write_parquet(df, path, overwrite=True)
            results.append(YFResult(yf_symbol, label, "1d", len(df), str(path)))
        except Exception as e:
            results.append(YFResult(yf_symbol, label, "1d", 0, "", str(e)[:200]))
    return results
