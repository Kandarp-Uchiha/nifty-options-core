"""
yfinance — global cross-asset daily OHLC (FREE).

Used for v2 direction-prediction features per docs/02-direction-prediction-v2-roadmap.md
Idea 2. These are overnight global-macro factors that close BEFORE NIFTY's next
session opens, so they carry true predictive (not coincident) information.

Symbol list and rationale (from the roadmap doc):
  ES=F        S&P 500 e-mini futures — single most reliable input
  ^GSPC       S&P 500 cash (used if ES=F missing days)
  DX-Y.NYB    US Dollar Index
  ^TNX        US 10y Treasury yield
  INR=X       USD/INR — direct effect on FII flows
  CL=F        WTI Crude oil — Indian-import sensitive
  ^HSI        Hang Seng — Asia open before NIFTY
  ^N225       Nikkei 225 — Asia open before NIFTY

Coverage caveat: yfinance daily data is reliable back ~10 years for these majors.
Holiday calendars differ (US closed July 4, India open; India closed Diwali, US open).
Feature builder forward-fills across calendar days then shifts to enforce no-lookahead.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from nifty_core.data.storage import daily_partition_path, write_parquet

logger = logging.getLogger(__name__)


CROSS_ASSET_SYMBOLS: dict[str, str] = {
    # yfinance ticker -> human label
    "ES=F":      "SP500_FUT",
    "^GSPC":     "SP500_CASH",
    "DX-Y.NYB":  "DXY",
    "^TNX":      "US_10Y_YIELD",
    "INR=X":     "USDINR",
    "CL=F":      "WTI_CRUDE",
    "^HSI":      "HANG_SENG",
    "^N225":     "NIKKEI",
}


@dataclass
class FetchResult:
    yf_symbol: str
    label: str
    rows: int
    path: str
    error: str | None = None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4), reraise=True)
def _fetch_daily(ticker_obj, period: str) -> pd.DataFrame:
    return ticker_obj.history(period=period, interval="1d", auto_adjust=False)


def fetch_cross_asset_daily(yf_symbol: str, period: str = "7y") -> pd.DataFrame:
    """Fetch a single symbol's daily OHLCV for the given lookback period."""
    import yfinance as yf
    t = yf.Ticker(yf_symbol)
    df = _fetch_daily(t, period=period)
    if df.empty:
        return df
    df = df.reset_index()
    df["yf_symbol"] = yf_symbol
    return df


def backfill_cross_asset_daily(period: str = "7y") -> list[FetchResult]:
    """Pull daily OHLC for every configured cross-asset symbol.

    Stored at data/raw/yfinance/cross_asset_daily/{label}/{today}.parquet, overwriting
    each run (the full window is re-pulled — small dataset, ~1800 rows × 8 symbols).
    """
    results: list[FetchResult] = []
    today = date.today()
    for yf_symbol, label in CROSS_ASSET_SYMBOLS.items():
        try:
            df = fetch_cross_asset_daily(yf_symbol, period=period)
            if df.empty:
                results.append(FetchResult(yf_symbol, label, 0, "", "empty result"))
                continue
            path = daily_partition_path("yfinance", f"cross_asset_daily/{label}", today)
            write_parquet(df, path, overwrite=True)
            results.append(FetchResult(yf_symbol, label, len(df), str(path)))
            logger.info("Saved %s -> %d rows", label, len(df))
        except Exception as e:
            results.append(FetchResult(yf_symbol, label, 0, "", str(e)[:200]))
            logger.warning("Failed %s: %s", label, e)
    return results
