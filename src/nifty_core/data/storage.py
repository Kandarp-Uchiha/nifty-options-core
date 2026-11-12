"""
Storage helpers — parquet partitioning + read helpers.

Layout convention:
    data/raw/<source>/<segment>/YYYYMMDD.parquet     # per trading day
    data/raw/<source>/<segment>/<symbol>/YYYYMMDD.parquet  # per symbol per day

Sources currently writing here:
    - nselib_fno         (EOD F&O bhavcopy)
    - yfinance_index     (1-min intraday for ^NSEI / ^NSEBANK / ^INDIAVIX)
    - breeze             (intraday option bars from ICICI Direct Breeze; primary)
    - truedata           (historical NFO option bars; reserved for future use)
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from nifty_core.paths import DATA_ROOT  # noqa: E402
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def daily_partition_path(
    source: str,
    segment: str,
    day: date | datetime,
    symbol: str | None = None,
) -> Path:
    """
    Resolve the partition path for a given day.

    With symbol:    data/raw/<source>/<segment>/<symbol>/YYYYMMDD.parquet
    Without symbol: data/raw/<source>/<segment>/YYYYMMDD.parquet
    """
    if isinstance(day, datetime):
        day = day.date()
    base = RAW_ROOT / source / segment
    if symbol:
        base = base / _safe_symbol(symbol)
    ensure_dir(base)
    return base / f"{day:%Y%m%d}.parquet"


def _safe_symbol(symbol: str) -> str:
    """Filesystem-safe transform: avoid characters problematic on Windows."""
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_").replace("|", "_").replace(" ", "_")


def write_parquet(df: pd.DataFrame, path: Path, overwrite: bool = False) -> Path:
    """Write a DataFrame to parquet (snappy compression). Skips if file exists unless overwrite."""
    if not overwrite and path.exists():
        return path
    ensure_dir(path.parent)
    df.to_parquet(path, index=False, compression="snappy")
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def list_existing_days(source: str, segment: str, symbol: str | None = None) -> list[date]:
    """Return sorted list of dates that already have parquet files written."""
    base = RAW_ROOT / source / segment
    if symbol:
        base = base / _safe_symbol(symbol)
    if not base.exists():
        return []
    days: list[date] = []
    for f in base.glob("*.parquet"):
        try:
            days.append(datetime.strptime(f.stem, "%Y%m%d").date())
        except ValueError:
            continue
    return sorted(days)


def missing_days(
    source: str,
    segment: str,
    wanted: Iterable[date],
    symbol: str | None = None,
) -> list[date]:
    """Return wanted days that don't yet have a parquet file. Useful for resumable backfills."""
    have = set(list_existing_days(source, segment, symbol))
    return [d for d in wanted if d not in have]
