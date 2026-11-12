"""Small option-chain helpers shared by research scripts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _round_atm(spot: float, step: int = 50) -> int:
    return int(round(spot / step) * step)


def _price_at(parquet_path: Path, hh: int, mm: int) -> float | None:
    try:
        df = pd.read_parquet(parquet_path, columns=["datetime", "close"])
        if df.empty:
            return None
        df["datetime"] = pd.to_datetime(df["datetime"])
        sub = df.loc[df["datetime"].dt.hour.eq(hh) & df["datetime"].dt.minute.eq(mm)]
        if sub.empty:
            return None
        v = float(sub["close"].iloc[0])
        return v if v > 0 else None
    except Exception:
        return None


round_atm = _round_atm
price_at = _price_at
