"""Minute-bar loading and ATM-plan helpers for option backtests."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from nifty_core.core.events import BarEvent, Right
from nifty_core.research.probes import BREEZE_OPTIONS_ROOT, atm_strike, load_index_daily


def build_atm_for_day(trading_days: list[date], step: int = 50) -> dict[date, int]:
    """Map trading day -> ATM strike (from previous-day NIFTY close)."""
    nifty = load_index_daily("NIFTY_50")["Close"]
    out: dict[date, int] = {}
    for d in trading_days:
        sub = nifty.loc[: pd.Timestamp(d - timedelta(days=1))]
        if sub.empty:
            continue
        out[d] = atm_strike(float(sub.iloc[-1]), step=step)
    return out


def _expected_contract_dir(stock: str, expiry: date, strike: int, right: Right) -> Path:
    return BREEZE_OPTIONS_ROOT / f"{stock}_{expiry:%Y%m%d}_{strike}_{right.value.replace('CE','CALL').replace('PE','PUT')}"


def _short_to_breeze_right(right: Right) -> str:
    return "CALL" if right == Right.CE else "PUT"


def load_day_bars_for_strategy(
    day: date,
    stock: str,
    expiry: date,
    strikes: list[int],
    rights: list[Right],
) -> list[BarEvent]:
    """Load all minute bars for the legs the strategy cares about on `day`."""
    events: list[BarEvent] = []
    for strike in strikes:
        for right in rights:
            sym = f"{stock}_{expiry:%Y%m%d}_{strike}_{_short_to_breeze_right(right)}"
            p = BREEZE_OPTIONS_ROOT / sym / f"{day:%Y%m%d}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            df["datetime"] = pd.to_datetime(df["datetime"])
            for row in df.itertuples(index=False):
                events.append(BarEvent(
                    ts=row.datetime.to_pydatetime() if hasattr(row.datetime, "to_pydatetime") else row.datetime,
                    stock_code=stock,
                    expiry=f"{expiry:%Y-%m-%d}",
                    strike=strike,
                    right=right,
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=int(row.volume) if hasattr(row, "volume") else 0,
                    open_interest=int(row.open_interest) if hasattr(row, "open_interest") else 0,
                ))
    events.sort(key=lambda e: e.ts)
    return events
