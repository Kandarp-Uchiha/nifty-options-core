"""
Vol-surface features per trading day from NIFTY weekly option EOD quotes.

Implements Idea 5 from docs/02-direction-prediction-v2-roadmap.md. For each
trading day, computes a small set of forward-looking implied-vol features that
capture skew, term structure, and put/call asymmetry.

Features built per day:
  atm_iv             ATM implied vol (front weekly, avg of ATM CE + ATM PE)
  atm_iv_chg_1d      day-over-day change in ATM IV (in IV points × 100)
  put_skew           IV(ATM-100 PE) - IV(ATM PE)   — crash-fear measure
  call_skew          IV(ATM+100 CE) - IV(ATM CE)   — upside-fear measure
  pc_skew_asym       put_skew - call_skew           — directional sentiment
  term_slope         IV(next-week ATM) - IV(this-week ATM)
                     (>0 = contango = calm; <0 = backwardation = stress)
  wing_avg_iv        avg IV of ATM±100 strikes (overall wing height)

All values .shift(1)-able for no-lookahead ML usage; caller is responsible.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from nifty_core.research.iv import implied_vol
from nifty_core.research.probes import (
    BREEZE_OPTIONS_ROOT, atm_strike, available_contracts, load_index_daily,
)

logger = logging.getLogger(__name__)

# Constants — Indian rate regime
RISK_FREE_RATE = 0.065   # 10y G-sec ~6.5%
DIV_YIELD = 0.012         # NIFTY trailing div yield ~1.2%
TRADING_DAYS_PER_YEAR = 252

# NIFTY weekly expiry weekday switched from Thursday (3) to Tuesday (1)
# effective 2025-09-01 (NSE/SEBI). Mirrors nifty_core.core.expiries.NIFTY_EXPIRY_SWITCH.
NIFTY_EXPIRY_SWITCH = date(2025, 9, 1)


def _is_valid_weekly_expiry(exp: date) -> bool:
    """Return True if exp is a Thursday before 2025-09-01 or Tuesday after."""
    expected_weekday = 1 if exp >= NIFTY_EXPIRY_SWITCH else 3
    return exp.weekday() == expected_weekday


def _eod_close(parquet_path: Path) -> float | None:
    """Return the last close in a 1-min option parquet file, or None if empty."""
    try:
        df = pd.read_parquet(parquet_path, columns=["close"])
        if df.empty:
            return None
        return float(df["close"].iloc[-1])
    except Exception:
        return None


def _index_eod_close(symbol: str = "NIFTY_50") -> pd.Series:
    return load_index_daily(symbol)["Close"]


def _build_contract_index() -> dict[tuple[date, int, str], date]:
    """Map (expiry, strike, right) -> expiry date for all available NIFTY contracts."""
    out: dict[tuple[date, int, str], date] = {}
    for exp, strike, right in available_contracts("NIFTY"):
        out[(exp, strike, right)] = exp
    return out


def _option_path(exp: date, strike: int, right: str, day: date) -> Path:
    """right is 'CE'/'PE'; on disk we store CALL/PUT directory names."""
    folder = "CALL" if right == "CE" else "PUT"
    return BREEZE_OPTIONS_ROOT / f"NIFTY_{exp:%Y%m%d}_{strike}_{folder}" / f"{day:%Y%m%d}.parquet"


def _iv_at(day: date, exp: date, strike: int, right: str, spot: float) -> float | None:
    """Compute EOD IV for one (day, expiry, strike, right) cell. None if missing/unsolvable."""
    p = _option_path(exp, strike, right, day)
    if not p.exists():
        return None
    px = _eod_close(p)
    if px is None or px <= 0:
        return None
    # Trading-day fraction of year. Floor at 1 to avoid 0-DTE singularity.
    dte = max((exp - day).days, 1)
    T = dte / TRADING_DAYS_PER_YEAR
    return implied_vol(
        price=px, S=spot, K=float(strike), T=T,
        r=RISK_FREE_RATE, q=DIV_YIELD, right=right,
    )


def _weekly_expiries_by_day(days: list[date]) -> dict[date, list[date]]:
    """For each trading day, list the available weekly expiries on/after that day,
    sorted earliest-first.

    Filters to only "valid" weekly expiries (Thursday before 2025-09-01, Tuesday
    after) — without this, stale phantom contract directories from a prior bad
    backfill leak into the surface and cause IV lookups to return None.
    """
    contracts = available_contracts("NIFTY")
    all_expiries = sorted({exp for exp, _, _ in contracts if _is_valid_weekly_expiry(exp)})
    out: dict[date, list[date]] = {}
    for d in days:
        out[d] = [e for e in all_expiries if e >= d][:5]  # at most 5 weeklies ahead
    return out


def build_vol_surface_features(
    start: date,
    end: date,
    *,
    step: int = 50,
    wing_distance: int = 100,
) -> pd.DataFrame:
    """Construct per-day vol-surface feature dataframe in [start, end].

    Returns DataFrame indexed by date with columns:
        atm_iv, atm_iv_chg_1d, put_skew, call_skew, pc_skew_asym,
        term_slope, wing_avg_iv
    """
    nifty_close = _index_eod_close("NIFTY_50")
    # Coerce to date->float dict for cheap lookup independent of index dtype
    nifty_by_date: dict[date, float] = {
        ts.date(): float(v) for ts, v in nifty_close.items()
    }
    trading_days = [d.date() for d in pd.bdate_range(start, end)]
    expiries_by_day = _weekly_expiries_by_day(trading_days)

    rows: list[dict] = []
    for d in trading_days:
        spot = nifty_by_date.get(d)
        if spot is None:
            continue
        atm = atm_strike(spot, step=step)

        weekly_list = expiries_by_day.get(d, [])
        if not weekly_list:
            continue
        front = weekly_list[0]
        # Skip days where the only expiry is today (0 DTE) — IV unreliable
        if front == d and len(weekly_list) >= 2:
            front = weekly_list[1]
        next_exp = weekly_list[1] if len(weekly_list) >= 2 else None
        if next_exp == d and len(weekly_list) >= 3:
            next_exp = weekly_list[2]

        # ATM IV (use CE + PE average to denoise put-call parity)
        iv_atm_ce = _iv_at(d, front, atm, "CE", spot)
        iv_atm_pe = _iv_at(d, front, atm, "PE", spot)
        if iv_atm_ce is None and iv_atm_pe is None:
            continue
        atm_iv = (
            (iv_atm_ce + iv_atm_pe) / 2
            if iv_atm_ce is not None and iv_atm_pe is not None
            else (iv_atm_ce or iv_atm_pe)
        )

        # Wing IVs
        iv_otm_put = _iv_at(d, front, atm - wing_distance, "PE", spot)
        iv_otm_call = _iv_at(d, front, atm + wing_distance, "CE", spot)

        put_skew = (iv_otm_put - (iv_atm_pe or atm_iv)) if iv_otm_put is not None else None
        call_skew = (iv_otm_call - (iv_atm_ce or atm_iv)) if iv_otm_call is not None else None
        pc_skew_asym = (
            (put_skew - call_skew)
            if put_skew is not None and call_skew is not None
            else None
        )

        wing_avg_iv = None
        if iv_otm_put is not None and iv_otm_call is not None:
            wing_avg_iv = (iv_otm_put + iv_otm_call) / 2

        # Term structure: next-week ATM IV minus this-week ATM IV
        term_slope = None
        if next_exp is not None:
            iv_next_ce = _iv_at(d, next_exp, atm, "CE", spot)
            iv_next_pe = _iv_at(d, next_exp, atm, "PE", spot)
            if iv_next_ce is not None and iv_next_pe is not None:
                next_atm_iv = (iv_next_ce + iv_next_pe) / 2
                term_slope = next_atm_iv - atm_iv

        rows.append({
            "date": pd.Timestamp(d),
            "atm_iv": atm_iv,
            "put_skew": put_skew,
            "call_skew": call_skew,
            "pc_skew_asym": pc_skew_asym,
            "term_slope": term_slope,
            "wing_avg_iv": wing_avg_iv,
            "nifty_close": spot,
            "atm_strike": atm,
            "front_expiry": front,
            "next_expiry": next_exp,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df["atm_iv_chg_1d"] = df["atm_iv"].diff() * 100  # change in IV-points (e.g. +2 = +200 bps)
    return df
