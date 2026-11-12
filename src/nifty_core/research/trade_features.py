"""
Build per-trade entry-time features for ML filtering of iron-condor entries.

For each trade (opened on day D at minute M), construct a feature vector using
ONLY information available before market open on day D — i.e. closing data from
day D-1. Combines:
  - Baseline market features (VIX, NIFTY momentum, day-of-week, PCR)
  - Vol-surface features (ATM IV, put skew, call skew, term slope)

The target for each trade is binary: close_pnl > 0.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from nifty_core.research.probes import NSELIB_BHAV_ROOT, load_index_daily
from nifty_core.research.vol_surface import build_vol_surface_features


VOL_SURFACE_COLS = [
    "atm_iv", "atm_iv_chg_5d", "iv_rank_252d", "put_skew", "call_skew",
    "pc_skew_asym", "term_slope", "wing_avg_iv",
]

# Candidate vol-surface extensions (added 2026-05-31, derived from existing cols).
# Kept separate so feature_lab can A/B them vs the proven 8-column base.
# (atm_iv_chg_5d + iv_rank_252d promoted into VOL_SURFACE_COLS after feature_lab
# forward-selection showed them as top-2 most-impactful new features.)
VOL_SURFACE_EXT_COLS = [
    "atm_iv_chg_1d",         # 1d change in ATM IV (was in core; demoted in favor of 5d)
    "atm_iv_chg_20d",        # 20d change in ATM IV
    "vrp_realized",          # IV premium over realized: atm_iv*100 - rv_5d
    "put_skew_chg_5d",       # 5d change in put_skew (skew velocity)
    "term_slope_chg_5d",     # 5d change in term_slope
    "wing_avg_iv_chg_5d",    # 5d change in average wing IV
]

BASELINE_COLS = [
    "vix_prev", "vix_chg_1d", "vix_chg_5d",
    "nifty_ret_1d", "nifty_ret_5d", "nifty_ret_10d", "nifty_ret_20d",
    "rv_5d", "rv_20d", "prev_up",
    "is_dow_0", "is_dow_1", "is_dow_2", "is_dow_3", "is_dow_4",
    "nifty_above_sma20", "pcr_eod",
]

ALL_FEATURE_COLS = BASELINE_COLS + VOL_SURFACE_COLS + VOL_SURFACE_EXT_COLS


def _compute_pcr_series() -> pd.Series:
    out = {}
    for p in NSELIB_BHAV_ROOT.glob("*.parquet"):
        try:
            d = pd.to_datetime(p.stem, format="%Y%m%d")
            bhav = pd.read_parquet(p, columns=["TckrSymb", "OptnTp", "OpnIntrst", "FinInstrmTp"])
            nifty_opt = bhav[(bhav["TckrSymb"] == "NIFTY") & (bhav["FinInstrmTp"] == "IDO")]
            put_oi = nifty_opt[nifty_opt["OptnTp"] == "PE"]["OpnIntrst"].astype(float).sum()
            call_oi = nifty_opt[nifty_opt["OptnTp"] == "CE"]["OpnIntrst"].astype(float).sum()
            if call_oi > 0:
                out[d] = put_oi / call_oi
        except Exception:
            continue
    return pd.Series(out, name="pcr_eod").sort_index()


def build_daily_feature_panel(start: date, end: date) -> pd.DataFrame:
    """Build a daily feature panel covering [start, end].

    Each row is one trading day. All features computed from prior-day data
    (shift(1)). Caller looks up trade entry date in the index to get features.

    Index is normalized to midnight to ensure consistent alignment across the
    yfinance daily series (which may carry tz-offset time components) and the
    vol-surface dates (which are pure dates).
    """
    nifty = load_index_daily("NIFTY_50")["Close"]
    vix = load_index_daily("INDIA_VIX")["Close"]
    df = pd.DataFrame({"nifty": nifty, "vix": vix}).dropna()
    df.index = pd.to_datetime(df.index).normalize()
    df["vix_prev"] = df["vix"].shift(1)
    df["vix_chg_1d"] = df["vix"].pct_change(1).shift(1) * 100
    df["vix_chg_5d"] = df["vix"].pct_change(5).shift(1) * 100
    df["nifty_ret_1d"] = df["nifty"].pct_change(1).shift(1) * 100
    df["nifty_ret_5d"] = df["nifty"].pct_change(5).shift(1) * 100
    df["nifty_ret_10d"] = df["nifty"].pct_change(10).shift(1) * 100
    df["nifty_ret_20d"] = df["nifty"].pct_change(20).shift(1) * 100
    df["rv_5d"] = (df["nifty"].pct_change(1).rolling(5).std() * (252 ** 0.5) * 100).shift(1)
    df["rv_20d"] = (df["nifty"].pct_change(1).rolling(20).std() * (252 ** 0.5) * 100).shift(1)
    df["prev_up"] = (df["nifty"].pct_change(1).shift(1) > 0).astype(int)
    df["dow"] = df.index.weekday
    for d in range(5):
        df[f"is_dow_{d}"] = (df["dow"] == d).astype(int)
    df["nifty_above_sma20"] = (
        (df["nifty"] > df["nifty"].rolling(20).mean()).shift(1).astype("Int64")
    )

    pcr = _compute_pcr_series()
    pcr.index = pd.to_datetime(pcr.index).normalize()
    df = df.join(pcr, how="left")
    df["pcr_eod"] = df["pcr_eod"].shift(1)

    # Vol surface (already shifted internally vs entry day via its own .shift(1))
    vs = build_vol_surface_features(start, end)
    if not vs.empty:
        vs.index = pd.to_datetime(vs.index).normalize()
        for col in VOL_SURFACE_COLS:
            if col in vs.columns:
                df[col] = vs[col].shift(1)

        # ---- Vol-surface EXTENSIONS (cheap derivatives of the base columns) ----
        # All computed from the raw vs frame then shift(1) for no-lookahead.
        df["atm_iv_chg_1d"] = (vs["atm_iv"].diff(1) * 100).shift(1)
        df["atm_iv_chg_5d"] = (vs["atm_iv"].diff(5) * 100).shift(1)
        df["atm_iv_chg_20d"] = (vs["atm_iv"].diff(20) * 100).shift(1)
        # 252-day rolling IV rank, normalized 0..1 (excludes today via shift)
        iv_rank = vs["atm_iv"].rolling(252, min_periods=60).apply(
            lambda s: (s.rank(pct=True).iloc[-1]), raw=False,
        )
        df["iv_rank_252d"] = iv_rank.shift(1)
        # VRP: IV minus realized (both in % space). Compute on raw (unshifted)
        # series, then shift(1) so the trade-day row gets t-1's VRP (no lookahead).
        rv_5d_raw = (df["nifty"].pct_change(1).rolling(5).std() * (252 ** 0.5) * 100)
        atm_iv_pct = vs["atm_iv"].reindex(df.index) * 100
        df["vrp_realized"] = (atm_iv_pct - rv_5d_raw).shift(1)
        # Skew/term-slope velocity (5d)
        df["put_skew_chg_5d"] = (vs["put_skew"].diff(5) * 100).shift(1)
        df["term_slope_chg_5d"] = (vs["term_slope"].diff(5) * 100).shift(1)
        df["wing_avg_iv_chg_5d"] = (vs["wing_avg_iv"].diff(5) * 100).shift(1)
    return df.loc[pd.Timestamp(start).normalize():pd.Timestamp(end).normalize()]


def attach_features_to_trades(
    trades: pd.DataFrame,
    feature_panel: pd.DataFrame,
    *,
    target_col: str = "pnl_inr",
) -> pd.DataFrame:
    """Join feature panel onto trades by entry date. Adds 'win' target column."""
    trades = trades.copy()
    trades["opened_at"] = pd.to_datetime(trades["opened_at"])
    trades["entry_date"] = trades["opened_at"].dt.normalize()
    panel = feature_panel.copy()
    panel.index = pd.to_datetime(panel.index).normalize()
    merged = trades.merge(
        panel[ALL_FEATURE_COLS], left_on="entry_date", right_index=True, how="left"
    )
    merged["win"] = (merged[target_col] > 0).astype(int)
    return merged
