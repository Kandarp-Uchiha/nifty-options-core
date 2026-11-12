"""
Backtest metrics — computes the Phase-2 success criteria table.

Per docs/02-hybrid-strategy-design.md "Measurement & success criteria":
  trade_count, win_rate, avg_pnl, median_pnl, max_single_trade_loss,
  profit_factor, sharpe_daily, max_drawdown, calmar_ratio
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from nifty_core.core.events import SpreadPosition


@dataclass
class TradeRow:
    spread_id: str
    opened_at: str
    closed_at: str
    regime: str
    pnl_inr: float
    n_legs: int


@dataclass
class Metrics:
    n_trades: int
    win_rate: float
    avg_pnl_inr: float
    median_pnl_inr: float
    total_pnl_inr: float
    best_pnl_inr: float
    worst_pnl_inr: float
    profit_factor: float
    sharpe_daily: float
    max_drawdown_inr: float
    max_drawdown_pct: float
    calmar_ratio: float
    by_regime: pd.DataFrame
    daily_pnl: pd.Series
    trades: pd.DataFrame


def compute_metrics(
    closed_positions: list[SpreadPosition],
    daily_pnl: dict[date, float],
    starting_capital_inr: float,
) -> Metrics:
    trade_rows = []
    for sp in closed_positions:
        trade_rows.append({
            "spread_id": sp.spread_group_id,
            "opened_at": sp.opened_at.isoformat() if sp.opened_at else "",
            "closed_at": sp.closed_at.isoformat() if sp.closed_at else "",
            "regime": sp.regime_at_entry or "UNK",
            "pnl_inr": float(sp.realised_pnl_inr),
        })
    trades = pd.DataFrame(trade_rows)

    if trades.empty:
        return Metrics(
            n_trades=0, win_rate=0.0, avg_pnl_inr=0.0, median_pnl_inr=0.0,
            total_pnl_inr=0.0, best_pnl_inr=0.0, worst_pnl_inr=0.0,
            profit_factor=0.0, sharpe_daily=0.0,
            max_drawdown_inr=0.0, max_drawdown_pct=0.0, calmar_ratio=0.0,
            by_regime=pd.DataFrame(), daily_pnl=pd.Series(dtype=float), trades=trades,
        )

    pnls = trades["pnl_inr"]
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf = float(wins.sum() / abs(losses.sum())) if not losses.empty and losses.sum() != 0 else float("inf")

    # Daily series
    days = sorted(daily_pnl.keys())
    daily = pd.Series([daily_pnl[d] for d in days], index=pd.to_datetime(days), name="daily_pnl")

    # Sharpe
    if daily.std() > 0:
        sharpe = float(daily.mean() / daily.std() * (252 ** 0.5))
    else:
        sharpe = 0.0

    # Drawdown on cumulative
    cum = daily.cumsum()
    peak = cum.cummax()
    drawdown = cum - peak
    max_dd_inr = float(drawdown.min())
    max_dd_pct = float(max_dd_inr / starting_capital_inr * 100) if starting_capital_inr else 0.0

    # Calmar = annualised return / |max DD %|
    days_elapsed = max((days[-1] - days[0]).days, 1)
    annualised_return_pct = float(cum.iloc[-1] / starting_capital_inr * (365 / days_elapsed) * 100)
    calmar = annualised_return_pct / abs(max_dd_pct) if max_dd_pct != 0 else float("inf")

    # By regime
    by_regime = trades.groupby("regime").agg(
        n=("pnl_inr", "count"),
        win_rate=("pnl_inr", lambda s: (s > 0).mean()),
        avg_pnl_inr=("pnl_inr", "mean"),
        med_pnl_inr=("pnl_inr", "median"),
        best=("pnl_inr", "max"),
        worst=("pnl_inr", "min"),
        total=("pnl_inr", "sum"),
    )

    return Metrics(
        n_trades=len(trades),
        win_rate=float((pnls > 0).mean()),
        avg_pnl_inr=float(pnls.mean()),
        median_pnl_inr=float(pnls.median()),
        total_pnl_inr=float(pnls.sum()),
        best_pnl_inr=float(pnls.max()),
        worst_pnl_inr=float(pnls.min()),
        profit_factor=pf,
        sharpe_daily=sharpe,
        max_drawdown_inr=max_dd_inr,
        max_drawdown_pct=max_dd_pct,
        calmar_ratio=calmar,
        by_regime=by_regime,
        daily_pnl=daily,
        trades=trades,
    )


def print_report(m: Metrics, title: str = "Backtest report") -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    if m.n_trades == 0:
        print("No trades executed.")
        return
    print(f"Trades              : {m.n_trades}")
    print(f"Win rate            : {m.win_rate:.1%}")
    print(f"Avg P&L per trade   : INR {m.avg_pnl_inr:+,.0f}")
    print(f"Median P&L          : INR {m.median_pnl_inr:+,.0f}")
    print(f"Best / Worst        : INR {m.best_pnl_inr:+,.0f} / {m.worst_pnl_inr:+,.0f}")
    print(f"Total P&L           : INR {m.total_pnl_inr:+,.0f}")
    print(f"Profit factor       : {m.profit_factor:.2f}")
    print(f"Daily Sharpe        : {m.sharpe_daily:+.2f}")
    print(f"Max drawdown        : INR {m.max_drawdown_inr:+,.0f}  ({m.max_drawdown_pct:+.2f}% of capital)")
    print(f"Calmar ratio        : {m.calmar_ratio:+.2f}")
    print()
    print("By VIX regime:")
    print(m.by_regime.to_string(float_format=lambda v: f"{v:+.3f}"))
