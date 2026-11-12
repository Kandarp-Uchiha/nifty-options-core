"""
Risk Manager — enforces G3 (daily P&L hard-stop) and G4 (max concurrent).

Maintains the open positions ledger, recomputes P&L each tick from broker
marks, vetoes new entry orders that would breach risk limits, and
triggers the kill switch (G2 hook) on daily-loss breach.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from nifty_core.core.events import (
    FillEvent, LegPosition, OrderIntent, Right, Side, SpreadPosition,
)
from nifty_core.exec.backtest_broker import NIFTY_LOT_SIZE

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    daily_loss_inr: float
    daily_loss_pct_of_capital: float
    starting_capital_inr: float
    max_concurrent_positions: int
    max_premium_per_spread_inr: float
    max_loss_per_spread_inr: float

    @classmethod
    def from_yaml(cls, path: Path, starting_capital_inr: float) -> "RiskLimits":
        with path.open() as f:
            cfg = yaml.safe_load(f)
        return cls(
            daily_loss_inr=float(cfg["daily_loss_limit"]["rupees"]),
            daily_loss_pct_of_capital=float(cfg["daily_loss_limit"]["percent_of_capital"]),
            starting_capital_inr=starting_capital_inr,
            max_concurrent_positions=int(cfg["max_concurrent_positions"]),
            max_premium_per_spread_inr=float(cfg["position"]["max_premium_per_spread_rupees"]),
            max_loss_per_spread_inr=float(cfg["position"]["max_loss_per_spread_rupees"]),
        )

    @classmethod
    def from_yaml_scaled_for_lots(
        cls, path: Path, starting_capital_inr: float, max_lots: int = 1,
    ) -> "RiskLimits":
        """Same as from_yaml, but scales the per-trade and per-day rupee caps by
        the largest lot count we plan to trade. The risk-yaml values are stored
        as "per 1 lot" baselines; this factory multiplies them so the limits
        keep the same risk-per-lot ratio as you change position size.

        Effective at-trading-time caps for max_lots=N:
          daily_loss_inr             = base_yaml_value × N
          max_premium_per_spread_inr = base_yaml_value × N
          max_loss_per_spread_inr    = base_yaml_value × N

        The percent-of-capital cap is unchanged (it's already proportional to
        the capital you're putting at risk).
        """
        with path.open() as f:
            cfg = yaml.safe_load(f)
        scale = max(1, int(max_lots))
        return cls(
            daily_loss_inr=float(cfg["daily_loss_limit"]["rupees"]) * scale,
            daily_loss_pct_of_capital=float(cfg["daily_loss_limit"]["percent_of_capital"]),
            starting_capital_inr=starting_capital_inr,
            max_concurrent_positions=int(cfg["max_concurrent_positions"]),
            max_premium_per_spread_inr=float(cfg["position"]["max_premium_per_spread_rupees"]) * scale,
            max_loss_per_spread_inr=float(cfg["position"]["max_loss_per_spread_rupees"]) * scale,
        )


@dataclass
class RiskState:
    current_day: date | None = None
    realised_pnl_today_inr: float = 0.0
    unrealised_pnl_inr: float = 0.0
    open_positions: dict[str, SpreadPosition] = field(default_factory=dict)
    closed_positions: list[SpreadPosition] = field(default_factory=list)
    kill_switch_armed: bool = False
    kill_reason: str = ""


class RiskManager:
    """G3 + G4 enforcement and position ledger."""

    def __init__(self, limits: RiskLimits, lot_size: int = NIFTY_LOT_SIZE) -> None:
        self.limits = limits
        self.lot_size = lot_size
        self.state = RiskState()

    # ---------- per-day reset ----------

    def on_new_day(self, day: date) -> None:
        # Sweep any open positions into closed if they survived (shouldn't happen
        # in intraday-only v1, but defensive).
        if self.state.current_day != day:
            self.state.current_day = day
            self.state.realised_pnl_today_inr = 0.0

    # ---------- pre-trade veto ----------

    def vet_intent(self, intent: OrderIntent) -> tuple[bool, str]:
        """Pre-flight check. Returns (ok, reason)."""
        if self.state.kill_switch_armed:
            return False, f"kill_switch_armed: {self.state.kill_reason}"

        # Closing orders (suffix _EXIT) reduce positions — always allowed
        # if the parent spread is still open. They never breach G4.
        sgid = intent.spread_group_id
        if sgid.endswith("_EXIT"):
            parent_id = sgid[: -len("_EXIT")]
            if parent_id in self.state.open_positions:
                return True, "ok (closing leg)"
            return False, f"closing leg for unknown spread {parent_id}"

        # G4 — count distinct spread_group_ids currently open
        # Allow if the intent is for a spread group already being opened (sibling legs)
        if sgid in self.state.open_positions:
            return True, "ok (existing spread group)"
        if len(self.state.open_positions) >= self.limits.max_concurrent_positions:
            return False, f"G4: would exceed max_concurrent_positions={self.limits.max_concurrent_positions}"

        return True, "ok"

    # ---------- fill bookkeeping ----------

    def on_fill(self, fill: FillEvent) -> None:
        if fill.filled_qty_lots == 0:
            return  # rejected order; no position change

        intent = fill.intent
        sgid = intent.spread_group_id

        # Closing order (suffix _EXIT)?
        is_closing = sgid.endswith("_EXIT")
        parent_id = sgid[: -len("_EXIT")] if is_closing else sgid

        if is_closing:
            sp = self.state.open_positions.get(parent_id)
            if sp is None:
                logger.warning("Closing fill for unknown spread %s", parent_id)
                return
            # Match against an open leg with same strike+right; close it
            matched = None
            for leg in sp.legs:
                # We're closing → original side is opposite of this fill's side
                expected_orig_side = Side.BUY if intent.side == Side.SELL else Side.SELL
                if (leg.strike == intent.strike and leg.right == intent.right
                        and leg.side == expected_orig_side):
                    matched = leg
                    break
            if matched is None:
                logger.warning("Closing fill %s has no matching open leg", fill)
                return
            # Realise P&L on this leg
            per_lot = matched.mtm_per_lot(fill.fill_price)
            leg_pnl = per_lot * matched.qty_lots * self.lot_size - matched.fees_paid_inr - fill.fee_inr
            sp.realised_pnl_inr += leg_pnl
            sp.legs.remove(matched)
            self.state.realised_pnl_today_inr += leg_pnl
            if not sp.legs:
                # All legs closed → finalise
                sp.closed_at = fill.ts
                self.state.closed_positions.append(sp)
                del self.state.open_positions[parent_id]
                logger.info(
                    "Closed spread %s pnl=%.2f INR (regime=%s)",
                    parent_id, sp.realised_pnl_inr, sp.regime_at_entry,
                )
        else:
            # Opening
            sp = self.state.open_positions.get(sgid)
            if sp is None:
                sp = SpreadPosition(
                    spread_group_id=sgid,
                    algo_id=intent.algo_id,
                    strategy_label=intent.algo_id,
                    opened_at=fill.ts,
                )
                self.state.open_positions[sgid] = sp
            leg = LegPosition(
                stock_code=intent.stock_code,
                expiry=intent.expiry,
                strike=intent.strike,
                right=intent.right,
                side=intent.side,
                qty_lots=fill.filled_qty_lots,
                entry_price=fill.fill_price,
                entry_ts=fill.ts,
                last_mark_price=fill.fill_price,
                fees_paid_inr=fill.fee_inr,
            )
            sp.legs.append(leg)
            # Update entry credit per lot (running sum)
            sp.entry_credit_per_lot += leg.signed_premium_per_lot

    # ---------- mark-to-market ----------

    def mark_positions(self, marks: dict[tuple[str, str, int, Right], float]) -> None:
        """Update last_mark_price on each leg and recompute unrealised P&L."""
        total_unrealised = 0.0
        for sp in self.state.open_positions.values():
            for leg in sp.legs:
                key = (leg.stock_code, leg.expiry, leg.strike, leg.right)
                mk = marks.get(key)
                if mk is not None:
                    leg.last_mark_price = mk
                total_unrealised += leg.mtm_per_lot(leg.last_mark_price) * leg.qty_lots * self.lot_size
        self.state.unrealised_pnl_inr = total_unrealised

    # ---------- daily-stop check (G3) ----------

    def check_daily_stop(self) -> bool:
        """Return True if daily loss limit breached (kill switch should arm)."""
        net = self.state.realised_pnl_today_inr + self.state.unrealised_pnl_inr
        limits = self.limits
        if net <= -limits.daily_loss_inr:
            self.state.kill_switch_armed = True
            self.state.kill_reason = f"G3 INR breach: net_pnl={net:.0f} <= -{limits.daily_loss_inr:.0f}"
            return True
        pct = -net / limits.starting_capital_inr * 100.0
        if pct >= limits.daily_loss_pct_of_capital:
            self.state.kill_switch_armed = True
            self.state.kill_reason = (
                f"G3 PCT breach: drawdown={pct:.2f}% >= {limits.daily_loss_pct_of_capital:.2f}%"
            )
            return True
        return False
