"""
DryRunBroker — Phase 1 live-trading shim.

Same Broker interface as the real Kotak Neo broker, but:
  - submit() returns a synthetic order_id, never calls a real API
  - drain_fills() simulates an instant fill at the mark + slippage
    (mirroring BacktestBroker's conservative model)
  - Every intent is written to a JSONL log with a timestamp

Purpose: prove the entire pipeline end-to-end
  Strategy \u2192 RiskManager \u2192 OrderRouter \u2192 Broker \u2192 Fill \u2192 PositionTracker
without putting any money at risk. When we're confident the
dry-run logs show sensible decisions and the position bookkeeping
is correct, we swap this for KotakNeoBroker.

Rule G7: orders still carry algo_id and spread_group_id; the
whitelist check still runs in the Order Router upstream.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from nifty_core.core.events import (
    FillEvent, OrderIntent, OrderStatus, OrderType, Right, Side,
)
from nifty_core.exec.broker import Broker

logger = logging.getLogger(__name__)

NIFTY_LOT_SIZE = 75


@dataclass
class DryRunBrokerConfig:
    """Mirrors BacktestBrokerConfig so dry-run P&L \u2248 backtest P&L."""
    lot_size: int = NIFTY_LOT_SIZE
    fill_slippage_pct: float = 0.005
    brokerage_per_leg_inr: float = 20.0
    stt_pct_on_sell_premium: float = 0.000625
    txn_charge_pct: float = 0.00053
    gst_pct_on_brokerage_and_txn: float = 0.18
    stamp_duty_pct_on_buy: float = 0.00003
    # Where to append the intent/fill log (jsonl). One line per event.
    log_path: Path = field(default_factory=lambda: Path("logs/dryrun_orders.jsonl"))


class DryRunBroker(Broker):
    """Live-shaped broker that prints what it WOULD do, places nothing."""

    def __init__(self, config: DryRunBrokerConfig | None = None) -> None:
        self.config = config or DryRunBrokerConfig()
        self._marks: dict[tuple[str, str, int, Right], float] = {}
        self._pending_fills: deque[FillEvent] = deque()
        self._order_counter = 0
        self._now: datetime | None = None
        self.config.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- runner integration ----------

    def set_clock(self, ts: datetime) -> None:
        self._now = ts

    def update_mark(self, stock_code: str, expiry: str, strike: int,
                    right: Right, mark: float) -> None:
        self._marks[(stock_code, expiry, strike, right)] = mark

    # ---------- Broker interface ----------

    def submit(self, intent: OrderIntent) -> str:
        self._order_counter += 1
        order_id = f"DRY-{self._order_counter:08d}"
        now = self._now or datetime.now()

        key = (intent.stock_code, intent.expiry, intent.strike, intent.right)
        mark = self._marks.get(key)

        # Log the intent first (we'd want this in production even on rejection).
        self._log({
            "ts": now.isoformat(timespec="seconds"),
            "event": "INTENT",
            "order_id": order_id,
            "algo_id": intent.algo_id,
            "spread_group_id": intent.spread_group_id,
            "stock": intent.stock_code,
            "expiry": intent.expiry,
            "strike": intent.strike,
            "right": intent.right.value,
            "side": intent.side.value,
            "qty_lots": intent.qty_lots,
            "order_type": intent.order_type.value,
            "limit_price": intent.limit_price,
            "mark_at_submit": mark,
        })

        if mark is None:
            # No mark \u2192 reject (mirrors backtest behaviour on missing data)
            fill = FillEvent(ts=now, intent=intent, fill_price=0.0,
                             filled_qty_lots=0,
                             status=OrderStatus.REJECTED,
                             broker_order_id=order_id, fee_inr=0.0)
            self._pending_fills.append(fill)
            self._log_fill(fill, reason="no mark for contract")
            return order_id

        # Compute conservative fill price (same model as BacktestBroker)
        slip = self.config.fill_slippage_pct
        if intent.order_type == OrderType.MARKET:
            fill_px = mark * (1 + slip) if intent.side == Side.BUY else mark * (1 - slip)
        else:
            limit = intent.limit_price or 0.0
            if intent.side == Side.BUY:
                proposed = mark * (1 + slip)
                fill_px = limit if proposed > limit else proposed
            else:
                proposed = mark * (1 - slip)
                fill_px = limit if proposed < limit else proposed

        # Compute fees (same model as BacktestBroker)
        premium = fill_px * self.config.lot_size * intent.qty_lots
        brokerage = self.config.brokerage_per_leg_inr
        txn = self.config.txn_charge_pct * premium
        gst = self.config.gst_pct_on_brokerage_and_txn * (brokerage + txn)
        stt = (self.config.stt_pct_on_sell_premium * premium) if intent.side == Side.SELL else 0.0
        stamp = (self.config.stamp_duty_pct_on_buy * premium) if intent.side == Side.BUY else 0.0
        fee = brokerage + txn + gst + stt + stamp

        fill = FillEvent(ts=now, intent=intent, fill_price=fill_px,
                         filled_qty_lots=intent.qty_lots,
                         status=OrderStatus.FILLED,
                         broker_order_id=order_id, fee_inr=fee)
        self._pending_fills.append(fill)
        self._log_fill(fill, reason="dry-run instant fill")
        return order_id

    def drain_fills(self) -> Iterable[FillEvent]:
        while self._pending_fills:
            yield self._pending_fills.popleft()

    # ---------- internal ----------

    def _log(self, entry: dict) -> None:
        try:
            with self.config.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except OSError as e:
            logger.error("dry-run log write failed: %s", e)

    def _log_fill(self, fill: FillEvent, reason: str) -> None:
        self._log({
            "ts": fill.ts.isoformat(timespec="seconds"),
            "event": "FILL",
            "order_id": fill.broker_order_id,
            "status": fill.status.value,
            "fill_price": fill.fill_price,
            "filled_qty_lots": fill.filled_qty_lots,
            "fee_inr": round(fill.fee_inr, 2),
            "reason": reason,
        })
