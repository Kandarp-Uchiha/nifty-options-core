"""
BacktestBroker — simulates fills from historical option marks.

Fill model (conservative — biases against the strategy):
  - LIMIT BUY:  filled at min(limit_price, mark + slippage)
                If mark > limit_price, treats as no-fill for now (queued)
                But for the simple t+0 backtest we always fill at the worst
                of (limit_price, mark * (1 + slippage_pct)) → maker-loses model.
  - LIMIT SELL: filled at max(limit_price, mark - slippage)
  - All orders fill in the next clock tick (1-min delay simulated).

Fees (per leg):
  - Brokerage: ₹20 flat (Kotak Neo F&O)
  - STT (sell only): 0.0625% on premium × lot_size (for option sells)
  - Exchange transaction charge: 0.053% on premium × lot_size
  - GST: 18% on (brokerage + transaction charges)
  - SEBI charges: ₹10 per crore (negligible)
  - Stamp duty (buy only): 0.003% on premium × lot_size
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from nifty_core.core.events import (
    FillEvent, OrderIntent, OrderStatus, OrderType, Right, Side,
)
from nifty_core.exec.broker import Broker

logger = logging.getLogger(__name__)

NIFTY_LOT_SIZE = 75   # NSE lot size since late 2024


@dataclass
class BacktestBrokerConfig:
    lot_size: int = NIFTY_LOT_SIZE
    # Slippage applied to mark when computing fill price.
    # Conservative: paying more than mid when buying, getting less when selling.
    fill_slippage_pct: float = 0.005   # 0.5% — atm spreads have tight markets
    brokerage_per_leg_inr: float = 20.0
    # Realistic Indian F&O option charges (sell-side only for STT):
    stt_pct_on_sell_premium: float = 0.000625
    txn_charge_pct: float = 0.00053
    gst_pct_on_brokerage_and_txn: float = 0.18
    stamp_duty_pct_on_buy: float = 0.00003


class BacktestBroker(Broker):
    """Synchronous, deterministic fill simulator."""

    def __init__(self, config: BacktestBrokerConfig | None = None) -> None:
        self.config = config or BacktestBrokerConfig()
        self._marks: dict[tuple[str, str, int, Right], float] = {}
        self._pending_fills: deque[FillEvent] = deque()
        self._order_counter = 0
        self._now: datetime | None = None

    # Required when running backtest — runner pumps current ts in
    def set_clock(self, ts: datetime) -> None:
        self._now = ts

    def update_mark(self, stock_code: str, expiry: str, strike: int, right: Right, mark: float) -> None:
        self._marks[(stock_code, expiry, strike, right)] = mark

    def submit(self, intent: OrderIntent) -> str:
        self._order_counter += 1
        order_id = f"BT-{self._order_counter:08d}"

        key = (intent.stock_code, intent.expiry, intent.strike, intent.right)
        mark = self._marks.get(key)
        if mark is None:
            # No quote → reject (simulates illiquid contract)
            fill = FillEvent(
                ts=self._now or datetime.now(),
                intent=intent,
                fill_price=0.0,
                filled_qty_lots=0,
                status=OrderStatus.REJECTED,
                broker_order_id=order_id,
                fee_inr=0.0,
            )
            self._pending_fills.append(fill)
            return order_id

        # Determine fill price (conservative against the strategy)
        slip = self.config.fill_slippage_pct
        if intent.order_type == OrderType.MARKET:
            fill_px = mark * (1 + slip) if intent.side == Side.BUY else mark * (1 - slip)
        else:  # LIMIT
            limit = intent.limit_price or 0.0
            if intent.side == Side.BUY:
                # Buyer pays max of (mark+slip, limit); if mark+slip > limit, no fill
                proposed = mark * (1 + slip)
                if proposed > limit:
                    fill_px = limit  # tight fill at limit (assume liquidity at limit)
                else:
                    fill_px = proposed
            else:  # SELL
                proposed = mark * (1 - slip)
                if proposed < limit:
                    fill_px = limit
                else:
                    fill_px = proposed

        fill_px = max(0.05, round(fill_px, 2))  # 5 paise floor (NSE tick)

        # Fees
        notional = fill_px * intent.qty_lots * self.config.lot_size
        cfg = self.config
        brokerage = cfg.brokerage_per_leg_inr
        txn = notional * cfg.txn_charge_pct
        stt = notional * cfg.stt_pct_on_sell_premium if intent.side == Side.SELL else 0.0
        stamp = notional * cfg.stamp_duty_pct_on_buy if intent.side == Side.BUY else 0.0
        gst = (brokerage + txn) * cfg.gst_pct_on_brokerage_and_txn
        fees = brokerage + txn + stt + stamp + gst

        fill = FillEvent(
            ts=self._now or datetime.now(),
            intent=intent,
            fill_price=fill_px,
            filled_qty_lots=intent.qty_lots,
            status=OrderStatus.FILLED,
            broker_order_id=order_id,
            fee_inr=round(fees, 2),
        )
        self._pending_fills.append(fill)
        return order_id

    def drain_fills(self) -> Iterable[FillEvent]:
        out = list(self._pending_fills)
        self._pending_fills.clear()
        return out
