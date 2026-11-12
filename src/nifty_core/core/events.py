"""
Core event-driven DTOs shared by backtest, paper, and live runners.

Rule G9: the strategy module sees the SAME event shapes regardless of source.
Only the event source changes (parquet vs WebSocket vs replay).

Rule G7: OrderIntent.algo_id is non-optional. There is no constructor
path that produces an order without it; this enforces the algo-id
whitelist requirement at build time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal


# ---------------- Enums ----------------

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Right(str, Enum):
    CE = "CE"
    PE = "PE"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"  # only used for emergency unwind (G6)


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


# ---------------- Events ----------------

@dataclass(frozen=True)
class BarEvent:
    """One option-contract minute bar. Source-agnostic."""
    ts: datetime              # IST naive (matches our parquet convention)
    stock_code: str           # "NIFTY"
    expiry: str               # "YYYY-MM-DD" ISO
    strike: int
    right: Right
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: int


@dataclass(frozen=True)
class ClockEvent:
    """Wall-clock heartbeat. Fired every minute even when no bars arrive.
    Used by Risk Manager (P&L recompute) and Trade Manager (time-stop)."""
    ts: datetime


# ---------------- Order intents & fills ----------------

@dataclass(frozen=True)
class OrderIntent:
    """A request from the strategy to place ONE leg of a spread.

    The Order Router groups intents into spread-orders for G6 legging.
    """
    algo_id: str              # G7 — must be in algos.yaml whitelist
    stock_code: str
    expiry: str
    strike: int
    right: Right
    side: Side
    qty_lots: int             # always positive
    order_type: OrderType
    limit_price: float | None  # None for MARKET
    spread_group_id: str       # ties legs together for G6 unwind

    def __post_init__(self) -> None:
        if not self.algo_id:
            raise ValueError("G7 violation: order without algo_id")
        if self.qty_lots <= 0:
            raise ValueError(f"qty_lots must be positive, got {self.qty_lots}")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")


@dataclass(frozen=True)
class FillEvent:
    """A confirmed fill for one leg of one order."""
    ts: datetime
    intent: OrderIntent
    fill_price: float
    filled_qty_lots: int
    status: OrderStatus
    broker_order_id: str
    fee_inr: float = 0.0


# ---------------- Position state ----------------

@dataclass
class LegPosition:
    """One leg of an open spread."""
    stock_code: str
    expiry: str
    strike: int
    right: Right
    side: Side
    qty_lots: int
    entry_price: float
    entry_ts: datetime
    last_mark_price: float    # MtM
    fees_paid_inr: float = 0.0

    @property
    def signed_premium_per_lot(self) -> float:
        """+ve = we received cash (short leg), -ve = we paid (long leg)."""
        return self.entry_price * (1 if self.side == Side.SELL else -1)

    def mtm_per_lot(self, current_price: float) -> float:
        """Mark-to-market P&L per lot at current price."""
        # SELL: profit when price drops (we sold high, can buy back lower)
        # BUY:  profit when price rises
        if self.side == Side.SELL:
            return (self.entry_price - current_price)
        else:
            return (current_price - self.entry_price)


@dataclass
class SpreadPosition:
    """A full open spread (1+ legs). Identified by spread_group_id."""
    spread_group_id: str
    algo_id: str
    strategy_label: str       # human-readable e.g. "iron_condor_LOW_VOL"
    legs: list[LegPosition] = field(default_factory=list)
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    realised_pnl_inr: float = 0.0
    # For metrics:
    regime_at_entry: str | None = None
    entry_credit_per_lot: float = 0.0   # net of all legs

    @property
    def is_open(self) -> bool:
        return self.closed_at is None and bool(self.legs)

    def net_premium_per_lot(self, marks: dict[tuple[int, Right], float]) -> float:
        """Signed net premium at current marks. +ve = net credit position
        (collected cash); MtM gain when this shrinks below entry credit."""
        total = 0.0
        for leg in self.legs:
            mark = marks.get((leg.strike, leg.right), leg.last_mark_price)
            sign = 1 if leg.side == Side.SELL else -1
            total += mark * sign
        return total

    def mtm_inr(self, marks: dict[tuple[int, Right], float], lot_size: int) -> float:
        """Total MtM P&L in rupees (sum across all legs × lot size × qty)."""
        total_per_lot = 0.0
        for leg in self.legs:
            mark = marks.get((leg.strike, leg.right), leg.last_mark_price)
            total_per_lot += leg.mtm_per_lot(mark) * leg.qty_lots
        return total_per_lot * lot_size - sum(l.fees_paid_inr for l in self.legs)
