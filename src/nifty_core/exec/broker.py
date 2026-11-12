"""
Broker abstraction — same interface for backtest, paper, and live.

Rule G9: strategies talk to this interface only. Substituting BacktestBroker
for LiveBroker is the entire difference between backtest and live.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from nifty_core.core.events import FillEvent, OrderIntent


class Broker(ABC):
    """Order placement + fill notification."""

    @abstractmethod
    def submit(self, intent: OrderIntent) -> str:
        """Submit one leg. Returns broker_order_id (synthetic in backtest)."""
        ...

    @abstractmethod
    def drain_fills(self) -> Iterable[FillEvent]:
        """Return any new fills since last call (then clear internal queue)."""
        ...

    @abstractmethod
    def update_mark(self, stock_code: str, expiry: str, strike: int, right, mark: float) -> None:
        """Inform broker of the latest mark for a contract (used for fill sim)."""
        ...
