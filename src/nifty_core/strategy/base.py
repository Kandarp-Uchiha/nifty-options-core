"""
Strategy base class — pure event-driven, source-agnostic (Rule G9).

A strategy receives:
  - on_bar(BarEvent)         per-leg minute bars
  - on_clock(ClockEvent)     wall-clock heartbeats (used for time-stop, regime checks)
  - on_fill(FillEvent)       confirmed fills from broker

A strategy emits:
  - returns list[OrderIntent] from each callback (may be empty)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from nifty_core.core.events import BarEvent, ClockEvent, FillEvent, OrderIntent


class Strategy(ABC):
    """Base class for all strategies (iron condor, iron fly, etc.)."""

    name: str = "abstract"
    algo_id: str = ""  # must be set by subclass; enforced by Risk Manager

    @abstractmethod
    def on_bar(self, event: BarEvent) -> Iterable[OrderIntent]:
        """Called for every option-contract minute bar visible to this strategy."""
        ...

    @abstractmethod
    def on_clock(self, event: ClockEvent) -> Iterable[OrderIntent]:
        """Called every wall-clock minute (used for time-stop, regime re-check)."""
        ...

    def on_fill(self, event: FillEvent) -> Iterable[OrderIntent]:
        """Default: no reaction. Subclasses override if they need to chain orders."""
        return ()
