"""NIFTY weekly-expiry calendar helpers (shared by every strategy)."""
from __future__ import annotations

from datetime import date, timedelta

# NIFTY weekly expiry weekday switched from Thursday (3) to Tuesday (1)
# effective 2025-09-01 (NSE/SEBI).
NIFTY_EXPIRY_SWITCH = date(2025, 9, 1)


def nifty_weekly_weekday(d: date) -> int:
    return 1 if d >= NIFTY_EXPIRY_SWITCH else 3


def list_weekly_expiries(start: date, end: date) -> list[date]:
    """All NIFTY weekly expiry dates in [start, end], honoring the weekday switch."""
    out: list[date] = []
    cur = start
    while cur <= end:
        if cur.weekday() == nifty_weekly_weekday(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def nearest_expiry_on_or_after(day: date, expiries: list[date]) -> date | None:
    for e in expiries:
        if e >= day:
            return e
    return None
