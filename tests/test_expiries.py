from datetime import date

from nifty_core.core.expiries import (
    NIFTY_EXPIRY_SWITCH,
    list_weekly_expiries,
    nearest_expiry_on_or_after,
    nifty_weekly_weekday,
)


def test_weekday_switches_from_thursday_to_tuesday():
    assert NIFTY_EXPIRY_SWITCH == date(2025, 9, 1)
    assert nifty_weekly_weekday(date(2025, 8, 28)) == 3
    assert nifty_weekly_weekday(date(2025, 9, 1)) == 1


def test_list_weekly_expiries_spans_the_switch():
    assert list_weekly_expiries(date(2025, 8, 25), date(2025, 9, 12)) == [
        date(2025, 8, 28),  # Thursday (old regime)
        date(2025, 9, 2),   # Tuesday (new regime)
        date(2025, 9, 9),
    ]


def test_nearest_expiry_on_or_after():
    exps = [date(2026, 1, 6), date(2026, 1, 13)]
    assert nearest_expiry_on_or_after(date(2026, 1, 6), exps) == date(2026, 1, 6)
    assert nearest_expiry_on_or_after(date(2026, 1, 7), exps) == date(2026, 1, 13)
    assert nearest_expiry_on_or_after(date(2026, 1, 14), exps) is None
