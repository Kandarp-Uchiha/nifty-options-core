import math

import pytest

from nifty_core.research.iv import bs_price, implied_vol

S, K, T, R, Q = 23_500.0, 23_500.0, 7 / 365, 0.065, 0.0


@pytest.mark.parametrize("right", ["CE", "PE"])
@pytest.mark.parametrize("sigma", [0.08, 0.14, 0.35])
def test_implied_vol_round_trips_bs_price(right, sigma):
    price = bs_price(S, K, T, R, Q, sigma, right)
    assert implied_vol(price, S, K, T, R, Q, right) == pytest.approx(sigma, abs=1e-3)


def test_put_call_parity():
    c = bs_price(S, 23_600.0, T, R, Q, 0.15, "CE")
    p = bs_price(S, 23_600.0, T, R, Q, 0.15, "PE")
    assert c - p == pytest.approx(S * math.exp(-Q * T) - 23_600.0 * math.exp(-R * T), abs=1e-6)


def test_price_below_intrinsic_has_no_iv():
    assert implied_vol(50.0, 24_000.0, 23_800.0, T, R, Q, "CE") is None


def test_expired_option_prices_at_intrinsic():
    assert bs_price(24_000.0, 23_800.0, 0.0, R, Q, 0.2, "CE") == 200.0
    assert bs_price(24_000.0, 23_800.0, 0.0, R, Q, 0.2, "PE") == 0.0
