"""
Minimal Black-Scholes pricing + implied-volatility inversion.

No scipy dependency — uses math.erf for the normal CDF and bisection for IV.
Adequate for daily-snapshot research (a few thousand inversions take milliseconds).

For NIFTY weekly options we use:
  - S       spot (NIFTY index level at evaluation time)
  - K       strike
  - T       time-to-expiry in *trading-days* fraction of year (252-day base)
            (trading-day basis matches IV intuition better than calendar days for
             short-DTE weeklies; the difference for 5 DTE ≈ 25 pp absolute IV.)
  - r       risk-free rate (annualized, decimal). Indian 10y G-sec ~6.5%.
  - q       dividend yield (annualized, decimal). NIFTY ~1.2%.
  - price   observed option premium
  - right   'CE' or 'PE'
"""
from __future__ import annotations

import math


SQRT_2 = math.sqrt(2.0)


def _N(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, right: str) -> float:
    """Black-Scholes-Merton option price (European, continuous dividends)."""
    if T <= 0 or sigma <= 0:
        # Intrinsic value at expiry / degenerate vol
        if right == "CE":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if right == "CE":
        return S * disc_q * _N(d1) - K * disc_r * _N(d2)
    # PE
    return K * disc_r * _N(-d2) - S * disc_q * _N(-d1)


def implied_vol(
    price: float,
    S: float, K: float, T: float, r: float, q: float, right: str,
    *,
    lo: float = 0.001,
    hi: float = 5.0,
    tol: float = 1e-5,
    max_iter: int = 60,
) -> float | None:
    """Solve BS for sigma given observed price. Bisection — robust if a solution exists.

    Returns annualized IV (e.g. 0.15 = 15%), or None if:
      - price < intrinsic (arbitrage / quote error)
      - price > theoretical max (deep ITM with weird quote)
      - solver fails to converge (rare; happens for near-zero time value)
    """
    if T <= 0 or price <= 0:
        return None

    # Intrinsic value (lower bound)
    intrinsic = max(S - K, 0.0) if right == "CE" else max(K - S, 0.0)
    if price < intrinsic - 1e-6:
        return None

    # Theoretical bounds at the search endpoints
    p_lo = bs_price(S, K, T, r, q, lo, right)
    p_hi = bs_price(S, K, T, r, q, hi, right)

    # If our observed price is outside [p_lo, p_hi], no solution in range
    if price <= p_lo:
        return lo
    if price >= p_hi:
        return None

    a, b = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        p_mid = bs_price(S, K, T, r, q, mid, right)
        if abs(p_mid - price) < tol:
            return mid
        if p_mid < price:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)
