"""
Token-bucket rate limiter — thread-safe, monotonic-clock based.

Used by the Breeze backfill to enforce a strict calls-per-minute cap across
all worker threads, replacing per-call time.sleep(polite_sleep_s).

Design:
  - Bucket holds up to `capacity` tokens.
  - Tokens refill at `rate_per_sec` continuously.
  - acquire() blocks until at least 1 token is available, then consumes it.
  - Thread-safe via a single mutex; tokens computed lazily on each acquire.

Why this and not asyncio: the Breeze SDK is sync (requests-based); a thread
pool over a sync bucket is the simplest model that composes with it.
"""
from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float | None = None) -> None:
        """
        Args:
          rate_per_sec: steady-state token refill rate (e.g. 100/60 ≈ 1.667 for 100/min)
          capacity:     max tokens (= burst budget). Defaults to rate_per_sec * 2.
        """
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity) if capacity is not None else max(1.0, rate_per_sec * 2)
        self._tokens = self.capacity  # start full so the first burst doesn't wait
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> float:
        """Block until n tokens are available, consume them. Return wait time in s."""
        wait_s = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                # Refill
                elapsed = now - self._last
                if elapsed > 0:
                    self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                    self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return wait_s
                # Compute wait until we have n tokens
                deficit = n - self._tokens
                to_sleep = deficit / self.rate
            time.sleep(to_sleep)
            wait_s += to_sleep

    @classmethod
    def per_minute(cls, per_minute: float, capacity: float | None = None) -> "TokenBucket":
        """Convenience constructor: rate specified as tokens-per-minute."""
        return cls(rate_per_sec=per_minute / 60.0, capacity=capacity)
