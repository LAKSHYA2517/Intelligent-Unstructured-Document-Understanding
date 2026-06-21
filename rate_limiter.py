import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class APIBudget:
    """
    Tracks the request budget for a single API provider.

    Free tier limits (as of 2024):
    Groq llama-3.1-8b-instant:  30 req/min, 14,400 req/day
    Groq llama-4-scout (vision): 15 req/min, 500 req/day
    Gemini Flash free:           15 req/min, 1,500 req/day
    """
    requests_per_minute: int
    requests_per_day:    int
    min_gap_seconds:     float

    burst_window_seconds: float = 10.0
    burst_max_fraction:   float = 0.5


API_BUDGETS = {
    "groq_text":   APIBudget(
        requests_per_minute=28,   # Leave 2 as buffer
        requests_per_day=14000,
        min_gap_seconds=2.2       # 60/28 = ~2.14s minimum gap
    ),
    "groq_vision": APIBudget(
        requests_per_minute=12,   # Leave 3 as buffer
        requests_per_day=450,
        min_gap_seconds=5.0       # 60/12 = 5s minimum gap
    ),
    "gemini_flash": APIBudget(
        requests_per_minute=13,
        requests_per_day=1400,
        min_gap_seconds=4.7
    ),
}

class RateLimitBudgetManager:
    def __init__(self, budget: APIBudget, provider_name: str):
        self.budget        = budget
        self.provider_name = provider_name
        self.logger        = logging.getLogger(__name__)
        self._lock         = threading.Lock()

        self._tokens       = float(budget.requests_per_minute)
        self._max_tokens   = float(budget.requests_per_minute)
        self._refill_rate  = budget.requests_per_minute / 60.0
        self._last_refill  = time.monotonic()

        self._request_times: deque = deque()

        self._daily_count  = 0
        self._day_start    = time.time()

        self._backoff_until: float = 0.0
        self._consecutive_429s: int = 0

    def _refill_tokens(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._refill_rate
        self._tokens = min(self._max_tokens, self._tokens + new_tokens)
        self._last_refill = now

    def _reset_daily_if_needed(self) -> None:
        now = time.time()
        if now - self._day_start > 86400:
            self._daily_count = 0
            self._day_start = now

    def wait_for_slot(self) -> float:
        with self._lock:
            self._reset_daily_if_needed()

            if self._daily_count >= self.budget.requests_per_day:
                raise RuntimeError(
                    f"{self.provider_name} daily budget exhausted "
                    f"({self._daily_count}/{self.budget.requests_per_day}). "
                    f"Retry tomorrow."
                )

            wait_start = time.monotonic()

            if time.monotonic() < self._backoff_until:
                sleep_time = self._backoff_until - time.monotonic()
                self.logger.info(
                    f"{self.provider_name}: In backoff, sleeping {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)

            self._refill_tokens()

            while self._tokens < 1.0:
                wait_needed = (1.0 - self._tokens) / self._refill_rate
                self.logger.debug(
                    f"{self.provider_name}: Token bucket empty, waiting {wait_needed:.2f}s"
                )
                time.sleep(wait_needed)
                self._refill_tokens()

            now = time.monotonic()
            if self._request_times:
                last_request = self._request_times[-1]
                gap = now - last_request
                if gap < self.budget.min_gap_seconds:
                    sleep_needed = self.budget.min_gap_seconds - gap
                    time.sleep(sleep_needed)

            self._tokens -= 1.0
            self._request_times.append(time.monotonic())
            self._daily_count += 1

            cutoff = time.monotonic() - 60.0
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()

            total_waited = time.monotonic() - wait_start
            return total_waited

    def register_429(self, retry_after_seconds: Optional[float] = None) -> None:
        with self._lock:
            self._consecutive_429s += 1
            if retry_after_seconds:
                backoff = retry_after_seconds + 5.0
            else:
                backoff = 30.0 * (2 ** min(self._consecutive_429s - 1, 3))

            self._backoff_until = time.monotonic() + backoff
            self._tokens = 0.0

            self.logger.warning(
                f"{self.provider_name}: Received 429 "
                f"(consecutive: {self._consecutive_429s}). "
                f"Backing off {backoff:.0f}s"
            )

    def register_success(self) -> None:
        with self._lock:
            self._consecutive_429s = 0

    @property
    def status(self) -> dict:
        with self._lock:
            self._refill_tokens()
            return {
                "provider":       self.provider_name,
                "tokens_available": round(self._tokens, 2),
                "daily_used":     self._daily_count,
                "daily_limit":    self.budget.requests_per_day,
                "daily_remaining": self.budget.requests_per_day - self._daily_count,
                "in_backoff":     time.monotonic() < self._backoff_until
            }
