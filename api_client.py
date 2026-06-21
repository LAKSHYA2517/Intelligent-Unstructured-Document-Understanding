import json
import re
import time
import logging
from typing import Optional, Callable, Any

from rate_limiter import RateLimitBudgetManager

logger = logging.getLogger(__name__)

def extract_retry_after(error_str: str) -> Optional[float]:
    """
    Parse retry-after time from error message.
    Handles multiple formats:
    - "retry in 45.2s"
    - "retry after 45 seconds"  
    - "rate_limit_exceeded... please retry in 1m30s"
    """
    match = re.search(r'retry in (\d+\.?\d*)s', error_str, re.IGNORECASE)
    if match:
        return float(match.group(1))

    match = re.search(r'retry after (\d+)', error_str, re.IGNORECASE)
    if match:
        return float(match.group(1))

    match = re.search(r'(\d+)m\s*(\d+)s', error_str, re.IGNORECASE)
    if match:
        return float(match.group(1)) * 60 + float(match.group(2))

    return None


def is_rate_limit_error(error: Exception) -> bool:
    """Distinguish rate limit errors from other errors."""
    error_str = str(error).lower()
    return any(signal in error_str for signal in [
        "429", "rate limit", "quota", "too many requests",
        "rate_limit_exceeded", "tokens per minute"
    ])


def is_connection_error(error: Exception) -> bool:
    """Distinguish connection errors from rate limit errors."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    return any(signal in error_str or signal in error_type for signal in [
        "connection", "timeout", "network", "remotedisconnected",
        "remoteprotocolerror", "connecterror", "readtimeout",
        "connectionreset", "brokenpipe"
    ])


class ManagedAPIClient:
    """
    Wraps a raw API client (Groq, Gemini, etc.) with:
    1. Pre-request budget management (prevents hitting limits)
    2. Post-error classification (rate limit vs connection vs other)
    3. Smart retry with correct strategy per error type
    4. Automatic 429 registration with budget manager
    """

    def __init__(
        self,
        client,
        budget_manager: RateLimitBudgetManager,
        max_retries: int = 3,
        connection_retry_delay: float = 10.0
    ):
        self.client          = client
        self.budget          = budget_manager
        self.max_retries     = max_retries
        self.conn_retry_delay = connection_retry_delay
        self.logger          = logging.getLogger(__name__)

    def call(self, api_func: Callable, *args, **kwargs) -> Any:
        """
        Execute an API call with full budget management and retry logic.
        """
        for attempt in range(self.max_retries):

            # STEP 1: Wait for a budget slot BEFORE making the call
            waited = self.budget.wait_for_slot()
            if waited > 1.0:
                self.logger.debug(
                    f"Waited {waited:.1f}s for rate limit slot "
                    f"(attempt {attempt + 1})"
                )

            # STEP 2: Make the API call
            try:
                result = api_func(*args, **kwargs)
                self.budget.register_success()
                return result

            # STEP 3a: Rate limit error — register with budget manager
            except Exception as e:
                if is_rate_limit_error(e):
                    retry_after = extract_retry_after(str(e))
                    self.budget.register_429(retry_after)

                    if attempt == self.max_retries - 1:
                        self.logger.error(
                            f"Rate limit: exhausted {self.max_retries} retries"
                        )
                        raise

                    self.logger.warning(
                        f"Rate limit on attempt {attempt + 1}, "
                        f"budget manager will handle backoff"
                    )
                    # Don't sleep here — budget manager handles it on next iteration
                    continue

                # STEP 3b: Connection error — different strategy
                elif is_connection_error(e):
                    if attempt == self.max_retries - 1:
                        self.logger.error(
                            f"Connection error: exhausted {self.max_retries} retries"
                        )
                        raise

                    self.logger.warning(
                        f"Connection error on attempt {attempt + 1}: {type(e).__name__}. "
                        f"Waiting {self.conn_retry_delay}s before retry"
                    )
                    time.sleep(self.conn_retry_delay)
                    continue

                # STEP 3c: Any other error — don't retry, raise immediately
                else:
                    raise

        raise RuntimeError(f"API call failed after {self.max_retries} attempts")
