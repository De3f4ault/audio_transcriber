"""LLM Circuit Breaker and Retry mechanism.

Provides an exponential backoff loop and Circuit Breaker to prevent cascading
failures when external or local LLMs (like Ollama or Gemini) become unresponsive
or return continuous 429 Too Many Requests errors.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger("memory.llm_caller")

T = TypeVar("T")


class RateLimitError(Exception):
    """Raised when an API rate limit (e.g., 429) is encountered."""
    pass


class CircuitBreaker:
    """A circuit breaker to prevent repeated calls to a failing service."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = 0.0

    def call(self, fn: Callable[[], T]) -> T:
        """Call the given function, respecting the circuit breaker state."""
        if self.state == "OPEN":
            if time.monotonic() - self.last_failure_time > self.recovery_timeout:
                # Transition to HALF_OPEN to allow one probe
                self.state = "HALF_OPEN"
                logger.info("CircuitBreaker transitioning to HALF_OPEN state.")
            else:
                raise RuntimeError("Circuit breaker is OPEN. Fast-failing request.")

        try:
            result = fn()
        except Exception as exc:
            self._record_failure()
            raise exc

        self._record_success()
        return result

    def _record_failure(self) -> None:
        """Record a failure and potentially trip the circuit."""
        self.failures += 1
        self.last_failure_time = time.monotonic()
        
        if self.state == "HALF_OPEN":
            # Probe failed, trip immediately
            self.state = "OPEN"
            logger.warning("CircuitBreaker probe failed. Transitioning back to OPEN state.")
        elif self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning("CircuitBreaker threshold reached (%d). Transitioning to OPEN state.", self.failures)

    def _record_success(self) -> None:
        """Record a success and close the circuit if necessary."""
        if self.state == "HALF_OPEN" or self.failures > 0:
            self.state = "CLOSED"
            self.failures = 0
            logger.info("CircuitBreaker success. Transitioning to CLOSED state.")


def _retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    jitter: bool = False,
) -> T:
    """Execute a function with exponential backoff on RateLimitError."""
    import random

    retries = 0
    while True:
        try:
            return fn()
        except RateLimitError as exc:
            if retries >= max_retries:
                logger.error("Max retries (%d) exceeded after RateLimitError.", max_retries)
                raise exc
            
            delay = base_delay * (2 ** retries)
            if jitter:
                delay = delay * random.uniform(0.5, 1.5)
                
            retries += 1
            logger.warning("Rate limit hit. Retrying in %.2fs (attempt %d/%d).", delay, retries, max_retries)
            time.sleep(delay)
