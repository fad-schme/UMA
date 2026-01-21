"""
Generic retry helpers for sync operations.

Used for external dependencies where transient failures are expected
(graph drivers, vector services).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_sync(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 8.0,
    jitter: float = 0.1,
) -> T:
    """
    Retry a synchronous function with exponential backoff.

    Raises the last exception on final failure.
    """
    delay = initial_delay
    attempt = 1
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_attempts:
                logger.exception("retry_sync failed after %d attempts.", attempt)
                raise
            sleep_for = delay + random.uniform(0, jitter)
            logger.warning(
                "retry_sync: attempt %d/%d failed (%s). Retrying in %.2fs.",
                attempt,
                max_attempts,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
            delay = min(delay * backoff_factor, max_delay)
            attempt += 1
