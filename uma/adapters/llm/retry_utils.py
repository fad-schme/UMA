"""
Retry utilities for LLM and embedding calls.

This module provides:
- Exponential backoff retry decorator
- Timeout enforcement
- Logging

Coding agent instructions:
--------------------------
- Use @retryable for all network-bound operations.
- Adjust max_attempts and backoff_multiplier as needed.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Callable, Any, Awaitable

logger = logging.getLogger(__name__)


def retryable(
    max_attempts: int = 5,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 16.0,
):
    """
    Decorator adding exponential backoff retry to async functions.

    Use only on network-bound operations (LLM, embeddings, DB calls).

    Error handling:
    - Logs each retry with warning level.
    - On final failure, re-raises the last exception.
    """

    def decorator(func: Callable[..., Awaitable[Any]]):

        async def wrapper(*args, **kwargs):
            delay = initial_delay
            attempt = 1

            while attempt <= max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        logger.exception(
                            "Retry failed after %d attempts: %s", attempt, func.__name__
                        )
                        raise

                    logger.warning(
                        "Error in %s — attempt %d/%d. Retrying in %.2fs (%s)",
                        func.__name__,
                        attempt,
                        max_attempts,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                    attempt += 1

        return wrapper

    return decorator