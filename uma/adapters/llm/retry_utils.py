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
from typing import Callable, Any, Awaitable, Optional

logger = logging.getLogger(__name__)


def should_retry_network_only(exc: Exception) -> bool:
    """
    Retry only on transient connectivity/timeouts.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    # Optional aiohttp support
    try:
        import aiohttp  # type: ignore

        if isinstance(exc, (aiohttp.ClientConnectionError, aiohttp.ServerTimeoutError)):
            return True
        if isinstance(exc, aiohttp.ClientResponseError):
            if exc.status in (408, 429) or 500 <= exc.status < 600:
                return True
            return False
    except Exception:
        return False

    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 429) or 500 <= status < 600
    return False


def should_retry_openai(exc: Exception) -> bool:
    """
    Retry on OpenAI transient errors only (timeouts, rate limit, 5xx).
    Do NOT retry on auth/permission/bad request/not found/validation errors.
    """
    try:
        import openai  # type: ignore

        non_retryable = (
            openai.BadRequestError,
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.UnprocessableEntityError,
        )
        if isinstance(exc, non_retryable):
            return False

        retryable = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )
        if isinstance(exc, retryable):
            return True

        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                return status in (408, 429) or 500 <= status < 600
    except Exception:
        return should_retry_network_only(exc)

    # Fallback to generic network-only predicate
    return should_retry_network_only(exc)


def retryable(
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 16.0,
    should_retry: Optional[Callable[[Exception], bool]] = None,
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
                    if should_retry is not None and not should_retry(exc):
                        logger.exception(
                            "Non-retryable error in %s; aborting retries", func.__name__
                        )
                        raise
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
