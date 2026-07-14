"""
Timing utilities for UMA.

Includes:
- time_block context manager
- @timed decorator

Coding agent instructions:
--------------------------
- Use timed() for any expensive operation.
"""

from __future__ import annotations

import logging
import time
import asyncio
from contextlib import contextmanager
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)


@contextmanager
def time_block(name: str, logger: logging.Logger | None = None, level: int = logging.DEBUG):
    """Synchronous timing context manager.

    Parameters
    ----------
    name:
        Label for the timed block.
    logger:
        Optional logger to use (defaults to module logger).
    level:
        Logging level to emit the timing message.
    """
    _logger = logger or globals()["logger"]
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        _logger.log(level, "time_block[%s]: %.3fs", name, dt)


def timed(fn: Callable):
    """Decorator that logs execution time. Supports async functions and
    preserves function metadata via `wraps`.
    """

    if asyncio.iscoroutinefunction(fn):

        @wraps(fn)
        async def _async_wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                return await fn(*args, **kwargs)
            finally:
                dt = time.time() - t0
                logger.debug("timed[%s]: %.3fs", fn.__name__, dt)

        return _async_wrapper

    else:

        @wraps(fn)
        def _wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.time() - t0
                logger.debug("timed[%s]: %.3fs", fn.__name__, dt)

        return _wrapper