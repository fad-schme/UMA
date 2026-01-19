"""
Telemetry helpers for UMA-3.

Provides:
- Logging decorators
- Trace/Span placeholders (can integrate with OTEL)
"""

from __future__ import annotations
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)


def log_call(name: str):
    """
    Decorator logging function calls at DEBUG level.
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug("CALL %s args=%s kwargs=%s", name, args, kwargs)
            return func(*args, **kwargs)
        return wrapper

    return decorator