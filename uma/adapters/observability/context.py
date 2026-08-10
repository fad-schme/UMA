"""
Request context helpers for UMA logging.

Provides request_id / trace_id contextvars and a context manager to set them.
"""

from __future__ import annotations

import contextvars
import uuid
from contextlib import contextmanager
from typing import Iterator, Optional

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def get_request_id() -> str:
    return _request_id_var.get()


def get_trace_id() -> str:
    return _trace_id_var.get()


@contextmanager
def request_context(
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    generate: bool = True,
) -> Iterator[tuple[str, str]]:
    """
    Context manager that sets request_id/trace_id for structured logging.
    """
    if generate and not request_id:
        request_id = str(uuid.uuid4())
    req_token = _request_id_var.set(request_id or "-")
    trace_token = _trace_id_var.set(trace_id or "-")
    try:
        yield (_request_id_var.get(), _trace_id_var.get())
    finally:
        _request_id_var.reset(req_token)
        _trace_id_var.reset(trace_token)
