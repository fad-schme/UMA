"""
Lightweight in-memory metrics for UMA.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator

_lock = threading.Lock()
_counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
_timers: dict[tuple[str, tuple[tuple[str, str], ...]], list] = {}


def _norm_tags(tags: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not tags:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in tags.items()))


def increment(name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
    key = (name, _norm_tags(tags))
    with _lock:
        _counters[key] = _counters.get(key, 0) + int(value)


def observe(name: str, value: float, tags: dict[str, str] | None = None) -> None:
    key = (name, _norm_tags(tags))
    with _lock:
        _timers.setdefault(key, []).append(float(value))


@contextmanager
def timed(name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        observe(name, time.time() - start, tags=tags)


def snapshot() -> dict[str, dict[str, float]]:
    """
    Return a snapshot of counters and timers (avg).
    """
    out: dict[str, dict[str, float]] = {"counters": {}, "timers_avg": {}}
    with _lock:
        for (name, tags), value in _counters.items():
            tag_key = ",".join(f"{k}={v}" for k, v in tags)
            out["counters"][f"{name}|{tag_key}"] = float(value)
        for (name, tags), values in _timers.items():
            tag_key = ",".join(f"{k}={v}" for k, v in tags)
            avg = sum(values) / len(values) if values else 0.0
            out["timers_avg"][f"{name}|{tag_key}"] = float(avg)
    return out
