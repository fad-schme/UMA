"""
dedupe.py
=========

Canonical deduplication helpers.
"""

from __future__ import annotations

from typing import Any, List


def dedupe_by_id(items: List[Any]) -> List[Any]:
    if not items:
        return []
    seen = set()
    out: List[Any] = []
    for it in items:
        key = None
        if isinstance(it, dict):
            key = it.get("id")
        else:
            key = getattr(it, "id", None)
        if key is None:
            key = id(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out
