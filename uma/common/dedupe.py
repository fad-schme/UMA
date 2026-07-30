"""
dedupe.py
=========

Canonical deduplication helpers.
"""

from __future__ import annotations

import hashlib
from typing import Any


def dedupe_by_id(items: list[Any]) -> list[Any]:
    """Return a deduplicated list of artifacts, keeping the first occurrence of each ID."""
    if not items:
        return []
    seen = set()
    out: list[Any] = []
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


def dedupe_evidence_by_text(items: list[dict]) -> list[dict]:
    """Deduplicate serialized evidence dicts by text content hash.

    Two items are considered duplicates when their normalized text is identical.
    First occurrence wins; order is preserved.
    """
    if not items:
        return []
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        text = (it.get("text") or "").strip()
        key = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest() if text else id(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out
