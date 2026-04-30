from __future__ import annotations

from typing import Any


def get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Normalize object/dict access for RLM snippets and domain models."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
