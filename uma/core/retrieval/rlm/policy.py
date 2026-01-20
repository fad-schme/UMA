# uma3/core/retrieval/rlm/policy.py

from __future__ import annotations
from typing import Any, Dict, List


def good_enough(counts: Dict[str, int]) -> bool:
    """
    Deterministic stopping heuristic.

    Keep SIMPLE and explainable.
    """
    if counts.get("facts", 0) >= 6:
        return True
    if counts.get("episodes", 0) >= 4 and counts.get("facts", 0) >= 2:
        return True
    if counts.get("skills", 0) >= 3 and counts.get("facts", 0) >= 2:
        return True
    if counts.get("graph", 0) >= 8:
        return True
    return False


def merge_unique(existing: List[Any], new: List[Any], max_items: int) -> List[Any]:
    """
    Merge + dedupe by `.id` or dict["id"].
    """
    seen = set()
    out = []

    def key(x):
        if hasattr(x, "id"):
            return x.id
        if isinstance(x, dict):
            return x.get("id")
        return id(x)

    for it in existing + new:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
        if len(out) >= max_items:
            break

    return out