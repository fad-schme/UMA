"""
Content integrity hashing for UMA memory artifacts.

This module owns the canonical representation used to compute content_hash
for facts, episodes, and skills. Stable hashing is a hard contract: the same
artifact must always hash to the same value, regardless of insertion order,
whitespace in JSON-serialized values, or platform.

This module is intentionally small. Do not add new artifact types here
without updating the corresponding store and dataclass in the same PR.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_fact_content(subject: str, predicate: str, object_: Any) -> str:
    """SHA-256 hex digest over a canonical fact representation."""
    payload = f"{subject}|{predicate}|{json.dumps(object_, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_episode_content(summary: str) -> str:
    """SHA-256 hex digest over an episode's canonical summary text."""
    return hashlib.sha256((summary or "").encode("utf-8")).hexdigest()


def hash_skill_content(name: str, plan: Any) -> str:
    """SHA-256 hex digest over a skill's canonical (name, plan) pair."""
    payload = f"{name}|{json.dumps(plan, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_chunk_content(text: str) -> str:
    """SHA-256 hex digest over a chunk's canonical text content."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
