"""
Core types for UMA-3 semantic memory.

This module defines the `Fact` dataclass, which is the canonical representation
of semantic knowledge in UMA-3. A Fact is typically something stable or
long-lived (e.g., user preferences, profile attributes, project relationships).

Coding agent instructions
-------------------------
- Keep this file in sync with other subsystems that use `Fact`.
- Avoid adding heavy logic here; it should remain mostly a data container.
- When adding fields, consider how they map to SQL and JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Fact:
    """
    Semantic fact in UMA-3.

    Attributes
    ----------
    id:
        Unique identifier for the fact. Use a UUID string.
    subject:
        Entity the fact is about (e.g., "user:123", "project:alpha").
    predicate:
        Relationship type (e.g., "prefers_tone", "works_on", "country").
    object:
        Value of the fact. Can be string or any JSON-serializable type.
    created_at:
        When the fact was first created.
    updated_at:
        Last time the fact was updated.
    source_ids:
        IDs of episodes/messages/tools that support this fact.
    confidence:
        Optional confidence score in [0,1]. Can be used in conflict resolution.
    meta:
        Free-form metadata, e.g. {"salience": 0.9, "llm_model": "gpt-4o"}.
    """

    id: str
    subject: str
    predicate: str
    object: Any
    created_at: datetime
    updated_at: datetime
    source_ids: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)