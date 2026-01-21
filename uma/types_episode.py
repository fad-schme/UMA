"""
Episodic memory datatypes for UMA.

Episodes store past events, conversations, or agent actions. They are the
foundation of case-based reasoning in UMA.
An Episode represents a single coherent event in the agent’s life:
- A user request
- A tool execution
- A reasoning trace
- A conversation snippet
- A completed task

Coding agent instructions:
--------------------------
- Keep Episode immutable except for 'meta'.
- Do NOT add heavy logic here.
- Keep this dataclass aligned with your DB schema.
- Ensure all fields are serializable and easy to embed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class Episode:
    """
    Episodic memory unit in UMA.

    Attributes
    ----------
    id:
        Unique string ID (UUID recommended).
    user_id:
        Entity associated with the episode (e.g. a user or agent).
    timestamp:
        When the episode happened.
    summary:
        Text summary of the event (short).
    raw:
        Optional raw transcript or details (may be long).
    tags:
        Optional list of tags for retrieval, e.g. ["error", "database"].
    embedding:
        Optional vector embedding for semantic search.
    meta:
        Free-form metadata dictionary.
    """

    id: str
    user_id: str
    timestamp: datetime
    summary: str
    raw: str | None = None
    tags: List[str] = field(default_factory=list)
    embedding: List[float] | None = None
    meta: Dict[str, Any] = field(default_factory=dict)