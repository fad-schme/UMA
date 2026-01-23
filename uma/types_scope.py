"""
UMA3 Ownership / Scope types.

This module defines the canonical ownership model for all memory objects stored by UMA3.
Ownership is intentionally simple:

- owner_type: "agent" | "user" | "project"
- owner_id:   str (identifier within that owner_type)

This allows UMA3 to:
- keep agent KB logically separate from user/project memory,
- enable controlled promotion/linking of memories across scopes,
- support future multi-agent scenarios by using owner_type="agent" with unique owner_id.

Conventions
-----------
- For users:   owner_type="user",    owner_id="<user_id>"
- For projects: owner_type="project", owner_id="<user_id>:<project_id>"  (recommended default)
- For agents:  owner_type="agent",   owner_id="<agent_id>"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

OwnerType = Literal["agent", "user", "project"]


@dataclass(frozen=True)
class Ownership:
    """
    Represents the ownership (scope) of a memory record.

    Notes
    -----
    Use this everywhere (facts, episodes, skills, graph nodes/edges, SQL rows, vector metadata)
    to guarantee consistent separation and retrieval filtering.
    """

    owner_type: OwnerType
    owner_id: str

    def __post_init__(self) -> None:
        if self.owner_type not in ("agent", "user", "project"):
            raise ValueError(f"Invalid owner_type: {self.owner_type!r}")
        if not isinstance(self.owner_id, str) or not self.owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")

    @property
    def scope_key(self) -> str:
        """
        Stable string key for partitioning and filtering.
        """
        return f"{self.owner_type}:{self.owner_id}"