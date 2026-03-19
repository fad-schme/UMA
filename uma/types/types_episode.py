"""
Episodic memory datatypes for UMA.

Episodes store past events, conversations, or agent actions. They are the
foundation of case-based reasoning in UMA.

Notes
-----
Episodes are scoped by ownership:
- owner_type="user"/owner_id="user:<user_id>" for user-level logs
- owner_type="agent"/owner_id=agent_id for agent-global episodes (rare but possible)
- owner_type="workspace"/owner_id=<workspace_id> for shared workspace logs
- owner_type="system"/owner_id=<system_id> for operational records

Coding agent instructions:
--------------------------
- Keep Episode mostly immutable (avoid heavy logic here).
- Keep this dataclass aligned with your DB schema.
- Ensure all fields are serializable and easy to embed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .types_owner import OwnerType


@dataclass
class Episode:
    """
    Episodic memory unit in UMA.

    Important: this dataclass must remain stable because it is used by:
      - stores/episodic_sql.py
      - core/episodic/*
      - graph/updater.py

    """

    # -------------------------
    # Core fields (DO NOT rename)
    # -------------------------
    id: str
    timestamp: datetime
    summary: str
    user_id: str

    # -------------------------
    # Existing optional fields
    # -------------------------
    raw: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # Existing timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # -------------------------
    # Ownership
    # -------------------------
    owner_type: OwnerType = "user"
    owner_id: str = ""

    # -------------------------
    # Additional fields already referenced in design (optional)
    # -------------------------
    salience: float = 0.0
    transcript: Optional[str] = None
    source: Optional[str] = None

    def validate(self) -> None:
        """
        Validate minimal invariants. Call this in ingestion paths.

        We intentionally keep validation strict:
        - owner_id must be present and non-empty.
        """
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Episode.id must be a non-empty string")

        if not isinstance(self.timestamp, datetime):
            raise ValueError("Episode.timestamp must be a datetime")

        if not self.summary or not isinstance(self.summary, str):
            raise ValueError("Episode.summary must be a non-empty string")

        if not self.user_id or not isinstance(self.user_id, str):
            raise ValueError("Episode.user_id must be a non-empty string")

        if self.owner_type not in ("agent", "user", "workspace", "system"):
            raise ValueError(f"Invalid owner_type: {self.owner_type!r}")

        if not isinstance(self.owner_id, str) or not self.owner_id.strip():
            raise ValueError("Episode.owner_id must be a non-empty string")

        if float(self.salience) < 0.0:
            raise ValueError("Episode.salience must be >= 0")

    @property
    def scope_key(self) -> str:
        """
        Canonical scope key used for filtering and metadata.
        """
        return f"{self.owner_type}:{self.owner_id}"
