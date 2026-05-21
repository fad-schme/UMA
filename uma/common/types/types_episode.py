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
      - `uma.stores.episodic_sql`
      - `uma.memory.episodic`
      - `uma.memory.graph`

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
    tenant_id: str = "default"
    workspace_id: Optional[str] = None
    session_id: Optional[str] = None
    origin_agent_id: Optional[str] = None
    origin_user_id: Optional[str] = None
    origin_session_id: Optional[str] = None
    scope_model_version: Optional[str] = "v2"

    # -------------------------
    # Security primitives (PR1 baseline: neutral defaults)
    # -------------------------
    trust_score: float = 0.5
    content_hash: Optional[str] = None

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

        ts = float(self.trust_score)
        if not (0.0 <= ts <= 1.0):
            raise ValueError("Episode.trust_score must be in [0, 1]")
        if self.content_hash is not None and not isinstance(self.content_hash, str):
            raise ValueError("Episode.content_hash must be a string when provided")
        if isinstance(self.content_hash, str) and not self.content_hash:
            raise ValueError("Episode.content_hash must be non-empty when provided")

        if float(self.salience) < 0.0:
            raise ValueError("Episode.salience must be >= 0")

    @property
    def scope_key(self) -> str:
        """
        Canonical scope key used for filtering and metadata.
        """
        return f"{self.owner_type}:{self.owner_id}"
