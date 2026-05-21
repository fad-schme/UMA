"""
Core types for UMA semantic memory.

This module defines the `Fact` dataclass, which is the canonical representation
of semantic knowledge in UMA.

This version includes ownership:
- owner_type: durable owner scope
- owner_id: string identifier of the owner

Coding agent instructions
-------------------------
- Keep this file in sync with subsystems that use `Fact`.
- Avoid heavy logic here; it should remain mostly a data container.
- When adding fields, consider how they map to SQL and JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .types_owner import OwnerType


@dataclass
class Fact:
    """
    A semantic memory unit.

    Attributes (existing names preserved)
    -----------------------------------
    id, subject, predicate, object, created_at, updated_at,
    source_ids, confidence, meta, salience, owner_type, owner_id
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

    # Ownership (NEW, safe defaults)
    owner_type: OwnerType = "user"
    owner_id: str = ""
    tenant_id: str = "default"
    workspace_id: Optional[str] = None
    session_id: Optional[str] = None
    origin_agent_id: Optional[str] = None
    origin_user_id: Optional[str] = None
    origin_session_id: Optional[str] = None
    scope_model_version: Optional[str] = "v2"

    # Security primitives (PR1 baseline: neutral defaults)
    trust_score: float = 0.5
    content_hash: Optional[str] = None

    # Optional metadata
    salience: float = 0.0

    def validate(self) -> None:
        """
        Validate minimal invariants; raise ValueError on invalid data.

        Notes
        -----
        - `object` is Any (JSON-serializable preferred). We do NOT enforce string.
        - confidence is optional; if provided, must be in [0,1]
        """
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Fact.id must be a non-empty string")
        if not self.subject or not isinstance(self.subject, str):
            raise ValueError("Fact.subject must be a non-empty string")
        if not self.predicate or not isinstance(self.predicate, str):
            raise ValueError("Fact.predicate must be a non-empty string")

        if self.owner_type not in ("agent", "user", "workspace", "system"):
            raise ValueError(f"Invalid owner_type: {self.owner_type!r}")
        if self.owner_id is not None and not isinstance(self.owner_id, str):
            raise ValueError("Fact.owner_id must be a string")

        if self.confidence is not None:
            c = float(self.confidence)
            if not (0.0 <= c <= 1.0):
                raise ValueError("Fact.confidence must be in [0, 1] when provided")

        ts = float(self.trust_score)
        if not (0.0 <= ts <= 1.0):
            raise ValueError("Fact.trust_score must be in [0, 1]")
        if self.content_hash is not None and not isinstance(self.content_hash, str):
            raise ValueError("Fact.content_hash must be a string when provided")
        if isinstance(self.content_hash, str) and not self.content_hash:
            raise ValueError("Fact.content_hash must be non-empty when provided")

        if float(self.salience) < 0.0:
            raise ValueError("Fact.salience must be >= 0")

    @property
    def scope_key(self) -> str:
        derived_owner_id = self.owner_id.strip() or self.subject or ""
        # If subject is user:<id>, still fine as a fallback.
        return f"{self.owner_type}:{derived_owner_id}"
