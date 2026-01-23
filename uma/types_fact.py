"""
Core types for UMA semantic memory.

This module defines the `Fact` dataclass, which is the canonical representation
of semantic knowledge in UMA.

This version includes ownership:
- owner_type: "agent" | "user" | "project"
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

from .types_scope import OwnerType


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

        if self.owner_type not in ("agent", "user", "project"):
            raise ValueError(f"Invalid owner_type: {self.owner_type!r}")
        if self.owner_id is not None and not isinstance(self.owner_id, str):
            raise ValueError("Fact.owner_id must be a string")

        if self.confidence is not None:
            c = float(self.confidence)
            if not (0.0 <= c <= 1.0):
                raise ValueError("Fact.confidence must be in [0, 1] when provided")

        if float(self.salience) < 0.0:
            raise ValueError("Fact.salience must be >= 0")

    @property
    def scope_key(self) -> str:
        derived_owner_id = self.owner_id.strip() or self.subject or ""
        # If subject is user:<id>, still fine as a fallback.
        return f"{self.owner_type}:{derived_owner_id}"