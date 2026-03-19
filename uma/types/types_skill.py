"""
Skill model for Procedural Memory in UMA.

Skills are procedural units (how-to, playbooks, instructions).
This version adds ownership so skills can live in:
- agent KB (global skills)
- user KB (personal procedures)
- workspace KB (shared procedures)
- system scope (operational procedures)

Coding agent instructions
-------------------------
- Treat Skill as a stable schema: do not add arbitrary fields without
  also updating the store and indexer.
- Ensure all fields are JSON-serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .types_owner import OwnerType


@dataclass
class Skill:
    # Existing fields (DO NOT rename)
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime

    # Ownership (NEW, safe defaults so old code doesn't break)
    owner_type: OwnerType = "user"
    owner_id: str = ""

    salience: float = 0.0
    tags: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # Older fields for procedural memory (keep as-is)
    trigger_phrases: List[str] = field(default_factory=list)
    trigger_patterns: List[str] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)
    example: str = ""
    embedding: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Skill.id must be a non-empty string")
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Skill.name must be a non-empty string")
        if not self.description or not isinstance(self.description, str):
            raise ValueError("Skill.description must be a non-empty string")

        if self.owner_type not in ("agent", "user", "workspace", "system"):
            raise ValueError(f"Invalid owner_type: {self.owner_type!r}")
        if self.owner_id is not None and not isinstance(self.owner_id, str):
            raise ValueError("Skill.owner_id must be a string")

        if float(self.salience) < 0.0:
            raise ValueError("Skill.salience must be >= 0")

    @property
    def scope_key(self) -> str:
        derived_owner_id = self.owner_id.strip() or self.id
        return f"{self.owner_type}:{derived_owner_id}"
