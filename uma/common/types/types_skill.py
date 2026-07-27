"""
Skill model for Procedural Memory in UMA.

Skills are procedural units (how-to, playbooks, instructions) and
agent profiles (the agent's scope description, focus areas, tags).
Both live in the procedural store, discriminated by ``kind``.

The store holds:
- agent KB skills (global procedures shared across users of an agent)
- user KB skills (personal procedures)
- workspace KB skills (shared procedures)
- system scope skills (operational procedures)
- agent profiles (``kind="agent_profile"``, one row per agent, used by
  the promotion qualifier to decide which user-owned facts/episodes
  qualify for promotion into the agent's KB)

Coding agent instructions
-------------------------
- Treat Skill as a stable schema: do not add arbitrary fields without
  also updating the store and indexer.
- Ensure all fields are JSON-serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from .types_owner import OwnerType


@dataclass
class Skill:
    # Existing fields (DO NOT rename)
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime

    # Discriminator (NEW):
    #   "procedural"    — a how-to / playbook / instructions row (default).
    #   "agent_profile" — the profile describing an agent's scope. One
    #                     row per agent, referenced by the promotion
    #                     qualifier to decide fact/episode promotability.
    kind: Literal["procedural", "agent_profile"] = "procedural"

    # Ownership (NEW, safe defaults so old code doesn't break)
    owner_type: OwnerType = "user"
    owner_id: str = ""
    tenant_id: str = "default"
    workspace_id: Optional[str] = None
    origin_agent_id: Optional[str] = None
    origin_user_id: Optional[str] = None
    origin_session_id: Optional[str] = None
    scope_model_version: Optional[str] = "v2"

    # Security primitives (PR1 baseline: neutral defaults)
    trust_score: float = 0.5
    content_hash: Optional[str] = None
    quarantined_at: Optional[datetime] = None

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

    # Agent-profile-only field (NEW):
    #   Populated on rows with kind="agent_profile". Empty list on
    #   procedural rows. Used by the promotion qualifier's deterministic
    #   scope-match layer (keyword hit against fact text).
    focus_areas: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Skill.id must be a non-empty string")
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Skill.name must be a non-empty string")
        if not self.description or not isinstance(self.description, str):
            raise ValueError("Skill.description must be a non-empty string")

        if self.kind not in ("procedural", "agent_profile"):
            raise ValueError(f"Invalid Skill.kind: {self.kind!r}")

        if self.owner_type not in ("agent", "user", "workspace", "system"):
            raise ValueError(f"Invalid owner_type: {self.owner_type!r}")
        if self.owner_id is not None and not isinstance(self.owner_id, str):
            raise ValueError("Skill.owner_id must be a string")

        ts = float(self.trust_score)
        if not (0.0 <= ts <= 1.0):
            raise ValueError("Skill.trust_score must be in [0, 1]")
        if self.content_hash is not None and not isinstance(self.content_hash, str):
            raise ValueError("Skill.content_hash must be a string when provided")
        if isinstance(self.content_hash, str) and not self.content_hash:
            raise ValueError("Skill.content_hash must be non-empty when provided")

        if float(self.salience) < 0.0:
            raise ValueError("Skill.salience must be >= 0")

        if not isinstance(self.focus_areas, list):
            raise ValueError("Skill.focus_areas must be a list of strings")
        for item in self.focus_areas:
            if not isinstance(item, str):
                raise ValueError("Skill.focus_areas entries must be strings")

    @property
    def scope_key(self) -> str:
        derived_owner_id = self.owner_id.strip() or self.id
        return f"{self.owner_type}:{derived_owner_id}"