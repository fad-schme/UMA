"""
Chunk types for document ingestion.

This dataclass represents authoritative document chunks stored in SQL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .types_owner import OwnerType
from .types_scope import DEFAULT_TENANT_ID


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    page_range: tuple[int, int]
    position: int
    source_path: str
    source_hash: str
    created_at: datetime
    updated_at: datetime

    owner_type: OwnerType = "user"
    owner_id: str = ""
    tenant_id: str = DEFAULT_TENANT_ID
    workspace_id: Optional[str] = None
    origin_agent_id: Optional[str] = None
    origin_user_id: Optional[str] = None
    origin_session_id: Optional[str] = None
    scope_model_version: Optional[str] = "v2"

    # Security primitives (trust_score only; content integrity lives in meta["text_hash"])
    trust_score: float = 0.5
    quarantined_at: Optional[datetime] = None

    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("Chunk.id must be non-empty")
        if not self.doc_id:
            raise ValueError("Chunk.doc_id must be non-empty")
        if not isinstance(self.page_range, tuple) or len(self.page_range) != 2:
            raise ValueError("Chunk.page_range must be (start, end)")
        if self.owner_type not in ("agent", "user", "workspace", "system"):
            raise ValueError(f"Invalid owner_type: {self.owner_type!r}")
        ts = float(self.trust_score)
        if not (0.0 <= ts <= 1.0):
            raise ValueError("Chunk.trust_score must be in [0, 1]")
