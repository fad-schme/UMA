"""
Chunk types for document ingestion.

This dataclass represents authoritative document chunks stored in SQL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Tuple

from .types_scope import OwnerType


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    page_range: Tuple[int, int]
    position: int
    source_path: str
    source_hash: str
    created_at: datetime
    updated_at: datetime

    owner_type: OwnerType = "user"
    owner_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("Chunk.id must be non-empty")
        if not self.doc_id:
            raise ValueError("Chunk.doc_id must be non-empty")
        if not isinstance(self.page_range, tuple) or len(self.page_range) != 2:
            raise ValueError("Chunk.page_range must be (start, end)")
        if self.owner_type not in ("agent", "user", "project"):
            raise ValueError(f"Invalid owner_type: {self.owner_type!r}")
