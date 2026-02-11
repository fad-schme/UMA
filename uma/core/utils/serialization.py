from __future__ import annotations

from typing import Any, Dict

from ...types_chunk import Chunk


def chunk_to_dict(ch: Chunk) -> Dict[str, Any]:
    return {
        "id": ch.id,
        "doc_id": ch.doc_id,
        "text": ch.text,
        "page_range": ch.page_range,
        "position": ch.position,
        "owner_type": ch.owner_type,
        "owner_id": ch.owner_id,
        "source_path": ch.source_path,
        "source_hash": ch.source_hash,
        "created_at": ch.created_at.isoformat() if getattr(ch, "created_at", None) else None,
        "updated_at": ch.updated_at.isoformat() if getattr(ch, "updated_at", None) else None,
        "meta": ch.meta or {},
        "paragraph_index_start": (ch.meta or {}).get("paragraph_index_start"),
        "paragraph_index_end": (ch.meta or {}).get("paragraph_index_end"),
        "char_start": (ch.meta or {}).get("char_start"),
        "char_end": (ch.meta or {}).get("char_end"),
    }

