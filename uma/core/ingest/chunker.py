from __future__ import annotations

import hashlib
import logging
from typing import List

from .types import NormalizedSection, DocumentChunk

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Simple heuristic: ~4 chars/token
    return max(1, int(len(text) / 4))


def _chunk_text(text: str, *, chunk_size_tokens: int, overlap_tokens: int) -> List[str]:
    if not text:
        return []
    if chunk_size_tokens <= 0:
        return [text]
    overlap_tokens = max(0, min(overlap_tokens, chunk_size_tokens - 1))

    approx_chars = chunk_size_tokens * 4
    overlap_chars = overlap_tokens * 4

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + approx_chars)
        chunks.append(text[start:end])
        if end >= length:
            break
        start = max(0, end - overlap_chars)
    return chunks


def _stable_chunk_id(doc_id: str, position: int, text: str) -> str:
    h = hashlib.sha256()
    h.update(doc_id.encode("utf-8"))
    h.update(str(position).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return f"chunk_{h.hexdigest()[:24]}"


def chunk_sections(
    sections: List[NormalizedSection],
    *,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> List[DocumentChunk]:
    """
    Split sections into retrieval chunks.

    Generates stable chunk IDs based on doc_id + position + content hash.
    """
    if not isinstance(sections, list):
        raise ValueError("chunk_sections: sections must be a list")

    chunks: List[DocumentChunk] = []
    position = 0

    for sec in sections:
        text = sec.text or ""
        if not text.strip():
            continue
        for chunk_text in _chunk_text(
            text,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        ):
            if not chunk_text.strip():
                continue
            position += 1
            chunk_id = _stable_chunk_id(sec.doc_id, position, chunk_text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=sec.doc_id,
                    text=chunk_text,
                    page_range=sec.page_range,
                    position=position,
                )
            )

    if not chunks:
        logger.warning("chunk_sections: no chunks produced")

    return chunks
