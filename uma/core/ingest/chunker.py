from __future__ import annotations

"""
Chunker: deterministic, sentence/paragraph-aligned chunking for document ingestion.

Behavior overview
-----------------
- Normalize into paragraphs, fall back to sentence groups (>=2 sentences).
- Prefer paragraph boundaries; never cut mid-sentence.
- Soft token cap with overlap aligned to sentence/paragraph boundaries.
- Avoid fragment-like starts/ends; merge short chunks backward.

Coding agent: Codex (GPT-5). If you modify this file, preserve determinism and
avoid introducing non-deterministic ordering or LLM calls.
"""

import hashlib
import logging
import re
from typing import List

from .types import NormalizedSection, DocumentChunk

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Simple heuristic: ~4 chars/token
    return max(1, int(len(text) / 4))


_PARA_SPLIT_RE = re.compile(r"\n{2,}")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p and p.strip()]
    return parts or [text.strip()]


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
    return parts or [text.strip()]


def _group_sentences(sentences: List[str], *, min_sentences: int = 2) -> List[str]:
    if not sentences:
        return []
    groups: List[str] = []
    buff: List[str] = []
    for s in sentences:
        buff.append(s)
        if len(buff) >= min_sentences:
            groups.append(" ".join(buff).strip())
            buff = []
    if buff:
        # Attach leftovers to previous group to avoid 1-sentence fragments.
        if groups:
            groups[-1] = f"{groups[-1]} {' '.join(buff).strip()}".strip()
        else:
            groups.append(" ".join(buff).strip())
    return groups


_MIN_CHUNK_CHARS = 80
_MIN_OVERLAP_CHARS = 60
_SOFT_OVERFLOW_FRACTION = 0.15


def _starts_like_fragment(text: str) -> bool:
    if not text:
        return False
    return text[:1].islower() and not text.strip().endswith((".", "!", "?"))


def _ends_like_fragment(text: str) -> bool:
    if not text:
        return False
    return not text.strip().endswith((".", "!", "?"))


def _is_overlap_worthy(text: str) -> bool:
    if not text:
        return False
    if len(text) < _MIN_OVERLAP_CHARS:
        return False
    if _starts_like_fragment(text):
        return False
    if _ends_like_fragment(text):
        return False
    return True


def _merge_short_chunks(chunks: List[str]) -> List[str]:
    if not chunks:
        return []
    merged: List[str] = []
    for ch in chunks:
        if not merged:
            merged.append(ch)
            continue
        if len(ch) < _MIN_CHUNK_CHARS:
            merged[-1] = f"{merged[-1]} {ch}".strip()
        else:
            merged.append(ch)
    if merged and len(merged[0]) < _MIN_CHUNK_CHARS and len(merged) > 1:
        merged[1] = f"{merged[0]} {merged[1]}".strip()
        merged = merged[1:]
    return merged


def _chunk_text(text: str, *, chunk_size_tokens: int, overlap_tokens: int) -> List[str]:
    if not text:
        return []
    if not isinstance(chunk_size_tokens, int):
        raise ValueError("chunk_size_tokens must be an int")
    if not isinstance(overlap_tokens, int):
        raise ValueError("overlap_tokens must be an int")
    if chunk_size_tokens <= 0:
        logger.warning("chunk_size_tokens <= 0; returning unchunked text")
        return [text]

    target_tokens = int(chunk_size_tokens)
    overlap_tokens = max(0, min(int(overlap_tokens), target_tokens - 1))
    if overlap_tokens and overlap_tokens >= target_tokens:
        logger.warning("overlap_tokens >= target_tokens; clamped to target_tokens - 1")

    paras = _split_paragraphs(text)
    units: List[str] = []
    unit_is_paragraph: List[bool] = []
    for p in paras:
        if _estimate_tokens(p) <= target_tokens:
            units.append(p)
            unit_is_paragraph.append(True)
            continue
        # Paragraph too long: split by sentences, keep ≥2 sentences per unit.
        sentences = _split_sentences(p)
        groups = _group_sentences(sentences, min_sentences=2)
        units.extend(groups)
        unit_is_paragraph.extend([False] * len(groups))

    chunks: List[str] = []
    current: List[tuple[str, bool]] = []
    current_tokens = 0
    current_para_count = 0

    def _emit_current(curr: List[tuple[str, bool]]) -> None:
        if not curr:
            return
        chunk_text = " ".join([u for u, _ in curr]).strip()
        if not chunk_text:
            return
        if chunks and (
            len(chunk_text) < _MIN_CHUNK_CHARS
            or _starts_like_fragment(chunk_text)
            or _ends_like_fragment(chunk_text)
        ):
            # Merge backward to avoid short/fragment chunks.
            chunks[-1] = f"{chunks[-1]} {chunk_text}".strip()
            return
        chunks.append(chunk_text)

    def _overlap_from_current(
        curr: List[tuple[str, bool]],
        *,
        overlap_tokens: int,
    ) -> List[tuple[str, bool]]:
        if overlap_tokens <= 0 or not curr:
            return []
        # Prefer last full paragraph when available.
        for text_unit, is_para in reversed(curr):
            if is_para and _is_overlap_worthy(text_unit):
                return [(text_unit, True)]
        # Otherwise, include trailing sentence groups until we cross overlap_tokens.
        picked: List[tuple[str, bool]] = []
        total_tokens = 0
        for text_unit, is_para in reversed(curr):
            if _is_overlap_worthy(text_unit):
                picked.append((text_unit, is_para))
                total_tokens += _estimate_tokens(text_unit)
                if total_tokens >= overlap_tokens:
                    break
        picked.reverse()
        return picked

    for unit, is_para in zip(units, unit_is_paragraph):
        unit_tokens = _estimate_tokens(unit)
        if not current:
            current.append((unit, is_para))
            current_tokens = unit_tokens
            current_para_count = 1 if is_para else 0
            continue

        soft_limit = int(target_tokens * (1.0 + _SOFT_OVERFLOW_FRACTION))
        if current_tokens + unit_tokens <= soft_limit:
            current.append((unit, is_para))
            current_tokens += unit_tokens
            current_para_count += 1 if is_para else 0
            continue

        # Emit current chunk
        curr_text = " ".join([u for u, _ in current]).strip()
        if curr_text and (_starts_like_fragment(curr_text) or _ends_like_fragment(curr_text)):
            # Avoid fragment boundaries: keep accumulating even if over soft cap.
            current.append((unit, is_para))
            current_tokens += unit_tokens
            current_para_count += 1 if is_para else 0
            continue

        _emit_current(current)

        # Start new chunk with paragraph-boundary overlap only
        overlap_units = _overlap_from_current(current, overlap_tokens=overlap_tokens)

        current = overlap_units + [(unit, is_para)]
        current_tokens = sum(_estimate_tokens(u) for u, _ in current)
        current_para_count = sum(1 for _, is_p in current if is_p)

    if current:
        _emit_current(current)

    return _merge_short_chunks(chunks)


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
    if not isinstance(chunk_size_tokens, int) or not isinstance(overlap_tokens, int):
        raise ValueError("chunk_sections: chunk_size_tokens/overlap_tokens must be ints")

    chunks: List[DocumentChunk] = []
    position = 0

    for sec in sections:
        if not isinstance(sec, NormalizedSection):
            raise ValueError("chunk_sections: sections must contain NormalizedSection items")
        text = sec.text or ""
        if not text.strip():
            continue
        try:
            emitted = _chunk_text(
                text,
                chunk_size_tokens=chunk_size_tokens,
                overlap_tokens=overlap_tokens,
            )
        except Exception:
            logger.exception("chunk_sections: failed to chunk section_id=%s", sec.section_id)
            continue

        for chunk_text in emitted:
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
    else:
        logger.debug(
            "chunk_sections: produced=%d chunk_size_tokens=%d overlap_tokens=%d",
            len(chunks),
            chunk_size_tokens,
            overlap_tokens,
        )

    return chunks
