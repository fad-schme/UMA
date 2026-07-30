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
from dataclasses import dataclass
from typing import Optional

from .types import NormalizedSection, DocumentChunk

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Simple heuristic: ~4 chars/token
    return max(1, int(len(text) / 4))


_PARA_SPLIT_RE = re.compile(r"\n{2,}")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p and p.strip()]
    return parts or [text.strip()]


def _split_paragraphs_with_indices(text: str) -> list[tuple[int, str]]:
    if not text:
        return []
    parts = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p and p.strip()]
    if not parts:
        parts = [text.strip()]
    return [(i, p) for i, p in enumerate(parts)]


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
    return parts or [text.strip()]


def _ensure_terminal(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    return t if _is_terminal(t) else f"{t}."


def _group_sentences(sentences: list[str], *, min_sentences: int = 2) -> list[str]:
    if not sentences:
        return []
    groups: list[str] = []
    buff: list[str] = []
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


@dataclass(frozen=True)
class _ChunkUnit:
    text: str
    is_paragraph: bool
    paragraph_index_start: int
    paragraph_index_end: int
    char_start: Optional[int] = None
    char_end: Optional[int] = None


def _starts_like_fragment(text: str) -> bool:
    if not text:
        return False
    t = text.lstrip()
    if not t:
        return False
    # Fragment-like start: begins with lowercase letter or punctuation/connector.
    first = t[:1]
    # NOTE: Lowercase starts are common in plain-text documents (e.g., "hello world")
    # and are not reliable evidence of mid-sentence cuts. We rely on sentence/terminal
    # punctuation rules + min length + no fragment ends for strictness.
    if first in (",", ";", ":"):
        return True
    return False


def _ends_like_fragment(text: str) -> bool:
    if not text:
        return False
    # Treat missing terminal punctuation as a fragment-like ending.
    # (The strict rule is enforced separately via _is_terminal/validate_chunk_text.)
    return not _is_terminal(text)


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


def _merge_short_chunks(chunks: list[str]) -> list[str]:
    if not chunks:
        return []
    merged: list[str] = []
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


# -----------------------------
# Strict finalization + validation
# -----------------------------

_TERMINAL_PUNCT = (".", "!", "?")


def _is_terminal(text: str) -> bool:
    return bool(text and text.strip().endswith(_TERMINAL_PUNCT))


def validate_chunk_text(text: str) -> None:
    """Validate a single chunk against strict chunker rules.

    This MUST be satisfied before persistence (SQL or vector).
    """
    if not isinstance(text, str):
        raise ValueError("chunk text must be a string")
    t = text.strip()
    if not t:
        raise ValueError("chunk text must be non-empty")
    if len(t) < _MIN_CHUNK_CHARS:
        raise ValueError(f"chunk too short (<{_MIN_CHUNK_CHARS} chars)")
    if _starts_like_fragment(t):
        raise ValueError("chunk starts like a fragment")
    if not _is_terminal(t):
        raise ValueError("chunk must end with terminal punctuation")


def validate_chunks(chunks: list[DocumentChunk]) -> None:
    """Validate DocumentChunk objects against strict chunker rules."""
    for i, ch in enumerate(chunks or []):
        try:
            validate_chunk_text(ch.text or "")
        except Exception as exc:
            raise ValueError(
                f"invalid chunk chunk_id={getattr(ch, 'chunk_id', None)} index={i}: {exc}"
            ) from exc


def validate_docchunk_structure(chunks: list[DocumentChunk]) -> None:
    """Validate structural metadata for the document/PDF chunking pipeline.

    Policy (doc ingestion only):
    - paragraph indices are mandatory (needed for retrieval expansion/snippet quality)
    - paragraph indices are section/page_range-local (not doc-global); scope must be preserved by callers
    - char offsets are optional; if one is present, require both + ordering
    """
    for i, ch in enumerate(chunks or []):
        if ch.paragraph_index_start is None or ch.paragraph_index_end is None:
            raise ValueError(
                f"invalid chunk chunk_id={getattr(ch, 'chunk_id', None)} index={i}: "
                "missing paragraph indices"
            )
        if not isinstance(ch.paragraph_index_start, int) or not isinstance(ch.paragraph_index_end, int):
            raise ValueError(
                f"invalid chunk chunk_id={getattr(ch, 'chunk_id', None)} index={i}: "
                "paragraph indices must be ints"
            )
        if ch.paragraph_index_start < 0:
            raise ValueError(
                f"invalid chunk chunk_id={getattr(ch, 'chunk_id', None)} index={i}: "
                "paragraph_index_start must be >= 0"
            )
        if ch.paragraph_index_start > ch.paragraph_index_end:
            raise ValueError(
                f"invalid chunk chunk_id={getattr(ch, 'chunk_id', None)} index={i}: "
                "paragraph_index_start > paragraph_index_end"
            )

        if (ch.char_start is None) != (ch.char_end is None):
            raise ValueError(
                f"invalid chunk chunk_id={getattr(ch, 'chunk_id', None)} index={i}: "
                "char_start/char_end must both be set or both be None"
            )
        if ch.char_start is not None and ch.char_end is not None:
            if not isinstance(ch.char_start, int) or not isinstance(ch.char_end, int):
                raise ValueError(
                    f"invalid chunk chunk_id={getattr(ch, 'chunk_id', None)} index={i}: "
                    "char_start/char_end must be ints"
                )
            if ch.char_start < 0:
                raise ValueError(
                    f"invalid chunk chunk_id={getattr(ch, 'chunk_id', None)} index={i}: "
                    "char_start must be >= 0"
                )
            if ch.char_start > ch.char_end:
                raise ValueError(
                    f"invalid chunk chunk_id={getattr(ch, 'chunk_id', None)} index={i}: "
                    "char_start > char_end"
                )


def _stable_chunk_id(
    doc_id: str,
    page_range: tuple[int, int],
    paragraph_index_start: int | None,
    paragraph_index_end: int | None,
    text: str,
) -> str:
    h = hashlib.sha256()
    h.update(doc_id.encode("utf-8"))
    h.update(str(page_range[0]).encode("utf-8"))
    h.update(str(page_range[1]).encode("utf-8"))
    h.update(str(paragraph_index_start if paragraph_index_start is not None else -1).encode("utf-8"))
    h.update(str(paragraph_index_end if paragraph_index_end is not None else -1).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return f"chunk_{h.hexdigest()[:24]}"


def finalize_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Finalize chunk texts with strict rules.

    This runs AFTER initial chunk emission but BEFORE any persistence.
    It may merge adjacent chunks and will reassign stable IDs/positions deterministically.

    Structural metadata contract:
    - paragraph_index_start = min(paragraph_index_start of merged inputs), None-safe
    - paragraph_index_end   = max(paragraph_index_end of merged inputs), None-safe
    - char_start            = min(char_start of merged inputs) if any present; else None
    - char_end              = max(char_end of merged inputs) if any present; else None
    """
    if not chunks:
        return []

    def _apply_strict_rules_to_chunk_objects(group_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Apply strict rules while preserving structural metadata.

        Merges are performed on DocumentChunk objects, not just text, so paragraph indices
        remain authoritative.
        """
        # Start with cleaned objects (normalized whitespace, drop empties).
        cleaned: list[DocumentChunk] = []
        for ch in group_chunks:
            t = " ".join((ch.text or "").split()).strip()
            if not t:
                continue
            cleaned.append(
                DocumentChunk(
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    text=t,
                    page_range=ch.page_range,
                    position=ch.position,
                    char_start=ch.char_start,
                    char_end=ch.char_end,
                    paragraph_index_start=ch.paragraph_index_start,
                    paragraph_index_end=ch.paragraph_index_end,
                )
            )
        if not cleaned:
            return []

        def _merge_two(a: DocumentChunk, b: DocumentChunk) -> DocumentChunk:
            merged_text = f"{a.text} {b.text}".strip()
            p_start = a.paragraph_index_start
            p_end = a.paragraph_index_end
            if b.paragraph_index_start is not None:
                p_start = b.paragraph_index_start if p_start is None else min(p_start, b.paragraph_index_start)
            if b.paragraph_index_end is not None:
                p_end = b.paragraph_index_end if p_end is None else max(p_end, b.paragraph_index_end)
            if a.paragraph_index_start is None:
                p_start = b.paragraph_index_start
            if a.paragraph_index_end is None:
                p_end = b.paragraph_index_end
            cs = a.char_start
            ce = a.char_end
            if b.char_start is not None:
                cs = b.char_start if cs is None else min(cs, b.char_start)
            if b.char_end is not None:
                ce = b.char_end if ce is None else max(ce, b.char_end)
            if a.char_start is None:
                cs = b.char_start
            if a.char_end is None:
                ce = b.char_end
            return DocumentChunk(
                chunk_id=a.chunk_id,
                doc_id=a.doc_id,
                text=merged_text,
                page_range=a.page_range,
                position=a.position,
                char_start=cs,
                char_end=ce,
                paragraph_index_start=p_start,
                paragraph_index_end=p_end,
            )

        # 1) Terminal enforcement: merge forward/backward until each chunk ends terminally.
        terminal_fixed: list[DocumentChunk] = []
        carry: DocumentChunk | None = None
        for ch in cleaned:
            if carry is None:
                candidate = ch
            else:
                candidate = _merge_two(carry, ch)
            if not candidate.text.strip():
                carry = None
                continue
            if not _is_terminal(candidate.text):
                carry = candidate
                continue
            terminal_fixed.append(candidate)
            carry = None
        if carry is not None:
            if terminal_fixed:
                terminal_fixed[-1] = _merge_two(terminal_fixed[-1], carry)
            else:
                terminal_fixed.append(carry)

        # 2) Merge short chunks backward.
        short_fixed: list[DocumentChunk] = []
        for ch in terminal_fixed:
            if not short_fixed:
                short_fixed.append(ch)
                continue
            if len(ch.text) < _MIN_CHUNK_CHARS:
                short_fixed[-1] = _merge_two(short_fixed[-1], ch)
            else:
                short_fixed.append(ch)
        if short_fixed and len(short_fixed[0].text) < _MIN_CHUNK_CHARS and len(short_fixed) > 1:
            short_fixed[1] = _merge_two(short_fixed[0], short_fixed[1])
            short_fixed = short_fixed[1:]

        # 3) Final pass: merge fragment-like or non-terminal leftovers backward.
        final: list[DocumentChunk] = []
        for ch in short_fixed:
            t = ch.text.strip()
            if not t:
                continue
            if final and (
                len(t) < _MIN_CHUNK_CHARS
                or _starts_like_fragment(t)
                or _ends_like_fragment(t)
                or not _is_terminal(t)
            ):
                final[-1] = _merge_two(final[-1], ch)
            else:
                final.append(ch)

        # Validate texts after merging; drop chunks that are still too short
        # (degenerate single-item groups such as title pages or TOC headers)
        # rather than hard-failing the whole document.
        valid: list[DocumentChunk] = []
        for i, ch in enumerate(final):
            try:
                validate_chunk_text(ch.text or "")
                valid.append(ch)
            except Exception as exc:
                logger.debug(
                    "finalize_chunks: dropping short chunk at index=%d after all merging: %s", i, exc
                )
        return valid

    # Group by (doc_id, page_range) preserving original order.
    #
    # Determinism note:
    # - Python dict preserves insertion order.
    # - Output order is therefore stable as long as the incoming `chunks` list is emitted
    #   in a stable order (e.g., page order from the parser/normalizer).
    # - Avoid introducing upstream iteration over sets or hash-based random ordering.
    grouped: dict[tuple[str, tuple[int, int]], list[DocumentChunk]] = {}
    for ch in chunks:
        key = (ch.doc_id, ch.page_range)
        grouped.setdefault(key, []).append(ch)

    out: list[DocumentChunk] = []
    position = 0
    for (doc_id, page_range), group in grouped.items():
        fixed_chunks = _apply_strict_rules_to_chunk_objects(group)
        for ch in fixed_chunks:
            position += 1
            chunk_id = _stable_chunk_id(
                doc_id,
                page_range,
                ch.paragraph_index_start,
                ch.paragraph_index_end,
                ch.text,
            )
            out.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=ch.text,
                    page_range=page_range,
                    position=position,
                    char_start=ch.char_start,
                    char_end=ch.char_end,
                    paragraph_index_start=ch.paragraph_index_start,
                    paragraph_index_end=ch.paragraph_index_end,
                )
            )

    # For the doc/PDF chunking pipeline, structural metadata must be present before persistence.
    validate_docchunk_structure(out)
    validate_chunks(out)
    return out


def _chunk_text(
    text: str, *, chunk_size_tokens: int, overlap_tokens: int
) -> list[tuple[str, int, int]]:
    if not text:
        return []
    if not isinstance(chunk_size_tokens, int):
        raise ValueError("chunk_size_tokens must be an int")
    if not isinstance(overlap_tokens, int):
        raise ValueError("overlap_tokens must be an int")
    if chunk_size_tokens <= 0:
        logger.warning("chunk_size_tokens <= 0; returning unchunked text")
        # Preserve contract: return structured tuples with deterministic paragraph indices.
        # Treat the entire text as a single paragraph unit.
        t = (text or "").strip()
        if not t:
            return []
        return [(t, 0, 0)]

    target_tokens = int(chunk_size_tokens)
    overlap_tokens = max(0, min(int(overlap_tokens), target_tokens - 1))
    if overlap_tokens and overlap_tokens >= target_tokens:
        logger.warning("overlap_tokens >= target_tokens; clamped to target_tokens - 1")

    paras = _split_paragraphs_with_indices(text)
    units: list[_ChunkUnit] = []
    for para_index, p in paras:
        p = _ensure_terminal(p)
        if _estimate_tokens(p) <= target_tokens:
            units.append(
                _ChunkUnit(
                    text=p,
                    is_paragraph=True,
                    paragraph_index_start=para_index,
                    paragraph_index_end=para_index,
                )
            )
            continue
        # Paragraph too long: split by sentences, keep ≥2 sentences per unit.
        sentences = _split_sentences(p)
        groups = _group_sentences(sentences, min_sentences=2)
        units.extend(
            [
                _ChunkUnit(
                    text=_ensure_terminal(g),
                    is_paragraph=False,
                    paragraph_index_start=para_index,
                    paragraph_index_end=para_index,
                )
                for g in groups
            ]
        )

    chunks: list[tuple[str, int, int]] = []
    current: list[_ChunkUnit] = []
    current_tokens = 0
    current_para_count = 0

    def _emit_current(curr: list[_ChunkUnit]) -> None:
        if not curr:
            return
        chunk_text = " ".join([u.text for u in curr]).strip()
        if not chunk_text:
            return
        p_start = min(u.paragraph_index_start for u in curr)
        p_end = max(u.paragraph_index_end for u in curr)
        if chunks and (
            len(chunk_text) < _MIN_CHUNK_CHARS
            or _starts_like_fragment(chunk_text)
            or _ends_like_fragment(chunk_text)
        ):
            # Merge backward to avoid short/fragment chunks.
            prev_text, prev_start, prev_end = chunks[-1]
            chunks[-1] = (f"{prev_text} {chunk_text}".strip(), min(prev_start, p_start), max(prev_end, p_end))
            return
        chunks.append((chunk_text, p_start, p_end))

    def _overlap_from_current(
        curr: list[_ChunkUnit],
        *,
        overlap_tokens: int,
    ) -> list[_ChunkUnit]:
        if overlap_tokens <= 0 or not curr:
            return []
        # Prefer last full paragraph when available.
        for unit in reversed(curr):
            if unit.is_paragraph and _is_overlap_worthy(unit.text):
                return [unit]
        # Otherwise, include trailing sentence groups until we cross overlap_tokens.
        picked: list[_ChunkUnit] = []
        total_tokens = 0
        for unit in reversed(curr):
            if _is_overlap_worthy(unit.text):
                picked.append(unit)
                total_tokens += _estimate_tokens(unit.text)
                if total_tokens >= overlap_tokens:
                    break
        picked.reverse()
        return picked

    for unit in units:
        unit_tokens = _estimate_tokens(unit.text)
        if not current:
            current.append(unit)
            current_tokens = unit_tokens
            current_para_count = 1 if unit.is_paragraph else 0
            continue

        soft_limit = int(target_tokens * (1.0 + _SOFT_OVERFLOW_FRACTION))
        if current_tokens + unit_tokens <= soft_limit:
            current.append(unit)
            current_tokens += unit_tokens
            current_para_count += 1 if unit.is_paragraph else 0
            continue

        # Emit current chunk
        curr_text = " ".join([u.text for u in current]).strip()
        if curr_text and (_starts_like_fragment(curr_text) or _ends_like_fragment(curr_text)):
            # Avoid fragment boundaries: keep accumulating even if over soft cap.
            current.append(unit)
            current_tokens += unit_tokens
            current_para_count += 1 if unit.is_paragraph else 0
            continue

        _emit_current(current)

        # Start new chunk with paragraph-boundary overlap only
        overlap_units = _overlap_from_current(current, overlap_tokens=overlap_tokens)

        current = overlap_units + [unit]
        current_tokens = sum(_estimate_tokens(u.text) for u in current)
        current_para_count = sum(1 for u in current if u.is_paragraph)

    if current:
        _emit_current(current)

    # Note: _merge_short_chunks is text-only. Strict merging/validation happens at finalize_chunks()
    # with DocumentChunk objects, so this stage returns raw structural tuples without merging.
    return chunks


def chunk_sections(
    sections: list[NormalizedSection],
    *,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[DocumentChunk]:
    """
    Split sections into retrieval chunks.

    Generates stable chunk IDs based on doc-local structural address (page_range + paragraph indices) + content hash.
    """
    if not isinstance(sections, list):
        raise ValueError("chunk_sections: sections must be a list")
    if not isinstance(chunk_size_tokens, int) or not isinstance(overlap_tokens, int):
        raise ValueError("chunk_sections: chunk_size_tokens/overlap_tokens must be ints")

    chunks: list[DocumentChunk] = []
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

        for chunk_text, p_start, p_end in emitted:
            if not chunk_text.strip():
                continue
            position += 1
            chunk_id = _stable_chunk_id(sec.doc_id, sec.page_range, p_start, p_end, chunk_text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=sec.doc_id,
                    text=chunk_text,
                    page_range=sec.page_range,
                    position=position,
                    paragraph_index_start=p_start,
                    paragraph_index_end=p_end,
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
