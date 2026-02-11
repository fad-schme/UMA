from __future__ import annotations

import logging
import re
from uuid import uuid4
from typing import List

from .types import ParsedDocument, NormalizedSection

logger = logging.getLogger(__name__)

_HEADER_FOOTER_RE = re.compile(r"^\s*\d+\s*$")
_WHITESPACE_RE = re.compile(r"[ \t\x0b\x0c]+")
_DEHYPHENATE_RE = re.compile(r"([A-Za-z])-\n([A-Za-z])")


def _dehyphenate_linebreaks(text: str) -> str:
    """
    Join hyphenation splits created by PDF extraction.

    Example: "inter-\nnal" -> "internal"
    This is intentionally conservative: alphabetic-only on both sides.
    """
    if not text:
        return ""
    return _DEHYPHENATE_RE.sub(r"\1\2", text)


def _reflow_soft_wrapped_lines(text: str) -> str:
    """
    Reflow "soft" line wraps inside a paragraph without removing blank-line boundaries.

    Conservative deterministic rules:
    - Preserve blank lines as paragraph boundaries.
    - If a line does not end terminally and the next line starts with lowercase,
      join with a space.
    - Otherwise preserve the newline.
    """
    if not text:
        return ""
    out_lines: List[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            out_lines.append("")
            i += 1
            continue
        cur = line.rstrip()
        # Merge forward while the next line looks like a soft wrap continuation.
        while i + 1 < len(lines):
            nxt = lines[i + 1]
            if not nxt.strip():
                break
            nxt_stripped = nxt.lstrip()
            if (
                cur
                and cur[-1] not in ".!?:;"
                and nxt_stripped
                and nxt_stripped[0].islower()
            ):
                cur = f"{cur} {nxt_stripped}".rstrip()
                i += 1
                continue
            break
        out_lines.append(cur)
        i += 1
    return "\n".join(out_lines)


def _drop_repeated_lines_across_pages(pages: List[str], *, min_repeats: int = 3) -> List[str]:
    """
    Drop lines that repeat across many pages (common headers/footers).

    This is deterministic and conservative: only exact matches after basic trimming.
    """
    if not pages:
        return []
    counts: dict[str, int] = {}
    per_page_lines: List[List[str]] = []
    for text in pages:
        lines = []
        for ln in (text or "").splitlines():
            s = ln.strip()
            if not s:
                lines.append("")
                continue
            lines.append(s)
        per_page_lines.append(lines)
        seen = set()
        for s in lines:
            if not s or s in seen:
                continue
            seen.add(s)
            counts[s] = counts.get(s, 0) + 1
    to_drop = {s for s, c in counts.items() if c >= int(min_repeats)}
    if not to_drop:
        return pages
    out_pages: List[str] = []
    for lines in per_page_lines:
        kept = [ln for ln in lines if not (ln and ln in to_drop)]
        out_pages.append("\n".join(kept))
    return out_pages


def _clean_page_text(text: str) -> str:
    if not text:
        return ""
    text = _dehyphenate_linebreaks(text)
    text = _reflow_soft_wrapped_lines(text)
    lines = []
    for line in text.splitlines():
        if _HEADER_FOOTER_RE.match(line.strip()):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_document(parsed: ParsedDocument) -> List[NormalizedSection]:
    """
    Normalize document pages into sections.

    Responsibilities:
    - Remove obvious headers/footers
    - Collapse whitespace
    - Preserve page boundaries
    """
    if parsed is None:
        raise ValueError("normalize_document: parsed document is required")

    # Remove repeated headers/footers across pages before per-page normalization.
    page_texts = [t for _, t in (parsed.pages or [])]
    page_texts = _drop_repeated_lines_across_pages(page_texts, min_repeats=3)

    sections: List[NormalizedSection] = []
    for (page_num, _), text in zip(parsed.pages, page_texts):
        cleaned = _clean_page_text(text or "")
        if not cleaned:
            continue
        section_id = f"sec_{uuid4().hex}"
        sections.append(
            NormalizedSection(
                section_id=section_id,
                doc_id=parsed.doc_id,
                text=cleaned,
                page_range=(page_num, page_num),
            )
        )

    if not sections:
        logger.warning("normalize_document: no sections produced for doc_id=%s", parsed.doc_id)

    return sections
