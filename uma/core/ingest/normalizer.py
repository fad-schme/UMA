from __future__ import annotations

import logging
import re
from uuid import uuid4
from typing import List

from .types import ParsedDocument, NormalizedSection

logger = logging.getLogger(__name__)

_HEADER_FOOTER_RE = re.compile(r"^\s*\d+\s*$")
_WHITESPACE_RE = re.compile(r"[ \t\x0b\x0c]+")


def _clean_page_text(text: str) -> str:
    if not text:
        return ""
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

    sections: List[NormalizedSection] = []
    for page_num, text in parsed.pages:
        cleaned = _clean_page_text(text)
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
