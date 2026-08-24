"""
uma.retrieve.rlm.entity_seed
=================================

Deterministic evidence entity extraction for topical graph seeding.

This module is intentionally lightweight and LLM-free. It produces a small list
of candidate entity strings that can be resolved to graph node ids.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from uma.common.text import extract_keywords_and_phrases


logger = logging.getLogger(__name__)
_RE_ACRONYM = re.compile(r"\b[A-Z]{2,10}\b")


def _safe_str(x: Any) -> str:
    try:
        return str(x or "").strip()
    except Exception as exc:
        logger.debug("_safe_str: string coercion failed: %s", exc, exc_info=True)
        return ""


def _fact_text(f: Any) -> str:
    try:
        meta = f.get("meta") if isinstance(f, dict) else getattr(f, "meta", None)
    except Exception as exc:
        logger.debug("_fact_text: fact metadata access failed: %s", exc, exc_info=True)
        meta = None
    if isinstance(meta, dict):
        ft = _safe_str(meta.get("fact_text"))
        if ft:
            return ft
    try:
        obj = f.get("object") if isinstance(f, dict) else getattr(f, "object", None)
        return _safe_str(obj)
    except Exception as exc:
        logger.debug("_fact_text: fact object access failed: %s", exc, exc_info=True)
        return ""


def _chunk_text(ch: Any) -> str:
    try:
        return _safe_str(ch.get("text") if isinstance(ch, dict) else getattr(ch, "text", None))
    except Exception as exc:
        logger.debug("_chunk_text: chunk text access failed: %s", exc, exc_info=True)
        return ""


def extract_candidate_entities(
    query_text: str,
    facts: list[Any],
    chunks: list[Any],
    limit: int = 5,
) -> list[str]:
    """
    Extract a bounded list of candidate entity strings for graph seeding.

    Deterministic approach:
    - keyphrases/keywords from extract_keywords_and_phrases(query_text)
    - capitalized acronyms in the query (IAM, VPC, KMS, TLS)
    - optional extra terms from a small evidence blob (facts/chunks)
    """
    limit_i = max(0, int(limit))
    if limit_i == 0:
        return []

    out: list[str] = []
    seen = set()

    q = _safe_str(query_text)
    if not q:
        return []

    # Acronyms first: they're usually high-precision topical anchors (IAM, VPC, KMS, TLS).
    for m in _RE_ACRONYM.findall(q):
        s = _safe_str(m)
        if not s:
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= limit_i:
            return out

    try:
        extracted = extract_keywords_and_phrases(q)
    except Exception:
        extracted = {"keyphrases": [], "keywords": []}

    for key in ("keyphrases", "keywords"):
        vals = extracted.get(key) if isinstance(extracted, dict) else None
        if isinstance(vals, list):
            for v in vals:
                s = _safe_str(v)
                if not s:
                    continue
                if s.lower() in seen:
                    continue
                seen.add(s.lower())
                out.append(s)
                if len(out) >= limit_i:
                    return out

    # Evidence blob: bounded and optional.
    blob_parts: list[str] = []
    for f in (facts or [])[:8]:
        t = _fact_text(f)
        if t:
            blob_parts.append(t)
    for ch in (chunks or [])[:3]:
        t = _chunk_text(ch)
        if t:
            blob_parts.append(t)

    blob = " ".join(blob_parts)
    if blob:
        blob = blob[:4000]
        try:
            e2 = extract_keywords_and_phrases(blob)
        except Exception:
            e2 = {"keyphrases": [], "keywords": []}
        for key in ("keyphrases", "keywords"):
            vals = e2.get(key) if isinstance(e2, dict) else None
            if isinstance(vals, list):
                for v in vals:
                    s = _safe_str(v)
                    if not s:
                        continue
                    if s.lower() in seen:
                        continue
                    seen.add(s.lower())
                    out.append(s)
                    if len(out) >= limit_i:
                        return out

    return out[:limit_i]
