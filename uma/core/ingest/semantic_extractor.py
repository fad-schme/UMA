from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from ..semantic.scorer import SalienceScorer
from .types import DocumentChunk, ExtractedFact

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_json(raw: str) -> Dict[str, Any] | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = _JSON_OBJECT_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _build_prompt(chunk_text: str) -> List[Dict[str, str]]:
    system = (
        "Extract structured facts from the document text.\n"
        "Return ONLY valid JSON in this schema:\n"
        "{\n"
        "  \"facts\": [\n"
        "    {\n"
        "      \"subject\": \"...\",\n"
        "      \"predicate\": \"...\",\n"
        "      \"object\": \"...\",\n"
        "      \"confidence\": 0.0-1.0\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": chunk_text},
    ]


def _normalize_subject(subject: str, *, doc_id: str | None = None) -> str:
    """
    Normalize subject to a stable namespace.
    """
    subj = (subject or "").strip()
    if not subj:
        return f"doc:{doc_id}" if doc_id else "doc:unknown"
    if subj.startswith(("doc:", "entity:", "user:", "project:", "agent:")):
        return subj
    if doc_id:
        return f"doc:{doc_id}:{subj}"
    return f"entity:{subj}"


async def extract_facts(
    chunks: List[DocumentChunk],
    *,
    llm: Any,
) -> List[ExtractedFact]:
    """
    LLM-based fact extraction from document chunks.

    Returns a list of ExtractedFact (no persistence here).
    """
    if not chunks:
        return []
    if llm is None or not hasattr(llm, "generate"):
        raise ValueError("extract_facts: llm with .generate() required")

    scorer = SalienceScorer()
    extracted: List[ExtractedFact] = []

    for chunk in chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue

        raw = ""
        try:
            raw = await llm.generate(_build_prompt(text), max_tokens=400, temperature=0.0)
        except Exception:
            logger.exception("extract_facts: llm generate failed for chunk=%s", chunk.chunk_id)
            continue

        data = _try_parse_json(raw)
        if data is None:
            # Repair pass with stricter instruction
            try:
                repair_prompt = [
                    {"role": "system", "content": "Return ONLY valid JSON. No prose."},
                    {"role": "user", "content": raw},
                ]
                repaired = await llm.generate(repair_prompt, max_tokens=300, temperature=0.0)
                data = _try_parse_json(repaired)
            except Exception:
                data = None
        if data is None:
            logger.info("extract_facts: skipping chunk due to invalid JSON (chunk=%s)", chunk.chunk_id)
            continue

        facts_payload = data.get("facts")
        if not isinstance(facts_payload, list):
            continue

        for item in facts_payload:
            if not isinstance(item, dict):
                continue
            subj = item.get("subject")
            pred = item.get("predicate")
            obj = item.get("object")
            conf = item.get("confidence", 0.7)
            if not subj or not pred or obj is None:
                continue
            try:
                confidence = float(conf)
                confidence = max(0.0, min(1.0, confidence))
            except Exception:
                confidence = 0.7

            # Use salience scorer from core semantic
            from ...types_fact import Fact
            from datetime import datetime, timezone
            fact_stub = Fact(
                id="stub",
                subject=str(subj),
                predicate=str(pred),
                object=obj,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                confidence=confidence,
                meta={},
            )
            salience = scorer.score(fact_stub)

            extracted.append(
                ExtractedFact(
                    subject=_normalize_subject(str(subj), doc_id=chunk.doc_id),
                    predicate=str(pred),
                    object=obj if isinstance(obj, str) else json.dumps(obj),
                    confidence=confidence,
                    salience=salience,
                    source_chunk_id=chunk.chunk_id,
                )
            )

    return extracted
