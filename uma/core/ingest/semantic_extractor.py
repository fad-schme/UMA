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


def _build_prompt(chunk_text: str, *, min_fact_words: int) -> List[Dict[str, str]]:
    system = (
        "Extract KB-grade facts from the document text.\n"
        "Each fact MUST be a complete, self-contained sentence in the object field.\n"
        f"Each object sentence must be at least {min_fact_words} words long.\n"
        "Prefer stable, durable claims over short fragments.\n"
        "Use predicate=STATES unless a stronger relation is explicit.\n"
        "Return ONLY valid JSON in this schema:\n"
        "{\n"
        "  \"facts\": [\n"
        "    {\n"
        "      \"subject\": \"...\",\n"
        "      \"predicate\": \"STATES\",\n"
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
    Normalize extracted subjects into a stable namespace.

    Document-derived facts should be discoverable via corpus-wide semantic search,
    so default to entity:* rather than doc-scoped subjects.
    """
    subj = (subject or "").strip()
    if not subj:
        return "entity:unknown"
    if subj.startswith(("doc:", "entity:", "user:", "agent:")):
        return subj
    return f"entity:{subj}"

def _word_count(text: str) -> int:
    return len([t for t in (text or "").strip().split() if t])

def _build_summary_prompt(doc_text: str, *, max_facts: int, min_fact_words: int) -> List[Dict[str, str]]:
    system = (
        "Summarize the document into KB-grade facts.\n"
        f"Return up to {max_facts} facts.\n"
        "Each fact MUST be a complete, self-contained sentence in the object field.\n"
        f"Each object sentence must be at least {min_fact_words} words long.\n"
        "Use predicate=SUMMARY for all facts.\n"
        "Return ONLY valid JSON in this schema:\n"
        "{\n"
        "  \"facts\": [\n"
        "    {\n"
        "      \"subject\": \"...\",\n"
        "      \"predicate\": \"SUMMARY\",\n"
        "      \"object\": \"...\",\n"
        "      \"confidence\": 0.0-1.0\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": doc_text},
    ]


async def extract_facts(
    chunks: List[DocumentChunk],
    *,
    llm: Any,
    min_fact_words: int = 0,
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
            raw = await llm.generate(
                _build_prompt(text, min_fact_words=max(0, int(min_fact_words))),
                max_tokens=400,
                temperature=0.0,
            )
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

            obj_text = obj if isinstance(obj, str) else json.dumps(obj)
            if min_fact_words and _word_count(obj_text) < int(min_fact_words):
                continue

            # Use salience scorer from core semantic
            from ...types_fact import Fact
            from datetime import datetime, timezone
            fact_stub = Fact(
                id="stub",
                subject=str(subj),
                predicate=str(pred),
                object=obj_text,
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
                    object=obj_text,
                    confidence=confidence,
                    salience=salience,
                    source_chunk_id=chunk.chunk_id,
                )
            )

    return extracted


async def extract_summary_facts(
    doc_text: str,
    *,
    llm: Any,
    max_facts: int = 5,
    min_fact_words: int = 0,
    doc_id: str | None = None,
) -> List[ExtractedFact]:
    if not doc_text or not isinstance(doc_text, str):
        return []
    if llm is None or not hasattr(llm, "generate"):
        raise ValueError("extract_summary_facts: llm with .generate() required")

    scorer = SalienceScorer()
    max_facts = max(0, int(max_facts))
    if max_facts == 0:
        return []

    raw = ""
    try:
        raw = await llm.generate(
            _build_summary_prompt(doc_text, max_facts=max_facts, min_fact_words=max(0, int(min_fact_words))),
            max_tokens=400,
            temperature=0.0,
        )
    except Exception:
        logger.exception("extract_summary_facts: llm generate failed")
        return []

    data = _try_parse_json(raw)
    if data is None:
        logger.info("extract_summary_facts: invalid JSON output")
        return []

    facts_payload = data.get("facts")
    if not isinstance(facts_payload, list):
        return []

    extracted: List[ExtractedFact] = []
    for item in facts_payload[:max_facts]:
        if not isinstance(item, dict):
            continue
        subj = item.get("subject")
        pred = item.get("predicate") or "SUMMARY"
        obj = item.get("object")
        conf = item.get("confidence", 0.7)
        if not subj or obj is None:
            continue
        try:
            confidence = float(conf)
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = 0.7

        obj_text = obj if isinstance(obj, str) else json.dumps(obj)
        if min_fact_words and _word_count(obj_text) < int(min_fact_words):
            continue

        from ...types_fact import Fact
        from datetime import datetime, timezone
        fact_stub = Fact(
            id="stub",
            subject=str(subj),
            predicate=str(pred),
            object=obj_text,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            confidence=confidence,
            meta={},
        )
        salience = scorer.score(fact_stub)

        extracted.append(
            ExtractedFact(
                subject=_normalize_subject(str(subj), doc_id=doc_id),
                predicate=str(pred),
                object=obj_text,
                confidence=confidence,
                salience=salience,
                source_chunk_id="",
            )
        )

    return extracted
