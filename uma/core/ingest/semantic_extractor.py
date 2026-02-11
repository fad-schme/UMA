from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from ..semantic.scorer import SalienceScorer
from .types import DocumentChunk, ExtractedFact

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)

_EXTRACT_KEYWORDS = (
    "architecture",
    "design",
    "principle",
    "principles",
    "control",
    "controls",
    "iam",
    "network",
    "segmentation",
    "encryption",
    "threat",
    "risk",
    "policy",
    "logging",
)

_EXTRACT_BOILERPLATE = (
    "all rights reserved",
    "copyright",
    "table of contents",
    "page ",
)


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


def _try_parse_json_list(raw: str) -> List[Any] | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else None
    except Exception:
        pass
    m = _JSON_LIST_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, list) else None
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


def _build_batch_prompt(
    items: List[Dict[str, str]],
    *,
    min_fact_words: int,
) -> List[Dict[str, str]]:
    system = (
        "Extract KB-grade facts from each chunk independently.\n"
        "IMPORTANT RULES:\n"
        "- Facts for a chunk_id MUST be derived ONLY from that chunk's text.\n"
        "- Do not mix information across chunks.\n"
        f"- Each object sentence must be at least {min_fact_words} words long.\n"
        "- Use predicate=STATES unless a stronger relation is explicit.\n"
        "Return ONLY valid JSON in this schema (keys MUST be chunk_ids from input):\n"
        "{\n"
        "  \"chunks\": {\n"
        "    \"chunk_id\": {\n"
        "      \"facts\": [\n"
        "        {\n"
        "          \"subject\": \"...\",\n"
        "          \"predicate\": \"STATES\",\n"
        "          \"object\": \"...\",\n"
        "          \"confidence\": 0.0-1.0\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    user = json.dumps({"chunks": items}, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _partition_batches_by_chars(
    chunks: List[DocumentChunk],
    *,
    batch_size_chunks: int,
    max_chars: int,
) -> List[List[DocumentChunk]]:
    if not chunks:
        return []
    batch_size_chunks = max(1, int(batch_size_chunks))
    max_chars = max(1000, int(max_chars))

    # Deterministic order: position then chunk_id.
    ordered = sorted(chunks, key=lambda c: (int(getattr(c, "position", 0) or 0), getattr(c, "chunk_id", "")))
    batches: List[List[DocumentChunk]] = []
    cur: List[DocumentChunk] = []
    cur_chars = 0
    for ch in ordered:
        t = (ch.text or "")
        est = len(t)
        if cur and (len(cur) >= batch_size_chunks or cur_chars + est > max_chars):
            batches.append(cur)
            cur = []
            cur_chars = 0
        cur.append(ch)
        cur_chars += est
    if cur:
        batches.append(cur)
    return batches


def _parse_facts_payload_for_chunk(
    *,
    chunk: DocumentChunk,
    payload: Any,
    min_fact_words: int,
    scorer: SalienceScorer,
) -> List[ExtractedFact]:
    if not isinstance(payload, dict):
        return []
    facts_payload = payload.get("facts")
    if not isinstance(facts_payload, list):
        return []

    extracted: List[ExtractedFact] = []
    for item in facts_payload:
        if not isinstance(item, dict):
            continue
        subj = item.get("subject")
        pred = item.get("predicate") or "STATES"
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
                subject=_normalize_subject(str(subj), doc_id=chunk.doc_id),
                predicate=str(pred),
                object=obj_text,
                confidence=confidence,
                salience=salience,
                source_chunk_id=chunk.chunk_id,
            )
        )
    return extracted


async def extract_facts_batch(
    chunks: List[DocumentChunk],
    *,
    llm: Any,
    min_fact_words: int = 0,
    batch_size_chunks: int = 4,
    max_chars: int = 12000,
) -> List[ExtractedFact]:
    """
    Batched fact extraction with strict chunk_id-keyed JSON and partial salvage.

    - Batches are deterministic (position, chunk_id).
    - If batch output is missing/invalid for some chunks, fall back per-chunk for only those.
    """
    if not chunks:
        return []
    if llm is None or not hasattr(llm, "generate"):
        raise ValueError("extract_facts_batch: llm with .generate() required")

    scorer = SalienceScorer()
    extracted: List[ExtractedFact] = []
    batches = _partition_batches_by_chars(chunks, batch_size_chunks=batch_size_chunks, max_chars=max_chars)

    for batch_idx, batch in enumerate(batches):
        items = [{"chunk_id": c.chunk_id, "text": (c.text or "")} for c in batch]
        raw = ""
        data = None
        try:
            raw = await llm.generate(
                _build_batch_prompt(items, min_fact_words=max(0, int(min_fact_words))),
                max_tokens=800,
                temperature=0.0,
            )
            data = _try_parse_json(raw)
        except Exception:
            logger.exception("extract_facts_batch: llm generate failed batch_idx=%d", batch_idx)
            data = None

        if data is None:
            # Repair pass
            try:
                repair_prompt = [
                    {"role": "system", "content": "Return ONLY valid JSON. No prose."},
                    {"role": "user", "content": raw},
                ]
                repaired = await llm.generate(repair_prompt, max_tokens=600, temperature=0.0)
                data = _try_parse_json(repaired)
            except Exception:
                data = None

        chunks_obj = data.get("chunks") if isinstance(data, dict) else None
        good_ids: set[str] = set()
        if isinstance(chunks_obj, dict):
            for c in batch:
                payload = chunks_obj.get(c.chunk_id)
                facts_for_chunk = _parse_facts_payload_for_chunk(
                    chunk=c,
                    payload=payload,
                    min_fact_words=min_fact_words,
                    scorer=scorer,
                )
                if facts_for_chunk or isinstance(payload, dict):
                    # consider payload handled even if no facts (valid empty)
                    good_ids.add(c.chunk_id)
                extracted.extend(facts_for_chunk)

        # Salvage: per-chunk fallback only for missing/invalid payloads.
        for c in batch:
            if c.chunk_id in good_ids:
                continue
            extracted.extend(await extract_facts_one(c, llm=llm, min_fact_words=min_fact_words, scorer=scorer))

    return extracted


async def extract_facts_one(
    chunk: DocumentChunk,
    *,
    llm: Any,
    min_fact_words: int = 0,
    scorer: SalienceScorer | None = None,
) -> List[ExtractedFact]:
    if chunk is None:
        return []
    if llm is None or not hasattr(llm, "generate"):
        raise ValueError("extract_facts_one: llm with .generate() required")

    text = (chunk.text or "").strip()
    if not text:
        return []

    raw = ""
    try:
        raw = await llm.generate(
            _build_prompt(text, min_fact_words=max(0, int(min_fact_words))),
            max_tokens=400,
            temperature=0.0,
        )
    except Exception:
        logger.exception("extract_facts_one: llm generate failed for chunk=%s", chunk.chunk_id)
        return []

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
        logger.info("extract_facts_one: skipping chunk due to invalid JSON (chunk=%s)", chunk.chunk_id)
        return []

    facts_payload = data.get("facts")
    if not isinstance(facts_payload, list):
        return []

    scorer = scorer or SalienceScorer()
    extracted: List[ExtractedFact] = []

    for item in facts_payload:
        if not isinstance(item, dict):
            continue
        subj = item.get("subject")
        pred = item.get("predicate") or "STATES"
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


def select_chunks_for_fact_extraction(
    chunks: List[DocumentChunk],
    *,
    max_chunks: int,
    max_per_page: int = 2,
    min_chars: int = 200,
) -> List[DocumentChunk]:
    """
    Deterministically rank and select chunks for LLM fact extraction.

    Replaces blunt "first N chunks" slicing with a stable heuristic that prefers:
    - substantive (longer) content
    - sentence-rich text
    - domain/heading cues
    - diversity across page_range (cap per page)
    """
    if not chunks:
        return []
    max_chunks = max(0, int(max_chunks))
    if max_chunks == 0:
        return []
    max_per_page = max(1, int(max_per_page))
    min_chars = max(0, int(min_chars))

    def _score(ch: DocumentChunk) -> float:
        t0 = (ch.text or "")
        t = t0.strip().lower()
        if not t:
            return -1e9
        if min_chars and len(t0) < min_chars:
            return -1e6
        length = min(len(t), 4000) / 4000.0
        kw = sum(1 for k in _EXTRACT_KEYWORDS if k in t)
        boiler = sum(1 for b in _EXTRACT_BOILERPLATE if b in t)
        punct = sum(t.count(p) for p in ".!?") / max(1.0, len(t) / 200.0)
        return 2.0 * length + 0.3 * float(kw) + 0.2 * float(punct) - 2.0 * float(boiler)

    ranked = sorted(
        chunks,
        key=lambda ch: (_score(ch), int(getattr(ch, "position", 0) or 0), getattr(ch, "chunk_id", "")),
        reverse=True,
    )

    out: List[DocumentChunk] = []
    per_page: Dict[tuple[int, int], int] = {}
    for ch in ranked:
        pr = getattr(ch, "page_range", None)
        if not isinstance(pr, tuple) or len(pr) != 2:
            continue
        per_page[pr] = per_page.get(pr, 0) + 1
        if per_page[pr] > max_per_page:
            continue
        out.append(ch)
        if len(out) >= max_chunks:
            break
    return out


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
        extracted.extend(
            await extract_facts_one(
                chunk,
                llm=llm,
                min_fact_words=min_fact_words,
                scorer=scorer,
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
