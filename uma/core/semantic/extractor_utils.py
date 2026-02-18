from __future__ import annotations

"""
uma.core.semantic.extractor_utils
================================

Shared helpers for semantic extraction.

This module contains implementation helpers used by `uma.core.semantic.extractor`.
It is *not* intended to be imported or used directly by other modules.

Most logic is intentionally kept here so `extractor.py` can remain the single,
canonical public API surface for fact extraction.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..ingest.types import DocumentChunk, ExtractedFact
from ..llm.controller import LLMCallContext, generate_json
from .scorer import SalienceScorer
from ..utils.user_query_helper import get_generic_terms

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FACTS_PER_CHUNK = 4
_DEFAULT_OBJECT_MAX_WORDS = 50
_DEFAULT_MAX_FACT_TOKENS = 120  # per fact, estimated
_DEFAULT_MIN_FACT_WORDS = 0

_MIN_EXTRACT_CHUNK_CHARS = 300  # skip LLM extraction for tiny chunks (usually headers/footers/fragments)

# LLM output token budget (for the call). Output is additionally bounded by hard caps above.
# Keep this high enough for JSON overhead + 4 facts, but not huge.
_DEFAULT_SINGLE_CALL_MAX_TOKENS = 700
_DEFAULT_BATCH_CALL_MAX_TOKENS = 1200
_DEFAULT_SUMMARY_CALL_MAX_TOKENS = 800

# Generic boilerplate suppression: not domain-specific
_BOILERPLATE_SNIPPETS = (
    "all rights reserved",
    "copyright",
    "table of contents",
    "page ",
)



def _fallback_fact_for_chunk(
    ch: DocumentChunk,
    *,
    doc_id: str,
    object_max_words: int,
    max_fact_tokens: int,
) -> ExtractedFact:
    """Deterministic, data-agnostic fallback used only when extraction yields 0 facts for an eligible chunk.

    IMPORTANT:
    - Must conform to ExtractedFact fields used by ingest_service.
    - Must always carry `source_chunk_id` so facts map back to chunks.
    """
    text = _normalize_ws(ch.text or "")
    # Take a short prefix (1–2 sentences-ish) without any domain assumptions.
    prefix = text
    # Try to cut at a sentence boundary early to keep it clean.
    m = re.search(r"(.+?[.!?])\s", text)
    if m:
        prefix = m.group(1)
    prefix = _truncate_words(prefix, int(object_max_words))

    subj = _normalize_subject("", doc_id=doc_id)
    obj = prefix
    # Enforce token cap deterministically (approx).
    while obj and _estimate_tokens(obj) > int(max_fact_tokens):
        obj = _truncate_words(obj, max(5, int(len(obj.split()) * 0.85)))

    if not obj:
        obj = _truncate_words(text, int(object_max_words))

    return ExtractedFact(
        subject=subj,
        predicate="STATES",
        object=obj,
        confidence=0.6,
        salience=0.0,
        source_chunk_id=str(getattr(ch, "chunk_id", "") or ""),
    )


# -----------------------------
# Small text helpers
# -----------------------------
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENT_END_RE = re.compile(r"[.!?]\s")
_LIST_LINE_RE = re.compile(r"^\s*([-*•]|\d+\.)\s+", re.MULTILINE)


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _estimate_tokens(text: str) -> int:
    # Cheap deterministic approximation: ~4 chars/token
    t = (text or "").strip()
    if not t:
        return 0
    return max(1, int(len(t) / 4))


def _truncate_words(text: str, max_words: int) -> str:
    if not text or max_words <= 0:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_float(v: Any, default: float = 0.7) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _normalize_subject(subject: str, *, doc_id: str) -> str:
    # Keep it generic: stable per-document subject label.
    s = _normalize_ws(subject)
    if not s:
        return f"Document({doc_id})"
    # Avoid extremely long subjects.
    return _truncate_words(s, 12)


# -----------------------------
# Prompt builder (no domain cues)
# -----------------------------
def _build_prompt(
    *,
    min_fact_words: int,
    mode: str,
    chunk_text: Optional[str] = None,
    items: Optional[List[Dict[str, str]]] = None,
    doc_text: Optional[str] = None,
    max_facts: Optional[int] = None,
    max_facts_per_chunk: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Keep prompts generic and schema-strict.
    Hard constraints (max facts, word caps, token caps) are enforced in code after parsing.
    """
    min_fact_words = max(0, int(min_fact_words))

    if mode == "batch":
        mf = int(max_facts_per_chunk) if max_facts_per_chunk is not None else _DEFAULT_MAX_FACTS_PER_CHUNK

        system = (
            "Extract KB-grade facts from each chunk independently.\n"
            "IMPORTANT RULES:\n"
            "- Facts for a chunk_id MUST be derived ONLY from that chunk's text.\n"
            "- Do not mix information across chunks.\n"
            f"- Each object sentence must be at least {min_fact_words} words long.\n"
            f"- Return AT MOST {mf} facts per chunk_id. If more are possible, choose the {mf} most salient.\n"
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
            "          \"confidence\": \"0.0-1.0\"\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  }\n"
            "}\n"
        )
        user = json.dumps({"chunks": items or []}, ensure_ascii=False)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if mode == "summary":
        mf = int(max_facts) if max_facts is not None else 5
        system = (
            "Summarize the document into KB-grade facts.\n"
            f"Return up to {mf} facts.\n"
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
            "      \"confidence\": \"0.0-1.0\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": doc_text or ""},
        ]

    # default: single chunk
    mf = int(max_facts_per_chunk) if max_facts_per_chunk is not None else _DEFAULT_MAX_FACTS_PER_CHUNK
    system = (
        "Extract KB-grade facts from the document text.\n"
        "Each fact MUST be a complete, self-contained sentence in the object field.\n"
        f"Each object sentence must be at least {min_fact_words} words long.\n"
        f"Return AT MOST {mf} facts. If more are possible, choose the {mf} most salient.\n"
        "Prefer stable, durable claims over short fragments.\n"
        "Use predicate=STATES unless a stronger relation is explicit.\n"
        "Return ONLY valid JSON in this schema:\n"
        "{\n"
        "  \"facts\": [\n"
        "    {\n"
        "      \"subject\": \"...\",\n"
        "      \"predicate\": \"STATES\",\n"
        "      \"object\": \"...\",\n"
        "      \"confidence\": \"0.0-1.0\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": chunk_text or ""},
    ]


# -----------------------------
# Chunk selection: generic, data-agnostic
# -----------------------------
def _boilerplate_penalty(text: str) -> float:
    t = (text or "").lower()
    if not t:
        return 0.0
    for s in _BOILERPLATE_SNIPPETS:
        if s in t:
            return 1.0
    return 0.0


def _sentence_density(text: str) -> float:
    t = text or ""
    if not t:
        return 0.0
    sents = len(_SENT_END_RE.findall(t)) + (1 if t.strip().endswith((".", "!", "?")) else 0)
    return sents / max(1.0, len(t) / 400.0)  # sentences per ~400 chars


def _list_density(text: str) -> float:
    t = text or ""
    if not t:
        return 0.0
    bullets = len(_LIST_LINE_RE.findall(t))
    return bullets / max(1.0, len(t) / 400.0)


def _definition_like_score(text: str) -> float:
    """
    Generic “claim marker” cues (not domain-specific).
    """
    t = (text or "").lower()
    if not t:
        return 0.0
    cues = (
        " is ",
        " are ",
        " means ",
        " defined as ",
        " consists of ",
        " includes ",
        " requires ",
        " must ",
        " shall ",
        " should ",
    )
    hits = sum(1 for c in cues if c in t)
    return float(hits)


def _generic_term_ratio(text: str) -> float:
    """
    Use the repo's generic term list to avoid over-scoring high-level boilerplatey sections.
    """
    generic = get_generic_terms()
    tokens = re.findall(r"[a-z0-9-]+", (text or "").lower())
    if not tokens:
        return 0.0
    # only count alphabetic-ish tokens; ignore numeric
    words = [t for t in tokens if any(ch.isalpha() for ch in t)]
    if not words:
        return 0.0
    g = sum(1 for w in words if w in generic)
    return g / max(1, len(words))


def _chunk_quality_score(ch: DocumentChunk) -> float:
    """
    Higher score => more likely to contain extractable facts.
    Data-agnostic: no domain lexicon.
    """
    text = (ch.text or "").strip()
    if not text:
        return 0.0

    length = len(text)
    if length < 120:
        return 0.0

    # Base: prefer medium length chunks (not too short, not huge)
    length_score = min(1.0, length / 1200.0)

    sent = _sentence_density(text)
    lst = _list_density(text)
    defs = _definition_like_score(text)

    boiler = _boilerplate_penalty(text)
    gen_ratio = _generic_term_ratio(text)

    # Combine: keep weights simple
    score = 0.0
    score += 1.2 * length_score
    score += 0.8 * min(sent, 5.0) / 5.0
    score += 0.6 * min(lst, 5.0) / 5.0
    score += 0.6 * min(defs, 6.0) / 6.0

    # Penalize boilerplate strongly; penalize high generic-term ratio moderately
    score -= 1.2 * boiler
    if gen_ratio > 0.25:
        score -= 0.4 * min(1.0, (gen_ratio - 0.25) / 0.5)

    return max(0.0, score)



# -----------------------------
# Batching: FIX max_chars and deterministic grouping
# -----------------------------
def _partition_batches_by_chars(
    chunks: List[DocumentChunk],
    *,
    batch_size_chunks: int,
    max_chars: int,
) -> List[List[DocumentChunk]]:
    if not chunks:
        return []
    batch_size_chunks = max(1, int(batch_size_chunks))
    max_chars = max(2000, int(max_chars))  # avoid pathological tiny batches

    ordered = sorted(chunks, key=lambda c: (int(getattr(c, "position", 0) or 0), getattr(c, "chunk_id", "")))

    batches: List[List[DocumentChunk]] = []
    cur: List[DocumentChunk] = []
    cur_chars = 0

    def _item_chars(ch: DocumentChunk) -> int:
        # estimate JSON overhead + text
        return 40 + len((ch.text or ""))

    for ch in ordered:
        ch_chars = _item_chars(ch)

        # If adding this chunk would exceed limits, flush current batch first
        if cur:
            if len(cur) >= batch_size_chunks or (cur_chars + ch_chars) > max_chars:
                batches.append(cur)
                cur = []
                cur_chars = 0

        cur.append(ch)
        cur_chars += ch_chars

    if cur:
        batches.append(cur)
    logger.debug(
        "_partition_batches_by_chars: total=%d batches=%d batch_size_chunks=%d max_chars=%d",
        len(chunks),
        len(batches),
        batch_size_chunks,
        max_chars,
    )
    return batches


# -----------------------------
# Fact parsing + enforcement (lean + shared)
# -----------------------------
def _coerce_object_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return _normalize_ws(obj)
    try:
        return _normalize_ws(json.dumps(obj, ensure_ascii=False))
    except Exception:
        return _normalize_ws(str(obj))


def _enforce_fact_limits(
    subj: str,
    pred: str,
    obj_text: str,
    *,
    object_max_words: int,
    max_fact_tokens: int,
) -> Tuple[str, str, str]:
    """
    Enforce data-agnostic size constraints.
    """
    subj_n = _truncate_words(_normalize_ws(subj), 12)
    pred_n = _truncate_words(_normalize_ws(pred), 5) or "STATES"
    obj_n = _truncate_words(_normalize_ws(obj_text), int(object_max_words))

    # Token cap: shrink object until under budget
    max_fact_tokens = max(20, int(max_fact_tokens))
    composed = f"{subj_n} {pred_n} {obj_n}".strip()
    if _estimate_tokens(composed) <= max_fact_tokens:
        return subj_n, pred_n, obj_n

    # Reduce object words progressively
    words = obj_n.split()
    while words and _estimate_tokens(f"{subj_n} {pred_n} {' '.join(words)}".strip()) > max_fact_tokens:
        # drop the last ~10% each loop for speed
        drop = max(1, int(len(words) * 0.1))
        words = words[:-drop]
    obj_n2 = " ".join(words).strip()
    return subj_n, pred_n, obj_n2


def _parse_facts_list(
    *,
    facts_payload: Any,
    chunk: Optional[DocumentChunk],
    min_fact_words: int,
    scorer: SalienceScorer,
    max_facts_per_chunk: int,
    object_max_words: int,
    max_fact_tokens: int,
    predicate_default: str,
) -> List[ExtractedFact]:
    if not isinstance(facts_payload, list):
        return []

    extracted: List[ExtractedFact] = []
    drop_counts: Dict[str, int] = {}
    in_count = len(facts_payload)
    dropped_object_previews: List[Tuple[str, int]] = []
    dropped_object_preview_limit = 3
    for item in facts_payload:
        try:
            if not isinstance(item, dict):
                drop_counts["item_not_dict"] = drop_counts.get("item_not_dict", 0) + 1
                continue

            subj = item.get("subject")
            pred = item.get("predicate") or predicate_default
            obj = item.get("object")
            conf = item.get("confidence", 0.7)

            if not subj:
                drop_counts["missing_subject"] = drop_counts.get("missing_subject", 0) + 1
                continue
            obj_text = _coerce_object_text(obj)
            if not obj_text:
                drop_counts["missing_object"] = drop_counts.get("missing_object", 0) + 1
                continue

            if min_fact_words and _word_count(obj_text) < int(min_fact_words):
                drop_counts["below_min_fact_words"] = drop_counts.get("below_min_fact_words", 0) + 1
                if logger.isEnabledFor(logging.DEBUG) and len(dropped_object_previews) < dropped_object_preview_limit:
                    subj_s = _truncate_words(_normalize_ws(str(subj)), 12)
                    pred_s = _truncate_words(_normalize_ws(str(pred)), 5) or predicate_default
                    obj_s = _truncate_words(_normalize_ws(str(obj_text)), 32)
                    sentence = f"{subj_s} {pred_s} {obj_s}".strip()
                    dropped_object_previews.append((sentence, _word_count(obj_text)))
                continue

            confidence = _clamp01(_safe_float(conf, 0.7))
            subj_n, pred_n, obj_n = _enforce_fact_limits(
                str(subj),
                str(pred),
                obj_text,
                object_max_words=object_max_words,
                max_fact_tokens=max_fact_tokens,
            )
            if not obj_n:
                drop_counts["empty_after_limits"] = drop_counts.get("empty_after_limits", 0) + 1
                continue

            # score salience using core scorer (predicate weighting should be agnostic there)
            from ...types import Fact
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            fact_stub = Fact(
                id="stub",
                subject=str(subj_n),
                predicate=str(pred_n),
                object=str(obj_n),
                created_at=now,
                updated_at=now,
                confidence=confidence,
                meta={},
            )
            salience = scorer.score(fact_stub)

            source_chunk_id = chunk.chunk_id if chunk is not None else "doc_summary"
            doc_id = chunk.doc_id if chunk is not None else "doc"
            extracted.append(
                ExtractedFact(
                    subject=_normalize_subject(str(subj_n), doc_id=doc_id),
                    predicate=str(pred_n),
                    object=str(obj_n),
                    confidence=confidence,
                    salience=salience,
                    source_chunk_id=str(source_chunk_id),
                )
            )
        except Exception:
            logger.exception("_parse_facts_list: failed parsing fact item; skipping")
            drop_counts["exception"] = drop_counts.get("exception", 0) + 1
            continue

    if logger.isEnabledFor(logging.DEBUG) and in_count and (len(extracted) == 0 or drop_counts):
        chunk_id = getattr(chunk, "chunk_id", None) if chunk is not None else "doc_summary"
        logger.debug(
            "_parse_facts_list: chunk_id=%s in=%d kept=%d dropped=%s min_fact_words=%d object_max_words=%d max_fact_tokens=%d",
            chunk_id,
            in_count,
            len(extracted),
            drop_counts,
            int(min_fact_words),
            int(object_max_words),
            int(max_fact_tokens),
        )
        if dropped_object_previews:
            logger.debug(
                "_parse_facts_list: chunk_id=%s dropped_below_min_fact_words_examples=%s",
                chunk_id,
                [{"words": wc, "fact": sent} for sent, wc in dropped_object_previews],
            )

    # prefer higher salience then confidence; enforce max_facts_per_chunk
    extracted.sort(key=lambda f: (-float(f.salience), -float(f.confidence), f.subject))
    return extracted[: max(0, int(max_facts_per_chunk))]


def _parse_chunk_payload(
    *,
    chunk: DocumentChunk,
    payload: Any,
    min_fact_words: int,
    scorer: SalienceScorer,
    max_facts_per_chunk: int,
    object_max_words: int,
    max_fact_tokens: int,
) -> List[ExtractedFact]:
    if not isinstance(payload, dict):
        return []
    facts_payload = payload.get("facts")
    if logger.isEnabledFor(logging.DEBUG):
        if facts_payload is None:
            logger.debug(
                "_parse_chunk_payload: chunk_id=%s reason=facts_missing keys=%s",
                getattr(chunk, "chunk_id", None),
                list(payload.keys())[:12],
            )
        elif not isinstance(facts_payload, list):
            logger.debug(
                "_parse_chunk_payload: chunk_id=%s reason=facts_not_list type=%s",
                getattr(chunk, "chunk_id", None),
                type(facts_payload).__name__,
            )
        elif not facts_payload:
            logger.debug(
                "_parse_chunk_payload: chunk_id=%s reason=facts_empty_list",
                getattr(chunk, "chunk_id", None),
            )
    return _parse_facts_list(
        facts_payload=facts_payload,
        chunk=chunk,
        min_fact_words=min_fact_words,
        scorer=scorer,
        max_facts_per_chunk=max_facts_per_chunk,
        object_max_words=object_max_words,
        max_fact_tokens=max_fact_tokens,
        predicate_default="STATES",
    )


# -----------------------------
# Public APIs (called by uma.core.semantic.extractor)
# -----------------------------
async def extract_facts_batch(
    chunks: List[DocumentChunk],
    *,
    llm: Any,
    min_fact_words: int = _DEFAULT_MIN_FACT_WORDS,
    batch_size_chunks: int = 4,
    max_chars: int = 12000,
    max_facts_per_chunk: int = _DEFAULT_MAX_FACTS_PER_CHUNK,
    object_max_words: int = _DEFAULT_OBJECT_MAX_WORDS,
    max_fact_tokens: int = _DEFAULT_MAX_FACT_TOKENS,
) -> List[ExtractedFact]:
    """
    Batched fact extraction with strict chunk_id-keyed JSON and partial salvage.

    - Batches are deterministic (position, chunk_id).
    - If batch output is missing/invalid for some chunks, fall back per-chunk for only those.
    - Output constraints are enforced in code (facts capped per chunk; object size; token cap).

    INVARIANT:
    - For eligible chunks (len(text) >= _MIN_EXTRACT_CHUNK_CHARS), 0 facts is NOT allowed.
      If both batch and fallback extraction yield 0 facts, generate a deterministic fallback fact.
    """
    if not chunks:
        return []
    if llm is None or not hasattr(llm, "generate"):
        raise ValueError("extract_facts_batch: llm with .generate() required")

    scorer = SalienceScorer()
    extracted: List[ExtractedFact] = []
    total_chunks = len(chunks)
    batch_failures = 0
    forced_fallbacks = 0

    logger.info(
        "extract_facts_batch: start chunks=%d batch_size_chunks=%d max_chars=%d max_facts_per_chunk=%d object_max_words=%d max_fact_tokens=%d",
        len(chunks),
        batch_size_chunks,
        max_chars,
        max_facts_per_chunk,
        object_max_words,
        max_fact_tokens,
    )

    batches = _partition_batches_by_chars(chunks, batch_size_chunks=batch_size_chunks, max_chars=max_chars)

    for batch_idx, batch in enumerate(batches):
        # Skip tiny chunks (headers/footers/fragments) to reduce wasted LLM calls.
        batch_for_llm: List[DocumentChunk] = []
        for c in batch:
            t = (c.text or "").strip()
            if not t:
                continue
            if len(t) < _MIN_EXTRACT_CHUNK_CHARS:
                logger.debug(
                    "extract_facts_batch: skip tiny chunk chunk_id=%s chars=%d",
                    c.chunk_id,
                    len(t),
                )
                continue
            batch_for_llm.append(c)

        items = [{"chunk_id": c.chunk_id, "text": (c.text or "")} for c in batch_for_llm]
        if not items:
            continue

        data: Optional[Dict[str, Any]] = None
        try:
            if logger.isEnabledFor(logging.DEBUG):
                # Avoid logging full text; log ids + approximate size.
                approx_chars = sum(len((it.get("text") or "")) for it in items)
                logger.debug(
                    "extract_facts_batch: batch_idx=%d sending_to_llm chunks=%d approx_chars=%d first_chunk_ids=%s",
                    batch_idx,
                    len(items),
                    approx_chars,
                    [it.get("chunk_id") for it in items[:3]],
                )
            data = await generate_json(
                llm=llm,
                messages=_build_prompt(
                    mode="batch",
                    items=items,
                    min_fact_words=max(0, int(min_fact_words)),
                    max_facts_per_chunk=max_facts_per_chunk,
                ),
                max_tokens=_DEFAULT_BATCH_CALL_MAX_TOKENS,
                ctx=LLMCallContext(op="ingest_fact_extract_batch"),
                repair_messages_fn=lambda bad: [
                    {
                        "role": "system",
                        "content": (
                            "You MUST return ONLY valid JSON matching EXACTLY this schema:\n"
                            "{\n"
                            "  \"chunks\": {\n"
                            "    \"chunk_id\": {\n"
                            "      \"facts\": [\n"
                            "        {\"subject\":\"...\",\"predicate\":\"STATES\",\"object\":\"...\",\"confidence\":\"0.0-1.0\"}\n"
                            "      ]\n"
                            "    }\n"
                            "  }\n"
                            "}\n"
                            f"Rules: keys MUST be only provided chunk_ids; AT MOST {int(max_facts_per_chunk)} facts per chunk; no prose; no markdown."
                        ),
                    },
                    {"role": "user", "content": bad or ""},
                ],
            )
        except Exception:
            logger.exception("extract_facts_batch: llm generate failed batch_idx=%d", batch_idx)
            data = None
            batch_failures += 1

        chunks_payload = data.get("chunks") if isinstance(data, dict) else None
        if not isinstance(chunks_payload, dict):
            chunks_payload = {}
        if logger.isEnabledFor(logging.DEBUG):
            returned_keys = list(chunks_payload.keys()) if isinstance(chunks_payload, dict) else []
            logger.debug(
                "extract_facts_batch: batch_idx=%d llm_payload_keys=%d missing_keys=%d",
                batch_idx,
                len(returned_keys),
                max(0, len(items) - len(returned_keys)),
            )

        # Per eligible chunk: parse batch payload; if missing/invalid => fallback to per-chunk;
        # if still empty => deterministic fallback fact.
        for c in batch_for_llm:
            facts_for_chunk: List[ExtractedFact] = []
            payload = chunks_payload.get(c.chunk_id)
            source = "batch"
            reason = ""

            try:
                if isinstance(payload, dict):
                    facts_for_chunk = _parse_chunk_payload(
                        chunk=c,
                        payload=payload,
                        min_fact_words=min_fact_words,
                        scorer=scorer,
                        max_facts_per_chunk=max_facts_per_chunk,
                        object_max_words=object_max_words,
                        max_fact_tokens=max_fact_tokens,
                    )
                    if not facts_for_chunk:
                        reason = "parsed_or_filtered_to_0"
                else:
                    source = "fallback"
                    reason = "missing_or_invalid_batch_payload"
            except Exception:
                facts_for_chunk = []
                source = "fallback"
                reason = "parse_exception"
                logger.exception("extract_facts_batch: parse failed chunk_id=%s", c.chunk_id)

            # If batch produced nothing, salvage via per-chunk extraction (only when payload missing/invalid OR batch broke)
            if not facts_for_chunk and not isinstance(payload, dict):
                try:
                    facts_for_chunk = await extract_facts_one(
                        c,
                        llm=llm,
                        min_fact_words=min_fact_words,
                        scorer=scorer,
                        max_facts=max_facts_per_chunk,
                        object_max_words=object_max_words,
                        max_fact_tokens=max_fact_tokens,
                    )
                    if not facts_for_chunk:
                        reason = reason or "per_chunk_extractor_returned_0"
                except Exception:
                    facts_for_chunk = []
                    reason = "fallback_exception"
                    logger.exception("extract_facts_batch: fallback failed chunk_id=%s", c.chunk_id)

            # Enforce invariant: eligible chunk must never yield 0 facts.
            if not facts_for_chunk:
                forced_fallbacks += 1
                logger.warning(
                    "extract_facts_batch: forcing deterministic fallback fact chunk_id=%s (eligible chunk yielded 0 facts)",
                    c.chunk_id,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "extract_facts_batch: forced_fallback batch_idx=%d chunk_id=%s source=%s reason=%s",
                        batch_idx,
                        c.chunk_id,
                        source,
                        reason or "unknown",
                    )
                facts_for_chunk = [
                    _fallback_fact_for_chunk(
                        c,
                        doc_id=str(getattr(c, "doc_id", "") or ""),
                        object_max_words=object_max_words,
                        max_fact_tokens=max_fact_tokens,
                    )
                ]
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "extract_facts_batch: chunk_ok batch_idx=%d chunk_id=%s source=%s facts=%d",
                        batch_idx,
                        c.chunk_id,
                        source,
                        len(facts_for_chunk),
                    )

            extracted.extend(facts_for_chunk)

    if batch_failures:
        logger.warning(
            "extract_facts_batch: completed with batch_failures=%d total_chunks=%d",
            batch_failures,
            total_chunks,
        )

    if forced_fallbacks:
        logger.warning(
            "extract_facts_batch: forced_fallback_facts=%d (eligible chunks yielded 0 facts after LLM parsing)",
            forced_fallbacks,
        )

    logger.info(
        "extract_facts_batch: done extracted_facts=%d total_chunks=%d",
        len(extracted),
        total_chunks,
    )
    return extracted


async def extract_facts_one(
    chunk: DocumentChunk,
    *,
    llm: Any,
    min_fact_words: int = _DEFAULT_MIN_FACT_WORDS,
    scorer: Optional[SalienceScorer] = None,
    max_facts: int = _DEFAULT_MAX_FACTS_PER_CHUNK,
    object_max_words: int = _DEFAULT_OBJECT_MAX_WORDS,
    max_fact_tokens: int = _DEFAULT_MAX_FACT_TOKENS,
) -> List[ExtractedFact]:
    if chunk is None:
        return []
    if llm is None or not hasattr(llm, "generate"):
        raise ValueError("extract_facts_one: llm with .generate() required")

    text = (chunk.text or "").strip()
    if not text:
        return []
    if len(text) < _MIN_EXTRACT_CHUNK_CHARS:
        logger.debug(
            "extract_facts_one: skip tiny chunk chunk_id=%s chars=%d",
            chunk.chunk_id,
            len(text),
        )
        return []

    messages = _build_prompt(
        mode="single",
        chunk_text=text,
        min_fact_words=max(0, int(min_fact_words)),
        max_facts_per_chunk=max_facts,
    )

    logger.debug(
        "extract_facts_one: calling llm chunk_id=%s chars=%d",
        chunk.chunk_id,
        len(text),
    )

    try:
        data = await generate_json(
            llm=llm,
            messages=messages,
            max_tokens=_DEFAULT_SINGLE_CALL_MAX_TOKENS,
            ctx=LLMCallContext(op="ingest_fact_extract_one"),
            repair_messages_fn=lambda bad: [
                {"role": "system", "content": "Return ONLY valid JSON. No prose."},
                {"role": "user", "content": bad or ""},
            ],
        )
    except Exception:
        try:
            logger.error(
                "extract_facts_one: LLM JSON parse failed chunk=%s text_preview=%s",
                chunk.chunk_id,
                (text[:400] if text else ""),
            )
        except Exception:
            logger.exception("extract_facts_one: failed to log preview")
        logger.exception("extract_facts_one: llm generate failed for chunk=%s", chunk.chunk_id)
        # Eligible chunk must never yield 0 facts: deterministic fallback
        return [
            _fallback_fact_for_chunk(
                chunk,
                doc_id=str(getattr(chunk, "doc_id", "") or ""),
                object_max_words=object_max_words,
                max_fact_tokens=max_fact_tokens,
            )
        ]

    facts_payload = data.get("facts") if isinstance(data, dict) else None
    scorer = scorer or SalienceScorer()
    out = _parse_facts_list(
        facts_payload=facts_payload,
        chunk=chunk,
        min_fact_words=min_fact_words,
        scorer=scorer,
        max_facts_per_chunk=max_facts,
        object_max_words=object_max_words,
        max_fact_tokens=max_fact_tokens,
        predicate_default="STATES",
    )

    # Enforce invariant: eligible chunks must never yield 0 facts.
    if not out:
        logger.warning("extract_facts_one: no facts extracted chunk_id=%s; forcing fallback fact", chunk.chunk_id)
        out = [
            _fallback_fact_for_chunk(
                chunk,
                doc_id=str(getattr(chunk, "doc_id", "") or ""),
                object_max_words=object_max_words,
                max_fact_tokens=max_fact_tokens,
            )
        ]
    return out


async def extract_facts(
    chunks: List[DocumentChunk],
    *,
    llm: Any,
    min_fact_words: int = _DEFAULT_MIN_FACT_WORDS,
    max_facts_per_chunk: int = _DEFAULT_MAX_FACTS_PER_CHUNK,
    object_max_words: int = _DEFAULT_OBJECT_MAX_WORDS,
    max_fact_tokens: int = _DEFAULT_MAX_FACT_TOKENS,
) -> List[ExtractedFact]:
    """
    Deterministic per-chunk extraction (no batching).
    Used when batch mode is disabled or for narrow debugging.
    """
    if not chunks:
        return []
    scorer = SalienceScorer()
    out: List[ExtractedFact] = []
    for ch in sorted(chunks, key=lambda c: (int(getattr(c, "position", 0) or 0), getattr(c, "chunk_id", ""))):
        out.extend(
            await extract_facts_one(
                ch,
                llm=llm,
                min_fact_words=min_fact_words,
                scorer=scorer,
                max_facts=max_facts_per_chunk,
                object_max_words=object_max_words,
                max_fact_tokens=max_fact_tokens,
            )
        )
    return out
