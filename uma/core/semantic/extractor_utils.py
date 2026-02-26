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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ...types import Fact
from ..ingest.types import DocumentChunk
from ..utils.user_query_helper import get_generic_terms
from .scorer import SalienceScorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Internal defaults (kept as implementation constants)
# ---------------------------------------------------------------------
_DEFAULT_MAX_FACTS_PER_CHUNK = 4
_DEFAULT_OBJECT_MAX_WORDS = 50
_DEFAULT_MAX_FACT_TOKENS = 120  # per fact, estimated
_DEFAULT_MIN_FACT_WORDS = 3

_MIN_EXTRACT_CHUNK_CHARS = 300  # skip LLM extraction for tiny chunks (usually headers/footers/fragments)

# LLM output token budget (for the call). Output is additionally bounded by hard caps above.
_DEFAULT_SINGLE_CALL_MAX_TOKENS = 700
_DEFAULT_BATCH_CALL_MAX_TOKENS = 1200
_DEFAULT_SUMMARY_CALL_MAX_TOKENS = 800

# Batch sizing defaults used by ingestion
_DEFAULT_BATCH_SIZE_CHUNKS = 4
_DEFAULT_BATCH_MAX_CHARS = 12000

# User fact token cap (keep tighter than chunk facts)
_DEFAULT_MAX_FACT_TOKENS_USER = 80

# Generic boilerplate suppression: not domain-specific
_BOILERPLATE_SNIPPETS = (
    "all rights reserved",
    "copyright",
    "table of contents",
    "page ",
)

# ---------------------------------------------------------------------
# Public exports (constants) expected by extractor.py
# ---------------------------------------------------------------------
DEFAULT_MAX_FACTS_PER_CHUNK = _DEFAULT_MAX_FACTS_PER_CHUNK
DEFAULT_OBJECT_MAX_WORDS = _DEFAULT_OBJECT_MAX_WORDS
DEFAULT_MAX_FACT_TOKENS = _DEFAULT_MAX_FACT_TOKENS
DEFAULT_MAX_FACT_TOKENS_USER = _DEFAULT_MAX_FACT_TOKENS_USER
DEFAULT_MIN_FACT_WORDS = _DEFAULT_MIN_FACT_WORDS

MIN_EXTRACT_CHUNK_CHARS = _MIN_EXTRACT_CHUNK_CHARS

DEFAULT_SINGLE_CALL_MAX_TOKENS = _DEFAULT_SINGLE_CALL_MAX_TOKENS
DEFAULT_BATCH_CALL_MAX_TOKENS = _DEFAULT_BATCH_CALL_MAX_TOKENS
DEFAULT_SUMMARY_CALL_MAX_TOKENS = _DEFAULT_SUMMARY_CALL_MAX_TOKENS

DEFAULT_BATCH_SIZE_CHUNKS = _DEFAULT_BATCH_SIZE_CHUNKS
DEFAULT_BATCH_MAX_CHARS = _DEFAULT_BATCH_MAX_CHARS

# ---------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------
def _fallback_object_for_chunk(
    ch: DocumentChunk,
    *,
    object_max_words: int,
    max_fact_tokens: int,
) -> str:
    """
    Deterministic, data-agnostic fallback object text used only when extraction yields 0 facts
    for an eligible chunk.

    Goal: produce a short, self-contained sentence-ish snippet that is stable and bounded.
    """
    text = _normalize_ws(ch.text or "")
    if not text:
        return ""

    # Prefer first sentence if we can find one; otherwise use prefix.
    prefix = text
    m = re.search(r"(.+?[.!?])\s", text)
    if m:
        prefix = m.group(1)

    obj = _truncate_words(prefix, int(object_max_words))

    # Enforce token-ish budget on object alone (cheap approximation).
    while obj and _estimate_tokens(obj) > int(max_fact_tokens):
        obj = _truncate_words(obj, max(5, int(len(obj.split()) * 0.85)))

    if not obj:
        obj = _truncate_words(text, int(object_max_words))

    return obj


def _fallback_fact_for_chunk_as_fact(
    ch: DocumentChunk,
    *,
    owner_type: str,
    owner_id: str,
    doc_id: str,
    source_path: str,
    source_hash: str,
    now: datetime,
    object_max_words: int,
    max_fact_tokens: int,
    scorer: SalienceScorer,
) -> Fact:
    """
    Deterministic fallback Fact. This is the *last resort* when:
    - Chunk is eligible for extraction, but parsing yields 0 facts
    - Batch/single fallback attempts still yield nothing

    Always sets extractor_fallback=True in meta.
    """
    obj = _fallback_object_for_chunk(
        ch,
        object_max_words=int(object_max_words),
        max_fact_tokens=int(max_fact_tokens),
    )

    subj = _normalize_subject("", doc_id=doc_id)
    src_chunk_id = str(getattr(ch, "chunk_id", "") or "")

    fact = Fact(
        id=f"fact_fallback_{src_chunk_id or 'chunk'}",
        subject=subj,
        predicate="STATES",
        object=obj,
        created_at=now,
        updated_at=now,
        source_ids=[src_chunk_id] if src_chunk_id else [],
        confidence=0.6,
        salience=0.0,
        owner_type=owner_type,
        owner_id=owner_id,
        meta={
            "source_type": "pdf",
            "domain": "kb_doc",
            "doc_id": doc_id,
            "source_path": source_path,
            "source_hash": source_hash,
            "fact_text": obj,
            "fact_type": "claim",
            "extractor_fallback": True,
        },
    )
    fact.salience = float(scorer.score(fact))
    return fact


# ---------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------
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
    s = _normalize_ws(subject)
    if not s:
        return f"Document({doc_id})"
    return _truncate_words(s, 12)


# ---------------------------------------------------------------------
# Prompt builder (no domain cues)
# ---------------------------------------------------------------------
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
        return [{"role": "system", "content": system}, {"role": "user", "content": doc_text or ""}]

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
    return [{"role": "system", "content": system}, {"role": "user", "content": chunk_text or ""}]


# ---------------------------------------------------------------------
# Chunk selection (generic, data-agnostic)
# ---------------------------------------------------------------------
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
    return sents / max(1.0, len(t) / 400.0)


def _list_density(text: str) -> float:
    t = text or ""
    if not t:
        return 0.0
    bullets = len(_LIST_LINE_RE.findall(t))
    return bullets / max(1.0, len(t) / 400.0)


def _definition_like_score(text: str) -> float:
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
    generic = get_generic_terms()
    tokens = re.findall(r"[a-z0-9-]+", (text or "").lower())
    if not tokens:
        return 0.0
    words = [t for t in tokens if any(ch.isalpha() for ch in t)]
    if not words:
        return 0.0
    g = sum(1 for w in words if w in generic)
    return g / max(1, len(words))


def _chunk_quality_score(ch: DocumentChunk) -> float:
    text = (ch.text or "").strip()
    if not text:
        return 0.0

    length = len(text)
    if length < 120:
        return 0.0

    length_score = min(1.0, length / 1200.0)

    sent = _sentence_density(text)
    lst = _list_density(text)
    defs = _definition_like_score(text)

    boiler = _boilerplate_penalty(text)
    gen_ratio = _generic_term_ratio(text)

    score = 0.0
    score += 1.2 * length_score
    score += 0.8 * min(sent, 5.0) / 5.0
    score += 0.6 * min(lst, 5.0) / 5.0
    score += 0.6 * min(defs, 6.0) / 6.0

    score -= 1.2 * boiler
    if gen_ratio > 0.25:
        score -= 0.4 * min(1.0, (gen_ratio - 0.25) / 0.5)

    return max(0.0, score)


def select_chunks_for_fact_extraction(
    chunks: List[DocumentChunk],
    *,
    max_chunks: Optional[int] = None,
    max_per_page: Optional[int] = None,
) -> List[DocumentChunk]:
    """
    Deterministically select the highest-quality chunks for extraction.

    - Sort by page/position + chunk_id for determinism
    - Score via generic heuristics
    - Apply max_per_page first (if provided), then global max_chunks
    """
    if not chunks:
        return []

    ordered = sorted(chunks, key=lambda c: (int(getattr(c, "position", 0) or 0), getattr(c, "chunk_id", "")))

    # Group by page_range when available; fallback to None.
    buckets: Dict[str, List[DocumentChunk]] = {}
    for ch in ordered:
        pr = getattr(ch, "page_range", None)
        key = str(pr) if pr is not None else "none"
        buckets.setdefault(key, []).append(ch)

    selected: List[DocumentChunk] = []

    for _, group in buckets.items():
        scored = [(float(_chunk_quality_score(c)), c) for c in group]
        scored.sort(key=lambda x: (-x[0], int(getattr(x[1], "position", 0) or 0), getattr(x[1], "chunk_id", "")))
        if isinstance(max_per_page, int) and max_per_page > 0:
            scored = scored[: int(max_per_page)]
        selected.extend([c for _, c in scored])

    # Global cap
    selected.sort(key=lambda c: (int(getattr(c, "position", 0) or 0), getattr(c, "chunk_id", "")))
    if isinstance(max_chunks, int) and max_chunks >= 0:
        selected = selected[: int(max_chunks)]
    return selected


# ---------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------
def _partition_batches_by_chars(
    chunks: List[DocumentChunk],
    *,
    batch_size_chunks: int,
    max_chars: int,
) -> List[List[DocumentChunk]]:
    if not chunks:
        return []
    batch_size_chunks = max(1, int(batch_size_chunks))
    max_chars = max(2000, int(max_chars))

    ordered = sorted(chunks, key=lambda c: (int(getattr(c, "position", 0) or 0), getattr(c, "chunk_id", "")))

    batches: List[List[DocumentChunk]] = []
    cur: List[DocumentChunk] = []
    cur_chars = 0

    def _item_chars(ch: DocumentChunk) -> int:
        return 40 + len((ch.text or ""))

    for ch in ordered:
        ch_chars = _item_chars(ch)

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


# ---------------------------------------------------------------------
# Fact parsing + enforcement (internal parsed dicts -> Fact later)
# ---------------------------------------------------------------------
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
    subj_n = _truncate_words(_normalize_ws(subj), 12)
    pred_n = _truncate_words(_normalize_ws(pred), 5) or "STATES"
    obj_n = _truncate_words(_normalize_ws(obj_text), int(object_max_words))

    max_fact_tokens = max(20, int(max_fact_tokens))
    composed = f"{subj_n} {pred_n} {obj_n}".strip()
    if _estimate_tokens(composed) <= max_fact_tokens:
        return subj_n, pred_n, obj_n

    words = obj_n.split()
    while words and _estimate_tokens(f"{subj_n} {pred_n} {' '.join(words)}".strip()) > max_fact_tokens:
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
) -> List[Dict[str, Any]]:
    """
    Parse the model payload into a small internal representation.

    Returns a list of dicts with keys:
      subject, predicate, object, confidence, salience, source_chunk_id

    This is intentionally *not* a public type. We convert to `Fact` in
    `parse_facts_list_into_facts(...)` so `extractor_utils` stays Fact-only externally.
    """
    if not isinstance(facts_payload, list):
        return []

    extracted: List[Dict[str, Any]] = []
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
                object_max_words=int(object_max_words),
                max_fact_tokens=int(max_fact_tokens),
            )
            if not obj_n:
                drop_counts["empty_after_limits"] = drop_counts.get("empty_after_limits", 0) + 1
                continue

            # Score salience using a stub Fact (owner fields intentionally omitted).
            now = datetime.now(timezone.utc)
            stub = Fact(
                id="stub",
                subject=str(subj_n),
                predicate=str(pred_n),
                object=str(obj_n),
                created_at=now,
                updated_at=now,
                confidence=confidence,
                meta={},
            )
            salience = float(scorer.score(stub))

            source_chunk_id = str(chunk.chunk_id) if chunk is not None else "doc_summary"
            docid = str(chunk.doc_id) if chunk is not None else "doc"

            extracted.append(
                {
                    "subject": _normalize_subject(str(subj_n), doc_id=docid),
                    "predicate": str(pred_n),
                    "object": str(obj_n),
                    "confidence": float(confidence),
                    "salience": float(salience),
                    "source_chunk_id": source_chunk_id,
                }
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

    extracted.sort(key=lambda d: (-float(d.get("salience", 0.0)), -float(d.get("confidence", 0.0)), str(d.get("subject", ""))))
    return extracted[: max(0, int(max_facts_per_chunk))]


# ---------------------------------------------------------------------
# Public helpers for FactExtractor (returning Fact)
# ---------------------------------------------------------------------
def safe_confidence(v: Any, default: float = 0.7) -> float:
    return _clamp01(_safe_float(v, default))


def coerce_object_text(obj: Any) -> str:
    return _coerce_object_text(obj)


def word_count(text: str) -> int:
    return _word_count(text)


def enforce_fact_limits(
    *,
    subj: str,
    pred: str,
    obj_text: str,
    object_max_words: int,
    max_fact_tokens: int,
) -> Tuple[str, str, str]:
    return _enforce_fact_limits(
        subj=subj,
        pred=pred,
        obj_text=obj_text,
        object_max_words=object_max_words,
        max_fact_tokens=max_fact_tokens,
    )


def parse_facts_list_into_facts(
    *,
    facts_payload: Any,
    chunk: Optional[DocumentChunk],
    min_fact_words: int,
    scorer: SalienceScorer,
    max_facts_per_chunk: int,
    object_max_words: int,
    max_fact_tokens: int,
    predicate_default: str,
    owner_type: str,
    owner_id: str,
    now: datetime,
    doc_id: str,
    source_path: str,
    source_hash: str,
) -> List[Fact]:
    parsed = _parse_facts_list(
        facts_payload=facts_payload,
        chunk=chunk,
        min_fact_words=int(min_fact_words),
        scorer=scorer,
        max_facts_per_chunk=int(max_facts_per_chunk),
        object_max_words=int(object_max_words),
        max_fact_tokens=int(max_fact_tokens),
        predicate_default=str(predicate_default),
    )

    out: List[Fact] = []
    for p in parsed:
        try:
            src_chunk_id = str(p.get("source_chunk_id", "") or "")
            subj = str(p.get("subject", "") or "")
            pred = str(p.get("predicate", "") or "STATES")
            obj = str(p.get("object", "") or "")
            conf = float(p.get("confidence", 0.7) or 0.7)
            sal = float(p.get("salience", 0.0) or 0.0)

            if not subj or not obj:
                continue

            meta = {
                "source_type": "pdf",
                "domain": "kb_doc",
                "doc_id": doc_id,
                "source_path": source_path,
                "source_hash": source_hash,
                "fact_text": obj,
                "fact_type": "summary" if pred == "SUMMARY" else "claim",
            }

            f = Fact(
                id=f"fact_{uuid_from_text(f'{doc_id}:{src_chunk_id}:{subj}:{pred}:{obj}')}",
                subject=subj,
                predicate=pred,
                object=obj,
                created_at=now,
                updated_at=now,
                source_ids=[src_chunk_id] if src_chunk_id else [],
                confidence=conf,
                salience=sal,
                owner_type=owner_type,
                owner_id=owner_id,
                meta=meta,
            )

            # Re-score to ensure consistent scorer behavior across paths.
            f.salience = float(scorer.score(f))
            out.append(f)
        except Exception:
            logger.exception("parse_facts_list_into_facts: failed converting parsed payload to Fact; skipping")

    out.sort(key=lambda f: (-float(f.salience), -float(f.confidence), f.subject))
    return out[: max(0, int(max_facts_per_chunk))]


def parse_chunk_payload_into_facts(
    *,
    chunk: DocumentChunk,
    payload: Any,
    min_fact_words: int,
    scorer: SalienceScorer,
    max_facts_per_chunk: int,
    object_max_words: int,
    max_fact_tokens: int,
    owner_type: str,
    owner_id: str,
    now: datetime,
    doc_id: str,
    source_path: str,
    source_hash: str,
) -> List[Fact]:
    if not isinstance(payload, dict):
        return []
    facts_payload = payload.get("facts")
    return parse_facts_list_into_facts(
        facts_payload=facts_payload,
        chunk=chunk,
        min_fact_words=min_fact_words,
        scorer=scorer,
        max_facts_per_chunk=max_facts_per_chunk,
        object_max_words=object_max_words,
        max_fact_tokens=max_fact_tokens,
        predicate_default="STATES",
        owner_type=owner_type,
        owner_id=owner_id,
        now=now,
        doc_id=doc_id,
        source_path=source_path,
        source_hash=source_hash,
    )


def fallback_fact_for_chunk(
    ch: DocumentChunk,
    *,
    owner_type: str,
    owner_id: str,
    doc_id: str,
    source_path: str,
    source_hash: str,
    now: datetime,
    object_max_words: int,
    max_fact_tokens: int,
    scorer: SalienceScorer,
) -> Fact:
    return _fallback_fact_for_chunk_as_fact(
        ch,
        owner_type=owner_type,
        owner_id=owner_id,
        doc_id=doc_id,
        source_path=source_path,
        source_hash=source_hash,
        now=now,
        object_max_words=int(object_max_words),
        max_fact_tokens=int(max_fact_tokens),
        scorer=scorer,
    )


def build_prompt(**kwargs: Any) -> List[Dict[str, str]]:
    return _build_prompt(**kwargs)


def partition_batches_by_chars(
    chunks: List[DocumentChunk],
    *,
    batch_size_chunks: int,
    max_chars: int,
) -> List[List[DocumentChunk]]:
    return _partition_batches_by_chars(chunks, batch_size_chunks=batch_size_chunks, max_chars=max_chars)


def batch_repair_messages(bad: str, *, max_facts_per_chunk: int) -> List[Dict[str, str]]:
    return [
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
                f"Rules: keys MUST be only provided chunk_ids; AT MOST {int(max_facts_per_chunk)} facts per chunk; "
                "no prose; no markdown."
            ),
        },
        {"role": "user", "content": bad or ""},
    ]


def single_repair_messages(bad: str, *, max_facts: int) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return ONLY valid JSON in this schema:\n"
                "{\n"
                "  \"facts\": [\n"
                "    {\"subject\":\"...\",\"predicate\":\"STATES\",\"object\":\"...\",\"confidence\":\"0.0-1.0\"}\n"
                "  ]\n"
                "}\n"
                f"Rules: AT MOST {int(max_facts)} facts; no prose; no markdown."
            ),
        },
        {"role": "user", "content": bad or ""},
    ]


# ---------------------------------------------------------------------
# Deterministic UUID helper (for stable fact ids when desired)
# ---------------------------------------------------------------------
def uuid_from_text(text: str) -> str:
    # Cheap stable UUID (not cryptographic): deterministic based on content
    import hashlib

    h = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
