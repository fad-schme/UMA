"""
semantic/query_pruner.py
========================

Query-time fact pruning utilities.

These helpers score/filter "Fact-like" objects for relevance to a user query.
They are intentionally tolerant of dict/object shapes because they sit at the
retrieval boundary.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Sequence

from pydantic import BaseModel, Field

from uma.adapters.llm.controller import LLMCallContext, generate_model

from uma.retrieve.user_query_helper import build_fact_embedding_text, extract_keywords_and_phrases, get_stopwords

logger = logging.getLogger(__name__)

class _ScoresPayload(BaseModel):
    scores: List[float] = Field(default_factory=list)


def describe_fact(fact: Any) -> str:
    try:
        meta = (fact.get("meta") if isinstance(fact, dict) else getattr(fact, "meta", None)) or {}
        if isinstance(meta, dict):
            excerpt = meta.get("excerpt") or meta.get("description")
            if excerpt:
                return str(excerpt).replace("\n", " ").strip()
    except Exception:
        logger.exception("describe_fact: failed to read meta")

    try:
        text = str(build_fact_embedding_text(fact) or "").strip()
        if text:
            return text
    except Exception:
        logger.exception("describe_fact: failed to build fact text")

    try:
        sub = fact.get("subject") if isinstance(fact, dict) else getattr(fact, "subject", "user")
        pred = fact.get("predicate") if isinstance(fact, dict) else getattr(fact, "predicate", "related_to")
        return f"{sub} {pred}"
    except Exception:
        return ""


def fallback_keep_by_query(query: str, facts: Sequence[Any]) -> List[int]:
    stop = set()
    if get_stopwords:
        try:
            stop = set(get_stopwords() or set())
        except Exception:
            stop = set()

    terms: List[str] = []
    if extract_keywords_and_phrases:
        try:
            extracted = extract_keywords_and_phrases(query)
            terms = (extracted.get("keywords") or []) + (extracted.get("keyphrases") or [])
            terms = [t for t in terms if isinstance(t, str) and t and t not in stop]
        except Exception:
            terms = []

    if not terms:
        terms = [t for t in re.split(r"\W+", (query or "").lower()) if t and len(t) >= 4 and t not in stop]
    if not terms:
        return []

    scored: List[tuple[int, int, int]] = []
    for idx, fact in enumerate(facts, start=1):
        text = describe_fact(fact).lower()
        if not text:
            continue
        matched_terms = sum(1 for t in terms if t in text)
        phrase_hits = sum(1 for t in terms if " " in t and t in text)
        if matched_terms > 0:
            scored.append((idx, matched_terms, phrase_hits))
    scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [idx for idx, _matched, _phrases in scored]


async def prune_facts_for_query(
    *,
    llm: Any,
    query_text: str,
    facts: List[Any],
    threshold: float = 0.6,
    max_keep: int = 12,
    max_candidates: int = 20,
) -> List[Any]:
    if not llm or not facts:
        return facts or []

    candidates = list(facts)[: max(1, int(max_candidates))]
    descriptions = [f"{i}. {describe_fact(f)}" for i, f in enumerate(candidates, start=1) if describe_fact(f)]
    if not descriptions:
        return facts

    system_prompt = (
        "You are a retrieval assistant that scores fact relevance.\n"
        "Given a user question and a numbered list of facts, decide "
        "which facts are directly useful to answer the question.\n"
        "\n"
        "Output format requirements:\n"
        "- Output ONLY a single JSON object.\n"
        "- The JSON MUST have exactly one key: \"scores\".\n"
        "- The value of \"scores\" MUST be an array of floats in [0,1].\n"
        "- The array MUST be the same length as the fact list, in the same order.\n"
        "- Do not include any other keys, comments, or text before or after the JSON.\n"
    )
    user_prompt = f"Question:\n{query_text}\n\nFacts:\n" + "\n".join(descriptions) + "\n\nReturn the JSON object now."

    try:
        payload = await generate_model(
            llm=llm,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            ctx=LLMCallContext(op="fact_prune"),
            model_validate=_ScoresPayload.model_validate,
            repair_messages_fn=lambda bad: [
                {
                    "role": "user",
                    "content": "Return ONLY valid JSON of the form "
                    "{\"scores\": [0.0, ...]} with the same number of scores. "
                    "No extra text.\n\nBad response:\n"
                    + (bad or ""),
                }
            ],
        )
    except Exception:
        logger.exception("semantic.query_pruner: LLM scoring failed")
        return facts

    scores = []
    try:
        raw_scores = payload.scores
        if isinstance(raw_scores, list):
            scores = []
            for x in raw_scores[: len(descriptions)]:
                try:
                    v = float(x)
                except Exception:
                    v = 0.0
                scores.append(min(1.0, max(0.0, v)))
            if len(scores) != len(descriptions):
                scores = []
    except Exception:
        scores = []
    if not scores:
        logger.warning(
            "semantic.query_pruner: LLM scores missing or mismatched; "
            "using keyword fallback for %d candidates",
            len(candidates),
        )
        selected = fallback_keep_by_query(query_text, candidates)
        return [candidates[i - 1] for i in selected if 1 <= i <= len(candidates)] or facts

    indexed = list(enumerate(scores, start=1))
    kept = [i for i, s in indexed if s >= float(threshold)]
    if not kept:
        indexed.sort(key=lambda it: it[1], reverse=True)
        kept = [i for i, _ in indexed[: min(int(max_keep), len(indexed))]]
    out = [candidates[i - 1] for i in kept if 1 <= i <= len(candidates)]
    return out or facts
