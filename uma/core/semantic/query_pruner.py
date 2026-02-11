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

try:
    from ..utils.user_query_helper import extract_keywords_and_phrases, get_stopwords
except Exception:  # pragma: no cover - optional
    extract_keywords_and_phrases = None
    get_stopwords = None

logger = logging.getLogger(__name__)


def describe_fact(fact: Any) -> str:
    try:
        meta = (fact.get("meta") if isinstance(fact, dict) else getattr(fact, "meta", None)) or {}
        if isinstance(meta, dict):
            excerpt = meta.get("excerpt") or meta.get("description")
            if excerpt:
                return str(excerpt).replace("\n", " ").strip()
    except Exception:
        pass

    try:
        obj = fact.get("object") if isinstance(fact, dict) else getattr(fact, "object", None)
        if isinstance(obj, dict):
            text = obj.get("text") or ""
        else:
            text = str(obj)
        if text and str(text).strip():
            return str(text).strip()
    except Exception:
        pass

    try:
        sub = fact.get("subject") if isinstance(fact, dict) else getattr(fact, "subject", "user")
        pred = fact.get("predicate") if isinstance(fact, dict) else getattr(fact, "predicate", "related_to")
        return f"{sub} {pred}"
    except Exception:
        return ""


def parse_scores_list(response: str, *, n: int) -> List[float]:
    response = (response or "").strip()
    try:
        parsed = json.loads(response)
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    raw_scores = parsed.get("scores")
    if not isinstance(raw_scores, list):
        return []
    scores: List[float] = []
    for x in raw_scores[:n]:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        scores.append(min(1.0, max(0.0, v)))
    return scores if len(scores) == n else []


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

    kept: List[int] = []
    for idx, fact in enumerate(facts, start=1):
        text = describe_fact(fact).lower()
        if any(t in text for t in terms):
            kept.append(idx)
    return kept


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
        response = await llm.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=128,
            temperature=0.0,
        )
    except Exception:
        logger.exception("semantic.query_pruner: LLM scoring failed")
        return facts

    scores = parse_scores_list(response, n=len(descriptions))
    if not scores:
        selected = fallback_keep_by_query(query_text, candidates)
        return [candidates[i - 1] for i in selected if 1 <= i <= len(candidates)] or facts

    indexed = list(enumerate(scores, start=1))
    kept = [i for i, s in indexed if s >= float(threshold)]
    if not kept:
        indexed.sort(key=lambda it: it[1], reverse=True)
        kept = [i for i, _ in indexed[: min(int(max_keep), len(indexed))]]
    out = [candidates[i - 1] for i in kept if 1 <= i <= len(candidates)]
    return out or facts

