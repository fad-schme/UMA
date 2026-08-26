"""
uma.retrieve.rlm.query_decomposition
=====================================

Splits a broad, open-ended query ("What activities does X partake in?") into
narrower sub-queries so chunk candidate-gathering can reach answer-bearing
turns that a single query embedding ranks far outside the baseline top-k.

Background: retrieval-ranking-gap ticket 02 (research) confirmed that for
broad/list-style questions, the scattered answer-bearing chunks routinely
rank 60-500+ positions outside a fixed vector-top-k pool — no single k is
both affordable and wide enough to close that gap. Decomposing the query and
searching each sub-query's own top-k, then merging, reached chunks a single
query could not.

Safety guarantee: never raises. Returns [] on any failure (missing LLM,
malformed response, timeout) so a broken decomposition step degrades to the
single-query search path rather than failing retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from uma.common.json_utils import try_parse_json_object

logger = logging.getLogger(__name__)

# Matches controller.py's _LLM_HOP_SKIP_SEVERITIES — kept as a separate
# constant (not imported from controller.py) to avoid a chunk-core ->
# controller import edge; every LLM hop in the RLM pipeline skips on the
# same two severities the boundary scan (scan_user_input) can report.
_LLM_HOP_SKIP_SEVERITIES = frozenset({"medium", "high"})

_SYSTEM_PROMPT = (
    "You are decomposing a broad, open-ended question about a specific person "
    "from a long conversation history into narrower search queries that "
    "together would surface all the scattered evidence needed to answer it "
    "fully. Each sub-query must probe a different concrete angle already "
    "implied by the question (a specific activity, event, feeling, or fact) — "
    "do not invent people, relationships, or details the question does not "
    "already reference.\n\n"
    'Return ONLY valid JSON: {"sub_queries": ["...", "..."]}\n'
    "No prose. No markdown."
)


async def decompose_query(
    llm: Any,
    query_text: str,
    *,
    max_sub_queries: int = 4,
    max_tokens: int = 200,
    query_scan_severity: Optional[str] = None,
) -> list[str]:
    """
    Ask `llm` for up to `max_sub_queries` narrower search queries covering
    distinct facets of `query_text`. Returns [] if `llm` is unavailable, the
    response is malformed, the call fails for any reason, or
    `query_scan_severity` is medium/high — the same boundary-scan gate every
    other LLM hop in the RLM pipeline honors (controller.py's
    `_llm_hops_disabled`): a query flagged strongly enough to skip LLM
    amplification on retrieved content must also skip LLM amplification of
    the query itself.
    """
    if llm is None:
        return []
    if (query_scan_severity or "").strip().lower() in _LLM_HOP_SKIP_SEVERITIES:
        logger.info(
            "decompose_query: skipped, query_scan_severity=%r", query_scan_severity
        )
        return []
    q = (query_text or "").strip()
    if not q:
        return []
    max_sub_queries = max(0, int(max_sub_queries))
    if max_sub_queries == 0:
        return []

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION: {q}"},
    ]
    try:
        raw = await llm.generate(messages, max_tokens=int(max_tokens), temperature=0.0)
    except Exception:
        logger.exception("decompose_query: LLM generate failed for query_text=%r", q)
        return []

    data: Optional[dict[str, Any]] = try_parse_json_object(raw)
    if not data:
        logger.debug("decompose_query: unparseable response for query_text=%r", q)
        return []

    raw_subs = data.get("sub_queries")
    if not isinstance(raw_subs, list):
        return []

    sub_queries: list[str] = []
    seen = {q.strip().lower()}
    for item in raw_subs:
        if not isinstance(item, str):
            continue
        s = item.strip()
        key = s.lower()
        if not s or key in seen:
            continue
        seen.add(key)
        sub_queries.append(s)
        if len(sub_queries) >= max_sub_queries:
            break
    return sub_queries
