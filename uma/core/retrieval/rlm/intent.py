"""
uma.core.retrieval.rlm.intent
=============================

Deterministic query intent classification for RLM routing.

Goals
-----
- Production-safe (no LLM calls)
- Conservative defaults (TOPICAL unless strongly personal)
- Small, unit-testable surface
"""

from __future__ import annotations

import re
from enum import Enum


class QueryIntent(str, Enum):
    TOPICAL = "topical"
    PERSONAL = "personal"
    MIXED = "mixed"


_RE_PERSONAL_MARKERS = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)
_RE_PERSONAL_QUERIES = re.compile(
    r"\b("
    r"what do i like|my preferences?|what did i do|did i|do i|"
    r"my background|my experience|remember|recall|"
    r"like|prefer|dislike|hate|love"
    r")\b",
    re.IGNORECASE,
)

# Keep topical indicators small and fairly “formal”; used only to decide MIXED vs PERSONAL.
_RE_TOPICAL_HINTS = re.compile(
    r"\b(architecture|design|best practices?|framework|strategy|structured|structure|"
    r"policy|principles?|implementation|multi-?tier|reference)\b",
    re.IGNORECASE,
)


def classify_query_intent(query_text: str) -> QueryIntent:
    """
    Classify query intent for routing between topical KB retrieval vs personal recall/profile.

    Heuristic rules (conservative):
    - PERSONAL only when strong first-person + preference/recall cues exist.
    - MIXED when strong personal cues also include topical indicators.
    - TOPICAL otherwise.
    """
    q = (query_text or "").strip()
    if not q:
        return QueryIntent.TOPICAL

    personal_markers = bool(_RE_PERSONAL_MARKERS.search(q))
    personal_query = bool(_RE_PERSONAL_QUERIES.search(q))

    # Personal requires both marker and an explicit personal cue.
    if personal_markers and personal_query:
        topical_hint = bool(_RE_TOPICAL_HINTS.search(q))
        return QueryIntent.MIXED if topical_hint else QueryIntent.PERSONAL

    return QueryIntent.TOPICAL

