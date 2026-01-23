from __future__ import annotations

import re


_STOPWORDS = {
    "what",
    "is",
    "the",
    "a",
    "an",
    "of",
    "about",
    "please",
    "tell",
    "me",
    "does",
    "do",
    "and",
    "or",
    "to",
    "for",
    "in",
    "on",
    "with",
    "who",
    "where",
    "when",
    "why",
    "how",
}


def extract_query_terms(text: str) -> list[str]:
    if not text:
        return []
    terms = [t for t in re.split(r"\W+", text.lower()) if len(t) > 2]
    return [t for t in terms if t not in _STOPWORDS]
