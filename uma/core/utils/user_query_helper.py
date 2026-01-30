from __future__ import annotations

import re


_STOPWORDS = {
    "what",
    "which",
    "whats",
    "what's",
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
    "did",
    "you",
    "your",
    "know",
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


def expand_query_terms(text: str) -> list[str]:
    """
    Expand short queries with simple domain-agnostic variants.
    """
    terms = extract_query_terms(text)
    if not terms:
        return []

    expanded = list(terms)
    joined = " ".join(terms)
    if "sso" in terms:
        expanded.extend(["single sign on", "single sign-on"])
    if "single" in terms and "sign" in terms and "on" in terms:
        expanded.append("sso")
    if "single sign on" in text.lower():
        expanded.append("sso")
    if joined:
        expanded.append(joined)
    if len(terms) == 1 and text:
        expanded.append(text.lower().strip())
    # de-dup while preserving order
    seen = set()
    out = []
    for t in expanded:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_fact_embedding_text(fact: object) -> str:
    """
    Build embedding text for a Fact-like object.

    Priority:
    1) meta.excerpt / meta.text / meta.description
    2) object.text / object.content
    3) subject predicate object
    """
    try:
        meta = None
        if isinstance(fact, dict):
            meta = fact.get("meta")
        else:
            meta = getattr(fact, "meta", None)
        if isinstance(meta, dict):
            for key in ("excerpt", "text", "description"):
                val = meta.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip().replace("\n", " ")
    except Exception:
        pass

    try:
        obj = fact.get("object") if isinstance(fact, dict) else getattr(fact, "object", None)
        if isinstance(obj, dict):
            for key in ("text", "content"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip().replace("\n", " ")
        elif isinstance(obj, str) and obj.strip():
            return obj.strip().replace("\n", " ")
    except Exception:
        pass

    try:
        subject = fact.get("subject") if isinstance(fact, dict) else getattr(fact, "subject", "")
        predicate = fact.get("predicate") if isinstance(fact, dict) else getattr(fact, "predicate", "")
        obj = fact.get("object") if isinstance(fact, dict) else getattr(fact, "object", "")
        return f"{subject} {predicate} {obj}".strip()
    except Exception:
        return ""
