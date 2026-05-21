from __future__ import annotations

import re
from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Set, Tuple

from uma.common.accessors import get_attr_or_key

logger = logging.getLogger(__name__)


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "in",
    "on",
    "at",
    "for",
    "to",
    "of",
    "by",
    "from",
    "with",
    "as",
    "about",
    "into",
    "through",
    "over",
    "after",
    "before",
    "between",
    "without",
    "within",
    "during",
    "above",
    "below",
    "up",
    "down",
    "out",
    "off",
    "again",
    "further",
    "once",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "them",
    "my",
    "your",
    "his",
    "their",
    "our",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    # Existing repo-specific fillers.
    "know",
    "please",
    "tell",
    "whats",
    "what's",
    # Previously domain stopwords (kept to reduce lexical noise).
    "explain",
    "describe",
    "help",
    "details",
    "guide",
    "need",
    "needs",
    "structured",
    "multi",
}

_GENERIC_TERMS = {
    "architecture",
    "core",
    "design",
    "framework",
    "guide",
    "modern",
    "overview",
    "principle",
    "principles",
    "system",
    "data", "information", "service", "model", "design",
    "user", "application", "environment", "feature", "process",
}


@dataclass(frozen=True)
class QueryTermSet:
    terms: List[str]
    phrases: List[str]
    entities: List[str]


def get_stopwords() -> Set[str]:
    # Return a copy to prevent accidental mutation of module-level constants.
    return set(_STOPWORDS)


def get_generic_terms() -> Set[str]:
    # Return a copy to prevent accidental mutation of module-level constants.
    return set(_GENERIC_TERMS)


def normalize_query_text(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    # Preserve hyphenated/concatenated tokens (e.g., "defense-in-depth", "on-prem")
    # for consistent substring matching against extracted keywords/phrases.
    cleaned = re.sub(r"[^a-z0-9\\-]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


# ------------------------- Keyword / phrase extraction -------------------------

MAX_PHRASE_LEN: int = 3
MIN_PHRASE_LEN: int = 2
MAX_KEYPHRASES: int = 12
MAX_KEYWORDS: int = 12


def _normalize_for_keywords(text: str) -> str:
    text = text.lower()
    # Preserve hyphenated/concatenated tokens as single units (e.g., "multi-tier").
    # Keep hyphens but avoid turning them into separators.
    # Step 1: convert non-word punctuation to spaces, but keep hyphens.
    # NOTE: `-` must not be escaped here; escaping it makes it a literal backslash + hyphen in a raw regex.
    text = re.sub(r"[^a-z0-9-\s]", " ", text)
    # Step 2: normalize "word - word" to "word-word" before collapsing whitespace.
    text = re.sub(r"\b([a-z0-9]+)\s*-\s*([a-z0-9]+)\b", r"\1-\2", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    return text.split()


def _simple_pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    stop = get_stopwords()
    tags: List[Tuple[str, str]] = []
    for token in tokens:
        if token in stop:
            tags.append((token, "O"))
        elif re.match(r".+ly$", token):
            tags.append((token, "O"))
        elif re.match(r".+ing$|.+ed$", token):
            tags.append((token, "O"))
        elif re.match(r".+tion$|.+ment$|.+ness$|.+ity$|.+ship$", token):
            tags.append((token, "N"))
        elif re.match(r".+ive$|.+ous$|.+ful$|.+less$|.+able$|.+al$", token):
            tags.append((token, "A"))
        elif len(token) <= 2:
            tags.append((token, "O"))
        else:
            tags.append((token, "N"))
    return tags


def _extract_short_phrases(
    tokens: List[str],
    tags: List[Tuple[str, str]],
    *,
    min_len: int = MIN_PHRASE_LEN,
    max_len: int = MAX_PHRASE_LEN,
) -> List[str]:
    stop = get_stopwords()
    phrases: Set[str] = set()
    length = len(tokens)
    pos_labels = [p for (_, p) in tags]

    for i in range(length):
        for n in range(min_len, max_len + 1):
            j = i + n
            if j > length:
                continue
            span_tokens = tokens[i:j]
            span_pos = pos_labels[i:j]

            if span_tokens[0] in stop or span_tokens[-1] in stop:
                continue
            if any(t in stop for t in span_tokens[1:-1]):
                continue

            pattern = "".join(span_pos)
            if len(span_tokens) == 2:
                if pattern not in {"AN", "NN", "NA"}:
                    continue
            elif len(span_tokens) == 3:
                if pattern not in {"AAN", "ANN", "NAN"}:
                    continue

            phrases.add(" ".join(span_tokens))

    return list(phrases)


def _build_index_map(tokens: List[str]) -> Dict[str, int]:
    first: Dict[str, int] = {}
    for i, token in enumerate(tokens):
        if token not in first:
            first[token] = i
    return first


def _score_term(term: str, first_occurrence_index: int, total_tokens: int) -> float:
    generic = get_generic_terms()
    words = term.split()
    length = len(words)
    base = float(length)
    position_bonus = 0.0

    if total_tokens > 1:
        position_frac = 1.0 - (first_occurrence_index / (total_tokens - 1))
    else:
        position_frac = 1.0
    position_bonus = 0.4 * position_frac

    score = base + position_bonus

    if length == 1 and term in generic:
        score -= 0.6

    return max(score, 0.0)


def extract_keywords_and_phrases(text: str) -> Dict[str, List[str]]:
    normalized = _normalize_for_keywords(text)
    if not normalized:
        return {"keyphrases": [], "keywords": []}

    tokens = _tokenize(normalized)
    if not tokens:
        return {"keyphrases": [], "keywords": []}

    tags = _simple_pos_tag(tokens)
    first_index_map = _build_index_map(tokens)
    total_tokens = len(tokens)

    stop = get_stopwords()
    generic = get_generic_terms()
    phrase_scores: Dict[str, float] = {}
    keyword_scores: Dict[str, float] = {}

    phrase_candidates = _extract_short_phrases(tokens, tags)
    for phrase in phrase_candidates:
        first_word = phrase.split()[0]
        idx = first_index_map.get(first_word, total_tokens - 1)
        phrase_scores[phrase] = _score_term(phrase, idx, total_tokens)

    for i, tok in enumerate(tokens):
        # Drop pure-numeric tokens to avoid polluting lexical term sets with IDs/years/etc.
        # Numeric ranges are better handled via semantic retrieval and/or explicit patterns.
        if tok.isdigit():
            continue
        if tok in stop:
            continue
        if tok in generic:
            continue
        score = _score_term(tok, i, total_tokens)
        if score <= 0:
            continue
        keyword_scores[tok] = max(keyword_scores.get(tok, 0.0), score)

    keyphrases_sorted = [
        t for t, _ in sorted(phrase_scores.items(), key=lambda kv: kv[1], reverse=True)
    ][:MAX_KEYPHRASES]

    keywords_sorted = [
        t for t, _ in sorted(keyword_scores.items(), key=lambda kv: kv[1], reverse=True)
    ][:MAX_KEYWORDS]

    return {"keyphrases": keyphrases_sorted, "keywords": keywords_sorted}


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_entities(text: str) -> List[str]:
    if not text:
        return []
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
    capitalized = re.findall(r"\b[A-Z][a-z][A-Za-z0-9\\-]+\b", text)
    raw = [t.lower() for t in (acronyms + capitalized) if t]
    stop = get_stopwords()
    return _unique_preserve_order([t for t in raw if t not in stop and len(t) >= 2])


def build_query_term_set(
    text: str,
    *,
    max_terms: int = 10,
    max_phrases: int = 4,
    min_term_len: int = 3,
) -> QueryTermSet:
    if not text or not isinstance(text, str):
        return QueryTermSet(terms=[], phrases=[], entities=[])

    extracted = extract_keywords_and_phrases(text)
    ranked_terms = [
        t for t in (extracted.get("keywords", []) or []) if isinstance(t, str) and len(t) >= min_term_len
    ]
    ranked_phrases = [p for p in (extracted.get("keyphrases", []) or []) if isinstance(p, str) and p.strip()]
    entities = _extract_entities(text)

    return QueryTermSet(
        terms=ranked_terms[: max_terms],
        phrases=ranked_phrases[: max_phrases],
        entities=entities,
    )


def text_matches_query_terms(
    text: str,
    term_set: QueryTermSet,
    *,
    min_term_matches: int = 2,
    max_terms_for_match: int = 6,
) -> bool:
    if not text:
        return False
    hay = normalize_query_text(text)
    if not hay:
        return False
    for phrase in term_set.phrases:
        if phrase and phrase in hay:
            return True
    terms = [t for t in term_set.terms if t][:max_terms_for_match]
    if not terms:
        return False
    matches = 0
    for term in terms:
        if term in hay:
            matches += 1
            if matches >= min_term_matches:
                return True
    return False


#
# NOTE:
# `extract_keywords_and_phrases()` is the single canonical extractor for keywords and phrases.
# Do not add parallel extractors/wrappers here.


def build_fact_embedding_text(fact: object) -> str:
    """
    Build embedding text for a Fact-like object.

    Priority:
    1) meta.excerpt / meta.text / meta.description, augmented with relation text
    2) relation text built from subject + predicate + object
    """
    subject = ""
    predicate = ""
    object_text = ""

    try:
        subject = str(get_attr_or_key(fact, "subject") or "").strip()
    except Exception:
        subject = ""
    try:
        predicate = str(get_attr_or_key(fact, "predicate") or "").strip()
    except Exception:
        predicate = ""
    try:
        obj = get_attr_or_key(fact, "object") if isinstance(fact, dict) else getattr(fact, "object", None)
        if isinstance(obj, dict):
            for key in ("text", "content"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    object_text = val.strip().replace("\n", " ")
                    break
            if not object_text and obj:
                object_text = str(obj).strip().replace("\n", " ")
        elif obj is not None:
            object_text = str(obj).strip().replace("\n", " ")
    except Exception:
        logger.exception("build_fact_embedding_text: failed to read fact object")

    relation_parts = [part for part in (subject, predicate, object_text) if part]
    relation_text = " ".join(relation_parts).strip()

    try:
        meta = None
        meta = get_attr_or_key(fact, "meta")
        if isinstance(meta, dict):
            for key in ("excerpt", "text", "description"):
                val = meta.get(key)
                if isinstance(val, str) and val.strip():
                    excerpt = val.strip().replace("\n", " ")
                    if relation_text and relation_text.lower() not in excerpt.lower():
                        return f"{relation_text}. {excerpt}".strip()
                    return excerpt
    except Exception:
        logger.exception("build_fact_embedding_text: failed to read fact meta")

    return relation_text
