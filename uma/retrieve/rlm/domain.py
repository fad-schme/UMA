"""
uma.retrieve.rlm.domain
=============================

Minimal retrieval-domain tagging and defaulting for RLM routing.

Important semantic boundary:
- `kind` and `kb_lane` are UMA's canonical storage taxonomy.
- `domain` is retrieval/routing metadata only.
- This module must not be treated as the classifier for persisted records.

Routing metadata is stored in item meta (no schema migrations):

    meta["domain"] ∈ {"kb_doc", "user_profile", "procedural", "system"}

Old data may have domain missing; in that case we apply deterministic defaults
as specified in the Phase 0 plan.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


DOMAIN_VALUES: set[str] = {"kb_doc", "user_profile", "procedural", "system"}

# Preference-like predicates for user_profile facts.
PREFERENCE_PREDICATES: set[str] = {
    "LIKES",
    "PREFERS",
    "DISLIKES",
    "LOVES",
    "HATES",
    "AVOIDS",
    "FAVORS",
}


def _get_meta(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        meta = item.get("meta")
        if isinstance(meta, dict):
            return meta
        meta = {}
        item["meta"] = meta
        return meta
    meta = getattr(item, "meta", None)
    if isinstance(meta, dict):
        return meta
    meta = {}
    try:
        setattr(item, "meta", meta)
    except Exception:
        logger.debug("domain: could not set meta on item=%r", type(item).__name__, exc_info=True)
    return meta


def get_domain(item: Any) -> Optional[str]:
    meta = _get_meta(item)
    raw = meta.get("domain") if isinstance(meta, dict) else None
    if not raw:
        return None
    try:
        d = str(raw).strip().lower()
    except (TypeError, ValueError):
        return None
    return d if d in DOMAIN_VALUES else None


def _get_attr_or_key(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _preference_like_object(obj: Any) -> bool:
    """
    Minimal heuristic for “preference-like” objects.
    """
    if obj is None:
        return False
    try:
        s = str(obj).strip()
    except (TypeError, ValueError):
        return False
    if not s:
        return False
    words = s.split()
    return 1 <= len(words) <= 6 and len(s) <= 60


def ensure_fact_domain(fact: Any) -> str:
    """
    Ensure retrieval-domain metadata for routing decisions.

    This is not part of the canonical storage taxonomy. It exists only so RLM
    can distinguish retrieval intent buckets such as `kb_doc` vs `user_profile`.

    Defaulting rules (Phase 0):
    - If predicate in {LIKES, PREFERS, DISLIKES, ...} => user_profile
    - OR subject looks like user:<id> (or "user") and object is preference-like => user_profile
    - Otherwise => kb_doc
    """
    meta = _get_meta(fact)
    existing = get_domain(fact)
    if existing:
        meta["domain"] = existing
        return existing

    pred = _get_attr_or_key(fact, "predicate", "") or ""
    pred_u = str(pred).strip().upper()
    subj = _get_attr_or_key(fact, "subject", "") or ""
    subj_s = str(subj).strip().lower()
    obj = _get_attr_or_key(fact, "object", None)

    if pred_u in PREFERENCE_PREDICATES:
        meta["domain"] = "user_profile"
        return "user_profile"

    if (subj_s.startswith("user:") or subj_s == "user") and _preference_like_object(obj):
        meta["domain"] = "user_profile"
        return "user_profile"

    meta["domain"] = "kb_doc"
    return "kb_doc"


def ensure_chunk_domain(chunk: Any) -> str:
    """
    Ensure retrieval-domain metadata for routing decisions.

    Defaulting rules (Phase 0):
    - If chunk has source_path set => kb_doc
    - Otherwise => kb_doc (safe default)
    """
    meta = _get_meta(chunk)
    existing = get_domain(chunk)
    if existing:
        meta["domain"] = existing
        return existing

    source_path = _get_attr_or_key(chunk, "source_path", "") or ""
    if str(source_path).strip():
        meta["domain"] = "kb_doc"
        return "kb_doc"

    meta["domain"] = "kb_doc"
    return "kb_doc"


def ensure_skill_domain(skill: Any) -> str:
    """
    Ensure retrieval-domain metadata for routing decisions.

    Defaulting rules (Phase 0): Skills => procedural.
    """
    meta = _get_meta(skill)
    existing = get_domain(skill)
    if existing:
        meta["domain"] = existing
        return existing
    meta["domain"] = "procedural"
    return "procedural"


def ensure_domains_for_facts(facts: Iterable[Any]) -> None:
    for f in facts or []:
        try:
            ensure_fact_domain(f)
        except Exception:
            logger.debug("domain: skipped malformed item in ensure_domains", exc_info=True)
            continue


def ensure_domains_for_chunks(chunks: Iterable[Any]) -> None:
    for ch in chunks or []:
        try:
            ensure_chunk_domain(ch)
        except Exception:
            logger.debug("domain: skipped malformed item in ensure_domains", exc_info=True)
            continue


def ensure_domains_for_skills(skills: Iterable[Any]) -> None:
    for s in skills or []:
        try:
            ensure_skill_domain(s)
        except Exception:
            logger.debug("domain: skipped malformed item in ensure_domains", exc_info=True)
            continue


def filter_facts_by_domains(facts: list[Any], allowed_domains: set[str]) -> list[Any]:
    """
    Filter facts by retrieval-domain metadata.

    Items missing meta['domain'] are defaulted deterministically.
    """
    if not facts:
        return []
    allowed = {d for d in (allowed_domains or set()) if d in DOMAIN_VALUES}
    if not allowed:
        return []
    out: list[Any] = []
    for f in facts:
        d = ensure_fact_domain(f)
        if d in allowed:
            out.append(f)
    return out
