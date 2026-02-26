from __future__ import annotations

from datetime import datetime, timezone

from uma.core.retrieval.rlm.intent import QueryIntent, classify_query_intent
from uma.core.retrieval.rlm.domain import ensure_fact_domain, filter_facts_by_domains
from uma.types.types_fact import Fact


def _fact(*, predicate: str, subject: str = "user", obj: str = "x", source_ids=None, meta=None) -> Fact:
    return Fact(
        id="fact_test",
        subject=subject,
        predicate=predicate,
        object=obj,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        source_ids=list(source_ids or []),
        meta=dict(meta or {}),
        owner_type="user",
        owner_id="user:test",
        salience=0.0,
        confidence=0.7,
    )


def test_intent_topical_default() -> None:
    q = "How should a secure cloud security architecture be structured for a multi-tier application?"
    assert classify_query_intent(q) == QueryIntent.TOPICAL


def test_intent_personal_preferences() -> None:
    assert classify_query_intent("What do I like?") == QueryIntent.PERSONAL
    assert classify_query_intent("What are my preferences?") == QueryIntent.PERSONAL


def test_filtering_excludes_user_profile_when_not_allowed() -> None:
    profile = _fact(predicate="LIKES", obj="sushi")
    kb = _fact(predicate="STATES", subject="Document(d1)", obj="x", source_ids=["chunk_123"])
    out = filter_facts_by_domains([profile, kb], allowed_domains={"kb_doc"})
    assert out == [kb]


def test_domain_defaulting_preference_predicate() -> None:
    f = _fact(predicate="likes", obj="sushi")
    assert ensure_fact_domain(f) == "user_profile"
    assert f.meta.get("domain") == "user_profile"


def test_domain_defaulting_otherwise_kb_doc() -> None:
    f = _fact(predicate="STATES", subject="Document(d1)", obj="x")
    assert ensure_fact_domain(f) == "kb_doc"
    assert f.meta.get("domain") == "kb_doc"
