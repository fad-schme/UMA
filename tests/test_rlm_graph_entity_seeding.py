from __future__ import annotations

from datetime import datetime, timezone

from uma.core.retrieval.rlm.decisions import deterministic_decision
from uma.types.types_fact import Fact


def _kb_fact(*, predicate: str = "SEGMENTED_INTO", text: str = "iam vpc kms") -> Fact:
    return Fact(
        id="fact_test",
        subject="cloud_security_architecture",
        predicate=predicate,
        object="network segmentation",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        source_ids=["chunk_1"],
        meta={"domain": "kb_doc", "fact_text": text, "source_path": "kb/doc.md"},
        owner_type="agent",
        owner_id="agent:test",
        salience=0.0,
        confidence=0.7,
    )


class _Coverage:
    needs_semantic = False
    needs_clusters = False


def test_topical_graph_expansion_seeds_from_entities_not_user_id() -> None:
    class _Pack:
        graph = []
        facts = [_kb_fact()]
        chunks = []
        steps = []
        query_text = "How should IAM and VPC be used in a multi-tier architecture?"
        intent = "topical"
        owner_type = "agent"
        owner_id = "agent:test"
        user_id = "user:123"

    decision = deterministic_decision(
        _Pack(),
        _Coverage(),
        cfg={
            "chunk_fallback_enabled": False,
            "graph_predicate_limit": 2,
        },
    )
    assert decision is not None
    actions = [a for a in decision.actions if a.action == "expand_graph"]
    assert actions, "expected topical graph expansion actions"
    assert all(a.subject != "user:123" for a in actions)
    assert actions[0].subject in {"IAM", "VPC"}


def test_personal_graph_expansion_keeps_user_anchor() -> None:
    class _Pack:
        graph = []
        facts = [
            Fact(
                id="fact_test",
                subject="user:123",
                predicate="LIKES",
                object="sushi",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                source_ids=[],
                meta={"domain": "user_profile", "fact_text": "user likes sushi"},
                owner_type="user",
                owner_id="user:123",
                salience=0.0,
                confidence=0.7,
            )
        ]
        chunks = []
        steps = []
        query_text = "What do I like?"
        intent = "personal"
        owner_type = "user"
        owner_id = "user:123"
        user_id = "user:123"

    decision = deterministic_decision(
        _Pack(),
        _Coverage(),
        cfg={
            "chunk_fallback_enabled": False,
            "graph_predicate_limit": 2,
            "next_predicate_scope": lambda _p, _limit: ["LIKES"],
        },
    )
    assert decision is not None
    actions = [a for a in decision.actions if a.action == "expand_graph"]
    assert actions
    assert actions[0].subject == "user:123"
    assert actions[0].predicate == "LIKES"
