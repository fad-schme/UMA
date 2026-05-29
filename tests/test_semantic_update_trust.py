from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.common.types import Fact, RuntimeContext


def _build_fact(
    *,
    fact_id: str,
    tenant_id: str,
    owner_type: str,
    owner_id: str,
    trust_score: float = 0.9,
) -> Fact:
    now = datetime.now(timezone.utc)
    return Fact(
        id=fact_id,
        subject="service",
        predicate="USES",
        object="postgres",
        created_at=now,
        updated_at=now,
        source_ids=["chunk-1"],
        confidence=0.95,
        meta={},
        owner_type=owner_type,  # type: ignore[arg-type]
        owner_id=owner_id,
        tenant_id=tenant_id,
        trust_score=trust_score,
        content_hash="fact-hash",
    )


@pytest.mark.asyncio
async def test_update_trust_updates_fact_and_records_audit_history(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_once",
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)

    ctx = RuntimeContext(
        tenant_id="default",
        agent_id=uma_memory.agent_id or "agent-default",
        request_id="req-trust-once",
        user_id="user:u1",
    )
    await uma_memory.semantic_core.update_trust(
        fact.id,
        0.5,
        reason="operator downgrade after review",
        ctx=ctx,
    )

    updated = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
    )
    assert updated is not None
    assert updated.trust_score == pytest.approx(0.5)
    assert len(updated.meta["trust_updates"]) == 1
    entry = updated.meta["trust_updates"][0]
    assert entry["prior_score"] == pytest.approx(0.9)
    assert entry["new_score"] == pytest.approx(0.5)
    assert entry["reason"] == "operator downgrade after review"
    assert entry["timestamp"]


@pytest.mark.asyncio
async def test_update_trust_accumulates_history_in_order(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_twice",
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)
    ctx = RuntimeContext(
        tenant_id="default",
        agent_id=uma_memory.agent_id or "agent-default",
        request_id="req-trust-twice",
        user_id="user:u1",
    )

    await uma_memory.semantic_core.update_trust(fact.id, 0.6, reason="first review", ctx=ctx)
    await uma_memory.semantic_core.update_trust(fact.id, 0.4, reason="second review", ctx=ctx)

    updated = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
    )
    assert updated is not None
    assert updated.trust_score == pytest.approx(0.4)
    assert [entry["reason"] for entry in updated.meta["trust_updates"]] == ["first review", "second review"]
    assert [entry["prior_score"] for entry in updated.meta["trust_updates"]] == [pytest.approx(0.9), pytest.approx(0.6)]
    assert [entry["new_score"] for entry in updated.meta["trust_updates"]] == [pytest.approx(0.6), pytest.approx(0.4)]


@pytest.mark.asyncio
async def test_update_trust_rejects_out_of_range_scores_without_mutation(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_invalid_score",
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)
    ctx = RuntimeContext(
        tenant_id="default",
        agent_id=uma_memory.agent_id or "agent-default",
        request_id="req-trust-invalid-score",
        user_id="user:u1",
    )

    with pytest.raises(ValueError, match="new_score must be a float in \\[0.0, 1.0\\]"):
        await uma_memory.semantic_core.update_trust(fact.id, 1.1, reason="too high", ctx=ctx)
    with pytest.raises(ValueError, match="new_score must be a float in \\[0.0, 1.0\\]"):
        await uma_memory.semantic_core.update_trust(fact.id, -0.1, reason="too low", ctx=ctx)

    unchanged = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
    )
    assert unchanged is not None
    assert unchanged.trust_score == pytest.approx(0.9)
    assert unchanged.meta.get("trust_updates") is None


@pytest.mark.asyncio
async def test_update_trust_rejects_empty_reason_without_mutation(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_empty_reason",
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)
    ctx = RuntimeContext(
        tenant_id="default",
        agent_id=uma_memory.agent_id or "agent-default",
        request_id="req-trust-empty-reason",
        user_id="user:u1",
    )

    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        await uma_memory.semantic_core.update_trust(fact.id, 0.5, reason="   ", ctx=ctx)

    unchanged = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
    )
    assert unchanged is not None
    assert unchanged.trust_score == pytest.approx(0.9)
    assert unchanged.meta.get("trust_updates") is None


@pytest.mark.asyncio
async def test_update_trust_hides_other_agent_fact_from_unrelated_context(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_other_agent",
        tenant_id="default",
        owner_type="agent",
        owner_id="agent-other",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)
    ctx = RuntimeContext(
        tenant_id="default",
        agent_id=uma_memory.agent_id or "agent-default",
        request_id="req-trust-other-agent",
        user_id="user:u1",
    )

    with pytest.raises(ValueError, match="not found"):
        await uma_memory.semantic_core.update_trust(fact.id, 0.5, reason="not visible", ctx=ctx)

    unchanged = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="default",
        owner_type="agent",
        owner_id="agent-other",
    )
    assert unchanged is not None
    assert unchanged.trust_score == pytest.approx(0.9)
    assert unchanged.meta.get("trust_updates") is None


@pytest.mark.asyncio
async def test_update_trust_hides_cross_tenant_fact_from_mismatched_context(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_other_tenant",
        tenant_id="tenant-b",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)
    ctx = RuntimeContext(
        tenant_id="tenant-a",
        agent_id=uma_memory.agent_id or "agent-default",
        request_id="req-trust-other-tenant",
        user_id="user:u1",
    )

    with pytest.raises(ValueError, match="not found"):
        await uma_memory.semantic_core.update_trust(fact.id, 0.5, reason="wrong tenant", ctx=ctx)

    unchanged = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="tenant-b",
        owner_type="user",
        owner_id="user:u1",
    )
    assert unchanged is not None
    assert unchanged.trust_score == pytest.approx(0.9)
    assert unchanged.meta.get("trust_updates") is None
