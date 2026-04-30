from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.memory.promotion import PromotionPolicy
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import Fact, RuntimeContext, SCOPE_MODEL_VERSION, TargetOwner


def _build_fact(
    *,
    fact_id: str,
    owner_type: str,
    owner_id: str,
    predicate: str = "USES",
    object_text: str = "kubernetes cluster orchestration for production workloads",
    session_id: str | None = None,
    agent_id: str | None = None,
    workspace_id: str | None = None,
) -> Fact:
    now = datetime.now(timezone.utc)
    return Fact(
        id=fact_id,
        subject="team",
        predicate=predicate,
        object=object_text,
        created_at=now,
        updated_at=now,
        source_ids=["chunk-source-1"],
        confidence=0.95,
        salience=0.92,
        meta={"source_type": "text"},
        owner_type=owner_type,  # type: ignore[arg-type]
        owner_id=owner_id,
        tenant_id=DEFAULT_TENANT_ID,
        workspace_id=workspace_id,
        session_id=session_id,
        origin_agent_id=agent_id,
        origin_user_id="user:u1",
        origin_session_id=session_id,
        scope_model_version=SCOPE_MODEL_VERSION,
    )


@pytest.mark.asyncio
async def test_session_to_user_promotion_creates_new_fact_and_preserves_original(uma_memory) -> None:
    memory = uma_memory
    assert memory.agent_id
    source = _build_fact(
        fact_id="fact_source_session_user",
        owner_type="user",
        owner_id="user:u1",
        session_id="session-a",
        agent_id=memory.agent_id,
    )
    embedding = (await memory.embedder.embed([str(source.object)]))[0]
    await memory.semantic_core.upsert_fact(source, embedding)

    policy = PromotionPolicy(agent_id=memory.agent_id)
    promoted = policy.promote(
        source,
        target_owner=TargetOwner(
            tenant_id=DEFAULT_TENANT_ID,
            owner_type="user",
            owner_id="user:u1",
        ),
        reason="test_session_to_user",
    )
    await memory.semantic_core.upsert_fact(promoted, embedding)

    assert promoted.id != source.id
    assert promoted.owner_type == "user"
    assert promoted.owner_id == "user:u1"
    assert promoted.session_id is None
    assert promoted.origin_session_id == "session-a"
    assert promoted.scope_model_version == SCOPE_MODEL_VERSION
    assert promoted.meta["promotion"]["source_fact_id"] == source.id
    assert promoted.meta["promotion"]["target_owner_type"] == "user"

    request = memory._build_retrieval_request(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=memory.agent_id,
            request_id="req-prom-user",
            user_id="user:u1",
        )
    )
    visible = await memory.memory_env.fetch_facts_by_ids(
        request,
        [source.id, promoted.id],
        owner_type="user",
        owner_id="user:u1",
    )
    assert {fact.id for fact in visible} == {promoted.id}


@pytest.mark.asyncio
async def test_session_to_workspace_promotion_is_explicit_and_does_not_broaden_user_visibility(uma_memory) -> None:
    memory = uma_memory
    assert memory.agent_id
    source = _build_fact(
        fact_id="fact_source_session_workspace",
        owner_type="user",
        owner_id="user:u1",
        session_id="session-a",
        agent_id=memory.agent_id,
    )
    embedding = (await memory.embedder.embed([str(source.object)]))[0]
    await memory.semantic_core.upsert_fact(source, embedding)

    policy = PromotionPolicy(agent_id=memory.agent_id)
    promoted = policy.promote(
        source,
        target_owner=TargetOwner(
            tenant_id=DEFAULT_TENANT_ID,
            owner_type="workspace",
            owner_id="workspace-1",
            workspace_id="workspace-1",
        ),
        reason="test_session_to_workspace",
    )
    await memory.semantic_core.upsert_fact(promoted, embedding)

    assert promoted.id != source.id
    assert promoted.owner_type == "workspace"
    assert promoted.owner_id == "workspace-1"
    assert promoted.workspace_id == "workspace-1"
    assert promoted.session_id is None
    assert promoted.meta["promotion"]["source_scope_kind"] == "session"
    assert promoted.meta["promotion"]["target_owner_type"] == "workspace"

    workspace_facts = await memory.semantic_core.list_facts_for_owner(
        owner_type="workspace",
        owner_id="workspace-1",
    )
    assert {fact.id for fact in workspace_facts} == {promoted.id}

    request = memory._build_retrieval_request(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=memory.agent_id,
            request_id="req-prom-workspace",
            user_id="user:u1",
        )
    )
    user_visible = await memory.memory_env.fetch_facts_by_ids(
        request,
        [promoted.id],
        owner_type="user",
        owner_id="user:u1",
    )
    assert user_visible == []


@pytest.mark.asyncio
async def test_user_to_agent_promotion_is_copy_based_and_idempotent(uma_memory) -> None:
    memory = uma_memory
    assert memory.agent_id
    source = _build_fact(
        fact_id="fact_source_user_agent",
        owner_type="user",
        owner_id="user:u1",
        session_id=None,
        agent_id=memory.agent_id,
    )
    embedding = (await memory.embedder.embed([str(source.object)]))[0]
    await memory.semantic_core.upsert_fact(source, embedding)

    policy = PromotionPolicy(agent_id=memory.agent_id)
    target_owner = TargetOwner(
        tenant_id=DEFAULT_TENANT_ID,
        owner_type="agent",
        owner_id=memory.agent_id,
    )
    promoted_a = policy.promote(source, target_owner=target_owner, reason="test_user_to_agent")
    promoted_b = policy.promote(source, target_owner=target_owner, reason="test_user_to_agent")
    assert promoted_a.id == promoted_b.id
    assert promoted_a.id != source.id
    assert promoted_a.owner_type == "agent"
    assert promoted_a.owner_id == memory.agent_id

    await memory.semantic_core.upsert_fact(promoted_a, embedding)
    await memory.semantic_core.upsert_fact(promoted_b, embedding)

    sem_conn = memory._stores["semantic"]._conn()
    try:
        rows = memory._stores["semantic"]._query_all(
            sem_conn,
            "SELECT id FROM facts WHERE id=?",
            params=[promoted_a.id],
            log_context="test_user_to_agent_promotion_idempotent",
        )
        assert len(rows) == 1
    finally:
        sem_conn.close()


@pytest.mark.asyncio
async def test_invalid_promotion_targets_are_rejected(uma_memory) -> None:
    memory = uma_memory
    assert memory.agent_id
    policy = PromotionPolicy(agent_id=memory.agent_id)
    session_fact = _build_fact(
        fact_id="fact_source_invalid_targets",
        owner_type="user",
        owner_id="user:u1",
        session_id="session-a",
        agent_id=memory.agent_id,
    )

    with pytest.raises(ValueError, match="tenant_id"):
        policy.promote(
            session_fact,
            target_owner=TargetOwner(
                tenant_id="other-tenant",
                owner_type="user",
                owner_id="user:u1",
            ),
        )

    with pytest.raises(ValueError, match="system scope"):
        policy.promote(
            session_fact,
            target_owner=TargetOwner(
                tenant_id=DEFAULT_TENANT_ID,
                owner_type="system",
                owner_id="system-global",
            ),
        )

    with pytest.raises(ValueError, match="only be promoted to user or workspace"):
        policy.promote(
            session_fact,
            target_owner=TargetOwner(
                tenant_id=DEFAULT_TENANT_ID,
                owner_type="agent",
                owner_id=memory.agent_id,
            ),
        )


@pytest.mark.asyncio
async def test_process_turn_without_policy_does_not_silently_promote(uma_memory) -> None:
    memory = uma_memory
    assert memory.promotion_policy is None

    await memory.process_turn(
        user_id="user:u1",
        user_msg="hello",
        assistant_reply="the team uses kubernetes cluster orchestration for production workloads.",
        extra_meta={"session_id": "session-a"},
    )

    sem_conn = memory._stores["semantic"]._conn()
    try:
        rows = memory._stores["semantic"]._query_all(
            sem_conn,
            "SELECT id FROM facts WHERE id LIKE 'fact_prom_%'",
            params=[],
            log_context="test_process_turn_without_policy_does_not_silently_promote",
        )
        assert rows == []
    finally:
        sem_conn.close()
