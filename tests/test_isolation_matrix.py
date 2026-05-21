from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.helpers.runtime import init_uma_for_tests
from uma import UMAMemory
from uma.api.runtime import UMARequestHandle, UMARuntime
from uma.retrieve.rlm.context_pack import ContextPack
from uma.memory.promotion import PromotionPolicy
from uma.memory.working_memory.core import SessionScope
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import Fact, RuntimeContext, SCOPE_MODEL_VERSION


def _build_fact(
    *,
    fact_id: str,
    tenant_id: str,
    owner_type: str,
    owner_id: str,
    object_text: str,
    session_id: str | None,
    agent_id: str,
    user_id: str = "user:u1",
) -> Fact:
    now = datetime.now(timezone.utc)
    return Fact(
        id=fact_id,
        subject="team",
        predicate="USES",
        object=object_text,
        created_at=now,
        updated_at=now,
        source_ids=["chunk-source-1"],
        confidence=0.95,
        salience=0.9,
        meta={"source_type": "text"},
        owner_type=owner_type,  # type: ignore[arg-type]
        owner_id=owner_id,
        tenant_id=tenant_id,
        workspace_id=None,
        session_id=session_id,
        origin_agent_id=agent_id,
        origin_user_id=user_id,
        origin_session_id=session_id,
        scope_model_version=SCOPE_MODEL_VERSION,
    )


def _fact_ids_for_user(memory, owner_id: str) -> list[str]:
    conn = memory._stores["semantic"]._conn()
    try:
        rows = memory._stores["semantic"]._query_all(
            conn,
            "SELECT id FROM facts WHERE owner_type=? AND owner_id=? ORDER BY id ASC",
            params=["user", owner_id],
            log_context="test_isolation_matrix_fact_ids",
        )
        return [row["id"] for row in rows]
    finally:
        conn.close()


class _EmptyController:
    async def retrieve_context(self, request, query_text):
        return ContextPack(
            user_id=request.normalized_user_id,
            query_text=query_text,
            owner_type="user",
            owner_id=request.normalized_user_id,
        )


@pytest.mark.asyncio
async def test_multi_tenant_isolation_holds_with_matching_scope_tokens(tmp_path: Path) -> None:
    memory = await init_uma_for_tests(tmp_path, agent_id="agent-tenant")
    try:
        embedding = (await memory.embedder.embed(["tenant isolation fact"]))[0]
        fact_a = _build_fact(
            fact_id="fact_tenant_a",
            tenant_id="tenant-a",
            owner_type="user",
            owner_id="user:u1",
            object_text="tenant a only",
            session_id="shared-session",
            agent_id="agent-tenant",
        )
        fact_b = _build_fact(
            fact_id="fact_tenant_b",
            tenant_id="tenant-b",
            owner_type="user",
            owner_id="user:u1",
            object_text="tenant b only",
            session_id="shared-session",
            agent_id="agent-tenant",
        )
        await memory.semantic_core.upsert_fact(fact_a, embedding)
        await memory.semantic_core.upsert_fact(fact_b, embedding)

        memory.working_memory.append(
            scope=SessionScope(
                tenant_id="tenant-a",
                agent_id="agent-tenant",
                session_id="shared-session",
                user_id="user:u1",
            ),
            role="user",
            content="tenant a wm",
        )
        memory.working_memory.append(
            scope=SessionScope(
                tenant_id="tenant-b",
                agent_id="agent-tenant",
                session_id="shared-session",
                user_id="user:u1",
            ),
            role="user",
            content="tenant b wm",
        )

        runtime = UMARuntime.from_memory(memory)
        memory._rlm_controller = _EmptyController()
        handle_a = runtime.bind(
            RuntimeContext(
                tenant_id="tenant-a",
                agent_id="agent-tenant",
                request_id="req-tenant-a",
                user_id="user:u1",
                session_id="shared-session",
            )
        )
        handle_b = runtime.bind(
            RuntimeContext(
                tenant_id="tenant-b",
                agent_id="agent-tenant",
                request_id="req-tenant-b",
                user_id="user:u1",
                session_id="shared-session",
            )
        )

        ctx_a, ctx_b = await asyncio.gather(
            handle_a.retrieve_context("tenant isolation"),
            handle_b.retrieve_context("tenant isolation"),
        )
        req_a = memory.runtime._build_retrieval_request(handle_a.context)
        req_b = memory.runtime._build_retrieval_request(handle_b.context)
        facts_a = await memory.memory_env.fetch_facts_by_ids(req_a, [fact_a.id, fact_b.id], owner_type="user", owner_id="user:u1")
        facts_b = await memory.memory_env.fetch_facts_by_ids(req_b, [fact_a.id, fact_b.id], owner_type="user", owner_id="user:u1")

        assert [msg.content for msg in ctx_a["working_memory"]] == ["tenant a wm"]
        assert [msg.content for msg in ctx_b["working_memory"]] == ["tenant b wm"]
        assert {fact.id for fact in facts_a} == {"fact_tenant_a"}
        assert {fact.id for fact in facts_b} == {"fact_tenant_b"}
    finally:
        memory.shutdown()


@pytest.mark.asyncio
async def test_multi_user_retrieval_isolates_user_owned_data_but_keeps_agent_kb_shared(uma_memory, tmp_path: Path) -> None:
    memory = uma_memory
    assert memory.agent_id
    runtime = UMARuntime.from_memory(memory)

    agent_doc = tmp_path / "agent_shared.txt"
    agent_doc.write_text(
        "Shared agent KB mentions overlap token and common policy guidance. This sentence keeps chunking valid.\n",
        encoding="utf-8",
    )
    user_a_doc = tmp_path / "user_a.txt"
    user_a_doc.write_text(
        "User alpha document mentions overlap token and alpha-only preference. This sentence keeps chunking valid.\n",
        encoding="utf-8",
    )
    user_b_doc = tmp_path / "user_b.txt"
    user_b_doc.write_text(
        "User beta document mentions overlap token and beta-only preference. This sentence keeps chunking valid.\n",
        encoding="utf-8",
    )

    await memory.ingest_document(str(agent_doc), owner_type="agent", owner_id=memory.agent_id)
    await memory.ingest_document(str(user_a_doc), owner_type="user", owner_id="user:u1")
    await memory.ingest_document(str(user_b_doc), owner_type="user", owner_id="user:u2")

    handle_a = runtime.bind(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=memory.agent_id,
            request_id="req-user-a",
            user_id="user:u1",
        )
    )
    handle_b = runtime.bind(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=memory.agent_id,
            request_id="req-user-b",
            user_id="user:u2",
        )
    )

    ctx_a, ctx_b = await asyncio.gather(
        handle_a.retrieve_context("overlap token"),
        handle_b.retrieve_context("overlap token"),
    )

    owner_pairs_a = {(getattr(chunk, "owner_type", None), getattr(chunk, "owner_id", None)) for chunk in ctx_a.get("chunks") or []}
    owner_pairs_b = {(getattr(chunk, "owner_type", None), getattr(chunk, "owner_id", None)) for chunk in ctx_b.get("chunks") or []}

    assert ("agent", memory.agent_id) in owner_pairs_a
    assert ("agent", memory.agent_id) in owner_pairs_b
    assert ("user", "user:u1") in owner_pairs_a
    assert ("user", "user:u2") not in owner_pairs_a
    assert ("user", "user:u2") in owner_pairs_b
    assert ("user", "user:u1") not in owner_pairs_b


@pytest.mark.asyncio
async def test_retrieval_and_process_turn_overlap_preserve_session_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = await init_uma_for_tests(tmp_path, agent_id="agent-overlap")
    try:
        runtime = UMARuntime.from_memory(memory)
        memory._rlm_controller = _EmptyController()

        await memory.process_turn(
            user_id="user:u1",
            user_msg="I like coffee in session a.",
            assistant_reply="Good to know.",
            session_id="session-a",
            extra_meta={"request_id": "req-seed-a"},
        )

        entered = asyncio.Event()
        release = asyncio.Event()
        original_store_episode = memory.pipeline._store_episode

        async def blocking_store_episode(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original_store_episode(*args, **kwargs)

        monkeypatch.setattr(memory.pipeline, "_store_episode", blocking_store_episode)

        turn_task = asyncio.create_task(
            memory.process_turn(
                user_id="user:u1",
                user_msg="I like tea in session b.",
                assistant_reply="Nice.",
                session_id="session-b",
                extra_meta={"request_id": "req-write-b"},
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=1.0)

        handle_a = runtime.bind(
            RuntimeContext(
                tenant_id=DEFAULT_TENANT_ID,
                agent_id="agent-overlap",
                request_id="req-read-a",
                user_id="user:u1",
                session_id="session-a",
            )
        )
        during_ctx = await handle_a.retrieve_context("coffee")

        release.set()
        await asyncio.wait_for(turn_task, timeout=2.0)

        req_a = memory.runtime._build_retrieval_request(
            RuntimeContext(
                tenant_id=DEFAULT_TENANT_ID,
                agent_id="agent-overlap",
                request_id="req-facts-a",
                user_id="user:u1",
                session_id="session-a",
            )
        )
        req_b = memory.runtime._build_retrieval_request(
            RuntimeContext(
                tenant_id=DEFAULT_TENANT_ID,
                agent_id="agent-overlap",
                request_id="req-facts-b",
                user_id="user:u1",
                session_id="session-b",
            )
        )
        fact_ids = _fact_ids_for_user(memory, "user:u1")
        facts_a = await memory.memory_env.fetch_facts_by_ids(req_a, fact_ids, owner_type="user", owner_id="user:u1")
        facts_b = await memory.memory_env.fetch_facts_by_ids(req_b, fact_ids, owner_type="user", owner_id="user:u1")

        during_wm = [msg.content for msg in during_ctx["working_memory"]]
        objects_a = {str(getattr(fact, "object", "")) for fact in facts_a}
        objects_b = {str(getattr(fact, "object", "")) for fact in facts_b}

        assert any("session a" in content for content in during_wm)
        assert all("session b" not in content for content in during_wm)
        assert any("coffee" in obj for obj in objects_a)
        assert all("tea" not in obj for obj in objects_a)
        assert any("tea" in obj for obj in objects_b)
    finally:
        memory.shutdown()


@pytest.mark.asyncio
async def test_cross_agent_visibility_requires_explicit_promotion(uma_memory) -> None:
    memory = uma_memory
    embedding = (await memory.embedder.embed(["promotion isolation fact"]))[0]
    source = _build_fact(
        fact_id="fact_source_session_agent_a",
        tenant_id=DEFAULT_TENANT_ID,
        owner_type="user",
        owner_id="user:u1",
        object_text="session local only",
        session_id="session-a",
        agent_id="agent-a",
    )
    await memory.semantic_core.upsert_fact(source, embedding)

    cross_agent_request = memory.runtime._build_retrieval_request(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-b",
            request_id="req-cross-agent",
            user_id="user:u1",
            session_id="session-b",
        )
    )

    before = await memory.memory_env.fetch_facts_by_ids(
        cross_agent_request,
        [source.id],
        owner_type="user",
        owner_id="user:u1",
    )
    assert before == []

    policy = PromotionPolicy(agent_id="agent-a")
    promoted = policy.promote(
        source,
        tenant_id=DEFAULT_TENANT_ID,
        owner_type="user",
        owner_id="user:u1",
        reason="test_cross_agent_visibility_requires_explicit_promotion",
    )
    await memory.semantic_core.upsert_fact(promoted, embedding)

    after = await memory.memory_env.fetch_facts_by_ids(
        cross_agent_request,
        [source.id, promoted.id],
        owner_type="user",
        owner_id="user:u1",
    )

    assert {fact.id for fact in after} == {promoted.id}
    assert after[0].session_id is None


@pytest.mark.asyncio
async def test_request_handle_retrieval_remains_isolated_under_overlap(uma_memory, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = uma_memory
    assert memory.agent_id
    barrier = threading.Barrier(2)
    seen_contexts: list[tuple[str, str, str]] = []

    async def fake_structured(self, query_text: str):
        await asyncio.to_thread(barrier.wait)
        seen_contexts.append((query_text, self.context.user_id or "", self.context.session_id or ""))
        return {"working_memory": [], "episodic": [], "facts": [], "chunks": [], "skills": [], "graph": [], "trace": [], "confidence": {}}

    monkeypatch.setattr(UMARequestHandle, "retrieve_context", fake_structured)
    runtime = UMARuntime.from_memory(memory)

    results = await asyncio.gather(
        runtime.bind(
            RuntimeContext(
                tenant_id=DEFAULT_TENANT_ID,
                agent_id=memory.agent_id,
                request_id="req-overlap-a",
                user_id="user:u1",
                session_id="legacy-user:user:u1",
            )
        ).retrieve_context("query-a"),
        runtime.bind(
            RuntimeContext(
                tenant_id=DEFAULT_TENANT_ID,
                agent_id=memory.agent_id,
                request_id="req-overlap-b",
                user_id="user:u2",
                session_id="legacy-user:user:u2",
            )
        ).retrieve_context("query-b"),
    )

    assert len(results) == 2
    assert sorted(seen_contexts) == [
        ("query-a", "user:u1", "legacy-user:user:u1"),
        ("query-b", "user:u2", "legacy-user:user:u2"),
    ]


def test_filter_items_by_owner_drops_foreign_user_items_and_keeps_agent_items() -> None:
    from types import SimpleNamespace
    from uma.api.runtime import UMARuntime

    agent_item = SimpleNamespace(owner_type="agent", owner_id="agent-default")
    own_item = SimpleNamespace(owner_type="user", owner_id="user:alice")
    foreign_item = SimpleNamespace(owner_type="user", owner_id="user:bob")
    no_owner_item = SimpleNamespace()  # no owner_type attribute — kept for safety

    result = UMARuntime._filter_items_by_owner(
        [agent_item, own_item, foreign_item, no_owner_item],
        requesting_user_id="user:alice",
    )

    assert agent_item in result
    assert own_item in result
    assert foreign_item not in result
    assert no_owner_item in result
