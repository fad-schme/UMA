"""Isolation and tenancy: DAT invariants, cross-tenant/agent isolation, ownership validation.

Covers cross-tenant impossibility by construction, per-user/agent isolation
at all lanes, ownership validation (validate_explicit_owner), scope types
(RuntimeContext/SessionScope/OwnershipRef), and public API surface contracts.
"""
from __future__ import annotations
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from tests.helpers.context_bundle import make_context_bundle
from tests.helpers.runtime import TEST_AGENT_ID, build_test_config
from tests.helpers.runtime import init_uma_for_tests
from typing import get_args
from uma.api.memory import UMAMemory
from uma.api.runtime import UMARuntime
from uma.common.identity import normalize_user_id
from uma.common.ownership import validate_explicit_owner
from uma.common.types import Chunk, Episode, RuntimeContext, Skill, Fact, OwnershipRef,  SessionScope, SCOPE_MODEL_VERSION
from uma.common.types.types_owner import OwnerType
from uma.common.types.types_scope import validate_agent_id, validate_owner_id, validate_owner_type, validate_request_id, validate_session_id, validate_tenant_id, validate_user_id, validate_workspace_id
from uma.memory.promotion import PromotionPolicy
from uma.retrieve.rlm.context_pack import ContextPack
from uma.retrieve.rlm.request import RetrievalRequest, RetrievalScope
from uma.common.types.types_scope import DEFAULT_TENANT_ID
import asyncio
import pytest
import threading
import yaml

AGENT_ID = TEST_AGENT_ID

# Barrier waits run inside `asyncio.to_thread`, i.e. on non-daemon pool threads
# that asyncio cannot cancel. Without a timeout, one party failing before it
# reaches the barrier leaves the other blocked forever and the interpreter
# hangs at shutdown joining it. A bounded wait turns that into a
# BrokenBarrierError and an honest test failure.
_BARRIER_TIMEOUT_S = 10.0


# ── test_isolation_matrix ──────────────────────────────────────────






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
    memory = await init_uma_for_tests(tmp_path)
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

        runtime = memory.runtime
        memory._rlm_controller = _EmptyController()
        ctx_a_context = RuntimeContext(
            tenant_id="tenant-a",
            agent_id="agent-tenant",
            request_id="req-tenant-a",
            user_id="user:u1",
            session_id="shared-session",
        )
        ctx_b_context = RuntimeContext(
            tenant_id="tenant-b",
            agent_id="agent-tenant",
            request_id="req-tenant-b",
            user_id="user:u1",
            session_id="shared-session",
        )

        ctx_a, ctx_b = await asyncio.gather(
            runtime.retrieve_context(ctx_a_context, query_text="tenant isolation"),
            runtime.retrieve_context(ctx_b_context, query_text="tenant isolation"),
        )
        req_a = memory.runtime._build_retrieval_request(ctx_a_context)
        req_b = memory.runtime._build_retrieval_request(ctx_b_context)
        facts_a = await memory.memory_env.fetch_facts_by_ids(req_a, [fact_a.id, fact_b.id], owner_type="user", owner_id="user:u1")
        facts_b = await memory.memory_env.fetch_facts_by_ids(req_b, [fact_a.id, fact_b.id], owner_type="user", owner_id="user:u1")

        assert [msg.content for msg in ctx_a.working_memory] == ["tenant a wm"]
        assert [msg.content for msg in ctx_b.working_memory] == ["tenant b wm"]
        assert {fact.id for fact in facts_a} == {"fact_tenant_a"}
        assert {fact.id for fact in facts_b} == {"fact_tenant_b"}
    finally:
        memory.shutdown()


@pytest.mark.asyncio
async def test_multi_user_retrieval_isolates_user_owned_data_but_keeps_agent_kb_shared(uma_memory, tmp_path: Path) -> None:
    memory = uma_memory
    assert AGENT_ID
    runtime = memory.runtime

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

    await memory.ingest_document(str(agent_doc), owner_type="agent", owner_id=AGENT_ID)
    await memory.ingest_document(str(user_a_doc), owner_type="user", owner_id="user:u1")
    await memory.ingest_document(str(user_b_doc), owner_type="user", owner_id="user:u2")

    ctx_a_context = RuntimeContext(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id=AGENT_ID,
        request_id="req-user-a",
        user_id="user:u1",
    )
    ctx_b_context = RuntimeContext(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id=AGENT_ID,
        request_id="req-user-b",
        user_id="user:u2",
    )

    ctx_a, ctx_b = await asyncio.gather(
        runtime.retrieve_context(ctx_a_context, query_text="overlap token"),
        runtime.retrieve_context(ctx_b_context, query_text="overlap token"),
    )

    owner_pairs_a = {(getattr(chunk, "owner_type", None), getattr(chunk, "owner_id", None)) for chunk in ctx_a.chunks}
    owner_pairs_b = {(getattr(chunk, "owner_type", None), getattr(chunk, "owner_id", None)) for chunk in ctx_b.chunks}

    assert ("agent", AGENT_ID) in owner_pairs_a
    assert ("agent", AGENT_ID) in owner_pairs_b
    assert ("user", "user:u1") in owner_pairs_a
    assert ("user", "user:u2") not in owner_pairs_a
    assert ("user", "user:u2") in owner_pairs_b
    assert ("user", "user:u1") not in owner_pairs_b


@pytest.mark.asyncio
async def test_working_memory_isolates_users_sharing_one_session_id(tmp_path: Path) -> None:
    """session_id carries no identity, so it cannot be the whole WM key.

    The caller supplies session_id (over MCP it is a plain tool argument),
    so two users can legitimately present the same (tenant, agent, session).
    Working memory must still separate them the way every durable lane does.
    """
    memory = await init_uma_for_tests(tmp_path)
    try:
        runtime = memory.runtime
        memory.working_memory.append(
            scope=SessionScope(
                tenant_id=DEFAULT_TENANT_ID,
                agent_id=AGENT_ID,
                session_id="shared-session",
                user_id="user:alpha",
            ),
            role="user",
            content="alpha only wm",
        )
        memory.working_memory.append(
            scope=SessionScope(
                tenant_id=DEFAULT_TENANT_ID,
                agent_id=AGENT_ID,
                session_id="shared-session",
                user_id="user:beta",
            ),
            role="user",
            content="beta only wm",
        )

        ctx_alpha, ctx_beta = await asyncio.gather(
            runtime.retrieve_context(
                RuntimeContext(
                    tenant_id=DEFAULT_TENANT_ID,
                    agent_id=AGENT_ID,
                    request_id="req-wm-alpha",
                    user_id="user:alpha",
                    session_id="shared-session",
                ),
                query_text="wm",
            ),
            runtime.retrieve_context(
                RuntimeContext(
                    tenant_id=DEFAULT_TENANT_ID,
                    agent_id=AGENT_ID,
                    request_id="req-wm-beta",
                    user_id="user:beta",
                    session_id="shared-session",
                ),
                query_text="wm",
            ),
        )

        assert [msg.content for msg in ctx_alpha.working_memory] == ["alpha only wm"]
        assert [msg.content for msg in ctx_beta.working_memory] == ["beta only wm"]
    finally:
        memory.shutdown()


@pytest.mark.asyncio
async def test_working_memory_key_normalizes_the_user_subject(tmp_path: Path) -> None:
    """The turn path writes "user:<id>"; the request path reads the raw id.

    Both must resolve to the same buffer, or process_turn's working memory
    would be invisible to the retrieval that follows it.
    """
    memory = await init_uma_for_tests(tmp_path)
    try:
        memory.working_memory.append(
            scope=SessionScope(
                tenant_id=DEFAULT_TENANT_ID,
                agent_id=AGENT_ID,
                session_id="s1",
                user_id=normalize_user_id("gamma"),
            ),
            role="user",
            content="written with canonical subject",
        )
        messages = memory.working_memory.get_context(
            SessionScope(
                tenant_id=DEFAULT_TENANT_ID,
                agent_id=AGENT_ID,
                session_id="s1",
                user_id="gamma",
            )
        )
        assert [msg.content for msg in messages] == ["written with canonical subject"]
    finally:
        memory.shutdown()


def test_working_memory_buffer_rejects_a_scope_without_a_user() -> None:
    from uma.memory.working_memory.buffer import WorkingMemoryBuffer

    buffer = WorkingMemoryBuffer(max_tokens=1000)
    scope = SessionScope(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id=AGENT_ID,
        session_id="s1",
    )
    with pytest.raises(ValueError, match="requires SessionScope.user_id"):
        buffer.append(scope=scope, role="user", content="no user on this scope")


@pytest.mark.asyncio
async def test_retrieval_and_process_turn_overlap_preserve_session_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = await init_uma_for_tests(tmp_path)
    try:
        runtime = memory.runtime
        memory._rlm_controller = _EmptyController()

        await memory.process_turn(
            user_id="user:u1",
            user_msg="I like coffee in session a.",
            assistant_reply="Good to know.",
            session_id="session-a",
            extra_meta={"request_id": "req-seed-a"},
            agent_id="agent-overlap",
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
                agent_id="agent-overlap",
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=1.0)

        ctx_a_context = RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-overlap",
            request_id="req-read-a",
            user_id="user:u1",
            session_id="session-a",
        )
        during_ctx = await runtime.retrieve_context(ctx_a_context, query_text="coffee")

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

        during_wm = [msg.content for msg in during_ctx.working_memory]
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
async def test_retrieval_remains_isolated_under_concurrent_requests(uma_memory, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = uma_memory
    assert AGENT_ID
    barrier = threading.Barrier(2)
    seen_contexts: list[tuple[str, str, str]] = []

    async def fake_structured(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
        lane_filter=None,
        include_debug: bool = False,
    ):
        await asyncio.to_thread(barrier.wait, _BARRIER_TIMEOUT_S)
        seen_contexts.append((query_text, runtime_context.user_id or "", runtime_context.session_id or ""))
        return make_context_bundle(query=query_text)

    monkeypatch.setattr(UMARuntime, "retrieve_context", fake_structured)
    runtime = memory.runtime

    ctx_a = RuntimeContext(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id=AGENT_ID,
        request_id="req-overlap-a",
        user_id="user:u1",
        session_id="session:user:u1",
    )
    ctx_b = RuntimeContext(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id=AGENT_ID,
        request_id="req-overlap-b",
        user_id="user:u2",
        session_id="session:user:u2",
    )

    results = await asyncio.gather(
        runtime.retrieve_context(ctx_a, query_text="query-a"),
        runtime.retrieve_context(ctx_b, query_text="query-b"),
    )

    assert len(results) == 2
    assert sorted(seen_contexts) == [
        ("query-a", "user:u1", "session:user:u1"),
        ("query-b", "user:u2", "session:user:u2"),
    ]


def _alice_scopes() -> tuple[RetrievalScope, ...]:
    return RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-default",
            request_id="req-filter",
            user_id="user:alice",
        )
    ).scopes


def test_filter_items_by_scope_admits_only_the_requests_own_scopes() -> None:
    from types import SimpleNamespace
    from uma.api.runtime import UMARuntime

    def item(owner_type: str, owner_id: str, tenant_id: str = DEFAULT_TENANT_ID):
        return SimpleNamespace(
            tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id
        )

    own_agent_item = item("agent", "agent-default")
    own_user_item = item("user", "user:alice")
    foreign_user_item = item("user", "user:bob")
    foreign_agent_item = item("agent", "agent-other")
    workspace_item = item("workspace", "ws-1")
    foreign_tenant_item = item("user", "user:alice", tenant_id="tenant-other")

    result = UMARuntime._filter_items_by_scope(
        [
            own_agent_item,
            own_user_item,
            foreign_user_item,
            foreign_agent_item,
            workspace_item,
            foreign_tenant_item,
        ],
        _alice_scopes(),
        DEFAULT_TENANT_ID,
    )

    assert result == [own_agent_item, own_user_item]


def test_filter_items_by_scope_fails_closed_on_unreadable_owner_or_tenant() -> None:
    """Every lane this runs on returns tenant- and owner-bearing domain objects.

    An item missing either did not come from a store, so it is dropped rather
    than waved through.
    """
    from types import SimpleNamespace
    from uma.api.runtime import UMARuntime

    no_owner_item = SimpleNamespace()
    half_owner_item = SimpleNamespace(
        tenant_id=DEFAULT_TENANT_ID, owner_type="user", owner_id=None
    )
    no_tenant_item = SimpleNamespace(
        tenant_id=None, owner_type="user", owner_id="user:alice"
    )

    assert UMARuntime._filter_items_by_scope(
        [no_owner_item, half_owner_item, no_tenant_item],
        _alice_scopes(),
        DEFAULT_TENANT_ID,
    ) == []


def test_filter_items_by_scope_matches_either_user_subject_form() -> None:
    """A row written as "alice" is the same principal as "user:alice"."""
    from types import SimpleNamespace
    from uma.api.runtime import UMARuntime

    raw_form = SimpleNamespace(
        tenant_id=DEFAULT_TENANT_ID, owner_type="user", owner_id="alice"
    )

    assert UMARuntime._filter_items_by_scope(
        [raw_form], _alice_scopes(), DEFAULT_TENANT_ID
    ) == [raw_form]


def test_filter_items_by_scope_covers_every_owner_bearing_lane() -> None:
    """Regression: episodic and skills used to bypass the filter entirely."""
    import inspect
    from uma.api.runtime import UMARuntime

    source = inspect.getsource(UMARuntime._assemble_public_context_result)
    for lane in ("episodic", "facts", "chunks", "skills"):
        assert f"{lane} = self._filter_items_by_scope(" in source, lane


# ── test_tenant_scoped_durable_boundaries ──────────────────────────────────────────






@pytest.mark.asyncio
async def test_chunk_retrieval_is_tenant_scoped_for_identical_owner_tuple(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb_a = (await memory.embedder.embed(["tenant alpha chunk"]))[0]
    emb_b = (await memory.embedder.embed(["tenant beta chunk"]))[0]
    now = datetime.now(timezone.utc)

    await memory.chunk_core.upsert_chunk(
        Chunk(
            id="chunk_tenant_a",
            doc_id="doc-tenant-a",
            text="tenant alpha chunk",
            page_range=(1, 1),
            position=1,
            source_path="/tmp/a.txt",
            source_hash="hash-a",
            created_at=now,
            updated_at=now,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb_a,
    )
    await memory.chunk_core.upsert_chunk(
        Chunk(
            id="chunk_tenant_b",
            doc_id="doc-tenant-b",
            text="tenant beta chunk",
            page_range=(1, 1),
            position=1,
            source_path="/tmp/b.txt",
            source_hash="hash-b",
            created_at=now,
            updated_at=now,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb_b,
    )

    found = await memory.chunk_core.search_chunks(
        query_embedding=emb_a,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        k=5,
    )

    assert [chunk.id for chunk in found] == ["chunk_tenant_a"]
    assert memory.chunk_core.store.vector_index._scopes["chunk_tenant_a"][0] == "tenant-a"
    assert memory.chunk_core.store.vector_index._scopes["chunk_tenant_b"][0] == "tenant-b"


@pytest.mark.asyncio
async def test_procedural_retrieval_is_tenant_scoped_for_identical_owner_tuple(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb_a = (await memory.embedder.embed(["tenant alpha procedure"]))[0]
    emb_b = (await memory.embedder.embed(["tenant beta procedure"]))[0]
    now = datetime.now(timezone.utc)

    await memory.procedural_core.add_skill(
        Skill(
            id="skill_tenant_a",
            name="Tenant Alpha Procedure",
            description="alpha",
            created_at=now,
            updated_at=now,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            trigger_phrases=["alpha"],
            trigger_patterns=[],
            plan={"steps": ["a"]},
            tools=["tool-a"],
            example="alpha",
            meta={},
        ),
        emb_a,
    )
    await memory.procedural_core.add_skill(
        Skill(
            id="skill_tenant_b",
            name="Tenant Beta Procedure",
            description="beta",
            created_at=now,
            updated_at=now,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            trigger_phrases=["beta"],
            trigger_patterns=[],
            plan={"steps": ["b"]},
            tools=["tool-b"],
            example="beta",
            meta={},
        ),
        emb_b,
    )

    owner = OwnershipRef(tenant_id="tenant-a", owner_type="user", owner_id=owner_id)
    found = await memory.procedural_core.search(query_embedding=emb_a, owner=owner, k=5)

    assert [skill.id for skill in found] == ["skill_tenant_a"]
    assert memory.procedural_core.store.vector_index._scopes["skill_tenant_a"][0] == "tenant-a"
    assert memory.procedural_core.store.vector_index._scopes["skill_tenant_b"][0] == "tenant-b"


@pytest.mark.asyncio
async def test_semantic_store_list_and_fetch_are_tenant_scoped_at_durable_boundary(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb = (await memory.embedder.embed(["tenant scoped fact"]))[0]
    now = datetime.now(timezone.utc)

    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_tenant_a",
            subject=owner_id,
            predicate="LIKES",
            object="alpha",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_tenant_b",
            subject=owner_id,
            predicate="LIKES",
            object="beta",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb,
    )

    listed = await memory.semantic_core.store.list_facts_for_owner(
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        limit=None,
    )
    fetched = await memory.semantic_core.store.fetch_by_ids(
        ids=["fact_tenant_a", "fact_tenant_b"],
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
    )

    assert [fact.id for fact in listed] == ["fact_tenant_a"]
    assert [fact.id for fact in fetched] == ["fact_tenant_a"]
    assert memory.semantic_core.vector_index()._scopes["fact_tenant_a"][0] == "tenant-a"
    assert memory.semantic_core.vector_index()._scopes["fact_tenant_b"][0] == "tenant-b"


@pytest.mark.asyncio
async def test_semantic_vector_search_is_tenant_scoped_for_identical_owner_tuple(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb_a = (await memory.embedder.embed(["semantic tenant alpha"]))[0]
    emb_b = (await memory.embedder.embed(["semantic tenant beta"]))[0]
    now = datetime.now(timezone.utc)

    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_search_tenant_a",
            subject=owner_id,
            predicate="USES",
            object="alpha",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb_a,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_search_tenant_b",
            subject=owner_id,
            predicate="USES",
            object="beta",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb_b,
    )

    found = await memory.semantic_core.store.search(
        emb_a,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        k=5,
    )

    assert [fact.id for fact in found] == ["fact_search_tenant_a"]


@pytest.mark.asyncio
async def test_episodic_store_list_and_fetch_are_tenant_scoped_at_durable_boundary(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb = (await memory.embedder.embed(["tenant scoped episode"]))[0]
    now = datetime.now(timezone.utc)

    await memory.episodic_core.add_episode(
        Episode(
            id="episode_tenant_a",
            timestamp=now,
            summary="alpha",
            raw="alpha",
            tags=[],
            embedding=emb,
            meta={},
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            user_id=owner_id,
        ),
        emb,
    )
    await memory.episodic_core.add_episode(
        Episode(
            id="episode_tenant_b",
            timestamp=now,
            summary="beta",
            raw="beta",
            tags=[],
            embedding=emb,
            meta={},
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            user_id=owner_id,
        ),
        emb,
    )

    listed = await memory.episodic_core.store.list_episodes("tenant-a", "user", owner_id)
    fetched = await memory.episodic_core.store.fetch_by_ids(
        ["episode_tenant_a", "episode_tenant_b"],
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
    )

    assert [episode.id for episode in listed] == ["episode_tenant_a"]
    assert [episode.id for episode in fetched] == ["episode_tenant_a"]
    assert memory.episodic_core.vector_index()._scopes["episode_tenant_a"][0] == "tenant-a"
    assert memory.episodic_core.vector_index()._scopes["episode_tenant_b"][0] == "tenant-b"


@pytest.mark.asyncio
async def test_episodic_vector_search_is_tenant_scoped_for_identical_owner_tuple(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb_a = (await memory.embedder.embed(["episodic tenant alpha"]))[0]
    emb_b = (await memory.embedder.embed(["episodic tenant beta"]))[0]
    now = datetime.now(timezone.utc)

    await memory.episodic_core.add_episode(
        Episode(
            id="episode_search_tenant_a",
            timestamp=now,
            summary="alpha",
            raw="alpha",
            tags=[],
            embedding=emb_a,
            meta={},
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            user_id=owner_id,
        ),
        emb_a,
    )
    await memory.episodic_core.add_episode(
        Episode(
            id="episode_search_tenant_b",
            timestamp=now,
            summary="beta",
            raw="beta",
            tags=[],
            embedding=emb_b,
            meta={},
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            user_id=owner_id,
        ),
        emb_b,
    )

    found = await memory.episodic_core.store.search(
        emb_a,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        k=5,
    )

    assert [episode.id for episode in found] == ["episode_search_tenant_a"]


@pytest.mark.asyncio
async def test_search_ids_requires_tenant_scope_filters(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    embedding = (await memory.embedder.embed(["search ids tenant"]))[0]
    now = datetime.now(timezone.utc)

    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_search_ids_tenant",
            subject=owner_id,
            predicate="USES",
            object="tenant-search-ids",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        embedding,
    )

    with pytest.raises(ValueError, match="tenant_id"):
        await memory.semantic_core.store.search_ids(
            embedding,
            tenant_id="",
            owner_type="user",
            owner_id=owner_id,
            log_context="missing_tenant_search_ids",
        )

    found = await memory.semantic_core.store.search_ids(
        embedding,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        log_context="tenant_search_ids",
    )

    assert [fact_id for fact_id, _score in found] == ["fact_search_ids_tenant"]


@pytest.mark.asyncio
async def test_low_level_store_reads_fail_clearly_without_explicit_tenant(uma_memory) -> None:
    memory = uma_memory

    with pytest.raises(ValueError, match="tenant_id"):
        await memory.chunk_core.store.fetch_by_ids(
            ["missing"],
            owner_type="user",
            owner_id=normalize_user_id("user:u1"),
        )

    with pytest.raises(ValueError, match="tenant_id"):
        await memory.semantic_core.store.list_facts_for_owner(
            owner_type="user",
            owner_id=normalize_user_id("user:u1"),
            limit=None,
        )

    with pytest.raises(ValueError, match="tenant_id"):
        await memory.episodic_core.store.list_episodes(
            owner_type="user",
            owner_id=normalize_user_id("user:u1"),
        )

    with pytest.raises(ValueError, match="tenant_id"):
        await memory.procedural_core.store.list_skills(
            owner_type="user",
            owner_id=normalize_user_id("user:u1"),
        )


# ── test_scope_and_ownership ──────────────────────────────────────────






# ---------------------------------------------------------------------------
# Scope model version
# ---------------------------------------------------------------------------

def test_scope_model_version_is_exported() -> None:
    assert SCOPE_MODEL_VERSION == "v2"


# ---------------------------------------------------------------------------
# RuntimeContext construction and validation
# ---------------------------------------------------------------------------

def test_runtime_context_construction_succeeds() -> None:
    ctx = RuntimeContext(
        tenant_id="tenant-1",
        agent_id="agent-1",
        request_id="req-1",
        user_id="user-1",
        workspace_id="workspace-1",
        session_id="session-1",
    )
    assert ctx.tenant_id == "tenant-1"
    assert ctx.agent_id == "agent-1"
    assert ctx.request_id == "req-1"
    assert ctx.user_id == "user-1"
    assert ctx.workspace_id == "workspace-1"
    assert ctx.session_id == "session-1"


@pytest.mark.parametrize("kwargs,message", [
    ({"tenant_id": "", "agent_id": "agent-1", "request_id": "req-1"}, "tenant_id"),
    ({"tenant_id": "tenant-1", "agent_id": "", "request_id": "req-1"}, "agent_id"),
    ({"tenant_id": "tenant-1", "agent_id": "agent-1", "request_id": ""}, "request_id"),
    ({"tenant_id": "tenant-1", "agent_id": "agent-1", "request_id": "req-1", "user_id": ""}, "user_id"),
    ({"tenant_id": "tenant-1", "agent_id": "agent-1", "request_id": "req-1", "workspace_id": ""}, "workspace_id"),
    ({"tenant_id": "tenant-1", "agent_id": "agent-1", "request_id": "req-1", "session_id": ""}, "session_id"),
])
def test_runtime_context_rejects_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RuntimeContext(**kwargs)


def test_runtime_scope_types_are_immutable() -> None:
    ctx = RuntimeContext(tenant_id="tenant-1", agent_id="agent-1", request_id="req-1")
    with pytest.raises(FrozenInstanceError):
        ctx.agent_id = "agent-2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SessionScope
# ---------------------------------------------------------------------------

def test_session_scope_construction_succeeds() -> None:
    scope = SessionScope(
        tenant_id="tenant-1",
        agent_id="agent-1",
        session_id="session-1",
        user_id="user-1",
        workspace_id="workspace-1",
    )
    assert scope.tenant_id == "tenant-1"
    assert scope.agent_id == "agent-1"
    assert scope.session_id == "session-1"


@pytest.mark.parametrize("kwargs,message", [
    ({"tenant_id": "", "agent_id": "agent-1", "session_id": "session-1"}, "tenant_id"),
    ({"tenant_id": "tenant-1", "agent_id": "", "session_id": "session-1"}, "agent_id"),
    ({"tenant_id": "tenant-1", "agent_id": "agent-1", "session_id": ""}, "session_id"),
    ({"tenant_id": "tenant-1", "agent_id": "agent-1", "session_id": "session-1", "user_id": ""}, "user_id"),
    ({"tenant_id": "tenant-1", "agent_id": "agent-1", "session_id": "session-1", "workspace_id": ""}, "workspace_id"),
])
def test_session_scope_rejects_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SessionScope(**kwargs)


# ---------------------------------------------------------------------------
# OwnershipRef
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("owner_type", ["agent", "user", "workspace", "system"])
def test_ownership_ref_accepts_supported_owner_types(owner_type: str) -> None:
    ref = OwnershipRef(tenant_id="tenant-1", owner_type=owner_type, owner_id="owner-1")
    assert ref.owner_type == owner_type


@pytest.mark.parametrize("kwargs,message", [
    ({"tenant_id": "", "owner_type": "user", "owner_id": "owner-1"}, "tenant_id"),
    ({"tenant_id": "tenant-1", "owner_type": "invalid", "owner_id": "owner-1"}, "owner_type"),
    ({"tenant_id": "tenant-1", "owner_type": "user", "owner_id": ""}, "owner_id"),
])
def test_ownership_types_reject_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OwnershipRef(**kwargs)


def test_persistent_ownership_types_are_immutable() -> None:
    owner = OwnershipRef(tenant_id="tenant-1", owner_type="user", owner_id="owner-1")
    with pytest.raises(FrozenInstanceError):
        owner.owner_id = "owner-2"  # type: ignore[misc]


def test_owner_type_literal_matches_supported_vocabulary() -> None:
    assert set(get_args(OwnerType)) == {"agent", "user", "workspace", "system"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def test_validation_helpers_accept_none_for_optional_ids() -> None:
    assert validate_user_id(None) is None
    assert validate_workspace_id(None) is None
    assert validate_session_id(None) is None


@pytest.mark.parametrize("validator,value,expected", [
    (validate_tenant_id, "tenant-1", "tenant-1"),
    (validate_agent_id, "agent-1", "agent-1"),
    (validate_request_id, "req-1", "req-1"),
    (validate_user_id, "user-1", "user-1"),
    (validate_workspace_id, "workspace-1", "workspace-1"),
    (validate_session_id, "session-1", "session-1"),
    (validate_owner_type, "workspace", "workspace"),
    (validate_owner_id, "owner-1", "owner-1"),
])
def test_validation_helpers_accept_valid_strings(validator, value: str, expected: str) -> None:
    assert validator(value) == expected


@pytest.mark.parametrize("validator,value,message", [
    (validate_tenant_id, "", "tenant_id"),
    (validate_agent_id, "", "agent_id"),
    (validate_request_id, "", "request_id"),
    (validate_user_id, "", "user_id"),
    (validate_workspace_id, "", "workspace_id"),
    (validate_session_id, "", "session_id"),
    (validate_owner_type, "project", "owner_type"),
    (validate_owner_id, "", "owner_id"),
])
def test_validation_helpers_reject_invalid_strings(validator, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validator(value)


def test_new_types_are_exported_from_uma_types() -> None:
    assert RuntimeContext.__module__ == "uma.common.types.types_scope"
    assert SessionScope.__module__ == "uma.common.types.types_scope"
    assert OwnershipRef.__module__ == "uma.common.types.types_scope"


# ---------------------------------------------------------------------------
# No duplicate or forbidden ownership resolvers remain in the codebase
# ---------------------------------------------------------------------------

def test_no_duplicate_ownership_resolvers() -> None:
    forbidden = [
        "TargetOwner", "target_owner", "make_target_owner", "resolve_target_owner",
        "select_target_owner", "_resolve_owner", "_resolve_ownership_ref",
        "_read_owner_ref", "_write_owner_from_skill", "_select_owner",
        "_select_ownership", "resolve_ownership_ref", "resolve_explicit_owner",
    ]
    root = Path(__file__).resolve().parents[1]
    for path in (root / "uma").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, f"{marker} remains in {path}"

    ownership_text = (root / "uma/common/ownership.py").read_text(encoding="utf-8")
    assert "def validate_explicit_owner(" in ownership_text


# ── test_write_owner_contracts ──────────────────────────────────────────

def test_explicit_write_owner_accepts_agent_user_and_workspace() -> None:
    agent_owner = validate_explicit_owner(
        owner_type="agent",
        owner_id="agent:alpha",
    )
    assert agent_owner == {
        "tenant_id": "default",
        "owner_type": "agent",
        "owner_id": "agent:alpha",
        "workspace_id": None,
    }

    user_owner = validate_explicit_owner(
        owner_type="user",
        owner_id="u1",
    )
    assert user_owner == {
        "tenant_id": "default",
        "owner_type": "user",
        "owner_id": "user:u1",
        "workspace_id": None,
    }

    workspace_owner = validate_explicit_owner(
        owner_type="workspace",
        owner_id="workspace:alpha",
    )
    assert workspace_owner == {
        "tenant_id": "default",
        "owner_type": "workspace",
        "owner_id": "workspace:alpha",
        "workspace_id": "workspace:alpha",
    }


def test_explicit_write_owner_accepts_system_scope_when_requested() -> None:
    owner = validate_explicit_owner(owner_type="system", owner_id="system:alpha")
    assert owner == {
        "tenant_id": "default",
        "owner_type": "system",
        "owner_id": "system:alpha",
        "workspace_id": None,
    }


def test_explicit_write_owner_preserves_tenant_and_workspace() -> None:
    owner = validate_explicit_owner(
        tenant_id="tenant-1",
        owner_type="workspace",
        owner_id="workspace:alpha",
        workspace_id="workspace:alpha",
    )
    assert owner == {
        "tenant_id": "tenant-1",
        "owner_type": "workspace",
        "owner_id": "workspace:alpha",
        "workspace_id": "workspace:alpha",
    }


@pytest.mark.asyncio
async def test_document_ingest_rejects_missing_owner_type(uma_memory, tmp_path) -> None:
    path = tmp_path / "missing-owner-type.txt"
    path.write_text("Explicit owner validation should reject missing owner_type.\n")
    with pytest.raises(ValueError, match="owner_type and owner_id are required"):
        await uma_memory.ingest_document(str(path), owner_id="user:u1")


@pytest.mark.asyncio
async def test_document_ingest_rejects_missing_owner_id(uma_memory, tmp_path) -> None:
    path = tmp_path / "missing-owner-id.txt"
    path.write_text("Explicit owner validation should reject missing owner_id.\n")
    with pytest.raises(ValueError, match="owner_type and owner_id are required"):
        await uma_memory.ingest_document(str(path), owner_type="user")


@pytest.mark.asyncio
async def test_promotion_rejects_missing_owner_type(uma_memory) -> None:
    policy = PromotionPolicy(agent_id=AGENT_ID)
    now = datetime.now(timezone.utc)
    fact = Fact(
        id="fact_missing_owner_type",
        subject="team",
        predicate="USES",
        object="kubernetes cluster orchestration for production workloads",
        created_at=now,
        updated_at=now,
        source_ids=["chunk-source-1"],
        confidence=0.95,
        salience=0.92,
        meta={"source_type": "text"},
        owner_type="user",
        owner_id="user:u1",
        tenant_id="default",
        scope_model_version=SCOPE_MODEL_VERSION,
    )
    with pytest.raises(ValueError, match="owner_type and owner_id are required"):
        policy.promote(fact, owner_id="user:u1")


@pytest.mark.asyncio
async def test_promotion_rejects_missing_owner_id(uma_memory) -> None:
    policy = PromotionPolicy(agent_id=AGENT_ID)
    now = datetime.now(timezone.utc)
    fact = Fact(
        id="fact_missing_owner_id",
        subject="team",
        predicate="USES",
        object="kubernetes cluster orchestration for production workloads",
        created_at=now,
        updated_at=now,
        source_ids=["chunk-source-1"],
        confidence=0.95,
        salience=0.92,
        meta={"source_type": "text"},
        owner_type="user",
        owner_id="user:u1",
        tenant_id="default",
        scope_model_version=SCOPE_MODEL_VERSION,
    )
    with pytest.raises(ValueError, match="owner_type and owner_id are required"):
        policy.promote(fact, owner_type="user")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner_type", "owner_id", "expected_owner_type", "expected_owner_id", "expected_workspace_id"),
    [
        ("user", "user:u1", "user", "user:u1", None),
        ("agent", "agent:alpha", "agent", "agent:alpha", None),
        ("workspace", "workspace:alpha", "workspace", "workspace:alpha", "workspace:alpha"),
    ],
)
@pytest.mark.asyncio
async def test_ingest_document_persists_explicit_owner_fields(
    uma_memory,
    tmp_path,
    owner_type: str,
    owner_id: str,
    expected_owner_type: str,
    expected_owner_id: str,
    expected_workspace_id: str | None,
) -> None:
    memory = uma_memory
    path = tmp_path / f"{expected_owner_type}-target-doc.txt"
    path.write_text(
        "UMA explicit ingest owner field test. "
        "This paragraph is deliberately long enough to survive chunking and fact extraction without "
        "falling below the extractor threshold. It includes additional concrete statements about a "
        "workspace playbook, operational checklist, ownership marker, and validation trail so the "
        "document-derived fact path is exercised reliably during ingestion.\n"
    )

    report = await memory.ingest_document(str(path), owner_type=owner_type, owner_id=owner_id)
    assert report.doc_id

    conn = memory.document_store._conn()
    try:
        rows = memory.document_store._query_all(
            conn,
            """
            SELECT owner_type, owner_id, tenant_id, workspace_id
            FROM documents
            WHERE doc_id=?
            """,
            params=[report.doc_id],
            log_context="test_write_owner_ingest_document",
        )
        assert rows
        assert rows[0]["tenant_id"] == "default"
        assert rows[0]["owner_type"] == expected_owner_type
        assert rows[0]["owner_id"] == expected_owner_id
        assert rows[0]["workspace_id"] == expected_workspace_id
    finally:
        conn.close()

    # Document-derived facts carry the scope the extractor was handed. The
    # ingest stage no longer re-stamps tenant or owner onto what comes back,
    # so this asserts the whole tuple for every owner type rather than just
    # the workspace case.
    conn = memory.semantic_core.store._conn()
    try:
        fact_rows = memory.semantic_core.store._query_all(
            conn,
            """
            SELECT tenant_id, owner_type, owner_id, workspace_id
            FROM facts
            WHERE owner_type=? AND owner_id=? AND meta LIKE ?
            """,
            params=[expected_owner_type, expected_owner_id, f"%{report.doc_id}%"],
            log_context="test_write_owner_ingest_facts",
        )
        assert fact_rows, f"no document-derived facts for {expected_owner_type}"
        assert all(row["tenant_id"] == "default" for row in fact_rows)
        assert all(row["owner_type"] == expected_owner_type for row in fact_rows)
        assert all(row["owner_id"] == expected_owner_id for row in fact_rows)
        assert all(row["workspace_id"] == expected_workspace_id for row in fact_rows)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_document_ingest_requires_explicit_owner_fields(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "owner-contract-doc.txt"
    path.write_text(
        "UMA document ingestion contract test. "
        "This passage is intentionally long enough to produce a valid chunk and fact extraction path.\n"
    )

    report = await memory.ingest_document(str(path), owner_type="user", owner_id="u1")
    assert report.doc_id

    # Query directly to avoid changing the public read surface in this PR.
    conn = memory.document_store._conn()
    try:
        rows = memory.document_store._query_all(
            conn,
            "SELECT owner_type, owner_id, tenant_id FROM documents WHERE owner_type=? AND owner_id=? ORDER BY ingested_at DESC LIMIT 1",
            params=["user", "user:u1"],
            log_context="test_write_owner_ingest_manifest",
        )
        assert rows
        assert rows[0]["owner_type"] == "user"
        assert rows[0]["owner_id"] == "user:u1"
        assert rows[0]["tenant_id"] == "default"
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_document_ingest_rejects_system_owner_scope(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "system-target-doc.txt"
    path.write_text(
        "UMA unsupported system ingest target test. "
        "This paragraph is intentionally long enough to exercise the validation path.\n"
    )

    with pytest.raises(ValueError, match="owner_type"):
        await memory.ingest_document(
            str(path),
            owner_type="system",
            owner_id="system:ops",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner_type", "owner_id", "expected_owner_type", "expected_owner_id", "expected_workspace_id"),
    [
        ("user", "user:u1", "user", "user:u1", None),
        ("agent", "agent:alpha", "agent", "agent:alpha", None),
        ("workspace", "workspace:alpha", "workspace", "workspace:alpha", "workspace:alpha"),
    ],
)
@pytest.mark.asyncio
async def test_procedural_add_skill_for_owner_persists_explicit_owner_fields(
    uma_memory,
    owner_type: str,
    owner_id: str,
    expected_owner_type: str,
    expected_owner_id: str,
    expected_workspace_id: str | None,
) -> None:
    memory = uma_memory
    now = datetime.now(timezone.utc)
    skill = Skill(
        id=f"skill_{expected_owner_type}",
        name="Explicit owner skill",
        description="Verifies explicit write-target persistence.",
        created_at=now,
        updated_at=now,
        trigger_phrases=["owner"],
        trigger_patterns=[],
        plan={"steps": ["verify"]},
        tools=["shell"],
        example="check owner",
        meta={},
    )
    embedding = (await memory.embedder.embed([skill.description]))[0]

    persisted = await memory.procedural_core.add_skill_for_owner(
        skill,
        embedding,
        tenant_id="tenant-1",
        owner_type=owner_type,
        owner_id=owner_id,
    )
    assert persisted is not None

    conn = memory.procedural_core.store._conn()
    try:
        rows = memory.procedural_core.store._query_all(
            conn,
            """
            SELECT tenant_id, owner_type, owner_id, workspace_id
            FROM skills WHERE id=?
            """,
            params=[skill.id],
            log_context="test_write_owner_skill_write",
        )
        assert rows
        assert rows[0]["tenant_id"] == "tenant-1"
        assert rows[0]["owner_type"] == expected_owner_type
        assert rows[0]["owner_id"] == expected_owner_id
        assert rows[0]["workspace_id"] == expected_workspace_id
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_procedural_add_skill_for_owner_rejects_system_scope(uma_memory) -> None:
    memory = uma_memory
    now = datetime.now(timezone.utc)
    skill = Skill(
        id="skill_system_reject",
        name="Unsupported target",
        description="Should not persist under system owner in PR4.",
        created_at=now,
        updated_at=now,
        trigger_phrases=["reject"],
        trigger_patterns=[],
        plan={"steps": ["reject"]},
        tools=["shell"],
        example="reject",
        meta={},
    )
    embedding = (await memory.embedder.embed([skill.description]))[0]

    persisted = await memory.procedural_core.add_skill_for_owner(
        skill,
        embedding,
        tenant_id="tenant-1",
        owner_type="system",
        owner_id="system:ops",
    )
    assert persisted is None

    conn = memory.procedural_core.store._conn()
    try:
        rows = memory.procedural_core.store._query_all(
            conn,
            "SELECT id FROM skills WHERE id=?",
            params=[skill.id],
            log_context="test_write_owner_system_reject",
        )
        assert rows == []
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("write_owner_type", "write_owner_id", "lookup_owner"),
    [
        (
            "user",
            "user:u1",
            OwnershipRef(tenant_id="tenant-1", owner_type="user", owner_id="user:u1"),
        ),
        (
            "agent",
            "agent:alpha",
            OwnershipRef(tenant_id="tenant-1", owner_type="agent", owner_id="agent:alpha"),
        ),
        (
            "workspace",
            "workspace:alpha",
            OwnershipRef(tenant_id="tenant-1", owner_type="workspace", owner_id="workspace:alpha"),
        ),
    ],
)
@pytest.mark.asyncio
async def test_procedural_reads_require_explicit_owner_scope(
    uma_memory,
    write_owner_type: str,
    write_owner_id: str,
    lookup_owner: OwnershipRef,
) -> None:
    memory = uma_memory
    now = datetime.now(timezone.utc)
    suffix = write_owner_type
    skill = Skill(
        id=f"skill_read_{suffix}",
        name="Scoped read skill",
        description="Verifies explicit procedural read scoping.",
        created_at=now,
        updated_at=now,
        trigger_phrases=["scoped read"],
        trigger_patterns=[],
        plan={"steps": ["read"]},
        tools=["shell"],
        example="read scope",
        meta={},
    )
    embedding = (await memory.embedder.embed([skill.description]))[0]
    persisted = await memory.procedural_core.add_skill_for_owner(
        skill,
        embedding,
        tenant_id="tenant-1",
        owner_type=write_owner_type,
        owner_id=write_owner_id,
    )
    assert persisted is not None
    assert not hasattr(memory, "user_id")

    query_embedding = (await memory.embedder.embed(["scoped read"]))[0]
    found = await memory.procedural_core.search(
        query_embedding=query_embedding,
        owner=lookup_owner,
        k=5,
    )
    assert [item.id for item in found] == [skill.id]

    loaded = await memory.procedural_core.get_skill(skill.id, owner=lookup_owner)
    assert loaded is not None
    assert loaded.id == skill.id

    listed = await memory.procedural_core.list_skills(owner=lookup_owner, limit=5)
    assert [item.id for item in listed] == [skill.id]


@pytest.mark.asyncio
async def test_procedural_reads_reject_system_scope(uma_memory) -> None:
    query_embedding = [0.0] * int(uma_memory.embedding_cfg.dimension)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.search(
            query_embedding=query_embedding,
            owner=OwnershipRef(tenant_id="default", owner_type="system", owner_id="system:ops"),
            k=5,
        )


@pytest.mark.asyncio
async def test_procedural_core_read_apis_fail_clearly_for_unsupported_scope(uma_memory) -> None:
    owner = OwnershipRef(tenant_id="default", owner_type="system", owner_id="system:ops")
    query_embedding = [0.0] * int(uma_memory.embedding_cfg.dimension)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.get_skill("skill-missing", owner=owner)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.list_skills(owner=owner, limit=5)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.fetch_by_ids(["skill-missing"], owner=owner)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.delete_skill("skill-missing", owner=owner)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.search(
            query_embedding=query_embedding,
            owner=owner,
            k=5,
        )


# ── test_public_scope_surfaces ──────────────────────────────────────────







async def _init_memory_with_procedural_feature(tmp_path) -> UMAMemory:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)

    cfg = build_test_config(db_root=db_root)
    cfg["features"] = {
        "load": [
            {
                "name": "procedural",
                "enabled": True,
                "provider": "uma.memory.procedural.feature:ProceduralFeature",
                "config": {"max_k": 5},
            }
        ],
        "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False},
    }

    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    memory = UMAMemory.from_yaml(str(cfg_path))
    memory._ensure_ingestion_ready()
    return memory


@pytest.mark.asyncio
async def test_public_procedural_reads_require_explicit_user_id(tmp_path) -> None:
    memory = await _init_memory_with_procedural_feature(tmp_path)
    try:
        now = datetime.now(timezone.utc)
        skill = Skill(
            id="skill_user_owned",
            name="Book a flight",
            description="Book a flight safely using the user-owned travel flow.",
            created_at=now,
            updated_at=now,
            owner_type="user",
            owner_id="user:u1",
            trigger_phrases=["book a flight"],
            trigger_patterns=[],
            plan={"steps": ["book"]},
            tools=["shell"],
            example="book a flight",
            meta={},
        )
        embedding = (await memory.embedder.embed([skill.description]))[0]
        add_result = await memory.procedural_add_skill(skill, embedding)
        assert add_result.ok

        find_result = await memory.procedural_find_skills(
            "book a flight",
            user_id="user:u1",
            k=5,
        )
        assert find_result.ok
        assert find_result.data
        assert find_result.data[0].id == "skill_user_owned"

        get_result = await memory.procedural_get_skill(
            "skill_user_owned",
            user_id="user:u1",
        )
        assert get_result.ok
        assert get_result.data is not None
        assert get_result.data.id == "skill_user_owned"
    finally:
        memory.shutdown()


@pytest.mark.asyncio
async def test_public_procedural_reads_no_longer_depend_on_ambient_memory_user_id(tmp_path) -> None:
    memory = await _init_memory_with_procedural_feature(tmp_path)
    try:
        result = await memory.procedural_find_skills("book a flight", user_id="", k=5)
        assert not result.ok
        assert "missing user_id" in result.errors
        assert not hasattr(memory, "user_id")
    finally:
        memory.shutdown()


@pytest.mark.asyncio
async def test_public_procedural_reads_accept_explicit_workspace_scope_without_broadening_retrieval(tmp_path) -> None:
    memory = await _init_memory_with_procedural_feature(tmp_path)
    try:
        now = datetime.now(timezone.utc)
        skill = Skill(
            id="skill_workspace_owned",
            name="Workspace rollout",
            description="Run the shared workspace rollout procedure safely.",
            created_at=now,
            updated_at=now,
            trigger_phrases=["workspace rollout"],
            trigger_patterns=[],
            plan={"steps": ["rollout"]},
            tools=["shell"],
            example="workspace rollout",
            meta={},
        )
        embedding = (await memory.embedder.embed([skill.description]))[0]
        add_result = await memory.procedural_add_skill(
            skill,
            embedding,
            owner_type="workspace",
            owner_id="workspace:alpha",
        )
        assert add_result.ok

        find_result = await memory.procedural_find_skills(
            "workspace rollout",
            owner_type="workspace",
            owner_id="workspace:alpha",
            k=5,
        )
        assert find_result.ok
        assert [item.id for item in find_result.data] == ["skill_workspace_owned"]

        get_result = await memory.procedural_get_skill(
            "skill_workspace_owned",
            owner_type="workspace",
            owner_id="workspace:alpha",
        )
        assert get_result.ok
        assert get_result.data is not None
        assert get_result.data.owner_type == "workspace"

        request = RetrievalRequest.from_runtime_context(
            RuntimeContext(
                tenant_id="default",
                agent_id="agent-default",
                request_id="req-procedural-workspace",
                user_id="user:u1",
                session_id="session:user:u1",
            )
        )
        assert [scope.owner_type for scope in request.scopes] == ["agent", "user"]
    finally:
        memory.shutdown()


def test_agent_identity_is_never_held_on_the_instance(uma_memory) -> None:
    """UMA is multi-agent on one shared runtime: the instance carries no
    agent identity and offers no way to bind one."""
    assert not hasattr(uma_memory, "agent_id")
    assert not hasattr(uma_memory, "set_context")
    assert not hasattr(uma_memory, "promotion_policy")


@pytest.mark.asyncio
async def test_one_instance_serves_distinct_agents_per_call(uma_memory) -> None:
    """Two agents retrieve through the same instance; each call's scope is
    built from that call's agent_id."""
    seen: list[str] = []

    def _hook(operation: str, ctx) -> None:
        if ctx is not None:
            seen.append(ctx.agent_id)

    uma_memory.set_rate_limit_hook(_hook)
    try:
        await uma_memory.retrieve_context(
            query_text="hello",
            agent_id="agent-a",
            user_id="user:u1",
        )
        await uma_memory.retrieve_context(
            query_text="hello",
            agent_id="agent-b",
            user_id="user:u1",
        )
    finally:
        uma_memory.set_rate_limit_hook(None)

    assert seen == ["agent-a", "agent-b"]


@pytest.mark.asyncio
async def test_retrieve_context_raises_when_agent_id_is_missing(tmp_path) -> None:
    """_resolve_runtime_context must raise rather than fall back to any
    instance-level or default agent identity."""
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    memory = UMAMemory.from_yaml(str(cfg_path))

    with pytest.raises(ValueError, match="agent_id"):
        await memory.retrieve_context(
            query_text="hello",
            user_id="user:u1",
        )
    memory.shutdown()


def test_session_local_filter_fails_closed_on_missing_isolation_fields() -> None:
    """Session-local items must prove their tenant and originating agent.

    Every fact written through the turn path stamps both, so a missing value
    means the row predates the column or did not come from a store. In a
    runtime serving many agents, neither is admissible on trust.
    """
    from types import SimpleNamespace

    from uma.retrieve.rlm.environment import UMAMemoryEnvironment

    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-default",
            request_id="req-session-filter",
            user_id="user:alice",
        )
    )
    object.__setattr__(request.context, "session_id", "session-1")

    def item(**kwargs):
        base = {
            "tenant_id": DEFAULT_TENANT_ID,
            "session_id": "session-1",
            "origin_agent_id": "agent-default",
        }
        base.update(kwargs)
        return SimpleNamespace(**base)

    own = item()
    no_tenant = item(tenant_id=None)
    foreign_tenant = item(tenant_id="tenant-other")
    no_origin_agent = item(origin_agent_id=None)
    foreign_agent = item(origin_agent_id="agent-other")
    foreign_session = item(session_id="session-2")
    # Not session-local: scoped by the owner filter instead, so it passes here.
    not_session_local = item(session_id=None, origin_agent_id=None)

    kept = UMAMemoryEnvironment._filter_session_local_items(
        request,
        [
            own,
            no_tenant,
            foreign_tenant,
            no_origin_agent,
            foreign_agent,
            foreign_session,
            not_session_local,
        ],
    )

    assert kept == [own, not_session_local]
