"""Concurrency and operational correctness: singleflight init, concurrent retrieval isolation, index rebuild.

Covers singleflight initialization guards, thread-safe working memory buffer,
concurrent retrieve_context request scope isolation, process_turn/retrieval
overlap, vector and graph index rebuild correctness.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from tests.helpers.context_bundle import make_context_bundle
from tests.helpers.runtime import TEST_AGENT_ID, build_test_config
from uma.api.memory import UMAMemory
from uma.common import maintenance as maintenance_module
from uma.common.config import UMAConfig
from uma.common.identity import normalize_user_id
from uma.common.initializers import providers as provider_init
from uma.common.initializers import runtime as runtime_init
from uma.common.types import Episode
from uma.common.types import Fact
from uma.common.types import SessionScope
from uma.common.types import Skill
from uma.ingest import ingest_service
from uma.memory.working_memory.buffer import WorkingMemoryBuffer
from uma.common.text import build_fact_embedding_text
from uma.common.types.types_scope import DEFAULT_TENANT_ID
import asyncio
import pytest
import threading
import time
import uma.api.memory as memory_module
import yaml

AGENT_ID = TEST_AGENT_ID

# Barrier waits run inside `asyncio.to_thread`, i.e. on non-daemon pool threads
# that asyncio cannot cancel. Without a timeout, one party failing before it
# reaches the barrier leaves the other blocked forever and the interpreter
# hangs at shutdown joining it. A bounded wait turns that into a
# BrokenBarrierError and an honest test failure.
_BARRIER_TIMEOUT_S = 10.0


# ── test_runtime_concurrency ──────────────────────────────────────────






def _build_uninitialized_memory(tmp_path: Path) -> UMAMemory:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    loaded = UMAConfig.load_yaml(str(cfg_path))
    return UMAMemory(loaded, config_path=str(cfg_path))


def test_overlapping_retrieval_first_use_is_singleflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _build_uninitialized_memory(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    calls = {"stores": 0}
    completions: list[str] = []

    def slow_ensure_stores(mem: UMAMemory) -> None:
        calls["stores"] += 1
        entered.set()
        assert release.wait(timeout=2.0)
        mem._stores = {"ready": True}

    monkeypatch.setattr(runtime_init, "ensure_stores", slow_ensure_stores)
    monkeypatch.setattr(runtime_init, "ensure_llm", lambda mem: setattr(mem, "llm", object()))
    monkeypatch.setattr(runtime_init, "ensure_embedder", lambda mem: setattr(mem, "embedder", object()))
    monkeypatch.setattr(
        runtime_init,
        "ensure_cores",
        lambda mem: (
            setattr(mem, "working_memory", object()),
            setattr(mem, "episodic_core", object()),
            setattr(mem, "semantic_core", object()),
            setattr(mem, "procedural_core", object()),
            setattr(mem, "chunk_core", object()),
        ),
    )
    monkeypatch.setattr(runtime_init, "ensure_graph", lambda mem: None)
    monkeypatch.setattr(runtime_init, "ensure_rlm", lambda mem: setattr(mem, "_rlm_controller", object()))

    def call(name: str) -> None:
        memory._ensure_retrieval_ready()
        completions.append(name)

    first = threading.Thread(target=call, args=("first",))
    second = threading.Thread(target=call, args=("second",))

    first.start()
    assert entered.wait(timeout=1.0)
    second.start()
    time.sleep(0.1)

    assert calls["stores"] == 1
    assert completions == []
    assert memory._retrieval_ready is False

    release.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert calls["stores"] == 1
    assert sorted(completions) == ["first", "second"]
    assert memory._retrieval_ready is True


def test_overlapping_ingestion_first_use_is_singleflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _build_uninitialized_memory(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    calls = {"features": 0}
    completions: list[str] = []

    monkeypatch.setattr(runtime_init, "ensure_stores", lambda mem: setattr(mem, "_stores", {"ready": True}))
    monkeypatch.setattr(runtime_init, "ensure_llm", lambda mem: setattr(mem, "llm", object()))
    monkeypatch.setattr(runtime_init, "ensure_embedder", lambda mem: setattr(mem, "embedder", object()))
    monkeypatch.setattr(
        runtime_init,
        "ensure_cores",
        lambda mem: (
            setattr(mem, "working_memory", object()),
            setattr(mem, "episodic_core", object()),
            setattr(mem, "semantic_core", object()),
            setattr(mem, "procedural_core", object()),
            setattr(mem, "chunk_core", object()),
        ),
    )

    def slow_ensure_features(mem: UMAMemory) -> None:
        calls["features"] += 1
        entered.set()
        assert release.wait(timeout=2.0)
        mem._features_initialized = True

    monkeypatch.setattr(runtime_init, "ensure_features", slow_ensure_features)
    monkeypatch.setattr(runtime_init, "ensure_pipeline", lambda mem: setattr(mem, "pipeline", object()))

    def call(name: str) -> None:
        memory._ensure_ingestion_ready()
        completions.append(name)

    first = threading.Thread(target=call, args=("first",))
    second = threading.Thread(target=call, args=("second",))

    first.start()
    assert entered.wait(timeout=1.0)
    second.start()
    time.sleep(0.1)

    assert calls["features"] == 1
    assert completions == []
    assert memory._ingestion_ready is False

    release.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert calls["features"] == 1
    assert sorted(completions) == ["first", "second"]
    assert memory._ingestion_ready is True


def test_cold_retrieval_and_ingestion_share_base_singleflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _build_uninitialized_memory(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    calls = {"stores": 0}
    completions: list[str] = []

    def slow_ensure_stores(mem: UMAMemory) -> None:
        calls["stores"] += 1
        entered.set()
        assert release.wait(timeout=2.0)
        mem._stores = {"ready": True}

    monkeypatch.setattr(runtime_init, "ensure_stores", slow_ensure_stores)
    monkeypatch.setattr(runtime_init, "ensure_llm", lambda mem: setattr(mem, "llm", object()))
    monkeypatch.setattr(runtime_init, "ensure_embedder", lambda mem: setattr(mem, "embedder", object()))
    monkeypatch.setattr(
        runtime_init,
        "ensure_cores",
        lambda mem: (
            setattr(mem, "working_memory", object()),
            setattr(mem, "episodic_core", object()),
            setattr(mem, "semantic_core", object()),
            setattr(mem, "procedural_core", object()),
            setattr(mem, "chunk_core", object()),
        ),
    )
    monkeypatch.setattr(runtime_init, "ensure_graph", lambda mem: None)
    monkeypatch.setattr(runtime_init, "ensure_rlm", lambda mem: setattr(mem, "_rlm_controller", object()))
    monkeypatch.setattr(runtime_init, "ensure_features", lambda mem: setattr(mem, "_features_initialized", True))
    monkeypatch.setattr(runtime_init, "ensure_pipeline", lambda mem: setattr(mem, "pipeline", object()))

    def call_retrieval() -> None:
        memory._ensure_retrieval_ready()
        completions.append("retrieval")

    def call_ingestion() -> None:
        memory._ensure_ingestion_ready()
        completions.append("ingestion")

    retrieval_thread = threading.Thread(target=call_retrieval)
    ingestion_thread = threading.Thread(target=call_ingestion)

    retrieval_thread.start()
    assert entered.wait(timeout=1.0)
    ingestion_thread.start()
    time.sleep(0.1)

    assert calls["stores"] == 1
    assert completions == []
    assert memory._base_ready is False

    release.set()
    retrieval_thread.join(timeout=2.0)
    ingestion_thread.join(timeout=2.0)

    assert calls["stores"] == 1
    assert sorted(completions) == ["ingestion", "retrieval"]
    assert memory._base_ready is True
    assert memory._retrieval_ready is True
    assert memory._ingestion_ready is True


def test_failed_llm_init_does_not_leave_memory_llm_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _build_uninitialized_memory(tmp_path)
    memory.llm = None
    memory.agent_llm = None
    memory.agent_llm_cfg = type("AgentCfg", (), {"provider": "openai"})()

    class _FakeLLM:
        def __init__(self, model: str) -> None:
            self.model = model

    def fake_get_llm_factory(provider: str):
        if provider == memory.llm_cfg.provider:
            return lambda cfg: _FakeLLM("uma-model")
        if provider == "openai":
            def _raise(_cfg):
                raise RuntimeError("agent llm init failed")
            return _raise
        return None

    monkeypatch.setattr(provider_init, "get_llm_factory", fake_get_llm_factory)

    with pytest.raises(RuntimeError, match="agent llm init failed"):
        provider_init.ensure_llm(memory)

    assert memory.llm is None
    assert memory.agent_llm is None


def test_failed_embedder_validation_does_not_leave_memory_embedder_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _build_uninitialized_memory(tmp_path)
    memory.embedder = None

    class _BadEmbedder:
        model = "bad-embedder"
        dimension = 1

    monkeypatch.setattr(
        provider_init,
        "get_embedder_factory",
        lambda provider: (lambda cfg: _BadEmbedder()) if provider == memory.embedding_cfg.provider else None,
    )

    with pytest.raises(ValueError, match="Embedder dimension mismatch"):
        provider_init.ensure_embedder(memory)

    assert memory.embedder is None


def test_failed_core_init_can_retry_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _build_uninitialized_memory(tmp_path)
    memory.llm = object()
    memory.embedder = object()
    memory._stores = {
        "episodic": object(),
        "semantic": object(),
        "procedural": object(),
        "chunk": object(),
    }
    semantic_calls = {"count": 0}

    monkeypatch.setattr(memory_module, "WorkingMemoryCore", lambda **kwargs: object())
    monkeypatch.setattr(memory_module, "EpisodeIndexer", lambda **kwargs: object())
    monkeypatch.setattr(memory_module, "EpisodicRetentionPolicy", lambda **kwargs: object())
    monkeypatch.setattr(memory_module, "EpisodicCore", lambda **kwargs: object())
    monkeypatch.setattr(memory_module, "ProceduralCore", lambda store: object())
    monkeypatch.setattr(memory_module, "ChunkCore", lambda store, memory: object())

    def fake_semantic_core(**kwargs):
        semantic_calls["count"] += 1
        if semantic_calls["count"] == 1:
            raise RuntimeError("semantic init failed")
        return object()

    monkeypatch.setattr(memory_module, "SemanticCore", fake_semantic_core)

    with pytest.raises(RuntimeError, match="semantic init failed"):
        memory._init_core_subsystems()

    assert memory.working_memory is None
    assert memory.episodic_core is None
    assert memory.semantic_core is None
    assert memory.procedural_core is None
    assert memory.chunk_core is None

    memory._init_core_subsystems()

    assert semantic_calls["count"] == 2
    assert memory.working_memory is not None
    assert memory.episodic_core is not None
    assert memory.semantic_core is not None
    assert memory.procedural_core is not None
    assert memory.chunk_core is not None


def test_warmup_failure_does_not_poison_foreground_ingestion_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _build_uninitialized_memory(tmp_path)
    calls = {"count": 0}

    def fake_init_ingestion_ready(mem: UMAMemory) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("warmup failed")
        mem._ingestion_ready = True

    class _ImmediateThread:
        def __init__(self, *, target, name: str, daemon: bool) -> None:
            self._target = target

        def start(self) -> None:
            self._target()

    monkeypatch.setattr(runtime_init, "init_ingestion_ready", fake_init_ingestion_ready)
    monkeypatch.setattr(memory_module, "init_ingestion_ready", fake_init_ingestion_ready)
    monkeypatch.setattr(runtime_init.threading, "Thread", _ImmediateThread)

    runtime_init.schedule_ingestion_warmup(memory)

    assert calls["count"] == 1
    assert memory._ingestion_ready is False
    assert memory._warmup_scheduled is False

    memory._ensure_ingestion_ready()

    assert calls["count"] == 2
    assert memory._ingestion_ready is True


def test_memory_runtime_singleton_is_thread_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _build_uninitialized_memory(tmp_path)
    created: list[object] = []
    entered = threading.Event()
    release = threading.Event()

    def fake_runtime_ctor(**kwargs: object) -> object:
        created.append(object())
        entered.set()
        assert release.wait(timeout=2.0)
        return created[-1]

    monkeypatch.setattr(memory_module, "UMARuntime", fake_runtime_ctor)

    seen: list[object] = []

    def resolve_runtime() -> None:
        seen.append(memory.runtime)

    first = threading.Thread(target=resolve_runtime)
    second = threading.Thread(target=resolve_runtime)
    first.start()
    assert entered.wait(timeout=1.0)
    second.start()
    time.sleep(0.1)
    release.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert len(created) == 1
    assert len(seen) == 2
    assert seen[0] is seen[1]


def test_working_memory_buffer_thread_safe_for_append_read_replace() -> None:
    buffer = WorkingMemoryBuffer(max_tokens=100)
    scope = SessionScope(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id="agent-test",
        session_id="session-test",
        user_id="user:u1",
    )
    failures: list[BaseException] = []

    def writer() -> None:
        try:
            for idx in range(100):
                buffer.append(scope, "user", f"msg-{idx}")
        except BaseException as exc:  # pragma: no cover - failure capture only
            failures.append(exc)

    def reader_replacer() -> None:
        try:
            for _ in range(100):
                ctx = buffer.get_context(scope)
                buffer.replace_messages(scope, ctx[-10:])
                buffer.total_tokens(scope)
        except BaseException as exc:  # pragma: no cover - failure capture only
            failures.append(exc)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader_replacer),
        threading.Thread(target=reader_replacer),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert failures == []
    assert isinstance(buffer.get_context(scope), list)


# ── test_public_memory_concurrency ──────────────────────────────────────────






def _empty_context(*, query: str):
    return make_context_bundle(query=query)


@pytest.mark.asyncio
async def test_retrieve_context_passes_explicit_request_scope_through(uma_memory, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = uma_memory
    seen: list[tuple[str, str, str, str]] = []

    async def fake_retrieve_context(bound_context, *, query_text: str, lane_filter=None, include_debug: bool = False):
        del lane_filter
        seen.append(
            (
                bound_context.user_id or "",
                bound_context.tenant_id,
                bound_context.request_id,
                bound_context.session_id or "",
            )
        )
        return _empty_context(query=query_text)

    monkeypatch.setattr(memory.runtime, "retrieve_context", fake_retrieve_context)

    result = await memory.retrieve_context(
        query_text="coffee",
        agent_id=AGENT_ID,
        user_id="user:u1",
        tenant_id="tenant-a",
        request_id="req-a",
        session_id="session-a",
    )

    assert result.product == "context"
    assert seen == [("user:u1", "tenant-a", "req-a", "session-a")]


@pytest.mark.asyncio
async def test_concurrent_retrieve_context_calls_keep_request_scope_isolated(
    uma_memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = uma_memory
    barrier = threading.Barrier(2)
    seen: list[tuple[str, str, str, str]] = []

    async def fake_retrieve_context(bound_context, *, query_text: str, lane_filter=None, include_debug: bool = False):
        del lane_filter
        await asyncio.to_thread(barrier.wait, _BARRIER_TIMEOUT_S)
        seen.append(
            (
                query_text,
                bound_context.user_id or "",
                bound_context.request_id,
                bound_context.session_id or "",
            )
        )
        return _empty_context(query=query_text)

    monkeypatch.setattr(memory.runtime, "retrieve_context", fake_retrieve_context)

    first, second = await asyncio.gather(
        memory.retrieve_context(
            query_text="query-a",
            user_id="user:u1",
            request_id="req-a",
            session_id="session-a",
            agent_id=AGENT_ID,
        ),
        memory.retrieve_context(
            query_text="query-b",
            user_id="user:u2",
            request_id="req-b",
            session_id="session-b",
            agent_id=AGENT_ID,
        ),
    )

    assert first.product == "context"
    assert second.product == "context"
    assert sorted(seen) == [
        ("query-a", "user:u1", "req-a", "session-a"),
        ("query-b", "user:u2", "req-b", "session-b"),
    ]


@pytest.mark.asyncio
async def test_retrieve_context_overlap_with_process_turn_keeps_each_call_scope(
    uma_memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = uma_memory
    barrier = threading.Barrier(2)
    seen_contexts: list[tuple[str, str, str, str]] = []
    seen_turns: list[tuple[str, str, str, str]] = []

    async def fake_retrieve_context(bound_context, *, query_text: str, lane_filter=None, include_debug: bool = False):
        del lane_filter
        await asyncio.to_thread(barrier.wait, _BARRIER_TIMEOUT_S)
        seen_contexts.append(
            (
                query_text,
                bound_context.user_id or "",
                bound_context.request_id,
                bound_context.session_id or "",
            )
        )
        return _empty_context(query=query_text)

    class _DummyPipeline:
        async def process_turn(
            self,
            *,
            agent_id: str,
            user_id: str,
            user_msg: str,
            assistant_reply: str,
            session_id: str,
            tenant_id: str = DEFAULT_TENANT_ID,
            workspace_id=None,
            extra_meta=None,
        ) -> None:
            del user_msg, assistant_reply
            extra_meta = dict(extra_meta or {})
            await asyncio.to_thread(barrier.wait, _BARRIER_TIMEOUT_S)
            seen_turns.append(
                (
                    user_id,
                    str(tenant_id or ""),
                    str(extra_meta.get("request_id") or ""),
                    str(session_id or ""),
                )
            )

    monkeypatch.setattr(memory.runtime, "retrieve_context", fake_retrieve_context)
    memory.pipeline = _DummyPipeline()

    await asyncio.gather(
        memory.retrieve_context(
            query_text="query-a",
            user_id="user:u1",
            request_id="req-a",
            session_id="session-a",
            agent_id=AGENT_ID,
        ),
        memory.process_turn(
            agent_id=AGENT_ID,
            user_id="user:u2",
            user_msg="hello",
            assistant_reply="world",
            session_id="session-b",
            tenant_id="tenant-b",
            workspace_id="workspace:beta",
            extra_meta={
                "request_id": "req-b",
            },
        ),
    )

    assert seen_contexts == [("query-a", "user:u1", "req-a", "session-a")]
    assert seen_turns == [("user:u2", "tenant-b", "req-b", "session-b")]


@pytest.mark.asyncio
async def test_bootstrap_overlap_with_retrieval_keeps_explicit_request_scope(
    uma_memory,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = uma_memory
    barrier = threading.Barrier(2)
    bootstrap_path = tmp_path / "MEMORY.md"
    bootstrap_path.write_text("- remembers coffee\n", encoding="utf-8")
    seen_bootstrap: list[tuple[str, str, str]] = []
    seen_contexts: list[tuple[str, str, str]] = []

    async def fake_ingest_memory_bootstrap(file_path, *, memory, runtime_context, config=None):
        del file_path, memory, config
        await asyncio.to_thread(barrier.wait, _BARRIER_TIMEOUT_S)
        seen_bootstrap.append(
            (
                runtime_context.user_id or "",
                runtime_context.request_id,
                runtime_context.session_id or "",
            )
        )
        return {"status": "ingested", "facts_created": 1}

    async def fake_retrieve_context(bound_context, *, query_text: str, lane_filter=None, include_debug: bool = False):
        del query_text, lane_filter
        await asyncio.to_thread(barrier.wait, _BARRIER_TIMEOUT_S)
        seen_contexts.append(
            (
                bound_context.user_id or "",
                bound_context.request_id,
                bound_context.session_id or "",
            )
        )
        return _empty_context(query="coffee")

    monkeypatch.setattr(ingest_service, "ingest_memory_bootstrap", fake_ingest_memory_bootstrap)
    monkeypatch.setattr(memory.runtime, "retrieve_context", fake_retrieve_context)

    bootstrap_result, retrieval_result = await asyncio.gather(
        memory.load_memory_bootstrap(
            str(bootstrap_path),
            user_id="user:bootstrap",
            tenant_id="tenant-bootstrap",
            request_id="req-bootstrap",
            session_id="session-bootstrap",
            agent_id=AGENT_ID,
        ),
        memory.retrieve_context(
            query_text="coffee",
            user_id="user:retrieve",
            tenant_id="tenant-retrieve",
            request_id="req-retrieve",
            session_id="session-retrieve",
            agent_id=AGENT_ID,
        ),
    )

    assert bootstrap_result["status"] == "ingested"
    assert retrieval_result.product == "context"
    assert seen_bootstrap == [("user:bootstrap", "req-bootstrap", "session-bootstrap")]
    assert seen_contexts == [("user:retrieve", "req-retrieve", "session-retrieve")]


# ── test_rebuild_indexes ──────────────────────────────────────────





@pytest.mark.asyncio
async def test_rebuild_vector_indexes(uma_memory):
    memory = uma_memory

    user_id = "user:123"
    owner_id = normalize_user_id(user_id)
    embedding = (await memory.embedder.embed(["hello"]))[0]

    episode = Episode(
        id="ep-1",
        timestamp=datetime.utcnow(),
        summary="hello",
        user_id=user_id,
        owner_type="user",
        owner_id=owner_id,
        raw="hello world",
        tags=["test"],
        embedding=embedding,
    )
    await memory.episodic_core.add_episode(episode, embedding)

    fact = Fact(
        id="fact_1",
        subject=owner_id,
        predicate="prefers",
        object="coffee",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=[episode.id],
        confidence=0.9,
        owner_type="user",
        owner_id=owner_id,
    )
    await memory.semantic_core.upsert_fact(fact, embedding)

    skill = Skill(
        id="skill_1",
        name="Make coffee",
        description="Brews a cup of coffee.",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        owner_type="user",
        owner_id=owner_id,
        trigger_phrases=["coffee"],
        trigger_patterns=[],
        plan={"steps": ["boil", "brew"]},
        tools=["kettle"],
        example="Make coffee",
        meta={"tag": "demo"},
    )
    await memory.procedural_core.add_skill(skill, embedding)

    memory.episodic_core.vector_index().delete([episode.id])
    memory.semantic_core.vector_index().delete([fact.id])
    memory.procedural_core.vector_index().delete([skill.id])

    result = await memory.rebuild_vector_indexes(owner_type="user", owner_id=owner_id)
    assert result.status in ("ok", "degraded")
    assert episode.id in memory.episodic_core.vector_index()._vectors
    assert fact.id in memory.semantic_core.vector_index()._vectors
    assert skill.id in memory.procedural_core.vector_index()._vectors
    assert memory.episodic_core.vector_index()._scopes[episode.id] == ("default", "user", owner_id)
    assert isinstance(memory.episodic_core.vector_index()._extra.get(episode.id), dict)
    assert memory.semantic_core.vector_index()._scopes[fact.id] == ("default", "user", owner_id)
    assert memory.semantic_core.vector_index()._extra[fact.id]["subject"] == owner_id
    assert memory.semantic_core.vector_index()._extra[fact.id]["predicate"] == "prefers"
    assert memory.procedural_core.vector_index()._scopes[skill.id] == ("default", "user", owner_id)
    assert memory.procedural_core.vector_index()._extra[skill.id]["name"] == "Make coffee"


@pytest.mark.asyncio
async def test_rebuild_derived_indexes_replays_graph_from_authoritative_scope(uma_memory):
    memory = uma_memory
    owner_id = normalize_user_id("user:u1")
    base_ts = datetime.utcnow()

    episode_1 = Episode(
        id="ep-graph-1",
        timestamp=base_ts,
        summary="first summary",
        user_id=owner_id,
        owner_type="user",
        owner_id=owner_id,
        tenant_id="tenant-a",
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
        workspace_id=None,
        meta={"turn_id": "turn-a"},
    )
    episode_2 = Episode(
        id="ep-graph-2",
        timestamp=base_ts + timedelta(seconds=1),
        summary="second summary",
        user_id=owner_id,
        owner_type="user",
        owner_id=owner_id,
        tenant_id="tenant-a",
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
        workspace_id=None,
        meta={"turn_id": "turn-b"},
    )
    embedding = (await memory.embedder.embed(["hello graph"]))[0]
    await memory.episodic_core.add_episode(episode_1, embedding)
    await memory.episodic_core.add_episode(episode_2, embedding)

    fact_1 = Fact(
        id="fact_graph_1",
        subject=owner_id,
        predicate="likes",
        object="tea",
        created_at=base_ts,
        updated_at=base_ts,
        source_ids=["chunk-a"],
        confidence=0.9,
        owner_type="user",
        owner_id=owner_id,
        tenant_id="tenant-a",
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
        meta={"turn_id": "turn-a"},
    )
    fact_2 = Fact(
        id="fact_graph_2",
        subject=owner_id,
        predicate="prefers",
        object="coffee",
        created_at=base_ts + timedelta(seconds=1),
        updated_at=base_ts + timedelta(seconds=1),
        source_ids=["chunk-b"],
        confidence=0.9,
        owner_type="user",
        owner_id=owner_id,
        tenant_id="tenant-a",
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
        meta={"turn_id": "turn-b"},
    )
    await memory.semantic_core.upsert_fact(fact_1, embedding)
    await memory.semantic_core.upsert_fact(fact_2, embedding)

    adapter = getattr(memory.graph_core, "adapter", None)
    assert adapter is not None
    adapter.queries.clear()

    result = await memory.rebuild_derived_indexes(
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        include_procedural=False,
    )

    assert result.status in ("ok", "degraded")
    assert result.graph.status == "ok"
    assert result.graph.episodes == 2
    assert result.graph.facts == 2
    assert result.graph.episode_fact_links == 2
    assert result.graph.temporal_links == 1

    assert memory.episodic_core.vector_index()._scopes[episode_1.id] == ("tenant-a", "user", owner_id)
    assert memory.semantic_core.vector_index()._scopes[fact_1.id] == ("tenant-a", "user", owner_id)

    params_list = [params or {} for _cypher, params in adapter.queries]
    assert any(params.get("episode_id") == "ep-graph-1" and params.get("tenant_id") == "tenant-a" for params in params_list)
    assert any(params.get("fact_id") == "fact_graph_1" and params.get("scope_model_version") == "v2" for params in params_list)
    assert any(params.get("ep_id") == "ep-graph-1" and params.get("fact_id") == "fact_graph_1" for params in params_list)
    assert any(params.get("a") == "ep-graph-1" and params.get("b") == "ep-graph-2" for params in params_list)

    first_semantic_meta = dict(memory.semantic_core.vector_index()._extra[fact_1.id])
    first_query_count = len(adapter.queries)

    result_again = await memory.rebuild_derived_indexes(
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        include_procedural=False,
    )

    assert result_again.status in ("ok", "degraded")
    assert result_again.graph == result.graph
    assert memory.semantic_core.vector_index()._extra[fact_1.id] == first_semantic_meta
    assert len(adapter.queries) == first_query_count * 2


@pytest.mark.asyncio
async def test_rebuild_vector_indexes_preserves_promoted_workspace_fact_scope(uma_memory):
    memory = uma_memory
    embedding = (await memory.embedder.embed(["workspace fact"]))[0]

    fact = Fact(
        id="fact_workspace_rebuild",
        subject="workspace:alpha",
        predicate="contains",
        object="runbook",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=["chunk-workspace"],
        confidence=0.7,
        owner_type="workspace",
        owner_id="workspace:alpha",
        tenant_id="tenant-w",
        workspace_id="workspace:alpha",
        session_id=None,
        origin_agent_id="agent-a",
        origin_user_id="user:u1",
        origin_session_id="session-a",
        meta={"promotion_source": "session"},
    )
    await memory.semantic_core.upsert_fact(fact, embedding)
    memory.semantic_core.vector_index().delete([fact.id])

    result = await memory.rebuild_vector_indexes(
        tenant_id="tenant-w",
        owner_type="workspace",
        owner_id="workspace:alpha",
        include_episodic=False,
        include_procedural=False,
    )

    assert result.status in ("ok", "degraded")
    assert memory.semantic_core.vector_index()._scopes[fact.id] == ("tenant-w", "workspace", "workspace:alpha")
    metadata = memory.semantic_core.vector_index()._extra[fact.id]
    assert metadata["subject"] == "workspace:alpha"
    assert metadata["predicate"] == "contains"


@pytest.mark.asyncio
async def test_rebuild_derived_indexes_is_tenant_scoped_for_identical_owner_tuple(uma_memory):
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    base_ts = datetime.utcnow()
    embedding = (await memory.embedder.embed(["tenant scoped rebuild"]))[0]

    await memory.episodic_core.add_episode(
        Episode(
            id="ep-tenant-a",
            timestamp=base_ts,
            summary="tenant a episode",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-a",
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        embedding,
    )
    await memory.episodic_core.add_episode(
        Episode(
            id="ep-tenant-b",
            timestamp=base_ts,
            summary="tenant b episode",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-b",
            session_id="session-b",
            origin_agent_id="agent-b",
            origin_user_id=owner_id,
            origin_session_id="session-b",
            meta={"turn_id": "turn-b"},
        ),
        embedding,
    )

    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_tenant_rebuild_a",
            subject=owner_id,
            predicate="LIKES",
            object="alpha",
            created_at=base_ts,
            updated_at=base_ts,
            source_ids=["chunk-a"],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        embedding,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_tenant_rebuild_b",
            subject=owner_id,
            predicate="LIKES",
            object="beta",
            created_at=base_ts,
            updated_at=base_ts,
            source_ids=["chunk-b"],
            confidence=0.9,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-b",
            origin_agent_id="agent-b",
            origin_user_id=owner_id,
            origin_session_id="session-b",
            meta={"turn_id": "turn-b"},
        ),
        embedding,
    )

    await memory.procedural_core.add_skill(
        Skill(
            id="skill_tenant_rebuild_a",
            name="Tenant A Skill",
            description="alpha",
            created_at=base_ts,
            updated_at=base_ts,
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
        embedding,
    )
    await memory.procedural_core.add_skill(
        Skill(
            id="skill_tenant_rebuild_b",
            name="Tenant B Skill",
            description="beta",
            created_at=base_ts,
            updated_at=base_ts,
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
        embedding,
    )

    memory.episodic_core.vector_index().delete(["ep-tenant-a", "ep-tenant-b"])
    memory.semantic_core.vector_index().delete(["fact_tenant_rebuild_a", "fact_tenant_rebuild_b"])
    memory.procedural_core.vector_index().delete(["skill_tenant_rebuild_a", "skill_tenant_rebuild_b"])
    memory.graph_core.adapter.queries.clear()

    result = await memory.rebuild_derived_indexes(
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
    )

    assert result.status in ("ok", "degraded")
    assert result.vector.report["episodic"].count == 1
    assert result.vector.report["semantic"].count == 1
    assert result.vector.report["procedural"].count == 1
    assert result.graph.episodes == 1
    assert result.graph.facts == 1

    assert "ep-tenant-a" in memory.episodic_core.vector_index()._vectors
    assert "ep-tenant-b" not in memory.episodic_core.vector_index()._vectors
    assert "fact_tenant_rebuild_a" in memory.semantic_core.vector_index()._vectors
    assert "fact_tenant_rebuild_b" not in memory.semantic_core.vector_index()._vectors
    assert "skill_tenant_rebuild_a" in memory.procedural_core.vector_index()._vectors
    assert "skill_tenant_rebuild_b" not in memory.procedural_core.vector_index()._vectors

    params_list = [params or {} for _cypher, params in memory.graph_core.adapter.queries]
    assert any(params.get("tenant_id") == "tenant-a" and params.get("episode_id") == "ep-tenant-a" for params in params_list)
    assert not any(params.get("tenant_id") == "tenant-b" for params in params_list)


@pytest.mark.asyncio
async def test_vector_rebuild_lock_prevents_overlapping_execution(uma_memory, monkeypatch: pytest.MonkeyPatch):
    memory = uma_memory
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = {"count": 0}
    completions: list[str] = []

    async def slow_unlocked(*args, **kwargs):
        calls["count"] += 1
        entered.set()
        await release.wait()
        return {"status": "ok", "report": {}}

    monkeypatch.setattr(maintenance_module, "_rebuild_vector_indexes_unlocked", slow_unlocked)

    async def call(name: str) -> None:
        await maintenance_module.rebuild_vector_indexes(memory, owner_type="user", owner_id="user:u1")
        completions.append(name)

    first = asyncio.create_task(call("first"))
    await entered.wait()
    second = asyncio.create_task(call("second"))
    await asyncio.sleep(0.05)

    assert calls["count"] == 1
    assert completions == []

    release.set()
    await asyncio.gather(first, second)

    assert calls["count"] == 2
    assert sorted(completions) == ["first", "second"]


@pytest.mark.asyncio
async def test_graph_rebuild_lock_prevents_overlapping_execution(uma_memory, monkeypatch: pytest.MonkeyPatch):
    memory = uma_memory
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = {"count": 0}
    completions: list[str] = []

    async def slow_unlocked(*args, **kwargs):
        calls["count"] += 1
        entered.set()
        await release.wait()
        return {"status": "ok", "episodes": 0, "facts": 0, "episode_fact_links": 0, "temporal_links": 0}

    monkeypatch.setattr(maintenance_module, "_rebuild_graph_from_authoritative_stores_unlocked", slow_unlocked)

    async def call(name: str) -> None:
        await maintenance_module._rebuild_graph_from_authoritative_stores(
            memory,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=normalize_user_id("user:u1"),
            include_graph=True,
        )
        completions.append(name)

    first = asyncio.create_task(call("first"))
    await entered.wait()
    second = asyncio.create_task(call("second"))
    await asyncio.sleep(0.05)

    assert calls["count"] == 1
    assert completions == []

    release.set()
    await asyncio.gather(first, second)

    assert calls["count"] == 2
    assert sorted(completions) == ["first", "second"]


@pytest.mark.asyncio
async def test_graph_rebuild_clears_scoped_materialization_before_replay(uma_memory):
    memory = uma_memory
    owner_id = normalize_user_id("user:u1")
    base_ts = datetime.utcnow()
    embedding = (await memory.embedder.embed(["graph rebuild scope"]))[0]

    await memory.episodic_core.add_episode(
        Episode(
            id="ep-scope-clear",
            timestamp=base_ts,
            summary="scope clear",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-clear",
            session_id="session-clear",
            origin_agent_id="agent-clear",
            origin_user_id=owner_id,
            origin_session_id="session-clear",
            meta={"turn_id": "turn-clear"},
        ),
        embedding,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_scope_clear",
            subject=owner_id,
            predicate="likes",
            object="clear-tea",
            created_at=base_ts,
            updated_at=base_ts,
            source_ids=["chunk-clear"],
            confidence=0.9,
            tenant_id="tenant-clear",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-clear",
            origin_agent_id="agent-clear",
            origin_user_id=owner_id,
            origin_session_id="session-clear",
            meta={"turn_id": "turn-clear"},
        ),
        embedding,
    )

    adapter = memory.graph_core.adapter
    adapter.queries.clear()

    result = await memory.rebuild_derived_indexes(
        tenant_id="tenant-clear",
        owner_type="user",
        owner_id=owner_id,
        include_procedural=False,
    )

    assert result.status in ("ok", "degraded")
    assert len(adapter.queries) >= 3
    clear_queries = adapter.queries[:3]
    clear_params = [params or {} for _cypher, params in clear_queries]
    assert all(params.get("tenant_id") == "tenant-clear" for params in clear_params)
    assert all(params.get("owner_type") == "user" for params in clear_params)
    assert all(params.get("owner_id") == owner_id for params in clear_params)
    assert "DELETE r" in clear_queries[0][0]
    assert "DETACH DELETE f" in clear_queries[1][0]
    assert "DETACH DELETE e" in clear_queries[2][0]


@pytest.mark.asyncio
async def test_graph_rebuild_clear_is_scoped_to_requested_owner(uma_memory):
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    base_ts = datetime.utcnow()
    embedding = (await memory.embedder.embed(["graph rebuild owner scope"]))[0]

    await memory.episodic_core.add_episode(
        Episode(
            id="ep-clear-a",
            timestamp=base_ts,
            summary="tenant a",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-a",
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        embedding,
    )
    await memory.episodic_core.add_episode(
        Episode(
            id="ep-clear-b",
            timestamp=base_ts + timedelta(seconds=1),
            summary="tenant b",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-b",
            session_id="session-b",
            origin_agent_id="agent-b",
            origin_user_id=owner_id,
            origin_session_id="session-b",
            meta={"turn_id": "turn-b"},
        ),
        embedding,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_clear_a",
            subject=owner_id,
            predicate="likes",
            object="alpha",
            created_at=base_ts,
            updated_at=base_ts,
            source_ids=["chunk-a"],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        embedding,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_clear_b",
            subject=owner_id,
            predicate="likes",
            object="beta",
            created_at=base_ts + timedelta(seconds=1),
            updated_at=base_ts + timedelta(seconds=1),
            source_ids=["chunk-b"],
            confidence=0.9,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-b",
            origin_agent_id="agent-b",
            origin_user_id=owner_id,
            origin_session_id="session-b",
            meta={"turn_id": "turn-b"},
        ),
        embedding,
    )

    adapter = memory.graph_core.adapter
    adapter.queries.clear()

    await memory.rebuild_derived_indexes(
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        include_procedural=False,
    )

    clear_params = [(params or {}) for _cypher, params in adapter.queries[:3]]
    assert all(params.get("tenant_id") == "tenant-a" for params in clear_params)
    assert not any(params.get("tenant_id") == "tenant-b" for params in clear_params)


@pytest.mark.asyncio
async def test_live_write_overlap_with_vector_rebuild_keeps_retrieval_scoped(uma_memory, monkeypatch: pytest.MonkeyPatch):
    memory = uma_memory
    owner_id = normalize_user_id("user:scope-a")
    other_owner_id = normalize_user_id("user:scope-b")
    base_ts = datetime.utcnow()

    existing_fact = Fact(
        id="fact_overlap_existing",
        subject=owner_id,
        predicate="LIKES",
        object="coffee",
        created_at=base_ts,
        updated_at=base_ts,
        source_ids=["chunk-existing"],
        confidence=0.9,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
    )
    live_fact = Fact(
        id="fact_overlap_live",
        subject=owner_id,
        predicate="LIKES",
        object="tea",
        created_at=base_ts + timedelta(seconds=1),
        updated_at=base_ts + timedelta(seconds=1),
        source_ids=["chunk-live"],
        confidence=0.9,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
    )
    other_fact = Fact(
        id="fact_overlap_other_tenant",
        subject=other_owner_id,
        predicate="LIKES",
        object="juice",
        created_at=base_ts,
        updated_at=base_ts,
        source_ids=["chunk-other"],
        confidence=0.9,
        tenant_id="tenant-b",
        owner_type="user",
        owner_id=other_owner_id,
        session_id="session-b",
        origin_agent_id="agent-b",
        origin_user_id=other_owner_id,
        origin_session_id="session-b",
    )

    existing_embedding, live_embedding, other_embedding = await memory.embedder.embed(
        [
            build_fact_embedding_text(existing_fact),
            build_fact_embedding_text(live_fact),
            build_fact_embedding_text(other_fact),
        ]
    )
    await memory.semantic_core.upsert_fact(existing_fact, existing_embedding)
    await memory.semantic_core.upsert_fact(other_fact, other_embedding)

    original_list_facts = memory.semantic_core.list_facts_for_owner
    entered = asyncio.Event()
    release = asyncio.Event()
    blocked = {"done": False}

    async def paused_list_facts_for_owner(*, tenant_id: str, owner_type: str, owner_id: str, limit=None):
        if (
            not blocked["done"]
            and tenant_id == "tenant-a"
            and owner_type == "user"
            and owner_id == normalize_user_id("user:scope-a")
        ):
            blocked["done"] = True
            entered.set()
            await release.wait()
        return await original_list_facts(
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            limit=limit,
        )

    monkeypatch.setattr(memory.semantic_core, "list_facts_for_owner", paused_list_facts_for_owner)

    rebuild_task = asyncio.create_task(
        memory.rebuild_vector_indexes(
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            include_episodic=False,
            include_procedural=False,
        )
    )
    await entered.wait()

    await memory.semantic_core.upsert_fact(live_fact, live_embedding)
    release.set()
    rebuild_result = await rebuild_task

    assert rebuild_result.status in ("ok", "degraded")

    tenant_a_results = await memory.semantic_core.search(
        query_embedding=live_embedding,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        k=10,
        query_text=build_fact_embedding_text(live_fact),
    )
    assert tenant_a_results
    assert all(getattr(fact, "tenant_id", None) == "tenant-a" for fact in tenant_a_results)
    assert all(getattr(fact, "owner_id", None) == owner_id for fact in tenant_a_results)
    assert not any(getattr(fact, "id", None) == other_fact.id for fact in tenant_a_results)

    tenant_b_results = await memory.semantic_core.search(
        query_embedding=other_embedding,
        tenant_id="tenant-b",
        owner_type="user",
        owner_id=other_owner_id,
        k=10,
        query_text=build_fact_embedding_text(other_fact),
    )
    assert tenant_b_results
    assert all(getattr(fact, "tenant_id", None) == "tenant-b" for fact in tenant_b_results)
    assert all(getattr(fact, "owner_id", None) == other_owner_id for fact in tenant_b_results)
    assert not any(getattr(fact, "id", None) == live_fact.id for fact in tenant_b_results)


@pytest.mark.asyncio
async def test_deferred_graph_update_overlap_with_graph_rebuild_keeps_scope_isolated(
    uma_memory,
    monkeypatch: pytest.MonkeyPatch,
):
    memory = uma_memory
    owner_id = normalize_user_id("user:graph-a")
    base_ts = datetime.utcnow()
    seed_embedding = (await memory.embedder.embed(["graph overlap seed"]))[0]

    await memory.episodic_core.add_episode(
        Episode(
            id="ep-graph-overlap-a",
            timestamp=base_ts,
            summary="seed episode",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-a",
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        seed_embedding,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_graph_overlap_a",
            subject=owner_id,
            predicate="LIKES",
            object="alpha",
            created_at=base_ts,
            updated_at=base_ts,
            source_ids=["chunk-a"],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        seed_embedding,
    )

    adapter = memory.graph_core.adapter
    adapter.queries.clear()

    original_list_episodes = memory.episodic_core.list_episodes
    entered = asyncio.Event()
    release = asyncio.Event()
    blocked = {"done": False}

    async def paused_list_episodes(tenant_id: str, owner_type: str, owner_id: str):
        if (
            not blocked["done"]
            and tenant_id == "tenant-a"
            and owner_type == "user"
            and owner_id == normalize_user_id("user:graph-a")
        ):
            blocked["done"] = True
            entered.set()
            await release.wait()
        return await original_list_episodes(tenant_id, owner_type, owner_id)

    monkeypatch.setattr(memory.episodic_core, "list_episodes", paused_list_episodes)

    rebuild_task = asyncio.create_task(
        memory.rebuild_derived_indexes(
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            include_procedural=False,
        )
    )
    await entered.wait()

    await memory.process_turn(
        agent_id=AGENT_ID,
        user_id="user:graph-b",
        user_msg="I like tea",
        assistant_reply="Noted that you like tea.",
        session_id="session-b",
        tenant_id="tenant-b",
        workspace_id="workspace-b",
        extra_meta={"request_id": "req-graph-b"},
    )

    release.set()
    rebuild_result = await rebuild_task

    assert rebuild_result.status in ("ok", "degraded")
    clear_queries = [
        (cypher, params or {})
        for cypher, params in adapter.queries
        if "DELETE r" in cypher or "DETACH DELETE f" in cypher or "DETACH DELETE e" in cypher
    ]
    assert len(clear_queries) == 3
    assert all(params.get("tenant_id") == "tenant-a" for _cypher, params in clear_queries)
    assert all(params.get("owner_type") == "user" for _cypher, params in clear_queries)
    assert all(params.get("owner_id") == owner_id for _cypher, params in clear_queries)

    params_list = [params or {} for _cypher, params in adapter.queries]
    assert any(params.get("tenant_id") == "tenant-b" and params.get("owner_id") == normalize_user_id("user:graph-b") for params in params_list)
    assert any(params.get("tenant_id") == "tenant-a" and params.get("owner_id") == owner_id for params in params_list)
    assert not any(
        (params.get("tenant_id") == "tenant-b")
        and (
            "DELETE r" in cypher
            or "DETACH DELETE f" in cypher
            or "DETACH DELETE e" in cypher
        )
        for cypher, params in adapter.queries
    )


@pytest.mark.asyncio
async def test_semantic_search_drops_vector_candidates_without_committed_sql_row(
    uma_memory,
    monkeypatch: pytest.MonkeyPatch,
):
    memory = uma_memory
    owner_id = normalize_user_id("user:transient")
    fact = Fact(
        id="fact_transient_visibility",
        subject=owner_id,
        predicate="LIKES",
        object="transient coffee",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=["chunk-transient"],
        confidence=0.9,
        tenant_id="tenant-transient",
        owner_type="user",
        owner_id=owner_id,
        session_id="session-transient",
        origin_agent_id="agent-transient",
        origin_user_id=owner_id,
        origin_session_id="session-transient",
    )
    embedding = (await memory.embedder.embed([build_fact_embedding_text(fact)]))[0]

    vector_index = memory.semantic_core.vector_index()
    real_upsert = vector_index.upsert
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def blocking_upsert(*args, **kwargs):
        real_upsert(*args, **kwargs)
        entered.set()
        if not release.wait(timeout=2.0):
            raise TimeoutError("timed out waiting to release vector upsert")

    monkeypatch.setattr(vector_index, "upsert", blocking_upsert)

    def writer() -> None:
        try:
            asyncio.run(memory.semantic_core.upsert_fact(fact, embedding))
        except BaseException as exc:  # pragma: no cover - failure capture only
            failures.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    assert entered.wait(timeout=1.0)

    transient_results = await memory.semantic_core.search(
        query_embedding=embedding,
        tenant_id="tenant-transient",
        owner_type="user",
        owner_id=owner_id,
        k=10,
        query_text=build_fact_embedding_text(fact),
    )
    assert transient_results == []

    release.set()
    thread.join(timeout=2.0)
    assert not failures

    committed_results = await memory.semantic_core.search(
        query_embedding=embedding,
        tenant_id="tenant-transient",
        owner_type="user",
        owner_id=owner_id,
        k=10,
        query_text=build_fact_embedding_text(fact),
    )
    assert [item.id for item in committed_results] == [fact.id]


# ── Fix #1 regression: SQL work is offloaded to a worker thread ────────
#
# BaseSQLStore._run_sync must dispatch its callable to a worker thread so
# that concurrent async callers overlap. If SQL calls were sync-in-async
# (the pre-fix state), two `gather`-ed calls would serialize on the event
# loop and take ~2× the single-call time.
#
# The threshold is generous (~1.6× the sync body) to tolerate CI scheduler
# jitter while still failing loudly if the offload regresses to a sync
# call.


@pytest.mark.asyncio
async def test_run_sync_offloads_blocking_work_to_worker_thread() -> None:
    """Two concurrent `_run_sync` calls with a 100ms sync body must
    finish in ~100ms (parallel), not ~200ms (serialized).

    Every store method that touches sqlite3 goes through `_run_sync`;
    this test is the canonical proof that the offload works. If it
    regresses, every store method regresses with it.
    """
    from uma.stores.base_sql_store import BaseSQLStore

    def _slow_body() -> str:
        # Sleep executes on the worker thread. If `_run_sync` ran the
        # callable inline on the event loop, `gather` would serialize
        # the two invocations.
        time.sleep(0.10)
        return "done"

    start = time.monotonic()
    results = await asyncio.gather(
        BaseSQLStore._run_sync(_slow_body),
        BaseSQLStore._run_sync(_slow_body),
    )
    elapsed = time.monotonic() - start

    assert results == ["done", "done"]
    assert elapsed < 0.16, (
        f"Two 100ms `_run_sync` calls took {elapsed:.3f}s; expected "
        f"~0.10s if the offload works. This suggests SQL work is "
        f"running on the event loop again (regression of Fix #1)."
    )
