from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest
import yaml

from tests.helpers.runtime import build_test_config
from uma.common.initializers import providers as provider_init
from uma.common.initializers import runtime as runtime_init
import uma.api.memory as memory_module
from uma.api.memory import UMAMemory
from uma.ingest.pipeline import MemoryPipeline
from uma.common.config import UMAConfig
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import RuntimeContext


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


class _DummyPipelineMemory:
    def __init__(self) -> None:
        self.pipeline_cfg = type("PipelineCfg", (), {"defer_post_turn": True, "post_turn_queue_max": 64})()


def _payload(*, session_id: str, index: int = 0, extra_meta: dict | None = None) -> dict:
    return {
        "user_id": "user:u1",
        "user_msg": f"user-{index}",
        "assistant_reply": f"assistant-{index}",
        "episode": None,
        "facts": None,
        "turn_context": RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-test",
            request_id=f"req-{session_id}-{index}",
            user_id="user:u1",
            session_id=session_id,
        ),
        "extra_meta": extra_meta or {"session_id": session_id, "index": index},
    }


@pytest.mark.asyncio
async def test_deferred_queue_captures_immutable_payload_copy() -> None:
    pipeline = MemoryPipeline(memory_client=_DummyPipelineMemory(), hooks=object())
    original_meta = {"session_id": "session-a", "index": 1}
    seen: list[dict] = []

    async def fake_run_post_turn_tasks(**kwargs):
        seen.append(
            {
                "session_id": kwargs["turn_context"].session_id,
                "extra_meta": dict(kwargs["extra_meta"]),
            }
        )

    pipeline._run_post_turn_tasks = fake_run_post_turn_tasks  # type: ignore[method-assign]

    assert pipeline._enqueue_post_turn(_payload(session_id="session-a", index=1, extra_meta=original_meta))
    original_meta["session_id"] = "mutated"
    original_meta["index"] = 999

    processed = await pipeline.process_post_turn_queue()

    assert processed == 1
    assert seen == [{"session_id": "session-a", "extra_meta": {"session_id": "session-a", "index": 1}}]


@pytest.mark.asyncio
async def test_deferred_queue_snapshots_episode_object_at_enqueue_time() -> None:
    pipeline = MemoryPipeline(memory_client=_DummyPipelineMemory(), hooks=object())
    seen: list[dict] = []
    episode = {"id": "ep-1", "meta": {"state": "original"}}

    async def fake_run_post_turn_tasks(**kwargs):
        seen.append(kwargs["episode"])

    pipeline._run_post_turn_tasks = fake_run_post_turn_tasks  # type: ignore[method-assign]

    payload = _payload(session_id="session-a", index=1)
    payload["episode"] = episode

    assert pipeline._enqueue_post_turn(payload)
    episode["id"] = "ep-mutated"
    episode["meta"]["state"] = "mutated"

    processed = await pipeline.process_post_turn_queue()

    assert processed == 1
    assert seen == [{"id": "ep-1", "meta": {"state": "original"}}]


@pytest.mark.asyncio
async def test_deferred_queue_snapshots_facts_at_enqueue_time() -> None:
    pipeline = MemoryPipeline(memory_client=_DummyPipelineMemory(), hooks=object())
    seen: list[list[dict]] = []
    facts = [{"id": "fact-1", "meta": {"score": 1.0}}]

    async def fake_run_post_turn_tasks(**kwargs):
        seen.append(kwargs["facts"])

    pipeline._run_post_turn_tasks = fake_run_post_turn_tasks  # type: ignore[method-assign]

    payload = _payload(session_id="session-a", index=1)
    payload["facts"] = facts

    assert pipeline._enqueue_post_turn(payload)
    facts[0]["id"] = "fact-mutated"
    facts[0]["meta"]["score"] = 9.0
    facts.append({"id": "fact-2", "meta": {"score": 2.0}})

    processed = await pipeline.process_post_turn_queue()

    assert processed == 1
    assert seen == [[{"id": "fact-1", "meta": {"score": 1.0}}]]


@pytest.mark.asyncio
async def test_post_turn_queue_enqueue_is_not_blocked_by_running_task() -> None:
    pipeline = MemoryPipeline(memory_client=_DummyPipelineMemory(), hooks=object())
    started = asyncio.Event()
    release = asyncio.Event()
    seen: list[str] = []

    async def slow_run_post_turn_tasks(**kwargs):
        seen.append(kwargs["turn_context"].session_id)
        started.set()
        await release.wait()

    pipeline._run_post_turn_tasks = slow_run_post_turn_tasks  # type: ignore[method-assign]

    assert pipeline._enqueue_post_turn(_payload(session_id="session-a"))
    drain_task = asyncio.create_task(pipeline.process_post_turn_queue())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    assert pipeline._enqueue_post_turn(_payload(session_id="session-b"))

    release.set()
    processed = await asyncio.wait_for(drain_task, timeout=1.0)
    remaining = await pipeline.process_post_turn_queue()

    assert processed == 1
    assert remaining == 1
    assert seen == ["session-a", "session-b"]


@pytest.mark.asyncio
async def test_post_turn_queue_preserves_distinct_sessions_under_overlap() -> None:
    pipeline = MemoryPipeline(memory_client=_DummyPipelineMemory(), hooks=object())
    processed_sessions: list[str] = []
    enqueue_results: list[bool] = []

    async def fake_run_post_turn_tasks(**kwargs):
        processed_sessions.append(kwargs["turn_context"].session_id)

    pipeline._run_post_turn_tasks = fake_run_post_turn_tasks  # type: ignore[method-assign]

    def enqueue(index: int) -> None:
        enqueue_results.append(
            pipeline._enqueue_post_turn(
                _payload(session_id=f"session-{index}", index=index)
            )
        )

    threads = [threading.Thread(target=enqueue, args=(idx,)) for idx in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    processed = await pipeline.process_post_turn_queue()

    assert all(enqueue_results)
    assert processed == 10
    assert sorted(processed_sessions) == [f"session-{idx}" for idx in range(10)]
