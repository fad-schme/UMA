from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest
import yaml

from tests.helpers.runtime import build_test_config
from uma.core.initializers import runtime as runtime_init
from uma.core.uma_memory import UMAMemory
from uma.core.utils.pipeline import MemoryPipeline
from uma.core.memory_config import UMAConfig
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.types import RuntimeContext


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
