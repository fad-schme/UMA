from __future__ import annotations

import asyncio
import threading

import pytest

from uma.ingest import ingest_service


def _empty_context(*, query: str) -> dict[str, object]:
    return {
        "product": "context",
        "query": query,
        "working_memory": [],
        "episodic": [],
        "facts": [],
        "chunks": [],
        "documents": [],
        "skills": [],
        "graph": [],
        "trace": [],
        "confidence": {},
    }


@pytest.mark.asyncio
async def test_retrieve_context_does_not_require_set_context_for_request_scope(uma_memory, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = uma_memory
    seen: list[tuple[str, str, str, str]] = []

    async def fake_retrieve_context(bound_context, *, query_text: str, lane_filter=None):
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
        user_id="user:u1",
        tenant_id="tenant-a",
        request_id="req-a",
        session_id="session-a",
    )

    assert result["product"] == "context"
    assert seen == [("user:u1", "tenant-a", "req-a", "session-a")]


@pytest.mark.asyncio
async def test_concurrent_retrieve_context_calls_keep_request_scope_isolated(
    uma_memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = uma_memory
    barrier = threading.Barrier(2)
    seen: list[tuple[str, str, str, str]] = []

    async def fake_retrieve_context(bound_context, *, query_text: str, lane_filter=None):
        del lane_filter
        await asyncio.to_thread(barrier.wait)
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
            tenant_id="tenant-a",
            request_id="req-a",
            session_id="session-a",
        ),
        memory.retrieve_context(
            query_text="query-b",
            user_id="user:u2",
            tenant_id="tenant-b",
            request_id="req-b",
            session_id="session-b",
        ),
    )

    assert first["product"] == "context"
    assert second["product"] == "context"
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

    async def fake_retrieve_context(bound_context, *, query_text: str, lane_filter=None):
        del lane_filter
        await asyncio.to_thread(barrier.wait)
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
            user_id: str,
            user_msg: str,
            assistant_reply: str,
            session_id: str,
            tenant_id: str = "default",
            workspace_id=None,
            extra_meta=None,
        ) -> None:
            del user_msg, assistant_reply
            extra_meta = dict(extra_meta or {})
            await asyncio.to_thread(barrier.wait)
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
            tenant_id="tenant-a",
            request_id="req-a",
            session_id="session-a",
        ),
        memory.process_turn(
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
        await asyncio.to_thread(barrier.wait)
        seen_bootstrap.append(
            (
                runtime_context.user_id or "",
                runtime_context.request_id,
                runtime_context.session_id or "",
            )
        )
        return {"status": "ingested", "facts_created": 1}

    async def fake_retrieve_context(bound_context, *, query_text: str, lane_filter=None):
        del query_text, lane_filter
        await asyncio.to_thread(barrier.wait)
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
        ),
        memory.retrieve_context(
            query_text="coffee",
            user_id="user:retrieve",
            tenant_id="tenant-retrieve",
            request_id="req-retrieve",
            session_id="session-retrieve",
        ),
    )

    assert bootstrap_result["status"] == "ingested"
    assert retrieval_result["product"] == "context"
    assert seen_bootstrap == [("user:bootstrap", "req-bootstrap", "session-bootstrap")]
    assert seen_contexts == [("user:retrieve", "req-retrieve", "session-retrieve")]
