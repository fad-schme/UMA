from __future__ import annotations

from typing import Any, Dict, List

import pytest

from uma.api.runtime import UMARuntime
from uma.api.memory import UMAMemory
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import RuntimeContext
from uma.retrieve.rlm.context_pack import ContextPack


@pytest.mark.asyncio
async def test_request_handle_retrieval_delegates_directly_to_runtime(uma_memory) -> None:
    memory = uma_memory
    runtime = UMARuntime.from_memory(memory)
    context = RuntimeContext(
        tenant_id="tenant-1",
        agent_id=memory.agent_id or "agent-default",
        request_id="req-1",
        user_id="user:u1",
        workspace_id="workspace:alpha",
        session_id="session-1",
    )
    handle = runtime.bind(context)
    seen: List[tuple[str, RuntimeContext, str]] = []

    async def fake_context(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        lane_filter=None,
    ) -> Dict[str, list]:
        seen.append(("context", bound_context, query_text))
        return {
            "product": "context",
            "query": query_text,
            "lane_filter": list(lane_filter or []),
            "facts": [],
            "chunks": [],
            "documents": [],
            "working_memory": [],
            "episodic": [],
            "skills": [],
            "graph": [],
        }

    async def fake_memory(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        memory_intent: str = "continuity",
    ) -> Dict[str, Any]:
        seen.append(("memory", bound_context, query_text))
        return {
            "product": "memory",
            "query": query_text,
            "memory_intent": memory_intent,
            "memories": [{"id": "mem-1", "provenance": {"valid": True, "source_chunk_ids": ["chunk-1"]}}],
            "compiled_answer": {"id": "mem-1", "provenance": {"valid": True, "source_chunk_ids": ["chunk-1"]}},
            "evidence": [],
            "supporting_evidence": [],
            "supporting_facts": [],
            "supporting_skills": [],
            "conflicts": [],
            "support_density": 0.0,
            "fallback": {"used": True, "mode": "evidence_only", "reason": "no_compiled_memory_available"},
            "memory_sources": [],
            "compiled_memory_index": [],
            "compiled_memory_log": [],
            "confidence": {},
            "provenance": {"valid": True, "source_chunk_ids": ["chunk-1"]},
            "retrieval_path": [],
        }

    async def fake_messages(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        render_mode: str = "openclaw_v1",
    ) -> Dict[str, Any]:
        seen.append(("messages", bound_context, f"{query_text}:{render_mode}"))
        return {"messages": [{"role": "system", "content": "rendered context"}], "meta": {"render_mode": render_mode}}

    runtime.retrieve_context = fake_context  # type: ignore[method-assign]
    runtime.retrieve_memory = fake_memory  # type: ignore[method-assign]
    runtime.get_context_messages = fake_messages  # type: ignore[method-assign]

    structured = await handle.retrieve_context("hello world", lane_filter=["semantic"])
    memory_payload = await handle.retrieve_memory("hello world")
    messages = await handle.get_context_messages("hello world", render_mode="raw_rendered")

    assert structured["product"] == "context"
    assert structured["lane_filter"] == ["semantic"]
    assert structured["documents"] == []
    assert memory_payload["product"] == "memory"
    assert memory_payload["memories"][0]["id"] == "mem-1"
    assert memory_payload["fallback"]["used"] is True
    assert messages["meta"]["render_mode"] == "raw_rendered"
    assert seen == [
        ("context", context, "hello world"),
        ("memory", context, "hello world"),
        ("messages", context, "hello world:raw_rendered"),
    ]


@pytest.mark.asyncio
async def test_request_handle_retrieval_requires_user_id_for_current_behavior(uma_memory) -> None:
    runtime = UMARuntime.from_memory(uma_memory)
    handle = runtime.bind(
        RuntimeContext(
            tenant_id="tenant-1",
            agent_id=uma_memory.agent_id or "agent-default",
            request_id="req-1",
        )
    )

    with pytest.raises(ValueError, match="user_id"):
        await handle.retrieve_context("hello world")


@pytest.mark.asyncio
async def test_umamemory_public_retrieval_surface_delegates_by_intent(uma_memory) -> None:
    memory = uma_memory
    seen: List[tuple[str, RuntimeContext, str]] = []

    async def fake_context(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        lane_filter=None,
    ) -> Dict[str, list]:
        seen.append(("context", bound_context, query_text))
        return {
            "product": "context",
            "query": query_text,
            "lane_filter": list(lane_filter or []),
            "facts": [],
            "chunks": [],
            "documents": [],
            "working_memory": [],
            "episodic": [],
            "skills": [],
            "graph": [],
        }

    async def fake_memory(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        memory_intent: str = "continuity",
    ) -> Dict[str, Any]:
        seen.append(("memory", bound_context, query_text))
        return {
            "product": "memory",
            "query": query_text,
            "memory_intent": memory_intent,
            "memories": [{"id": "mem-1", "provenance": {"valid": True, "source_chunk_ids": ["chunk-1"]}}],
            "compiled_answer": {"id": "mem-1", "provenance": {"valid": True, "source_chunk_ids": ["chunk-1"]}},
            "evidence": [],
            "supporting_evidence": [],
            "supporting_facts": [],
            "supporting_skills": [],
            "conflicts": [],
            "support_density": 0.0,
            "fallback": {"used": True, "mode": "evidence_only", "reason": "no_compiled_memory_available"},
            "memory_sources": [],
            "compiled_memory_index": [],
            "compiled_memory_log": [],
            "confidence": {},
            "provenance": {"valid": True, "source_chunk_ids": ["chunk-1"]},
            "retrieval_path": [],
        }

    memory.runtime.retrieve_context = fake_context  # type: ignore[method-assign]
    memory.runtime.retrieve_memory = fake_memory  # type: ignore[method-assign]

    context = await memory.retrieve_context(
        query_text="hello world",
        user_id="user:u1",
        tenant_id="tenant-1",
        request_id="req-ctx",
        workspace_id="workspace:alpha",
        session_id="session-1",
    )
    memory_result = await memory.retrieve_memory(
        query_text="hello world",
        user_id="user:u1",
        tenant_id="tenant-1",
        request_id="req-mem",
        workspace_id="workspace:alpha",
        session_id="session-1",
    )

    assert context["product"] == "context"
    assert context["documents"] == []
    assert memory_result["product"] == "memory"
    assert memory_result["memories"][0]["id"] == "mem-1"
    assert memory_result["fallback"]["used"] is True
    assert seen == [
        (
            "context",
            RuntimeContext(
                tenant_id="tenant-1",
                agent_id=memory.agent_id or "agent-default",
                request_id="req-ctx",
                user_id="user:u1",
                workspace_id="workspace:alpha",
                session_id="session-1",
            ),
            "hello world",
        ),
        (
            "memory",
            RuntimeContext(
                tenant_id="tenant-1",
                agent_id=memory.agent_id or "agent-default",
                request_id="req-mem",
                user_id="user:u1",
                workspace_id="workspace:alpha",
                session_id="session-1",
            ),
            "hello world",
        ),
    ]


@pytest.mark.asyncio
async def test_runtime_memory_retrieval_surfaces_explicit_evidence_only_fallback(uma_memory) -> None:
    runtime = UMARuntime.from_memory(uma_memory)
    context = RuntimeContext(
        tenant_id="tenant-1",
        agent_id=uma_memory.agent_id or "agent-default",
        request_id="req-memory-fallback",
        user_id="user:u1",
    )

    async def fake_context(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        lane_filter=None,
    ) -> Dict[str, list]:
        assert lane_filter == ["wiki", "raw", "semantic", "episodic"]
        return {
            "product": "context",
            "query": query_text,
            "lane_filter": [],
            "working_memory": [],
            "episodic": [],
            "facts": [],
            "chunks": [],
            "documents": [],
            "skills": [],
            "graph": [],
            "trace": [],
            "confidence": {},
            "provenance": {"valid": False, "source_chunk_ids": [], "invalid_reasons": ["missing_source_chunk_ids"]},
        }

    runtime.retrieve_context = fake_context  # type: ignore[method-assign]

    result = await runtime.retrieve_memory(
        context,
        query_text="memory query",
        memory_intent="continuity",
    )

    assert result["product"] == "memory"
    assert result["memory_intent"] == "continuity"
    assert len(result["memories"]) == 1
    assert result["fallback"]["used"] is True
    assert result["fallback"]["mode"] == "evidence_only"
    assert result["fallback"]["reason"] == "no_compiled_memory_available"
    assert result["compiled_answer"]["provenance"]["valid"] is False
    assert "missing_source_chunk_ids" in result["compiled_answer"]["provenance"]["invalid_reasons"]


@pytest.mark.asyncio
async def test_runtime_context_trace_surfaces_lane_plan(uma_memory) -> None:
    runtime = UMARuntime.from_memory(uma_memory)
    context = RuntimeContext(
        tenant_id="tenant-1",
        agent_id=uma_memory.agent_id or "agent-default",
        request_id="req-context-plan",
        user_id="user:u1",
    )

    class FakeController:
        async def retrieve_context(self, request, query_text):
            assert request.plan is not None
            pack = ContextPack(
                user_id=request.normalized_user_id,
                query_text=query_text,
                agent_id=request.context.agent_id,
                intent=request.plan.query_intent,
                active_lanes=list(request.plan.participating_lanes),
                active_domains=list(request.plan.active_domains),
                lane_plan=request.plan.to_trace(),
            )
            pack.steps.append({"step": 0, "phase": "plan", **request.plan.to_trace()})
            return pack

    runtime.ensure_retrieval_ready = lambda: None  # type: ignore[method-assign]
    uma_memory._rlm_controller = FakeController()

    result = await runtime.retrieve_context(
        context,
        query_text="What do I like?",
    )

    assert result["product"] == "context"
    assert result["active_lanes"] == ["profile", "procedural"]
    lane_plan = next(step for step in result["trace"] if step.get("event") == "lane_plan")
    assert lane_plan["product"] == "context"
    assert lane_plan["participating_lanes"] == ["profile", "procedural"]
    assert lane_plan["active_domains"] == ["user_profile", "procedural"]

def test_umamemory_retrieval_shims_are_removed() -> None:
    assert hasattr(UMAMemory, "retrieve_context")
    assert hasattr(UMAMemory, "retrieve_memory")
    assert not hasattr(UMAMemory, "fetch_memory")
    assert not hasattr(UMAMemory, "get_structured_context")  # type: ignore[name-defined]
    assert not hasattr(UMAMemory, "get_rendered_context")  # type: ignore[name-defined]
    assert not hasattr(UMAMemory, "get_context_messages")  # type: ignore[name-defined]


@pytest.mark.asyncio
async def test_bound_context_workspace_id_does_not_broaden_retrieval_owner_support(
    uma_memory,
    tmp_path,
) -> None:
    memory = uma_memory
    assert memory.agent_id, "test runtime must set agent_id"

    agent_doc = tmp_path / "agent_doc.txt"
    agent_doc.write_text(
        (
            "Agent KB document. It contains hello world and an agent-only guideline. "
            "This sentence is padding to ensure strict chunk validation passes for ingestion in CI. "
            "Additional padding words to reach the minimum chunk length requirement.\n"
        ),
        encoding="utf-8",
    )
    user_doc = tmp_path / "user_doc.txt"
    user_doc.write_text(
        (
            "User document. It also contains hello world but is user-owned. "
            "This sentence is padding to ensure strict chunk validation passes for ingestion in CI. "
            "Additional padding words to reach the minimum chunk length requirement.\n"
        ),
        encoding="utf-8",
    )

    await memory.ingest_document(str(agent_doc), owner_type="agent", owner_id=memory.agent_id)
    await memory.ingest_document(str(user_doc), owner_type="user", owner_id="user:u1")

    runtime = UMARuntime.from_memory(memory)
    handle = runtime.bind(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=memory.agent_id,
            request_id="req-workspace-inert",
            user_id="user:u1",
            workspace_id="workspace:alpha",
        )
    )

    ctx = await handle.retrieve_context("hello world")
    owner_types = {getattr(chunk, "owner_type", None) for chunk in list(ctx.get("chunks") or [])}
    chunk_lanes = {
        (getattr(chunk, "meta", {}) or {}).get("kb_lane")
        for chunk in list(ctx.get("chunks") or [])
    }

    assert owner_types
    assert owner_types.issubset({"agent", "user"})
    assert "workspace" not in owner_types
    assert "system" not in owner_types
    assert chunk_lanes == {"raw"}
