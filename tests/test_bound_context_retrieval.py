from __future__ import annotations

from typing import Any, Dict, List

import pytest

from uma import UMARequestHandle, UMARuntime
from uma.core.uma_memory import UMAMemory
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.types import RuntimeContext


@pytest.mark.asyncio
async def test_request_handle_retrieval_delegates_to_memory_bridge(uma_memory) -> None:
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

    async def fake_structured(bound_context: RuntimeContext, *, query_text: str) -> Dict[str, list]:
        seen.append(("structured", bound_context, query_text))
        return {"facts": [], "chunks": [], "working_memory": [], "episodic": [], "skills": [], "graph": []}

    async def fake_rendered(bound_context: RuntimeContext, *, query_text: str) -> str:
        seen.append(("rendered", bound_context, query_text))
        return "rendered context"

    async def fake_messages(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        render_mode: str = "openclaw_v1",
    ) -> Dict[str, Any]:
        seen.append(("messages", bound_context, f"{query_text}:{render_mode}"))
        return {"messages": [{"role": "system", "content": "rendered context"}], "meta": {"render_mode": render_mode}}

    memory._retrieve_structured_context_for_context = fake_structured  # type: ignore[method-assign]
    memory._retrieve_rendered_context_for_context = fake_rendered  # type: ignore[method-assign]
    memory._get_context_messages_for_context = fake_messages  # type: ignore[method-assign]

    structured = await handle.retrieve_structured_context("hello world")
    rendered = await handle.retrieve_rendered_context("hello world")
    messages = await handle.get_context_messages("hello world", render_mode="raw_rendered")

    assert structured["facts"] == []
    assert rendered == "rendered context"
    assert messages["meta"]["render_mode"] == "raw_rendered"
    assert seen == [
        ("structured", context, "hello world"),
        ("rendered", context, "hello world"),
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
        await handle.retrieve_structured_context("hello world")

def test_umamemory_retrieval_shims_are_removed() -> None:
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
            tenant_id="tenant-1",
            agent_id=memory.agent_id,
            request_id="req-workspace-inert",
            user_id="user:u1",
            workspace_id="workspace:alpha",
        )
    )

    ctx = await handle.retrieve_structured_context("hello world")
    owner_types = {getattr(chunk, "owner_type", None) for chunk in list(ctx.get("chunks") or [])}

    assert owner_types
    assert owner_types.issubset({"agent", "user"})
    assert "workspace" not in owner_types
    assert "system" not in owner_types
