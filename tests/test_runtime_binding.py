from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from uma import UMAMemory
from uma.api.runtime import UMARequestHandle, UMARuntime
from uma.common.types import RuntimeContext


def test_runtime_bind_returns_immutable_handle() -> None:
    runtime = UMARuntime()
    context = RuntimeContext(
        tenant_id="tenant-1",
        agent_id="agent-1",
        request_id="req-1",
        user_id="user-1",
        workspace_id="workspace-1",
        session_id="session-1",
    )

    handle = runtime.bind(context)

    assert isinstance(handle, UMARequestHandle)
    assert handle.context == context
    assert handle.tenant_id == "tenant-1"
    assert handle.agent_id == "agent-1"
    assert handle.request_id == "req-1"
    assert handle.user_id == "user-1"
    assert handle.workspace_id == "workspace-1"
    assert handle.session_id == "session-1"

    with pytest.raises(FrozenInstanceError):
        handle.context = RuntimeContext(tenant_id="tenant-2", agent_id="agent-2", request_id="req-2")  # type: ignore[misc]


def test_multiple_handles_can_bind_distinct_contexts_safely() -> None:
    runtime = UMARuntime(metadata={"name": "shared"})

    handle_a = runtime.bind(
        RuntimeContext(
            tenant_id="tenant-1",
            agent_id="agent-a",
            request_id="req-a",
            user_id="user-1",
            session_id="session-a",
        )
    )
    handle_b = runtime.bind(
        RuntimeContext(
            tenant_id="tenant-1",
            agent_id="agent-b",
            request_id="req-b",
            user_id="user-2",
            session_id="session-b",
        )
    )

    assert handle_a.runtime is runtime
    assert handle_b.runtime is runtime
    assert handle_a.context != handle_b.context
    assert handle_a.agent_id == "agent-a"
    assert handle_b.agent_id == "agent-b"
    assert handle_a.user_id == "user-1"
    assert handle_b.user_id == "user-2"


def test_runtime_does_not_mutate_when_binding_distinct_contexts() -> None:
    runtime = UMARuntime(metadata={"source": "test"}, stores={"semantic": object()})
    before_metadata = dict(runtime.metadata)
    before_stores = dict(runtime.stores)

    runtime.bind(RuntimeContext(tenant_id="tenant-1", agent_id="agent-a", request_id="req-a"))
    runtime.bind(RuntimeContext(tenant_id="tenant-2", agent_id="agent-b", request_id="req-b"))

    assert runtime.metadata == before_metadata
    assert runtime.stores == before_stores
    assert not hasattr(runtime, "context")
    assert not hasattr(runtime, "request_id")
    assert runtime.agent_id is None
    assert not hasattr(runtime, "session_id")
    assert not hasattr(runtime, "user_id")


def test_runtime_bind_requires_runtime_context() -> None:
    runtime = UMARuntime()
    with pytest.raises(TypeError, match="RuntimeContext"):
        runtime.bind("not-a-context")  # type: ignore[arg-type]


def test_runtime_constructor_preserves_shared_infrastructure_references() -> None:
    stores = {"semantic": object()}
    features = {"procedural": object()}
    runtime = UMARuntime(
        config=object(),
        raw_config=object(),
        stores=stores,
        llm=object(),
        agent_llm=object(),
        embedder=object(),
        document_store=object(),
        graph_service=object(),
        ranking_service=object(),
        feature_registry=features,
        metadata={"version": "test"},
    )

    assert runtime.stores.keys() == stores.keys()
    assert runtime.feature_registry.keys() == features.keys()
    assert runtime.metadata["version"] == "test"


def test_runtime_from_memory_coexists_with_umamemory_fixture(uma_memory) -> None:
    runtime = UMARuntime.from_memory(uma_memory)

    assert runtime.memory_bridge is uma_memory
    assert runtime.config is getattr(uma_memory, "cfg", None)
    assert runtime.raw_config is getattr(uma_memory, "raw_config", None)
    assert runtime.document_store is getattr(uma_memory, "document_store", None)
    assert runtime.graph_service is getattr(uma_memory, "graph_core", None)
    assert runtime.metadata.get("source") == "UMAMemory"

    handle = runtime.bind(
        RuntimeContext(
            tenant_id="tenant-test",
            agent_id="agent-test",
            request_id="req-test",
            user_id="user-test",
            session_id="session-test",
        )
    )

    assert handle.runtime is runtime
    assert handle.context.agent_id == "agent-test"
