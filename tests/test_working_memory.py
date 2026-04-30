from __future__ import annotations

import asyncio

from uma.adapters.llm.callable_adapter import CallableLLMAdapter
from uma.common.config_types import WorkingMemorySettings
from uma.memory.working_memory.core import WorkingMemoryCore, legacy_session_scope_for_user
from uma.common.types import SessionScope

from tests.helpers.providers import fake_llm


class _MemoryClient:
    def __init__(self, wm_cfg: WorkingMemorySettings) -> None:
        self.working_memory_cfg = wm_cfg


def test_working_memory_chunked_compaction_produces_summary():
    llm = CallableLLMAdapter(callable_fn=fake_llm, name="tests.fake_llm")
    wm_cfg = WorkingMemorySettings(
        max_tokens=120,
        warning_ratio=0.2,
        hard_limit_ratio=0.9,
        chunk_size=2,
        keep_recent_messages=1,
        keep_recent_token_fraction=0.0,
    )
    mem = _MemoryClient(wm_cfg)
    wm = WorkingMemoryCore(llm=llm, memory_client=mem)

    scope = SessionScope(
        tenant_id="tenant-1",
        agent_id="agent-a",
        session_id="session-1",
        user_id="user:u1",
    )
    for i in range(6):
        wm.append(scope=scope, role="user", content=f"msg {i} " + "word " * 8)

    asyncio.run(wm.compact(scope=scope))

    ctx = wm.get_context(scope)
    assert ctx
    assert ctx[0].role == "summary"


def test_working_memory_emergency_prune_keeps_recent_messages():
    llm = CallableLLMAdapter(callable_fn=fake_llm, name="tests.fake_llm")
    wm_cfg = WorkingMemorySettings(
        max_tokens=20,
        warning_ratio=0.1,
        hard_limit_ratio=0.9,
        chunk_size=2,
        keep_recent_messages=1,
        keep_recent_token_fraction=0.0,
    )
    mem = _MemoryClient(wm_cfg)
    wm = WorkingMemoryCore(llm=llm, memory_client=mem)

    scope = SessionScope(
        tenant_id="tenant-1",
        agent_id="agent-a",
        session_id="session-2",
        user_id="user:u2",
    )
    for _i in range(12):
        wm.append(scope=scope, role="user", content="word " * 10)

    asyncio.run(wm.compact(scope=scope))

    ctx = wm.get_context(scope)
    assert len(ctx) == 10
    assert all(msg.role != "summary" for msg in ctx)


def test_working_memory_isolated_across_sessions_for_same_user_and_agent():
    llm = CallableLLMAdapter(callable_fn=fake_llm, name="tests.fake_llm")
    wm_cfg = WorkingMemorySettings(
        max_tokens=120,
        warning_ratio=0.2,
        hard_limit_ratio=0.9,
        chunk_size=2,
        keep_recent_messages=1,
        keep_recent_token_fraction=0.0,
    )
    mem = _MemoryClient(wm_cfg)
    wm = WorkingMemoryCore(llm=llm, memory_client=mem)

    scope_a = SessionScope(
        tenant_id="tenant-1",
        agent_id="agent-a",
        session_id="session-a",
        user_id="user:u1",
    )
    scope_b = SessionScope(
        tenant_id="tenant-1",
        agent_id="agent-a",
        session_id="session-b",
        user_id="user:u1",
    )

    wm.append(scope=scope_a, role="user", content="alpha session")
    wm.append(scope=scope_b, role="user", content="beta session")

    assert [msg.content for msg in wm.get_context(scope_a)] == ["alpha session"]
    assert [msg.content for msg in wm.get_context(scope_b)] == ["beta session"]


def test_working_memory_isolated_across_agents_for_same_user_and_session_token():
    llm = CallableLLMAdapter(callable_fn=fake_llm, name="tests.fake_llm")
    wm_cfg = WorkingMemorySettings(
        max_tokens=120,
        warning_ratio=0.2,
        hard_limit_ratio=0.9,
        chunk_size=2,
        keep_recent_messages=1,
        keep_recent_token_fraction=0.0,
    )
    mem = _MemoryClient(wm_cfg)
    wm = WorkingMemoryCore(llm=llm, memory_client=mem)

    scope_a = SessionScope(
        tenant_id="tenant-1",
        agent_id="agent-a",
        session_id="shared-token",
        user_id="user:u1",
    )
    scope_b = SessionScope(
        tenant_id="tenant-1",
        agent_id="agent-b",
        session_id="shared-token",
        user_id="user:u1",
    )

    wm.append(scope=scope_a, role="user", content="agent a only")
    wm.append(scope=scope_b, role="user", content="agent b only")

    assert [msg.content for msg in wm.get_context(scope_a)] == ["agent a only"]
    assert [msg.content for msg in wm.get_context(scope_b)] == ["agent b only"]


def test_working_memory_isolated_across_tenants():
    llm = CallableLLMAdapter(callable_fn=fake_llm, name="tests.fake_llm")
    wm_cfg = WorkingMemorySettings(
        max_tokens=120,
        warning_ratio=0.2,
        hard_limit_ratio=0.9,
        chunk_size=2,
        keep_recent_messages=1,
        keep_recent_token_fraction=0.0,
    )
    mem = _MemoryClient(wm_cfg)
    wm = WorkingMemoryCore(llm=llm, memory_client=mem)

    scope_a = SessionScope(tenant_id="tenant-a", agent_id="agent-a", session_id="session-1", user_id="user:u1")
    scope_b = SessionScope(tenant_id="tenant-b", agent_id="agent-a", session_id="session-1", user_id="user:u1")

    wm.append(scope=scope_a, role="user", content="tenant a")
    wm.append(scope=scope_b, role="user", content="tenant b")

    assert [msg.content for msg in wm.get_context(scope_a)] == ["tenant a"]
    assert [msg.content for msg in wm.get_context(scope_b)] == ["tenant b"]


def test_legacy_session_scope_bridge_is_agent_scoped():
    scope_a = legacy_session_scope_for_user(
        tenant_id="tenant-1",
        agent_id="agent-a",
        user_id="user:u1",
    )
    scope_b = legacy_session_scope_for_user(
        tenant_id="tenant-1",
        agent_id="agent-b",
        user_id="user:u1",
    )

    assert scope_a.session_id.startswith("legacy-user:")
    assert scope_a.session_id == scope_b.session_id
    assert scope_a.agent_id != scope_b.agent_id


def test_compaction_only_mutates_current_session_bucket():
    llm = CallableLLMAdapter(callable_fn=fake_llm, name="tests.fake_llm")
    wm_cfg = WorkingMemorySettings(
        max_tokens=120,
        warning_ratio=0.2,
        hard_limit_ratio=0.9,
        chunk_size=2,
        keep_recent_messages=1,
        keep_recent_token_fraction=0.0,
    )
    mem = _MemoryClient(wm_cfg)
    wm = WorkingMemoryCore(llm=llm, memory_client=mem)

    scope_a = SessionScope(tenant_id="tenant-1", agent_id="agent-a", session_id="session-a", user_id="user:u1")
    scope_b = SessionScope(tenant_id="tenant-1", agent_id="agent-a", session_id="session-b", user_id="user:u1")

    for i in range(6):
        wm.append(scope=scope_a, role="user", content=f"a{i} " + "word " * 8)
    wm.append(scope=scope_b, role="user", content="session b untouched")

    asyncio.run(wm.compact(scope=scope_a))

    assert wm.get_context(scope_a)[0].role == "summary"
    assert [msg.content for msg in wm.get_context(scope_b)] == ["session b untouched"]


def test_reset_only_clears_current_session_bucket():
    llm = CallableLLMAdapter(callable_fn=fake_llm, name="tests.fake_llm")
    wm_cfg = WorkingMemorySettings(
        max_tokens=120,
        warning_ratio=0.2,
        hard_limit_ratio=0.9,
        chunk_size=2,
        keep_recent_messages=1,
        keep_recent_token_fraction=0.0,
    )
    mem = _MemoryClient(wm_cfg)
    wm = WorkingMemoryCore(llm=llm, memory_client=mem)

    scope_a = SessionScope(tenant_id="tenant-1", agent_id="agent-a", session_id="session-a", user_id="user:u1")
    scope_b = SessionScope(tenant_id="tenant-1", agent_id="agent-a", session_id="session-b", user_id="user:u1")

    wm.append(scope=scope_a, role="user", content="session a message")
    wm.append(scope=scope_b, role="user", content="session b message")

    wm.reset(scope_a)

    assert wm.get_context(scope_a) == []
    assert [msg.content for msg in wm.get_context(scope_b)] == ["session b message"]
