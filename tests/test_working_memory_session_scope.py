from __future__ import annotations

import pytest

from uma import UMARuntime
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.core.working_memory.core import legacy_session_scope_for_user
from uma.types import RuntimeContext, SessionScope


@pytest.mark.asyncio
async def test_bound_retrieval_uses_only_current_session_working_memory(uma_memory) -> None:
    memory = uma_memory
    assert memory.working_memory is not None
    assert memory.agent_id

    scope_a = SessionScope(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id=memory.agent_id,
        session_id="session-a",
        user_id="user:u1",
    )
    scope_b = SessionScope(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id=memory.agent_id,
        session_id="session-b",
        user_id="user:u1",
    )
    memory.working_memory.append(scope=scope_a, role="user", content="alpha memory")
    memory.working_memory.append(scope=scope_b, role="user", content="beta memory")

    runtime = UMARuntime.from_memory(memory)
    ctx = await runtime.bind(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=memory.agent_id,
            request_id="req-session-a",
            user_id="user:u1",
            session_id="session-a",
        )
    ).retrieve_structured_context("hello world")

    wm_contents = [msg.content for msg in ctx["working_memory"]]
    assert wm_contents == ["alpha memory"]


@pytest.mark.asyncio
async def test_bound_retrieval_without_session_does_not_fallback_to_broad_working_memory(uma_memory) -> None:
    memory = uma_memory
    assert memory.working_memory is not None
    assert memory.agent_id

    legacy_scope = SessionScope(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id=memory.agent_id,
        session_id="legacy-user:user:u1",
        user_id="user:u1",
    )
    memory.working_memory.append(scope=legacy_scope, role="user", content="legacy memory")

    runtime = UMARuntime.from_memory(memory)
    ctx = await runtime.bind(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=memory.agent_id,
            request_id="req-no-session",
            user_id="user:u1",
        )
    ).retrieve_structured_context("hello world")

    assert ctx["working_memory"] == []


@pytest.mark.asyncio
async def test_process_turn_uses_explicit_session_scope_for_working_memory(tmp_path) -> None:
    from tests.helpers.runtime import init_uma_for_tests

    memory = await init_uma_for_tests(tmp_path, agent_id="agent-wm")
    try:
        await memory.process_turn(
            user_id="user:u1",
            user_msg="first",
            assistant_reply="reply one",
            extra_meta={"session_id": "session-a"},
        )
        await memory.process_turn(
            user_id="user:u1",
            user_msg="second",
            assistant_reply="reply two",
            extra_meta={"session_id": "session-b"},
        )

        assert memory.working_memory is not None
        scope_a = SessionScope(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-wm",
            session_id="session-a",
            user_id="user:u1",
        )
        scope_b = SessionScope(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-wm",
            session_id="session-b",
            user_id="user:u1",
        )

        contents_a = [msg.content for msg in memory.working_memory.get_context(scope_a)]
        contents_b = [msg.content for msg in memory.working_memory.get_context(scope_b)]

        assert "first" in contents_a
        assert "reply one" in contents_a
        assert "second" not in contents_a
        assert "reply two" not in contents_a
        assert "second" in contents_b
        assert "reply two" in contents_b
    finally:
        memory.shutdown()


@pytest.mark.asyncio
async def test_process_turn_without_session_id_uses_explicit_legacy_wm_bridge_only(tmp_path) -> None:
    from tests.helpers.runtime import init_uma_for_tests

    memory = await init_uma_for_tests(tmp_path, agent_id="agent-wm")
    try:
        await memory.process_turn(
            user_id="user:u1",
            user_msg="legacy first",
            assistant_reply="legacy reply",
        )

        assert memory.working_memory is not None
        legacy_scope = legacy_session_scope_for_user(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-wm",
            user_id="user:u1",
        )

        contents = [msg.content for msg in memory.working_memory.get_context(legacy_scope)]
        assert "legacy first" in contents
        assert "legacy reply" in contents

        runtime = UMARuntime.from_memory(memory)
        ctx = await runtime.bind(
            RuntimeContext(
                tenant_id=DEFAULT_TENANT_ID,
                agent_id="agent-wm",
                request_id="req-no-session-bridge",
                user_id="user:u1",
            )
        ).retrieve_structured_context("legacy")

        assert ctx["working_memory"] == []
    finally:
        memory.shutdown()
