from __future__ import annotations

import pytest

from uma.stores.base_sql_store import DEFAULT_TENANT_ID


@pytest.mark.asyncio
async def test_rlm_lane_recall_scopes_user_only(uma_memory, tmp_path):
    memory = uma_memory
    assert memory.agent_id, "test runtime must set agent_id"
    agent_doc = tmp_path / "agent_doc.txt"
    agent_doc.write_text(
        (
            "Agent KB document. It contains hello world but does not mention the recall cue. "
            "This sentence is padding to ensure strict chunk validation passes for ingestion in CI. "
            "Additional padding words to reach the minimum chunk length requirement.\n"
        ),
        encoding="utf-8",
    )
    user_doc = tmp_path / "user_doc.txt"
    user_doc.write_text(
        (
            "User document. Remember last time we talked about hello world and a private user note. "
            "This sentence is padding to ensure strict chunk validation passes for ingestion in CI. "
            "Additional padding words to reach the minimum chunk length requirement.\n"
        ),
        encoding="utf-8",
    )

    await memory.ingest_document(str(agent_doc), owner_type="agent", owner_id=memory.agent_id)
    await memory.ingest_document(str(user_doc), owner_type="user", owner_id="user:u1")

    ctx = await memory.set_context(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id=memory.agent_id,
        request_id="req-recall-user-only",
        user_id="user:u1",
        session_id="legacy-user:user:u1",
    ).retrieve_context(query_text="remember last time hello world")
    facts = ctx.get("facts") or []
    chunks = ctx.get("chunks") or []
    assert all(getattr(f, "owner_type", None) == "user" for f in facts)
    assert all(getattr(f, "owner_id", None) == "user:u1" for f in facts)
    assert all(getattr(c, "owner_type", None) == "user" for c in chunks)
    assert all(getattr(c, "owner_id", None) == "user:u1" for c in chunks)


@pytest.mark.asyncio
async def test_rlm_lane_kb_scopes_agent_and_user(uma_memory, tmp_path):
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

    ctx = await memory.set_context(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id=memory.agent_id,
        request_id="req-recall-kb",
        user_id="user:u1",
        session_id="legacy-user:user:u1",
    ).retrieve_context(query_text="hello world")
    chunks = list(ctx.get("chunks") or [])
    assert any(getattr(c, "owner_type", None) == "agent" for c in chunks)
    assert any(getattr(c, "owner_type", None) == "user" for c in chunks)
