from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_process_turn_is_idempotent_by_turn_id(uma_memory):
    mem = uma_memory

    # Use an assistant reply that triggers deterministic fact extraction.
    await mem.process_turn(
        user_id="user:u1",
        user_msg="I like coffee.",
        assistant_reply="Good choice.",
        session_id="session-a",
    )
    await mem.process_turn(
        user_id="user:u1",
        user_msg="I like coffee.",
        assistant_reply="Good choice.",
        session_id="session-a",
    )

    # Episodes are appended per call even when the derived turn_id is identical.
    conn = mem._stores["episodic"]._conn()
    try:
        rows = mem._stores["episodic"]._query_all(
            conn,
            "SELECT COUNT(*) AS n FROM episodes WHERE owner_type=? AND owner_id=?",
            params=["user", "user:u1"],
            log_context="test_episode_count",
        )
        assert int(rows[0]["n"]) == 2
    finally:
        conn.close()

    # Semantic facts remain stable across retries because fact IDs are content-derived.
    conn = mem._stores["semantic"]._conn()
    try:
        rows = mem._stores["semantic"]._query_all(
            conn,
            "SELECT COUNT(*) AS n FROM facts WHERE owner_type=? AND owner_id=?",
            params=["user", "user:u1"],
            log_context="test_fact_count",
        )
        assert int(rows[0]["n"]) == 3
    finally:
        conn.close()
