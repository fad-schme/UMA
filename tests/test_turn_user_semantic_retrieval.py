from __future__ import annotations

import pytest

from tests.helpers.runtime import init_uma_for_tests


def _fact_objects(payload: dict) -> set[str]:
    return {
        str(item.get("object") or item.get("text") or "").lower()
        for item in list(payload.get("facts") or [])
        if isinstance(item, dict)
    }


@pytest.mark.asyncio
async def test_process_turn_user_message_becomes_retrievable(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)

    await memory.process_turn(
        user_id="user:u1",
        user_msg="I am researching adoption agencies and I am interested in counseling or mental health work.",
        assistant_reply="Thanks, I will remember that context.",
        session_id="session-user-turn-facts",
    )

    recalled_adoption = await memory.retrieve_memory(
        query_text="adoption agencies",
        user_id="user:u1",
        tenant_id="default",
        request_id="req-user-turn-adoption",
        session_id="session-user-turn-facts",
    )
    recalled_mental_health = await memory.retrieve_memory(
        query_text="mental health",
        user_id="user:u1",
        tenant_id="default",
        request_id="req-user-turn-mental-health",
        session_id="session-user-turn-facts",
    )

    assert any("adoption agenc" in obj for obj in _fact_objects(recalled_adoption))
    assert any("mental health" in obj or obj == "counseling" for obj in _fact_objects(recalled_mental_health))
