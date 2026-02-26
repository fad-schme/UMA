from __future__ import annotations

import pytest

from tests.helpers.runtime import init_uma_for_tests


@pytest.mark.asyncio
async def test_pipeline_updates_graph_with_facts_and_temporal_links(tmp_path):
    mem = await init_uma_for_tests(
        tmp_path,
        graph_backend="tests.helpers.graph_adapter:RecordingGraphAdapter",
        graph_config={},
    )
    try:
        await mem.process_turn(
            user_id="user:u1",
            user_msg="hello",
            assistant_reply="user likes sushi.",
        )
        await mem.process_turn(
            user_id="user:u1",
            user_msg="hello again",
            assistant_reply="user likes pizza.",
        )

        adapter = getattr(mem.graph_core, "adapter", None)
        queries = getattr(adapter, "queries", None)
        assert isinstance(queries, list) and queries, "expected graph adapter to record cypher queries"

        cyphers = [q for q, _params in queries]
        assert any("HAS_EPISODE" in c for c in cyphers)
        assert any("MERGE (f:Fact" in c or "MERGE (f:Fact" in c.replace("\n", " ") for c in cyphers)
        assert any("PRECEDES" in c for c in cyphers), "expected temporal PRECEDES/FOLLOWS edges"
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass
