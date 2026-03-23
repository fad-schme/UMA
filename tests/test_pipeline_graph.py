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
            extra_meta={"session_id": "session-a"},
        )
        await mem.process_turn(
            user_id="user:u1",
            user_msg="hello again",
            assistant_reply="user likes pizza.",
            extra_meta={"session_id": "session-a"},
        )

        adapter = getattr(mem.graph_core, "adapter", None)
        queries = getattr(adapter, "queries", None)
        assert isinstance(queries, list) and queries, "expected graph adapter to record cypher queries"

        cyphers = [q for q, _params in queries]
        assert any("HAS_EPISODE" in c for c in cyphers)
        assert any("MERGE (f:Fact" in c or "MERGE (f:Fact" in c.replace("\n", " ") for c in cyphers)
        assert any("PRECEDES" in c for c in cyphers), "expected temporal PRECEDES/FOLLOWS edges"
        assert any((params or {}).get("tenant_id") == "default" for _c, params in queries)
        assert any((params or {}).get("owner_type") == "user" for _c, params in queries)
        assert any((params or {}).get("scope_model_version") == "v2" for _c, params in queries)
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass
