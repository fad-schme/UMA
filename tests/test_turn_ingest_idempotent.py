import asyncio

from uma.core.uma_memory import UMAMemory
from uma.core.memory_config import UMAConfig


def _good_embedder(texts=None, **kwargs):
    texts = texts or []
    return [[0.0, 0.0, 0.0] for _ in texts]


def _good_llm(messages=None, **kwargs):
    # Return minimal valid JSON for Semantic FactExtractor.
    return '{"facts":[{"predicate":"LIKES","object":"coffee","confidence":0.9,"source_ids":[]}]}'


def test_process_turn_is_idempotent_by_turn_id(tmp_path):
    cfg = {
        "storage": {
            "db_root": str(tmp_path),
            "sql_backend": "sqlite",
            "vector_backend": "inmemory",
            "graph_backend": "disabled",
        },
        "working_memory": {"max_tokens": 100, "warning_ratio": 0.7, "hard_limit_ratio": 0.95, "chunk_size": 10},
        "embedding": {
            "provider": "tests.test_turn_ingest_idempotent:_good_embedder",
            "model": "x",
            "dimension": 3,
            "config": {"preflight": False},
        },
        "llm": {
            "provider": "tests.test_turn_ingest_idempotent:_good_llm",
            "model": "x",
            "config": {"preflight": False},
        },
        "retrieval": {"max_episodes": 1, "max_facts": 1, "max_skills": 1, "max_graph_items": 1},
        "consolidation": {"enabled": False, "cluster_similarity": 0.75, "max_episodes_per_cycle": 10, "prune_min_fact_salience": 0.2},
        "features": {"load": [], "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False}},
    }

    mem = UMAMemory(UMAConfig(cfg))
    mem.initialize()

    async def run():
        await mem.process_turn(user_id="user:u1", user_msg="hello", assistant_reply="hi")
        await mem.process_turn(user_id="user:u1", user_msg="hello", assistant_reply="hi")

    asyncio.run(run())

    # Expect only one episode row due to turn_id idempotency guard.
    conn = mem._stores["episodic"]._conn()
    try:
        rows = mem._stores["episodic"]._query_all(
            conn,
            "SELECT COUNT(*) AS n FROM episodes WHERE owner_type=? AND owner_id=?",
            params=["user", "user:u1"],
            log_context="test_episode_count",
        )
        assert int(rows[0]["n"]) == 1
    finally:
        conn.close()

    # Facts should also be idempotent on retries (no duplicate rows).
    conn = mem._stores["semantic"]._conn()
    try:
        rows = mem._stores["semantic"]._query_all(
            conn,
            "SELECT COUNT(*) AS n FROM facts WHERE owner_type=? AND owner_id=?",
            params=["user", "user:u1"],
            log_context="test_fact_count",
        )
        assert int(rows[0]["n"]) == 1
    finally:
        conn.close()
