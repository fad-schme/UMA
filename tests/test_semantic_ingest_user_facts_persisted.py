import yaml
import pytest

from uma.core.uma_memory import UMAMemory
from uma.core.utils.identity import normalize_user_id


async def fake_llm(messages=None, **_kwargs):
    # Must match FactExtractor.extract_user_facts schema.
    return (
        "{"
        '"facts": ['
        '{"predicate":"LIKES","object":"sushi","confidence":0.9,"source_ids":[]},'
        '{"predicate":"LIKES","object":"pizza","confidence":0.9,"source_ids":[]}'
        "]}"
    )


async def fake_embed(texts=None, **_kwargs):
    texts = texts or []
    # Deterministic, valid shape.
    return [[0.1, 0.1, 0.1] for _ in texts]


@pytest.mark.asyncio
async def test_semantic_core_ingest_persists_multiple_user_facts(tmp_path):
    db_root = tmp_path / "db"
    db_root.mkdir()

    cfg = {
        "storage": {
            "db_root": str(db_root) + "/",
            "sql_backend": "sqlite",
            "vector_backend": "inmemory",
            "graph_backend": "disabled",
        },
        "working_memory": {
            "max_tokens": 512,
            "warning_ratio": 0.7,
            "hard_limit_ratio": 0.95,
            "chunk_size": 10,
        },
        "embedding": {
            "provider": "tests.test_semantic_ingest_user_facts_persisted:fake_embed",
            "model": "x",
            "dimension": 3,
        },
        "llm": {
            "provider": "tests.test_semantic_ingest_user_facts_persisted:fake_llm",
            "model": "x",
        },
        "semantic": {
            "salience_threshold": 0.1,
        },
        "retrieval": {
            "max_episodes": 5,
            "max_facts": 10,
            "max_skills": 5,
            "max_graph_items": 1,
        },
        "consolidation": {
            "enabled": False,
            "cluster_similarity": 0.75,
            "max_episodes_per_cycle": 50,
            "prune_min_fact_salience": 0.2,
        },
        "features": {
            "load": [],
        },
    }

    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    memory = UMAMemory.from_yaml(str(cfg_path))
    memory._ensure_ingestion_ready()

    user_id = "user:123"
    subject = normalize_user_id(user_id)

    persisted = await memory.semantic_core.ingest(subject, "user likes sushi and pizza", extra_meta={"turn_id": "t1"})
    assert persisted

    facts = await memory.semantic_core.list_facts_for_owner(owner_type="user", owner_id=subject, limit=None)
    likes = [f for f in facts if getattr(f, "subject", None) == subject and getattr(f, "predicate", "") == "LIKES"]
    objects = {str(getattr(f, "object", "")) for f in likes}
    assert {"sushi", "pizza"}.issubset(objects)
