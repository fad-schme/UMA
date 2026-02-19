import yaml
import pytest
from datetime import datetime

from uma.core.uma_memory import UMAMemory
from uma.core.utils.identity import normalize_user_id
from uma.types import Fact


async def fake_llm(messages=None, **kwargs):
    return "ok"


async def fake_embed(texts=None, **kwargs):
    texts = texts or []
    return [[0.1, 0.1, 0.1] for _ in texts]


@pytest.mark.asyncio
async def test_semantic_upsert_allows_multiple_objects_same_predicate(tmp_path):
    """
    Regression test:
    - We must NOT drop distinct objects for the same (owner, subject, predicate).
      e.g., user LIKES sushi AND user LIKES pizza should both persist and be retrievable.
    """
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
            "provider": "tests.test_semantic_upsert_multiple_objects:fake_embed",
            "model": "x",
            "dimension": 3,
        },
        "llm": {
            "provider": "tests.test_semantic_upsert_multiple_objects:fake_llm",
            "model": "x",
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
    owner_id = normalize_user_id(user_id)
    now = datetime.utcnow()

    sushi = Fact(
        id="fact_sushi",
        subject=owner_id,
        predicate="LIKES",
        object="sushi",
        created_at=now,
        updated_at=now,
        source_ids=[],
        confidence=0.8,
        owner_type="user",
        owner_id=owner_id,
        meta={},
        salience=0.9,
    )
    pizza = Fact(
        id="fact_pizza",
        subject=owner_id,
        predicate="LIKES",
        object="pizza",
        created_at=now,
        updated_at=now,
        source_ids=[],
        confidence=0.8,
        owner_type="user",
        owner_id=owner_id,
        meta={},
        salience=0.9,
    )

    await memory.semantic_core.upsert_fact(sushi, [1.0, 0.0, 0.0])
    await memory.semantic_core.upsert_fact(pizza, [0.9, 0.0, 0.0])

    facts = await memory.semantic_core.list_facts_for_owner(owner_type="user", owner_id=owner_id, limit=None)
    likes = [f for f in facts if f.subject == owner_id and f.predicate == "LIKES"]
    objects = {str(getattr(f, "object", "")) for f in likes}
    assert {"sushi", "pizza"}.issubset(objects)

    # Vector retrieval should be able to return both when k is large enough.
    found = await memory.semantic_core.search(
        query_embedding=[1.0, 0.0, 0.0],
        owner_type="user",
        owner_id=owner_id,
        k=10,
        offset=0,
        filters=None,
        query_text=None,
    )
    found_ids = {getattr(f, "id", None) for f in found}
    assert {"fact_sushi", "fact_pizza"}.issubset(found_ids)
