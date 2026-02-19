import yaml
import pytest
from datetime import datetime

from uma.core.uma_memory import UMAMemory
from uma.core.utils.identity import normalize_user_id
from uma.types import Episode
from uma.types import Fact
from uma.types import Skill


async def fake_llm(messages=None, **kwargs):
    return "ok"


async def fake_embed(texts=None, **kwargs):
    texts = texts or []
    return [[0.1, 0.1, 0.1] for _ in texts]


@pytest.mark.asyncio
async def test_rebuild_vector_indexes(tmp_path):
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
            "provider": "tests.test_rebuild_indexes:fake_embed",
            "model": "x",
            "dimension": 3,
        },
        "llm": {
            "provider": "tests.test_rebuild_indexes:fake_llm",
            "model": "x",
        },
        "retrieval": {
            "max_episodes": 5,
            "max_facts": 5,
            "max_skills": 5,
            "max_graph_items": 5,
        },
        "consolidation": {
            "enabled": True,
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
    embedding = [0.1, 0.1, 0.1]

    episode = Episode(
        id="ep-1",
        timestamp=datetime.utcnow(),
        summary="hello",
        user_id=user_id,
        owner_type="user",
        owner_id=user_id,
        raw="hello world",
        tags=["test"],
        embedding=embedding,
    )
    await memory.episodic_core.add_episode(episode, embedding)

    fact = Fact(
        id="fact_1",
        subject=normalize_user_id(user_id),
        predicate="prefers",
        object="coffee",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=[episode.id],
        confidence=0.9,
        owner_type="user",
        owner_id=normalize_user_id(user_id),
    )
    await memory.semantic_core.upsert_fact(fact, embedding)

    skill = Skill(
        id="skill-1",
        name="Make coffee",
        description="Brews a cup of coffee.",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        owner_type="user",
        owner_id=user_id,
        trigger_phrases=["coffee"],
        trigger_patterns=[],
        plan={"steps": ["boil", "brew"]},
        tools=["kettle"],
        example="Make coffee",
        meta={"tag": "demo"},
    )
    await memory.procedural_core.add_skill(skill, embedding)

    memory.episodic_core.vector_index().delete([episode.id])
    memory.semantic_core.vector_index().delete([fact.id])
    memory.procedural_core.vector_index().delete([skill.id])

    result = await memory.rebuild_vector_indexes(owner_type="user", owner_id=user_id)
    assert result["status"] in ("ok", "degraded")
    assert episode.id in memory.episodic_core.vector_index()._vectors
    assert fact.id in memory.semantic_core.vector_index()._vectors
    assert skill.id in memory.procedural_core.vector_index()._vectors
