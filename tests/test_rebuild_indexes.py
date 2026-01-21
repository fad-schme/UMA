import yaml
import pytest
from datetime import datetime

from uma.core.uma_memory import UMAMemory
from uma.core.utils.identity import ensure_user_subject
from uma.types_episode import Episode
from uma.types_fact import Fact
from uma.types_skill import Skill


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
            "dimension": 3,
        },
        "llm": {
            "provider": "tests.test_rebuild_indexes:fake_llm",
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
    memory.initialize()

    user_id = "user-123"
    embedding = [0.1, 0.1, 0.1]

    episode = Episode(
        id="ep-1",
        user_id=user_id,
        timestamp=datetime.utcnow(),
        summary="hello",
        raw="hello world",
        tags=["test"],
        embedding=embedding,
    )
    await memory.episodic_store.add_episode(episode, embedding)

    fact = Fact(
        id="fact-1",
        subject=ensure_user_subject(user_id),
        predicate="prefers",
        object="coffee",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=[episode.id],
        confidence=0.9,
    )
    await memory.semantic_store.upsert_fact(fact, embedding)

    skill = Skill(
        id="skill-1",
        name="Make coffee",
        trigger_phrases=["coffee"],
        trigger_patterns=[],
        plan={"steps": ["boil", "brew"]},
        tools=["kettle"],
        example="Make coffee",
        meta={"tag": "demo"},
    )
    await memory.procedural_store.add_skill(skill, embedding)

    memory.episodic_store.vector_index.delete([episode.id])
    memory.semantic_store.vector_index.delete([fact.id])
    memory.procedural_store.vector_index.delete([skill.id])

    result = await memory.rebuild_vector_indexes(user_id=user_id)
    assert result["status"] in ("ok", "degraded")
    assert episode.id in memory.episodic_store.vector_index._vectors
    assert fact.id in memory.semantic_store.vector_index._vectors
    assert skill.id in memory.procedural_store.vector_index._vectors
