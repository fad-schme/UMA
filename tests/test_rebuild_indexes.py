from datetime import datetime

import pytest

from uma.core.utils.identity import normalize_user_id
from uma.types import Episode
from uma.types import Fact
from uma.types import Skill


@pytest.mark.asyncio
async def test_rebuild_vector_indexes(uma_memory):
    memory = uma_memory

    user_id = "user:123"
    owner_id = normalize_user_id(user_id)
    embedding = (await memory.embedder.embed(["hello"]))[0]

    episode = Episode(
        id="ep-1",
        timestamp=datetime.utcnow(),
        summary="hello",
        user_id=user_id,
        owner_type="user",
        owner_id=owner_id,
        raw="hello world",
        tags=["test"],
        embedding=embedding,
    )
    await memory.episodic_core.add_episode(episode, embedding)

    fact = Fact(
        id="fact_1",
        subject=owner_id,
        predicate="prefers",
        object="coffee",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=[episode.id],
        confidence=0.9,
        owner_type="user",
        owner_id=owner_id,
    )
    await memory.semantic_core.upsert_fact(fact, embedding)

    skill = Skill(
        id="skill_1",
        name="Make coffee",
        description="Brews a cup of coffee.",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        owner_type="user",
        owner_id=owner_id,
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

    result = await memory.rebuild_vector_indexes(owner_type="user", owner_id=owner_id)
    assert result["status"] in ("ok", "degraded")
    assert episode.id in memory.episodic_core.vector_index()._vectors
    assert fact.id in memory.semantic_core.vector_index()._vectors
    assert skill.id in memory.procedural_core.vector_index()._vectors
