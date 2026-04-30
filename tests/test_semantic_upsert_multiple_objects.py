from datetime import datetime

import pytest

from uma.common.identity import normalize_user_id
from uma.common.types import Fact


@pytest.mark.asyncio
async def test_semantic_upsert_allows_multiple_objects_same_predicate(uma_memory):
    """
    Regression test:
    - We must NOT drop distinct objects for the same (owner, subject, predicate).
      e.g., user LIKES sushi AND user LIKES pizza should both persist and be retrievable.
    """
    memory = uma_memory

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

    sushi_emb, pizza_emb = await memory.embedder.embed(["sushi", "pizza"])
    await memory.semantic_core.upsert_fact(sushi, sushi_emb)
    await memory.semantic_core.upsert_fact(pizza, pizza_emb)

    facts = await memory.semantic_core.list_facts_for_owner(owner_type="user", owner_id=owner_id, limit=None)
    likes = [f for f in facts if f.subject == owner_id and f.predicate == "LIKES"]
    objects = {str(getattr(f, "object", "")) for f in likes}
    assert {"sushi", "pizza"}.issubset(objects)

    # Vector retrieval should return the correct fact when queried near its embedding.
    found_sushi = await memory.semantic_core.search(
        query_embedding=sushi_emb,
        owner_type="user",
        owner_id=owner_id,
        k=10,
        offset=0,
        filters=None,
        query_text=None,
    )
    assert any(getattr(f, "id", None) == "fact_sushi" for f in found_sushi)

    found_pizza = await memory.semantic_core.search(
        query_embedding=pizza_emb,
        owner_type="user",
        owner_id=owner_id,
        k=10,
        offset=0,
        filters=None,
        query_text=None,
    )
    assert any(getattr(f, "id", None) == "fact_pizza" for f in found_pizza)
