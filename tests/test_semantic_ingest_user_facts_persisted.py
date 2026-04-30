import pytest

from uma.common.identity import normalize_user_id


@pytest.mark.asyncio
async def test_semantic_core_ingest_persists_multiple_user_facts(uma_memory):
    memory = uma_memory

    user_id = "user:123"
    owner_id = normalize_user_id(user_id)

    persisted = await memory.semantic_core.ingest(
        owner_id,
        "user likes sushi and pizza",
        extra_meta={"turn_id": "t1"},
    )
    assert persisted

    facts = await memory.semantic_core.list_facts_for_owner(owner_type="user", owner_id=owner_id, limit=None)
    likes = [f for f in facts if getattr(f, "owner_id", None) == owner_id and getattr(f, "predicate", "") == "LIKES"]
    assert likes and all(getattr(f, "subject", None) == "user" for f in likes)
    objects = {str(getattr(f, "object", "")) for f in likes}
    assert {"sushi", "pizza"}.issubset(objects)
