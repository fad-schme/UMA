from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.common.identity import normalize_user_id
from uma.common.types import Chunk, Episode, Fact, OwnershipRef, Skill


@pytest.mark.asyncio
async def test_chunk_retrieval_is_tenant_scoped_for_identical_owner_tuple(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb_a = (await memory.embedder.embed(["tenant alpha chunk"]))[0]
    emb_b = (await memory.embedder.embed(["tenant beta chunk"]))[0]
    now = datetime.now(timezone.utc)

    await memory.chunk_core.upsert_chunk(
        Chunk(
            id="chunk_tenant_a",
            doc_id="doc-tenant-a",
            text="tenant alpha chunk",
            page_range=(1, 1),
            position=1,
            source_path="/tmp/a.txt",
            source_hash="hash-a",
            created_at=now,
            updated_at=now,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb_a,
    )
    await memory.chunk_core.upsert_chunk(
        Chunk(
            id="chunk_tenant_b",
            doc_id="doc-tenant-b",
            text="tenant beta chunk",
            page_range=(1, 1),
            position=1,
            source_path="/tmp/b.txt",
            source_hash="hash-b",
            created_at=now,
            updated_at=now,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb_b,
    )

    found = await memory.chunk_core.search_chunks(
        query_embedding=emb_a,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        k=5,
    )

    assert [chunk.id for chunk in found] == ["chunk_tenant_a"]
    assert memory.chunk_core.store.vector_index._metadata["chunk_tenant_a"]["tenant_id"] == "tenant-a"
    assert memory.chunk_core.store.vector_index._metadata["chunk_tenant_b"]["tenant_id"] == "tenant-b"


@pytest.mark.asyncio
async def test_procedural_retrieval_is_tenant_scoped_for_identical_owner_tuple(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb_a = (await memory.embedder.embed(["tenant alpha procedure"]))[0]
    emb_b = (await memory.embedder.embed(["tenant beta procedure"]))[0]
    now = datetime.now(timezone.utc)

    await memory.procedural_core.add_skill(
        Skill(
            id="skill_tenant_a",
            name="Tenant Alpha Procedure",
            description="alpha",
            created_at=now,
            updated_at=now,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            trigger_phrases=["alpha"],
            trigger_patterns=[],
            plan={"steps": ["a"]},
            tools=["tool-a"],
            example="alpha",
            meta={},
        ),
        emb_a,
    )
    await memory.procedural_core.add_skill(
        Skill(
            id="skill_tenant_b",
            name="Tenant Beta Procedure",
            description="beta",
            created_at=now,
            updated_at=now,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            trigger_phrases=["beta"],
            trigger_patterns=[],
            plan={"steps": ["b"]},
            tools=["tool-b"],
            example="beta",
            meta={},
        ),
        emb_b,
    )

    owner = OwnershipRef(tenant_id="tenant-a", owner_type="user", owner_id=owner_id)
    found = await memory.procedural_core.search(query_embedding=emb_a, owner=owner, k=5)

    assert [skill.id for skill in found] == ["skill_tenant_a"]
    assert memory.procedural_core.store.vector_index._metadata["skill_tenant_a"]["tenant_id"] == "tenant-a"
    assert memory.procedural_core.store.vector_index._metadata["skill_tenant_b"]["tenant_id"] == "tenant-b"


@pytest.mark.asyncio
async def test_semantic_store_list_and_fetch_are_tenant_scoped_at_durable_boundary(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb = (await memory.embedder.embed(["tenant scoped fact"]))[0]
    now = datetime.now(timezone.utc)

    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_tenant_a",
            subject=owner_id,
            predicate="LIKES",
            object="alpha",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_tenant_b",
            subject=owner_id,
            predicate="LIKES",
            object="beta",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb,
    )

    listed = await memory.semantic_core.store.list_facts_for_owner(
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        limit=None,
    )
    fetched = await memory.semantic_core.store.fetch_by_ids(
        ids=["fact_tenant_a", "fact_tenant_b"],
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
    )

    assert [fact.id for fact in listed] == ["fact_tenant_a"]
    assert [fact.id for fact in fetched] == ["fact_tenant_a"]
    assert memory.semantic_core.vector_index()._metadata["fact_tenant_a"]["tenant_id"] == "tenant-a"
    assert memory.semantic_core.vector_index()._metadata["fact_tenant_b"]["tenant_id"] == "tenant-b"


@pytest.mark.asyncio
async def test_semantic_vector_search_is_tenant_scoped_for_identical_owner_tuple(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb_a = (await memory.embedder.embed(["semantic tenant alpha"]))[0]
    emb_b = (await memory.embedder.embed(["semantic tenant beta"]))[0]
    now = datetime.now(timezone.utc)

    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_search_tenant_a",
            subject=owner_id,
            predicate="USES",
            object="alpha",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb_a,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_search_tenant_b",
            subject=owner_id,
            predicate="USES",
            object="beta",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        emb_b,
    )

    found = await memory.semantic_core.store.search(
        emb_a,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        k=5,
    )

    assert [fact.id for fact in found] == ["fact_search_tenant_a"]


@pytest.mark.asyncio
async def test_episodic_store_list_and_fetch_are_tenant_scoped_at_durable_boundary(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb = (await memory.embedder.embed(["tenant scoped episode"]))[0]
    now = datetime.now(timezone.utc)

    await memory.episodic_core.add_episode(
        Episode(
            id="episode_tenant_a",
            timestamp=now,
            summary="alpha",
            raw="alpha",
            tags=[],
            embedding=emb,
            meta={},
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            user_id=owner_id,
        ),
        emb,
    )
    await memory.episodic_core.add_episode(
        Episode(
            id="episode_tenant_b",
            timestamp=now,
            summary="beta",
            raw="beta",
            tags=[],
            embedding=emb,
            meta={},
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            user_id=owner_id,
        ),
        emb,
    )

    listed = await memory.episodic_core.store.list_episodes("tenant-a", "user", owner_id)
    fetched = await memory.episodic_core.store.fetch_by_ids(
        ["episode_tenant_a", "episode_tenant_b"],
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
    )

    assert [episode.id for episode in listed] == ["episode_tenant_a"]
    assert [episode.id for episode in fetched] == ["episode_tenant_a"]
    assert memory.episodic_core.vector_index()._metadata["episode_tenant_a"]["tenant_id"] == "tenant-a"
    assert memory.episodic_core.vector_index()._metadata["episode_tenant_b"]["tenant_id"] == "tenant-b"


@pytest.mark.asyncio
async def test_episodic_vector_search_is_tenant_scoped_for_identical_owner_tuple(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    emb_a = (await memory.embedder.embed(["episodic tenant alpha"]))[0]
    emb_b = (await memory.embedder.embed(["episodic tenant beta"]))[0]
    now = datetime.now(timezone.utc)

    await memory.episodic_core.add_episode(
        Episode(
            id="episode_search_tenant_a",
            timestamp=now,
            summary="alpha",
            raw="alpha",
            tags=[],
            embedding=emb_a,
            meta={},
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            user_id=owner_id,
        ),
        emb_a,
    )
    await memory.episodic_core.add_episode(
        Episode(
            id="episode_search_tenant_b",
            timestamp=now,
            summary="beta",
            raw="beta",
            tags=[],
            embedding=emb_b,
            meta={},
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            user_id=owner_id,
        ),
        emb_b,
    )

    found = await memory.episodic_core.store.search(
        emb_a,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        k=5,
    )

    assert [episode.id for episode in found] == ["episode_search_tenant_a"]


@pytest.mark.asyncio
async def test_search_ids_requires_tenant_scope_filters(uma_memory) -> None:
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    embedding = (await memory.embedder.embed(["search ids tenant"]))[0]
    now = datetime.now(timezone.utc)

    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_search_ids_tenant",
            subject=owner_id,
            predicate="USES",
            object="tenant-search-ids",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            meta={},
        ),
        embedding,
    )

    with pytest.raises(ValueError, match="tenant_id, owner_type and owner_id"):
        await memory.semantic_core.store.search_ids(
            embedding,
            filters={"owner_type": "user", "owner_id": owner_id},
            log_context="missing_tenant_search_ids",
        )

    found = await memory.semantic_core.store.search_ids(
        embedding,
        filters={"tenant_id": "tenant-a", "owner_type": "user", "owner_id": owner_id},
        log_context="tenant_search_ids",
        id_prefix="fact_",
    )

    assert [fact_id for fact_id, _score in found] == ["fact_search_ids_tenant"]


@pytest.mark.asyncio
async def test_low_level_store_reads_fail_clearly_without_explicit_tenant(uma_memory) -> None:
    memory = uma_memory

    with pytest.raises(ValueError, match="tenant_id"):
        await memory.chunk_core.store.fetch_by_ids(
            ["missing"],
            owner_type="user",
            owner_id=normalize_user_id("user:u1"),
        )

    with pytest.raises(ValueError, match="tenant_id"):
        await memory.semantic_core.store.list_facts_for_owner(
            owner_type="user",
            owner_id=normalize_user_id("user:u1"),
            limit=None,
        )

    with pytest.raises(ValueError, match="tenant_id"):
        await memory.episodic_core.store.list_episodes(
            owner_type="user",
            owner_id=normalize_user_id("user:u1"),
        )

    with pytest.raises(ValueError, match="tenant_id"):
        await memory.procedural_core.store.list_skills(
            owner_type="user",
            owner_id=normalize_user_id("user:u1"),
        )
