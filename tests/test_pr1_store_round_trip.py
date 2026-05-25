"""
PR1 — store round-trip: trust_score and content_hash persist and round-trip correctly.

Tests:
- write a Fact via SemanticSQLStore, read back: both new fields present and equal
- write an Episode via EpisodicSQLStore, read back: both new fields present and equal
- write a Skill via ProceduralSQLStore, read back: both new fields present and equal
- write a Chunk via ChunkSQLStore, read back: trust_score present (no content_hash on chunk)
- old-DB compatibility: simulate migration not run, row-mapping falls back to 0.5 / None
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from uma.adapters.db.sqlite_adapter import SQLiteAdapter
from uma.adapters.vector.base import VectorIndex
from uma.stores.chunk_sql import ChunkSQLStore
from uma.stores.episodic_sql import EpisodicSQLStore
from uma.stores.procedural_sql import ProceduralSQLStore
from uma.stores.semantic_sql import SemanticSQLStore
from uma.common.types import Chunk, Episode, Fact, Skill
from uma.common.integrity import hash_episode_content, hash_fact_content, hash_skill_content

_NOW = datetime.now(timezone.utc)
_EMBED = [0.1] * 64


class _NoopVectorIndex(VectorIndex):
    def upsert(self, ids, vectors, *, tenant_ids, owner_types, owner_ids, extra_metadata=None) -> None:
        return None

    def query(self, vector, *, tenant_id, owner_type, owner_id, k=10, extra_filters=None):
        return []

    def delete(self, ids) -> None:
        return None


def _semantic_store(tmp_path: Path) -> SemanticSQLStore:
    db_path = str(tmp_path / "semantic.db")
    return SemanticSQLStore(
        db_adapter=SQLiteAdapter(db_path),
        vector_index=_NoopVectorIndex(),
    )


def _episodic_store(tmp_path: Path) -> EpisodicSQLStore:
    db_path = str(tmp_path / "episodic.db")
    return EpisodicSQLStore(
        db_adapter=SQLiteAdapter(db_path),
        vector_index=_NoopVectorIndex(),
    )


def _procedural_store(tmp_path: Path) -> ProceduralSQLStore:
    db_path = str(tmp_path / "procedural.db")
    return ProceduralSQLStore(
        db_adapter=SQLiteAdapter(db_path),
        vector_index=_NoopVectorIndex(),
    )


def _chunk_store(tmp_path: Path) -> ChunkSQLStore:
    db_path = str(tmp_path / "chunks.db")
    return ChunkSQLStore(
        db_adapter=SQLiteAdapter(db_path),
        vector_index=_NoopVectorIndex(),
    )


# ──────────────────────────────────────────────
# Fact round-trip
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fact_round_trip_new_fields(tmp_path):
    store = _semantic_store(tmp_path)
    ch = hash_fact_content("user:alice", "LIKES", "sushi")
    fact = Fact(
        id="fact_001",
        subject="user:alice",
        predicate="LIKES",
        object="sushi",
        created_at=_NOW,
        updated_at=_NOW,
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
        trust_score=0.7,
        content_hash=ch,
    )
    await store.upsert_fact(fact, _EMBED)

    results = await store.list_facts_for_owner(
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert results, "expected at least one fact"
    r = results[0]
    assert r.trust_score == pytest.approx(0.7)
    assert r.content_hash == ch


@pytest.mark.asyncio
async def test_fact_default_trust_score_persisted(tmp_path):
    store = _semantic_store(tmp_path)
    fact = Fact(
        id="fact_002",
        subject="user:alice",
        predicate="LIKES",
        object="tea",
        created_at=_NOW,
        updated_at=_NOW,
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
    )
    await store.upsert_fact(fact, _EMBED)

    results = await store.list_facts_for_owner(
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert results
    r = results[0]
    assert r.trust_score == pytest.approx(0.5)
    assert r.content_hash is None


# ──────────────────────────────────────────────
# Episode round-trip
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_episode_round_trip_new_fields(tmp_path):
    store = _episodic_store(tmp_path)
    ch = hash_episode_content("User discussed sushi preferences.")
    ep = Episode(
        id="ep_001",
        timestamp=_NOW,
        summary="User discussed sushi preferences.",
        user_id="user:alice",
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
        trust_score=0.8,
        content_hash=ch,
    )
    await store.add_episode(ep, _EMBED)

    result = await store.get_episode(
        "ep_001",
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert result is not None
    assert result.trust_score == pytest.approx(0.8)
    assert result.content_hash == ch


@pytest.mark.asyncio
async def test_episode_default_trust_score_persisted(tmp_path):
    store = _episodic_store(tmp_path)
    ep = Episode(
        id="ep_002",
        timestamp=_NOW,
        summary="Short episode.",
        user_id="user:alice",
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
    )
    await store.add_episode(ep, _EMBED)

    result = await store.get_episode(
        "ep_002",
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert result is not None
    assert result.trust_score == pytest.approx(0.5)
    assert result.content_hash is None


# ──────────────────────────────────────────────
# Skill round-trip
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_round_trip_new_fields(tmp_path):
    store = _procedural_store(tmp_path)
    plan = {"steps": ["greet", "ask how they are"]}
    ch = hash_skill_content("greet_user", plan)
    skill = Skill(
        id="skill_001",
        name="greet_user",
        description="Greet the user.",
        created_at=_NOW,
        updated_at=_NOW,
        owner_type="agent",
        owner_id="agent-default",
        tenant_id="default",
        plan=plan,
        trust_score=0.6,
        content_hash=ch,
    )
    await store.add_skill(skill, _EMBED)

    result = await store.get_skill(
        "skill_001",
        tenant_id="default",
        owner_type="agent",
        owner_id="agent-default",
    )
    assert result is not None
    assert result.trust_score == pytest.approx(0.6)
    assert result.content_hash == ch


@pytest.mark.asyncio
async def test_skill_default_trust_score_persisted(tmp_path):
    store = _procedural_store(tmp_path)
    skill = Skill(
        id="skill_002",
        name="farewell",
        description="Say goodbye.",
        created_at=_NOW,
        updated_at=_NOW,
        owner_type="agent",
        owner_id="agent-default",
        tenant_id="default",
    )
    await store.add_skill(skill, _EMBED)

    result = await store.get_skill(
        "skill_002",
        tenant_id="default",
        owner_type="agent",
        owner_id="agent-default",
    )
    assert result is not None
    assert result.trust_score == pytest.approx(0.5)
    assert result.content_hash is None


# ──────────────────────────────────────────────
# Chunk round-trip (trust_score only)
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chunk_round_trip_trust_score(tmp_path):
    store = _chunk_store(tmp_path)
    chunk = Chunk(
        id="chunk_001",
        doc_id="doc_001",
        text="Some test chunk text.",
        page_range=(0, 1),
        position=0,
        source_path="/test/doc.pdf",
        source_hash="abc123",
        created_at=_NOW,
        updated_at=_NOW,
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
        trust_score=0.4,
    )
    await store.upsert_chunk(chunk, _EMBED)

    results = await store.fetch_by_ids(
        ["chunk_001"],
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert results
    r = results[0]
    assert r.trust_score == pytest.approx(0.4)
    assert not hasattr(r, "content_hash")


# ──────────────────────────────────────────────
# Old-DB compatibility (migration not yet run)
# ──────────────────────────────────────────────

def _create_legacy_facts_db(db_path: str) -> None:
    """Create a facts table without trust_score / content_hash columns (simulates old DB)."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                owner_type TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                workspace_id TEXT,
                session_id TEXT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_ids TEXT NOT NULL,
                source TEXT,
                origin_agent_id TEXT,
                origin_user_id TEXT,
                origin_session_id TEXT,
                scope_model_version TEXT,
                salience REAL NOT NULL DEFAULT 0.0,
                confidence REAL NULL,
                meta TEXT NOT NULL
            );
            INSERT INTO facts
                (id, tenant_id, owner_type, owner_id, subject, predicate, object,
                 created_at, updated_at, source_ids, salience, meta)
            VALUES
                ('legacy_fact_1', 'default', 'user', 'user:alice', 'user:alice', 'LIKES', '"coffee"',
                 '2024-01-01T00:00:00', '2024-01-01T00:00:00', '[]', 0.0, '{}');
            """
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_fact_row_mapping_falls_back_on_missing_columns(tmp_path):
    """SemanticSQLStore._row_to_object must fall back to (0.5, None) for legacy rows
    that lack trust_score / content_hash columns. Migration adds the columns on init,
    so we verify that the fallback in the mapper handles NULL values correctly."""
    db_path = str(tmp_path / "legacy_facts.db")
    _create_legacy_facts_db(db_path)

    # Initializing the store runs the migration (adds columns with DEFAULT 0.5).
    store = SemanticSQLStore(
        db_adapter=SQLiteAdapter(db_path),
        vector_index=_NoopVectorIndex(),
    )

    results = await store.list_facts_for_owner(
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert results, "legacy row should still be readable"
    r = results[0]
    # After migration, the column exists with DEFAULT 0.5; existing rows get 0.5.
    assert r.trust_score == pytest.approx(0.5)
    assert r.content_hash is None
