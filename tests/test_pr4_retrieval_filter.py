"""
test_pr4_retrieval_filter.py
==============================
Verifies that quarantined records are silently excluded from all retrieval
paths: vector search (fetch_by_ids), list_*, and lexical search.
Active records with the same scope must still be returned.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.adapters.db.sqlite_adapter import SQLiteAdapter
from uma.adapters.vector.base import VectorIndex
from uma.stores.semantic_sql import SemanticSQLStore
from uma.stores.episodic_sql import EpisodicSQLStore
from uma.stores.procedural_sql import ProceduralSQLStore
from uma.stores.chunk_sql import ChunkSQLStore
from uma.common.types import Fact, Episode, Skill, Chunk


class _NoopVI(VectorIndex):
    def upsert(self, ids, vectors, *, tenant_ids, owner_types, owner_ids, extra_metadata=None): pass
    def query(self, vector, *, tenant_id, owner_type, owner_id, k=10, extra_filters=None): return []
    def delete(self, ids): pass


_NOW = datetime.now(timezone.utc)
_VEC = [0.1, 0.2, 0.3, 0.4]
_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:alice")


def _semantic_store(tmp_path):
    return SemanticSQLStore(
        db_adapter=SQLiteAdapter(str(tmp_path / "s.db")),
        vector_index=_NoopVI(),
    )

def _episodic_store(tmp_path):
    return EpisodicSQLStore(
        db_adapter=SQLiteAdapter(str(tmp_path / "e.db")),
        vector_index=_NoopVI(),
    )

def _procedural_store(tmp_path):
    return ProceduralSQLStore(
        db_adapter=SQLiteAdapter(str(tmp_path / "p.db")),
        vector_index=_NoopVI(),
    )

def _chunk_store(tmp_path):
    return ChunkSQLStore(
        db_adapter=SQLiteAdapter(str(tmp_path / "c.db")),
        vector_index=_NoopVI(),
    )


# ---------------------------------------------------------------------------
# Semantic: quarantined excluded, active visible
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_mixed_only_active_returned(tmp_path):
    store = _semantic_store(tmp_path)

    active = Fact(
        id="fact_active", subject="user:alice", predicate="likes", object="coffee",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.7,
    )
    quarantined = Fact(
        id="fact_quarantined", subject="user:alice", predicate="uses", object="exploit",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.upsert_fact(active, _VEC)
    await store.upsert_fact(quarantined, _VEC)

    rows = await store.list_facts_for_owner(**_SCOPE)
    ids = {r.id for r in rows}
    assert "fact_active" in ids
    assert "fact_quarantined" not in ids


@pytest.mark.asyncio
async def test_semantic_fetch_by_ids_omits_quarantined(tmp_path):
    store = _semantic_store(tmp_path)
    active = Fact(
        id="fa", subject="user:alice", predicate="p", object="v",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.8,
    )
    qed = Fact(
        id="fq", subject="user:alice", predicate="p2", object="v2",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.upsert_fact(active, _VEC)
    await store.upsert_fact(qed, _VEC)

    fetched = await store.fetch_by_ids(["fa", "fq"], **_SCOPE)
    assert [r.id for r in fetched] == ["fa"]


@pytest.mark.asyncio
async def test_semantic_lexical_search_matches_turn_facts_with_owner_scope(tmp_path):
    store = _semantic_store(tmp_path)
    scope_a = dict(tenant_id="default", owner_type="user", owner_id="user:A")
    scope_b = dict(tenant_id="default", owner_type="user", owner_id="user:B")

    fact_a = Fact(
        id="fact_a",
        subject="user:A",
        predicate="current projects or research topics",
        object="adoption agencies",
        created_at=_NOW,
        updated_at=_NOW,
        **scope_a,
        trust_score=0.8,
    )
    fact_b = Fact(
        id="fact_b",
        subject="user:B",
        predicate="current projects or research topics",
        object="adoption agencies",
        created_at=_NOW,
        updated_at=_NOW,
        **scope_b,
        trust_score=0.8,
    )
    await store.upsert_fact(fact_a, _VEC)
    await store.upsert_fact(fact_b, _VEC)

    results = await store.lexical_search("adoption agencies", **scope_a, k=10)
    ids = {row.id for row in results}
    assert "fact_a" in ids
    assert "fact_b" not in ids


# ---------------------------------------------------------------------------
# Episodic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_episodic_list_recent_excludes_quarantined(tmp_path):
    store = _episodic_store(tmp_path)

    ep_ok = Episode(
        id="ep_ok", timestamp=_NOW, summary="clean message",
        user_id="user:alice", **_SCOPE, trust_score=0.7,
    )
    ep_bad = Episode(
        id="ep_bad", timestamp=_NOW, summary="poisoned",
        user_id="user:alice", **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.add_episode(ep_ok, _VEC)
    await store.add_episode(ep_bad, _VEC)

    recent = await store.list_recent(tenant_id="default", owner_type="user", owner_id="user:alice", n=10)
    ids = {r.id for r in recent}
    assert "ep_ok" in ids
    assert "ep_bad" not in ids


@pytest.mark.asyncio
async def test_episodic_fetch_by_ids_excludes_quarantined(tmp_path):
    store = _episodic_store(tmp_path)
    ep = Episode(
        id="ep_q", timestamp=_NOW, summary="bad",
        user_id="user:alice", **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.add_episode(ep, _VEC)

    result = await store.fetch_by_ids(["ep_q"], tenant_id="default", owner_type="user", owner_id="user:alice")
    assert result == []


# ---------------------------------------------------------------------------
# Procedural
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_procedural_list_excludes_quarantined(tmp_path):
    store = _procedural_store(tmp_path)
    sk_ok = Skill(
        id="sk_ok", name="clean_skill", description="fine", created_at=_NOW, updated_at=_NOW,
        **_SCOPE, trust_score=0.8,
    )
    sk_bad = Skill(
        id="sk_bad", name="bad_skill", description="poisoned", created_at=_NOW, updated_at=_NOW,
        **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.add_skill(sk_ok, _VEC)
    await store.add_skill(sk_bad, _VEC)

    skills = await store.list_skills(**_SCOPE)
    ids = {r.id for r in skills}
    assert "sk_ok" in ids
    assert "sk_bad" not in ids


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chunk_fetch_by_ids_excludes_quarantined(tmp_path):
    store = _chunk_store(tmp_path)
    ch = Chunk(
        id="ch_q", doc_id="doc1", text="bad content",
        page_range=(0, 1), position=0, source_path="x.pdf", source_hash="h",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.upsert_chunk(ch, _VEC)

    fetched = await store.fetch_by_ids(["ch_q"], **_SCOPE)
    assert fetched == []


@pytest.mark.asyncio
async def test_chunk_lexical_search_excludes_quarantined(tmp_path):
    store = _chunk_store(tmp_path)

    long_suffix = " " + ("x" * 60)  # pad to exceed lexical_search min_len=80
    ch_ok = Chunk(
        id="ch_ok", doc_id="doc2", text="the secret recipe for chocolate cake is vanilla" + long_suffix,
        page_range=(0, 1), position=0, source_path="x.pdf", source_hash="h2",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.7,
    )
    ch_bad = Chunk(
        id="ch_bad", doc_id="doc2", text="the secret recipe for chocolate cake is poison" + long_suffix,
        page_range=(1, 2), position=1, source_path="x.pdf", source_hash="h2",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.upsert_chunk(ch_ok, _VEC)
    await store.upsert_chunk(ch_bad, _VEC)

    results = await store.lexical_search("secret recipe chocolate cake", **_SCOPE, k=10)
    ids = {r.id for r in results}
    assert "ch_ok" in ids
    assert "ch_bad" not in ids
