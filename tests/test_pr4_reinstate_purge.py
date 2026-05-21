"""
test_pr4_reinstate_purge.py
==============================
Focused tests for reinstate and purge operations across all four store types
(semantic, episodic, procedural, raw/chunk).

Covers:
  - reinstate_quarantined_record directly on each store
  - purge via management API (delete_fact / delete_episode / delete_skill / delete_chunk)
  - audit log is appended on reinstate
  - reinstated record is visible in normal retrieval
  - purged record is gone from all views
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
    def upsert(self, ids, vectors, metadata=None): pass
    def query(self, vector, k=10, filters=None): return []
    def delete(self, ids): pass


_NOW = datetime.now(timezone.utc)
_VEC = [0.1, 0.2, 0.3, 0.4]
_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:test")
_AUDIT = {"action": "reinstate", "reason": "false positive", "reinstated_at": _NOW.isoformat()}


def _sem(tmp_path):
    return SemanticSQLStore(SQLiteAdapter(str(tmp_path / "s.db")), _NoopVI())

def _ep(tmp_path):
    return EpisodicSQLStore(SQLiteAdapter(str(tmp_path / "e.db")), _NoopVI())

def _proc(tmp_path):
    return ProceduralSQLStore(SQLiteAdapter(str(tmp_path / "p.db")), _NoopVI())

def _chunk(tmp_path):
    return ChunkSQLStore(SQLiteAdapter(str(tmp_path / "c.db")), _NoopVI())


# ---------------------------------------------------------------------------
# Semantic: reinstate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_reinstate_makes_fact_active(tmp_path):
    store = _sem(tmp_path)
    fact = Fact(
        id="f1", subject="user:test", predicate="p", object="v",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.upsert_fact(fact, _VEC)

    ok = await store.reinstate_quarantined_record("f1", **_SCOPE, audit_entry=_AUDIT)
    assert ok is True

    facts = await store.list_facts_for_owner(**_SCOPE)
    assert any(f.id == "f1" for f in facts)


@pytest.mark.asyncio
async def test_semantic_reinstate_appends_audit_log(tmp_path):
    store = _sem(tmp_path)
    fact = Fact(
        id="f2", subject="user:test", predicate="p", object="v",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.upsert_fact(fact, _VEC)
    await store.reinstate_quarantined_record("f2", **_SCOPE, audit_entry=_AUDIT)

    facts = await store.list_facts_for_owner(**_SCOPE)
    f = next(x for x in facts if x.id == "f2")
    audit_log = f.meta.get("security", {}).get("audit_log", [])
    assert any(e.get("action") == "reinstate" for e in audit_log)


@pytest.mark.asyncio
async def test_semantic_reinstate_nonexistent_returns_false(tmp_path):
    store = _sem(tmp_path)
    ok = await store.reinstate_quarantined_record("nonexistent", **_SCOPE, audit_entry=_AUDIT)
    assert ok is False


@pytest.mark.asyncio
async def test_semantic_purge_via_delete_fact(tmp_path):
    store = _sem(tmp_path)
    fact = Fact(
        id="fpurge", subject="user:test", predicate="p", object="v",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.upsert_fact(fact, _VEC)
    await store.delete_fact("fpurge", **_SCOPE)

    all_facts = await store.list_facts_for_owner(**_SCOPE, include_quarantined=True)
    assert not any(f.id == "fpurge" for f in all_facts)


# ---------------------------------------------------------------------------
# Episodic: reinstate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_episodic_reinstate_makes_episode_active(tmp_path):
    store = _ep(tmp_path)
    ep = Episode(
        id="e1", timestamp=_NOW, summary="bad", user_id="user:test",
        **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.add_episode(ep, _VEC)

    ok = await store.reinstate_quarantined_record("e1", **_SCOPE, audit_entry=_AUDIT)
    assert ok is True

    active = await store.list_episodes("default", "user", "user:test")
    assert any(e.id == "e1" for e in active)


@pytest.mark.asyncio
async def test_episodic_purge_via_delete_episode(tmp_path):
    store = _ep(tmp_path)
    ep = Episode(
        id="epurge", timestamp=_NOW, summary="bad", user_id="user:test",
        **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.add_episode(ep, _VEC)
    await store.delete_episode("epurge", **_SCOPE)

    all_eps = await store.list_episodes("default", "user", "user:test", include_quarantined=True)
    assert not any(e.id == "epurge" for e in all_eps)


# ---------------------------------------------------------------------------
# Procedural: reinstate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_procedural_reinstate_makes_skill_active(tmp_path):
    store = _proc(tmp_path)
    sk = Skill(
        id="sk1", name="bad_skill", description="poisoned", created_at=_NOW, updated_at=_NOW,
        **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.add_skill(sk, _VEC)

    ok = await store.reinstate_quarantined_record("sk1", **_SCOPE, audit_entry=_AUDIT)
    assert ok is True

    skills = await store.list_skills(**_SCOPE)
    assert any(s.id == "sk1" for s in skills)


@pytest.mark.asyncio
async def test_procedural_purge_via_delete_skill(tmp_path):
    store = _proc(tmp_path)
    sk = Skill(
        id="skpurge", name="bad", description="bad", created_at=_NOW, updated_at=_NOW,
        **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.add_skill(sk, _VEC)
    await store.delete_skill("skpurge", **_SCOPE)

    all_skills = await store.list_skills(**_SCOPE, include_quarantined=True)
    assert not any(s.id == "skpurge" for s in all_skills)


# ---------------------------------------------------------------------------
# Chunk: reinstate + purge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chunk_reinstate_makes_chunk_active(tmp_path):
    store = _chunk(tmp_path)
    ch = Chunk(
        id="ch1", doc_id="d1", text="bad text", page_range=(0, 1), position=0,
        source_path="f.pdf", source_hash="h", created_at=_NOW, updated_at=_NOW,
        **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.upsert_chunk(ch, _VEC)

    ok = await store.reinstate_quarantined_record("ch1", **_SCOPE, audit_entry=_AUDIT)
    assert ok is True

    active = await store.fetch_by_ids(["ch1"], **_SCOPE)
    assert any(c.id == "ch1" for c in active)


@pytest.mark.asyncio
async def test_chunk_purge_via_delete_chunk(tmp_path):
    store = _chunk(tmp_path)
    ch = Chunk(
        id="chpurge", doc_id="d2", text="bad", page_range=(0, 1), position=0,
        source_path="f.pdf", source_hash="h", created_at=_NOW, updated_at=_NOW,
        **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await store.upsert_chunk(ch, _VEC)
    await store.delete_chunk("chpurge", **_SCOPE)

    all_chunks = await store.list_chunks_for_owner(**_SCOPE, include_quarantined=True)
    assert not any(c.id == "chpurge" for c in all_chunks)


# ---------------------------------------------------------------------------
# Idempotent reinstate: reinstating an already-active record is a no-op
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reinstate_active_record_is_harmless(tmp_path):
    """Reinstating an already-active fact (quarantined_at=None) is safe and returns True."""
    store = _sem(tmp_path)
    fact = Fact(
        id="f_already_active", subject="user:test", predicate="p", object="v",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.7, quarantined_at=None,
    )
    await store.upsert_fact(fact, _VEC)

    ok = await store.reinstate_quarantined_record("f_already_active", **_SCOPE, audit_entry=_AUDIT)
    assert ok is True

    facts = await store.list_facts_for_owner(**_SCOPE)
    assert any(f.id == "f_already_active" for f in facts)
