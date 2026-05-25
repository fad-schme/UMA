"""
test_pr4_management_api.py
============================
Unit tests for the three quarantine management functions:
  list_quarantined, reinstate_quarantined, purge_quarantined
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from uma.api.management import (
    QuarantinedRecord,
    list_quarantined,
    reinstate_quarantined,
    purge_quarantined,
)
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
_SCOPE = dict(owner_type="user", owner_id="user:alice", tenant_id="default")


def _make_stores(tmp_path):
    def _db(name):
        return SQLiteAdapter(str(tmp_path / f"{name}.db"))

    return {
        "semantic": SemanticSQLStore(db_adapter=_db("sem"), vector_index=_NoopVI()),
        "episodic": EpisodicSQLStore(db_adapter=_db("ep"), vector_index=_NoopVI()),
        "procedural": ProceduralSQLStore(db_adapter=_db("proc"), vector_index=_NoopVI()),
        "chunk": ChunkSQLStore(db_adapter=_db("chunk"), vector_index=_NoopVI()),
    }


def _mock_memory(stores):
    m = MagicMock()
    m._stores = stores
    return m


# ---------------------------------------------------------------------------
# list_quarantined
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_quarantined_returns_only_quarantined_records(tmp_path):
    stores = _make_stores(tmp_path)
    memory = _mock_memory(stores)

    active = Fact(
        id="fact_a", subject="user:alice", predicate="likes", object="cats",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.8,
    )
    qed = Fact(
        id="fact_q", subject="user:alice", predicate="ignores", object="rules",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
        meta={"security": {"injection_scan": {"severity": "high", "matched_rules": ["jailbreak"]}}},
    )
    await stores["semantic"].upsert_fact(active, _VEC)
    await stores["semantic"].upsert_fact(qed, _VEC)

    records = await list_quarantined(memory, **_SCOPE, lane="semantic")
    assert len(records) == 1
    assert records[0].id == "fact_q"
    assert records[0].lane == "semantic"
    assert records[0].quarantined_at is not None
    assert records[0].severity == "high"


@pytest.mark.asyncio
async def test_list_quarantined_all_lanes_aggregates(tmp_path):
    stores = _make_stores(tmp_path)
    memory = _mock_memory(stores)

    fact_q = Fact(
        id="fq", subject="user:alice", predicate="p", object="v",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    ep_q = Episode(
        id="eq", timestamp=_NOW, summary="bad", user_id="user:alice",
        **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await stores["semantic"].upsert_fact(fact_q, _VEC)
    await stores["episodic"].add_episode(ep_q, _VEC)

    records = await list_quarantined(memory, **_SCOPE)
    lanes = {r.lane for r in records}
    assert "semantic" in lanes
    assert "episodic" in lanes


@pytest.mark.asyncio
async def test_list_quarantined_empty_when_nothing_quarantined(tmp_path):
    stores = _make_stores(tmp_path)
    memory = _mock_memory(stores)

    fact = Fact(
        id="fclean", subject="user:alice", predicate="p", object="v",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.8,
    )
    await stores["semantic"].upsert_fact(fact, _VEC)

    records = await list_quarantined(memory, **_SCOPE)
    assert records == []


@pytest.mark.asyncio
async def test_list_quarantined_unknown_lane_raises(tmp_path):
    stores = _make_stores(tmp_path)
    memory = _mock_memory(stores)
    with pytest.raises(ValueError, match="unknown lane"):
        await list_quarantined(memory, **_SCOPE, lane="nonexistent")


# ---------------------------------------------------------------------------
# reinstate_quarantined
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reinstate_clears_quarantined_at(tmp_path):
    stores = _make_stores(tmp_path)
    memory = _mock_memory(stores)

    fact_q = Fact(
        id="fact_reinstate", subject="user:alice", predicate="p", object="v",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await stores["semantic"].upsert_fact(fact_q, _VEC)

    result = await reinstate_quarantined(
        memory, record_id="fact_reinstate", lane="semantic", reason="false positive", **_SCOPE
    )
    assert result is True

    facts = await stores["semantic"].list_facts_for_owner(**_SCOPE)
    assert any(f.id == "fact_reinstate" for f in facts)


@pytest.mark.asyncio
async def test_reinstate_appends_audit_log(tmp_path):
    stores = _make_stores(tmp_path)
    memory = _mock_memory(stores)

    fact_q = Fact(
        id="fact_audit", subject="user:alice", predicate="q", object="r",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await stores["semantic"].upsert_fact(fact_q, _VEC)

    await reinstate_quarantined(
        memory, record_id="fact_audit", lane="semantic", reason="admin decision", **_SCOPE
    )

    facts = await stores["semantic"].list_facts_for_owner(**_SCOPE)
    reinstated = next(f for f in facts if f.id == "fact_audit")
    audit_log = reinstated.meta.get("security", {}).get("audit_log", [])
    assert any(entry.get("action") == "reinstate" for entry in audit_log)
    assert any(entry.get("reason") == "admin decision" for entry in audit_log)


@pytest.mark.asyncio
async def test_reinstate_returns_false_for_nonexistent_record(tmp_path):
    stores = _make_stores(tmp_path)
    memory = _mock_memory(stores)

    result = await reinstate_quarantined(
        memory, record_id="does_not_exist", lane="semantic", reason="test", **_SCOPE
    )
    assert result is False


# ---------------------------------------------------------------------------
# purge_quarantined
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_purge_deletes_quarantined_record(tmp_path):
    stores = _make_stores(tmp_path)
    memory = _mock_memory(stores)

    fact_q = Fact(
        id="fact_purge", subject="user:alice", predicate="p", object="v",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.0, quarantined_at=_NOW,
    )
    await stores["semantic"].upsert_fact(fact_q, _VEC)

    result = await purge_quarantined(
        memory, record_id="fact_purge", lane="semantic", reason="malicious content", **_SCOPE
    )
    assert result is True

    all_facts = await stores["semantic"].list_facts_for_owner(**_SCOPE, include_quarantined=True)
    assert not any(f.id == "fact_purge" for f in all_facts)


@pytest.mark.asyncio
async def test_purge_active_record_returns_false(tmp_path):
    stores = _make_stores(tmp_path)
    memory = _mock_memory(stores)

    fact = Fact(
        id="fact_active_purge", subject="user:alice", predicate="p", object="v",
        created_at=_NOW, updated_at=_NOW, **_SCOPE, trust_score=0.8,
    )
    await stores["semantic"].upsert_fact(fact, _VEC)

    result = await purge_quarantined(
        memory, record_id="fact_active_purge", lane="semantic", reason="test", **_SCOPE
    )
    assert result is False

    facts = await stores["semantic"].list_facts_for_owner(**_SCOPE)
    assert any(f.id == "fact_active_purge" for f in facts)


@pytest.mark.asyncio
async def test_purge_unknown_lane_raises(tmp_path):
    stores = _make_stores(tmp_path)
    memory = _mock_memory(stores)
    with pytest.raises(ValueError, match="unknown lane"):
        await purge_quarantined(memory, record_id="x", lane="invalid", reason="test", **_SCOPE)
