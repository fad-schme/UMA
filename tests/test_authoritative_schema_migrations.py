from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from uma.adapters.db.sqlite_adapter import SQLiteAdapter
from uma.adapters.vector.base import VectorIndex
from uma.stores.chunk_sql import ChunkSQLStore
from uma.stores.document_sql import DocumentRecord, DocumentSQLStore
from uma.stores.episodic_sql import EpisodicSQLStore
from uma.stores.procedural_sql import ProceduralSQLStore
from uma.stores.semantic_sql import SemanticSQLStore
from uma.types import Chunk, Episode, Fact, Skill, SCOPE_MODEL_VERSION


class _NoopVectorIndex(VectorIndex):
    def upsert(self, ids, vectors, metadata=None) -> None:
        return None

    def query(self, vector, k=10, filters=None):
        return []

    def delete(self, ids) -> None:
        return None


def _columns(db_path: str, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row[1]) for row in rows}
    finally:
        conn.close()


def _indexes(db_path: str, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
            [table],
        ).fetchall()
        return {str(row[0]) for row in rows}
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_document_store_migrates_legacy_rows_and_persists_scope_fields(tmp_path) -> None:
    db_path = str(tmp_path / "documents.sqlite")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                owner_type TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                meta TEXT NOT NULL
            );
            INSERT INTO documents(doc_id, source_path, source_hash, ingested_at, owner_type, owner_id, meta)
            VALUES ('doc_old', '/tmp/old.txt', 'hash-old', '2026-01-01T00:00:00+00:00', 'user', 'user:u1', '{}');
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = DocumentSQLStore(db_adapter=SQLiteAdapter(db_path))
    record = await store.get_by_owner_and_hash(owner_type="user", owner_id="user:u1", source_hash="hash-old")
    assert record is not None
    assert record.doc_id == "doc_old"
    assert record.tenant_id == "default"
    assert record.workspace_id is None

    await store.upsert_document(
        DocumentRecord(
            doc_id="doc_new",
            source_path="/tmp/new.txt",
            source_hash="hash-new",
            ingested_at=datetime.now(timezone.utc),
            tenant_id="tenant-1",
            owner_type="workspace",
            owner_id="workspace:alpha",
            workspace_id="workspace:alpha",
            origin_agent_id="agent-1",
            origin_user_id="user-1",
            origin_session_id="session-1",
            scope_model_version=SCOPE_MODEL_VERSION,
            meta={"kind": "manifest"},
        )
    )

    assert {
        "tenant_id",
        "workspace_id",
        "origin_agent_id",
        "origin_user_id",
        "origin_session_id",
        "scope_model_version",
    }.issubset(_columns(db_path, "documents"))
    assert {
        "idx_documents_tenant_owner",
        "idx_documents_tenant_owner_hash",
    }.issubset(_indexes(db_path, "documents"))

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT tenant_id, workspace_id, origin_agent_id, origin_user_id, origin_session_id, scope_model_version
            FROM documents WHERE doc_id='doc_new'
            """
        ).fetchone()
        assert row == ("tenant-1", "workspace:alpha", "agent-1", "user-1", "session-1", SCOPE_MODEL_VERSION)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_chunk_store_migrates_legacy_rows_and_round_trips_scope_fields(tmp_path) -> None:
    db_path = str(tmp_path / "chunks.sqlite")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                position INTEGER NOT NULL,
                source_path TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                owner_type TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                meta TEXT NOT NULL
            );
            INSERT INTO chunks VALUES (
                'chunk_old', 'doc_old', 'legacy text', 1, 1, 0, '/tmp/old.txt', 'hash-old',
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 'user', 'user:u1', '{}'
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = ChunkSQLStore(db_adapter=SQLiteAdapter(db_path), vector_index=_NoopVectorIndex())
    legacy = await store.fetch_by_ids(["chunk_old"], owner_type="user", owner_id="user:u1")
    assert len(legacy) == 1
    assert legacy[0].tenant_id == "default"
    assert legacy[0].workspace_id is None

    now = datetime.now(timezone.utc)
    await store.upsert_chunk(
        Chunk(
            id="chunk_new",
            doc_id="doc_new",
            text="new chunk",
            page_range=(2, 3),
            position=1,
            source_path="/tmp/new.txt",
            source_hash="hash-new",
            created_at=now,
            updated_at=now,
            tenant_id="tenant-1",
            owner_type="workspace",
            owner_id="workspace:alpha",
            workspace_id="workspace:alpha",
            origin_agent_id="agent-1",
            origin_user_id="user-1",
            origin_session_id="session-1",
            scope_model_version=SCOPE_MODEL_VERSION,
            meta={},
        ),
        embedding=[0.0, 0.0, 0.0],
    )

    stored = await store.fetch_by_ids(
        ["chunk_new"],
        tenant_id="tenant-1",
        owner_type="workspace",
        owner_id="workspace:alpha",
    )
    assert len(stored) == 1
    assert stored[0].tenant_id == "tenant-1"
    assert stored[0].workspace_id == "workspace:alpha"
    assert stored[0].origin_session_id == "session-1"
    assert stored[0].scope_model_version == SCOPE_MODEL_VERSION


@pytest.mark.asyncio
async def test_semantic_store_migrates_legacy_rows_and_persists_new_scope_columns(tmp_path) -> None:
    db_path = str(tmp_path / "facts.sqlite")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE facts (
                id TEXT PRIMARY KEY,
                owner_type TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_ids TEXT NOT NULL,
                source TEXT,
                salience REAL NOT NULL,
                confidence REAL NULL,
                meta TEXT NOT NULL
            );
            INSERT INTO facts VALUES (
                'fact_old', 'user', 'user:u1', 'user:u1', 'LIKES', '\"coffee\"',
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', '[]', NULL, 0.0, NULL, '{}'
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = SemanticSQLStore(db_adapter=SQLiteAdapter(db_path), vector_index=_NoopVectorIndex())
    legacy = await store.list_facts_for_owner(owner_type="user", owner_id="user:u1", limit=None)
    assert len(legacy) == 1
    assert legacy[0].tenant_id == "default"
    assert legacy[0].session_id is None

    now = datetime.now(timezone.utc)
    await store.upsert_fact(
        Fact(
            id="fact_new",
            subject="workspace:alpha",
            predicate="USES",
            object="sso",
            created_at=now,
            updated_at=now,
            owner_type="workspace",
            owner_id="workspace:alpha",
            tenant_id="tenant-1",
            workspace_id="workspace:alpha",
            session_id="session-1",
            origin_agent_id="agent-1",
            origin_user_id="user-1",
            origin_session_id="session-1",
            scope_model_version=SCOPE_MODEL_VERSION,
            meta={"turn_id": "t1"},
        ),
        embedding=[0.0, 0.0, 0.0],
    )

    stored = await store.list_facts_for_owner(
        tenant_id="tenant-1",
        owner_type="workspace",
        owner_id="workspace:alpha",
        limit=None,
    )
    assert len(stored) == 1
    assert stored[0].tenant_id == "tenant-1"
    assert stored[0].workspace_id == "workspace:alpha"
    assert stored[0].session_id == "session-1"
    assert stored[0].origin_agent_id == "agent-1"
    assert stored[0].scope_model_version == SCOPE_MODEL_VERSION


@pytest.mark.asyncio
async def test_episodic_store_migrates_legacy_rows_and_persists_session_scope_fields(tmp_path) -> None:
    db_path = str(tmp_path / "episodes.sqlite")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                owner_type TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                summary TEXT NOT NULL,
                raw TEXT,
                tags TEXT NOT NULL,
                meta TEXT NOT NULL,
                embedding TEXT
            );
            CREATE TABLE episode_clusters (
                id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                episode_ids TEXT NOT NULL,
                owner_type TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                latest_timestamp TEXT NOT NULL
            );
            CREATE TABLE episode_cluster_members (
                cluster_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                PRIMARY KEY (cluster_id, episode_id)
            );
            INSERT INTO episodes VALUES (
                'ep_old', 'user', 'user:u1', 'user:u1', '2026-01-01T00:00:00+00:00',
                'legacy summary', 'legacy raw', '[]', '{}', NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = EpisodicSQLStore(db_adapter=SQLiteAdapter(db_path), vector_index=_NoopVectorIndex())
    legacy = await store.list_episodes(owner_type="user", owner_id="user:u1")
    assert len(legacy) == 1
    assert legacy[0].tenant_id == "default"
    assert legacy[0].session_id is None
    assert {"tenant_id", "session_id", "scope_model_version"}.issubset(_columns(db_path, "episode_clusters"))

    now = datetime.now(timezone.utc)
    await store.add_episode(
        Episode(
            id="ep_new",
            user_id="user-1",
            timestamp=now,
            summary="session summary",
            raw="full transcript",
            owner_type="workspace",
            owner_id="workspace:alpha",
            tenant_id="tenant-1",
            workspace_id="workspace:alpha",
            session_id="session-1",
            origin_agent_id="agent-1",
            origin_user_id="user-1",
            origin_session_id="session-1",
            scope_model_version=SCOPE_MODEL_VERSION,
            meta={},
        ),
        embedding=[0.0, 0.0, 0.0],
    )

    stored = await store.list_episodes("tenant-1", "workspace", "workspace:alpha")
    assert len(stored) == 1
    assert stored[0].tenant_id == "tenant-1"
    assert stored[0].workspace_id == "workspace:alpha"
    assert stored[0].session_id == "session-1"
    assert stored[0].origin_session_id == "session-1"
    assert stored[0].scope_model_version == SCOPE_MODEL_VERSION


@pytest.mark.asyncio
async def test_procedural_store_migrates_legacy_rows_and_persists_provenance_fields(tmp_path) -> None:
    db_path = str(tmp_path / "skills.sqlite")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                trigger_phrases TEXT NOT NULL,
                trigger_patterns TEXT NOT NULL,
                plan TEXT NOT NULL,
                tools TEXT NOT NULL,
                example TEXT NOT NULL,
                meta TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                owner_type TEXT NOT NULL,
                owner_id TEXT NOT NULL
            );
            INSERT INTO skills VALUES (
                'skill_old', 'Legacy Skill', '[]', '[]', '{}', '[]', 'example', '{}',
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', 'user', 'user:u1'
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = ProceduralSQLStore(db_adapter=SQLiteAdapter(db_path), vector_index=_NoopVectorIndex())
    legacy = await store.list_skills(owner_type="user", owner_id="user:u1")
    assert len(legacy) == 1
    assert legacy[0].tenant_id == "default"
    assert legacy[0].workspace_id is None

    now = datetime.now(timezone.utc)
    await store.add_skill(
        Skill(
            id="skill_new",
            name="Workspace Skill",
            description="shared",
            created_at=now,
            updated_at=now,
            owner_type="workspace",
            owner_id="workspace:alpha",
            tenant_id="tenant-1",
            workspace_id="workspace:alpha",
            origin_agent_id="agent-1",
            origin_user_id="user-1",
            origin_session_id="session-1",
            scope_model_version=SCOPE_MODEL_VERSION,
            trigger_phrases=["deploy"],
            trigger_patterns=[],
            plan={"steps": ["run"]},
            tools=["shell"],
            example="deploy app",
            meta={},
        ),
        embedding=[0.0, 0.0, 0.0],
    )

    stored = await store.list_skills(tenant_id="tenant-1", owner_type="workspace", owner_id="workspace:alpha")
    assert len(stored) == 1
    assert stored[0].tenant_id == "tenant-1"
    assert stored[0].workspace_id == "workspace:alpha"
    assert stored[0].origin_user_id == "user-1"
    assert stored[0].scope_model_version == SCOPE_MODEL_VERSION
