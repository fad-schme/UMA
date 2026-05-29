from __future__ import annotations

from pathlib import Path

import pytest

from uma.ingest.ingest_service import capture_source


def _manifest_rows(memory, *, tenant_id: str, owner_type: str, owner_id: str, source_path: str) -> list[dict]:
    conn = memory.document_store._conn()
    try:
        rows = memory.document_store._query_all(
            conn,
            """
            SELECT *
            FROM documents
            WHERE tenant_id = ? AND owner_type = ? AND owner_id = ? AND source_path = ?
            ORDER BY ingested_at ASC
            """,
            params=[tenant_id, owner_type, owner_id, source_path],
            log_context="test_manifest_supersession_rows",
        )
        return list(rows)
    finally:
        conn.close()


def _chunk_ids_for_doc(memory, *, tenant_id: str, owner_type: str, owner_id: str, doc_id: str) -> list[str]:
    conn = memory.chunk_core.store._conn()
    try:
        rows = memory.chunk_core.store._query_all(
            conn,
            """
            SELECT id
            FROM chunks
            WHERE tenant_id = ? AND owner_type = ? AND owner_id = ? AND doc_id = ?
            ORDER BY position ASC, id ASC
            """,
            params=[tenant_id, owner_type, owner_id, doc_id],
            log_context="test_manifest_supersession_chunks",
        )
        return [row["id"] for row in rows]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_same_source_same_content_keeps_existing_manifest_gate_behavior(uma_memory, tmp_path: Path) -> None:
    path = tmp_path / "manifest-idempotent.txt"
    path.write_text(
        "This source stays identical across both ingests. The paragraph is long enough to produce a valid chunk.\n",
        encoding="utf-8",
    )

    first = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        tenant_id="default",
        memory=uma_memory,
    )
    second = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        tenant_id="default",
        memory=uma_memory,
    )

    rows = _manifest_rows(
        uma_memory,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        source_path=str(path),
    )
    assert first.skipped is False
    assert second.skipped is True
    assert len(rows) == 1
    meta = rows[0]["meta"]
    assert '"supersedes"' not in meta
    assert '"superseded_by"' not in meta
    assert '"superseded_at"' not in meta


@pytest.mark.asyncio
async def test_same_source_different_content_records_manifest_supersession_and_preserves_prior_chunks(
    uma_memory,
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest-supersede.txt"
    path.write_text(
        "Version one explains the original deployment checklist. It preserves a chunk for the first manifest.\n",
        encoding="utf-8",
    )
    first = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        tenant_id="default",
        memory=uma_memory,
    )

    path.write_text(
        "Version two explains the revised deployment checklist. It must create a second manifest and new chunks.\n",
        encoding="utf-8",
    )
    second = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        tenant_id="default",
        memory=uma_memory,
    )

    first_manifest = await uma_memory.document_store.get_by_owner_and_hash(
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        source_hash=first.parsed.source_hash,
    )
    second_manifest = await uma_memory.document_store.get_by_owner_and_hash(
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        source_hash=second.parsed.source_hash,
    )

    assert first_manifest is not None
    assert second_manifest is not None
    assert first_manifest.doc_id != second_manifest.doc_id
    assert first_manifest.meta["superseded_by"] == second_manifest.doc_id
    assert first_manifest.meta["superseded_at"]
    assert second_manifest.meta["supersedes"] == first_manifest.doc_id
    assert _chunk_ids_for_doc(
        uma_memory,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        doc_id=first_manifest.doc_id,
    )
    assert _chunk_ids_for_doc(
        uma_memory,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        doc_id=second_manifest.doc_id,
    )


@pytest.mark.asyncio
async def test_three_successive_source_versions_form_immediate_supersession_chain(uma_memory, tmp_path: Path) -> None:
    path = tmp_path / "manifest-chain.txt"
    doc_ids: list[str] = []
    source_hashes: list[str] = []

    for idx, text in enumerate(
        [
            "Version one documents the original process. This sentence keeps the chunk coherent.\n",
            "Version two documents the updated process. This sentence keeps the chunk coherent.\n",
            "Version three documents the latest process. This sentence keeps the chunk coherent.\n",
        ],
        start=1,
    ):
        path.write_text(text, encoding="utf-8")
        capture = await capture_source(
            str(path),
            owner_type="user",
            owner_id="user:u1",
            tenant_id="default",
            memory=uma_memory,
        )
        doc_ids.append(capture.parsed.doc_id)
        source_hashes.append(capture.parsed.source_hash)

    manifests = [
        await uma_memory.document_store.get_by_owner_and_hash(
            tenant_id="default",
            owner_type="user",
            owner_id="user:u1",
            source_hash=source_hash,
        )
        for source_hash in source_hashes
    ]

    assert all(manifest is not None for manifest in manifests)
    assert manifests[0].meta["superseded_by"] == doc_ids[1]
    assert manifests[1].meta["supersedes"] == doc_ids[0]
    assert manifests[1].meta["superseded_by"] == doc_ids[2]
    assert manifests[2].meta["supersedes"] == doc_ids[1]
    assert "superseded_by" not in manifests[2].meta


@pytest.mark.asyncio
async def test_manifest_supersession_does_not_cross_tenant_or_owner_boundaries(uma_memory, tmp_path: Path) -> None:
    path = tmp_path / "manifest-isolation.txt"
    path.write_text(
        "Tenant alpha ingests this source first. The text is long enough for one valid chunk.\n",
        encoding="utf-8",
    )
    alpha = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        tenant_id="tenant-alpha",
        memory=uma_memory,
    )

    path.write_text(
        "Tenant beta ingests a changed version of the same file path. No cross-tenant supersession is allowed.\n",
        encoding="utf-8",
    )
    beta = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        tenant_id="tenant-beta",
        memory=uma_memory,
    )

    alpha_manifest = await uma_memory.document_store.get_by_owner_and_hash(
        tenant_id="tenant-alpha",
        owner_type="user",
        owner_id="user:u1",
        source_hash=alpha.parsed.source_hash,
    )
    beta_manifest = await uma_memory.document_store.get_by_owner_and_hash(
        tenant_id="tenant-beta",
        owner_type="user",
        owner_id="user:u1",
        source_hash=beta.parsed.source_hash,
    )

    assert alpha_manifest is not None
    assert beta_manifest is not None
    assert "superseded_by" not in alpha_manifest.meta
    assert "supersedes" not in alpha_manifest.meta
    assert "superseded_by" not in beta_manifest.meta
    assert "supersedes" not in beta_manifest.meta
