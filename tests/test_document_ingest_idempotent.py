from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_ingest_document_is_idempotent_by_owner_and_hash(uma_memory, tmp_path):
    memory = uma_memory

    # Prepare a stable text file to ingest twice.
    p = tmp_path / "doc.txt"
    p.write_text("hello world.\n" * 200, encoding="utf-8")

    report1 = await memory.ingest_document(str(p), owner_type="user", owner_id="user:u1")
    assert report1.doc_id
    assert report1.chunks_created >= 0
    assert report1.facts_created >= 0

    # Second ingest should be a refresh-only (no new chunks/facts) for same owner+hash+signature.
    report2 = await memory.ingest_document(str(p), owner_type="user", owner_id="user:u1")
    assert report2.doc_id == report1.doc_id
    assert report2.chunks_created == 0
    assert report2.facts_created == 0

    # Chunk count should remain stable after second ingest.
    conn = memory._stores["chunk"]._conn()
    try:
        rows = memory._stores["chunk"]._query_all(
            conn,
            "SELECT COUNT(*) AS n FROM chunks WHERE owner_type=? AND owner_id=?",
            params=["user", "user:u1"],
            log_context="test_chunk_count",
        )
        assert int(rows[0]["n"]) == report1.chunks_created

        # Ensure chunk meta includes deterministic text_hash and chunking params.
        if report1.chunks_created > 0:
            meta_rows = memory._stores["chunk"]._query_all(
                conn,
                "SELECT meta FROM chunks WHERE owner_type=? AND owner_id=? LIMIT 1",
                params=["user", "user:u1"],
                log_context="test_chunk_meta",
            )
            assert meta_rows
            meta = json.loads(meta_rows[0]["meta"])
            assert isinstance(meta.get("text_hash"), str) and len(meta["text_hash"]) >= 32
            assert meta.get("chunker_version") == "doc_chunk_v2"
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_ingest_persists_terminal_chunks(uma_memory, tmp_path):
    """
    Invariant: what gets stored (and therefore embedded) must be finalized/terminal.
    """
    memory = uma_memory

    p = tmp_path / "doc.txt"
    p.write_text(("A" * 400) + ".\n\n" + ("B" * 400) + ".", encoding="utf-8")

    report = await memory.ingest_document(str(p), owner_type="user", owner_id="user:u1")
    assert report.chunks_created > 0

    conn = memory._stores["chunk"]._conn()
    try:
        rows = memory._stores["chunk"]._query_all(
            conn,
            "SELECT text FROM chunks WHERE owner_type=? AND owner_id=? ORDER BY position ASC",
            params=["user", "user:u1"],
            log_context="test_chunk_terminality",
        )
        texts = [(r["text"] or "").strip() for r in rows]
        assert texts
        assert all(t.endswith((".", "!", "?")) for t in texts if t), "expected stored chunks to be terminal"
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_ingest_document_reingests_when_signature_changes(uma_memory, tmp_path):
    memory = uma_memory

    p = tmp_path / "doc.txt"
    p.write_text("hello world.\n" * 200, encoding="utf-8")

    report1 = await memory.ingest_document(str(p), owner_type="user", owner_id="user:u1")
    assert report1.chunks_created > 0

    # Re-run ingest with a different chunk_size_tokens so the ingest signature changes.
    from uma.core.ingest.ingest_service import ingest_document as _ingest
    from uma.core.ingest.types import IngestConfig

    report2 = await _ingest(
        str(p),
        owner_type="user",
        owner_id="user:u1",
        config=IngestConfig(chunk_size_tokens=60, overlap_tokens=10),
        memory=memory,
    )
    assert report2.doc_id == report1.doc_id
    assert report2.chunks_created > 0

    from uma.core.ingest.parser import parse_file

    parsed = parse_file(str(p))
    manifest = await memory.document_store.get_by_owner_and_hash(
        owner_type="user",
        owner_id="user:u1",
        source_hash=parsed.source_hash,
    )
    assert manifest is not None

    sig = (manifest.meta or {}).get("ingest_signature") or {}
    assert sig.get("chunk_size_tokens") == 60
    assert sig.get("extractor_version") == "doc_fact_extract_v1"
    assert sig.get("splitter_version") == "doc_normalize_v1"
    assert sig.get("chunker_version") == "doc_chunk_v2"
    history = (manifest.meta or {}).get("ingest_history")
    assert isinstance(history, list)
    assert len(history) >= 2
    assert history[-1]["signature"]["chunk_size_tokens"] == 60
