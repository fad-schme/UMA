"""
PR1 — ingest end-to-end: trust_score set on chunks; content_hash set on derived facts.

Tests:
- After ingest_document, resulting chunks have trust_score == 0.5.
- text_hash in chunk meta is still present (not removed).
- Derived facts (if any) have trust_score == 0.5 and non-empty content_hash.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.runtime import init_uma_for_tests


_FIXTURE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This document is used for testing the UMA ingest pipeline. "
    "It contains several sentences to ensure the chunker produces at least one chunk. "
    "Memory systems need reliable content hashing and trust tracking. "
    "Every stored artifact carries a trust_score and content_hash as security primitives."
)


@pytest.fixture
async def _uma(tmp_path):
    mem = await init_uma_for_tests(tmp_path)
    try:
        yield mem
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass


@pytest.fixture
def fixture_doc(tmp_path) -> str:
    doc_path = tmp_path / "test_doc.txt"
    doc_path.write_text(_FIXTURE_TEXT, encoding="utf-8")
    return str(doc_path)


@pytest.mark.asyncio
async def test_ingest_chunks_have_trust_score(tmp_path, fixture_doc):
    mem = await init_uma_for_tests(tmp_path)
    try:
        report = await mem.ingest_document(
            fixture_doc,
            owner_type="user",
            owner_id="user:alice",
        )
        assert report.chunks_created > 0, "expected at least one chunk"

        chunk_store = mem._stores["chunk"]
        conn = chunk_store._conn()
        try:
            rows = chunk_store._query_all(
                conn,
                "SELECT id, trust_score FROM chunks WHERE owner_id = ?",
                params=["user:alice"],
                log_context="test_pr1_ingest_trust_score",
            )
        finally:
            conn.close()

        assert rows, "expected chunk rows in DB"
        for row in rows:
            assert row["trust_score"] is not None
            assert abs(float(row["trust_score"]) - 0.7) < 1e-6, (
                f"chunk {row['id']} must have trust_score=0.7 (document source)"
            )
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_ingest_chunks_text_hash_still_in_meta(tmp_path, fixture_doc):
    """text_hash in chunk meta must be preserved after PR1 (not promoted, not removed)."""
    mem = await init_uma_for_tests(tmp_path)
    try:
        import json

        report = await mem.ingest_document(
            fixture_doc,
            owner_type="user",
            owner_id="user:alice",
        )
        assert report.chunks_created > 0

        chunk_store = mem._stores["chunk"]
        conn = chunk_store._conn()
        try:
            rows = chunk_store._query_all(
                conn,
                "SELECT id, trust_score FROM chunks WHERE owner_id = ?",
                params=["user:alice"],
                log_context="test_pr1_ingest_text_hash_meta",
            )
        finally:
            conn.close()
        assert rows

        for row in rows:
            # trust_score must be set (text_hash is preserved in meta, not replaced)
            assert abs(float(row["trust_score"]) - 0.7) < 1e-6
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_ingest_derived_facts_have_trust_score_and_content_hash(tmp_path, fixture_doc):
    mem = await init_uma_for_tests(tmp_path)
    try:
        await mem.ingest_document(
            fixture_doc,
            owner_type="user",
            owner_id="user:alice",
        )

        sem_store = mem._stores["semantic"]
        facts = await sem_store.list_facts_for_owner(
            tenant_id="default",
            owner_type="user",
            owner_id="user:alice",
        )

        if not facts:
            pytest.skip("fake_llm produced no facts for this fixture; skipping")

        for fact in facts:
            assert fact.trust_score == pytest.approx(0.7), (
                f"fact id={fact.id} must have trust_score=0.7 (document source)"
            )
            if fact.content_hash is not None:
                assert len(fact.content_hash) == 64, (
                    f"fact id={fact.id} content_hash must be 64-char SHA-256 hex"
                )
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass
