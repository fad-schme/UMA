import asyncio

from uma.core.uma_memory import UMAMemory
from uma.core.memory_config import UMAConfig


def _good_embedder(texts):
    return [[0.0, 0.0, 0.0] for _ in texts]


def _good_llm(messages, **kwargs):
    # Return a minimal valid JSON payload for extract_facts()
    return '{"facts": []}'


def _good_embedder_kwargs(texts, **kwargs):
    # CallableEmbedderAdapter may pass model/config kwargs.
    return _good_embedder(texts)


def test_ingest_document_is_idempotent_by_owner_and_hash(tmp_path):
    # Prepare a stable text file to ingest twice.
    p = tmp_path / "doc.txt"
    p.write_text("hello world\n" * 200, encoding="utf-8")

    cfg = {
        "storage": {
            "db_root": str(tmp_path),
            "sql_backend": "sqlite",
            "vector_backend": "inmemory",
            "graph_backend": "disabled",
        },
        "working_memory": {"max_tokens": 100, "warning_ratio": 0.7, "hard_limit_ratio": 0.95},
        "embedding": {
            "provider": "tests.test_document_ingest_idempotent:_good_embedder",
            "model": "x",
            "dimension": 3,
            "config": {"preflight": False},
        },
        "llm": {
            "provider": "tests.test_document_ingest_idempotent:_good_llm",
            "model": "x",
            "config": {"preflight": False},
        },
        "retrieval": {"max_episodes": 1, "max_facts": 1, "max_skills": 1, "max_graph_items": 1},
        "consolidation": {"enabled": False, "cluster_similarity": 0.75, "max_episodes_per_cycle": 10, "prune_min_fact_salience": 0.2},
        "features": {"load": [], "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False}},
    }

    mem = UMAMemory(UMAConfig(cfg))
    mem.initialize()

    report1 = asyncio.run(mem.ingest_document(str(p), owner_type="user", owner_id="u1"))
    assert report1.doc_id

    # Second ingest should be a refresh-only (no new chunks/facts) for same owner+hash+signature.
    report2 = asyncio.run(mem.ingest_document(str(p), owner_type="user", owner_id="u1"))
    assert report2.doc_id == report1.doc_id
    assert report2.chunks_created == 0
    assert report2.facts_created == 0

    # Chunk count should remain stable after second ingest.
    conn = mem._stores["chunk"]._conn()
    try:
        rows = mem._stores["chunk"]._query_all(conn, "SELECT COUNT(*) AS n FROM chunks", params={}, log_context="test_chunk_count")
        assert int(rows[0]["n"]) == report1.chunks_created

        # Ensure chunk meta includes deterministic text_hash and chunking params (from the first ingest).
        if report1.chunks_created > 0:
            meta_rows = mem._stores["chunk"]._query_all(
                conn,
                "SELECT meta FROM chunks LIMIT 1",
                params={},
                log_context="test_chunk_meta",
            )
            assert meta_rows
            import json
            meta = json.loads(meta_rows[0]["meta"])
            assert isinstance(meta.get("text_hash"), str) and len(meta["text_hash"]) >= 32
            assert meta.get("chunk_size_tokens") == 500
            assert meta.get("overlap_tokens") == 80
            assert meta.get("chunker_version") == "doc_chunk_v2"
    finally:
        conn.close()


def test_ingest_document_reingests_when_signature_changes(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello world\n" * 200, encoding="utf-8")

    cfg = {
        "storage": {
            "db_root": str(tmp_path),
            "sql_backend": "sqlite",
            "vector_backend": "inmemory",
            "graph_backend": "disabled",
        },
        "working_memory": {"max_tokens": 100, "warning_ratio": 0.7, "hard_limit_ratio": 0.95},
        "embedding": {
            "provider": "tests.test_document_ingest_idempotent:_good_embedder_kwargs",
            "model": "x",
            "dimension": 3,
            "config": {"preflight": False},
        },
        "llm": {
            "provider": "tests.test_document_ingest_idempotent:_good_llm",
            "model": "x",
            "config": {"preflight": False},
        },
        "retrieval": {"max_episodes": 1, "max_facts": 1, "max_skills": 1, "max_graph_items": 1},
        "consolidation": {"enabled": False, "cluster_similarity": 0.75, "max_episodes_per_cycle": 10, "prune_min_fact_salience": 0.2},
        "features": {"load": [], "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False}},
    }

    mem = UMAMemory(UMAConfig(cfg))
    mem.initialize()

    report1 = asyncio.run(mem.ingest_document(str(p), owner_type="user", owner_id="u1"))
    assert report1.chunks_created > 0

    # Directly call core ingest with a different chunk_size_tokens so the ingest signature changes.
    from uma.core.ingest.ingest_service import ingest_document as _ingest
    from uma.core.ingest.types import IngestConfig

    report2 = asyncio.run(
        _ingest(
            str(p),
            owner_type="user",
            owner_id="u1",
            config=IngestConfig(chunk_size_tokens=60, overlap_tokens=10),
            memory=mem,
        )
    )
    assert report2.doc_id == report1.doc_id
    assert report2.chunks_created > 0
    # Manifest should now record the updated signature for this owner+hash.
    # Fetch by (owner, source_hash) via parse_file.
    from uma.core.ingest.parser import parse_file
    parsed = parse_file(str(p))
    manifest = asyncio.run(
        mem.document_store.get_by_owner_and_hash(
            owner_type="user",
            owner_id="user:u1",
            source_hash=parsed.source_hash,
        )
    )
    assert manifest is not None
    # Re-ingest changes derived artifacts, but the final manifest write is keyed by parsed.doc_id.
    # We assert the ingest signature itself reflects the new parameters.
    sig = (manifest.meta or {}).get("ingest_signature") or {}
    assert sig.get("chunk_size_tokens") == 60
    assert sig.get("embedding_model") == "x"
    assert sig.get("extractor_version") == "doc_fact_extract_v1"
    assert sig.get("splitter_version") == "doc_normalize_v1"
    assert sig.get("chunker_version") == "doc_chunk_v2"
    history = (manifest.meta or {}).get("ingest_history")
    assert isinstance(history, list)
    assert len(history) >= 2
    assert history[-1]["signature"]["chunk_size_tokens"] == 60
