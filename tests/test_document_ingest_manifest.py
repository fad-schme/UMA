import asyncio
from datetime import datetime, timezone
import yaml

from uma.core.uma_memory import UMAMemory
from uma.stores.document_sql import DocumentRecord


def _good_embedder(texts):
    return [[0.0, 0.0, 0.0] for _ in texts]


def _good_llm(messages, **kwargs):
    return "ok"


def test_document_manifest_persistence(tmp_path):
    cfg = {
        "storage": {
            "db_root": str(tmp_path),
            "sql_backend": "sqlite",
            "vector_backend": "inmemory",
            "graph_backend": "disabled",
        },
        "working_memory": {
            "max_tokens": 100,
            "warning_ratio": 0.7,
            "hard_limit_ratio": 0.95,
            "chunk_size": 10,
        },
        "embedding": {
            "provider": "tests.test_document_ingest_manifest:_good_embedder",
            "model": "x",
            "dimension": 3,
            "config": {"preflight": False},
        },
        "llm": {
            "provider": "tests.test_document_ingest_manifest:_good_llm",
            "model": "x",
            "config": {"preflight": False},
        },
        "retrieval": {"max_episodes": 1, "max_facts": 1, "max_skills": 1, "max_graph_items": 1},
        "consolidation": {"enabled": False, "cluster_similarity": 0.75, "max_episodes_per_cycle": 10, "prune_min_fact_salience": 0.2},
        "features": {"load": [], "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False}},
    }

    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    mem = UMAMemory.from_yaml(str(cfg_path))

    record = DocumentRecord(
        doc_id="doc_test",
        source_path="/tmp/doc.txt",
        source_hash="hash123",
        ingested_at=datetime.now(timezone.utc),
        owner_type="user",
        owner_id="u1",
        meta={},
    )

    asyncio.run(mem.document_store.upsert_document(record))

    # Verify record exists by querying directly
    conn = mem.document_store._conn()
    try:
        rows = mem.document_store._query_all(
            conn,
            "SELECT * FROM documents WHERE doc_id=?",
            params=["doc_test"],
            log_context="test_document_manifest",
        )
        assert rows and rows[0]["source_hash"] == "hash123"
    finally:
        conn.close()
