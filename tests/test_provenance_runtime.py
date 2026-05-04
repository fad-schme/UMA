from __future__ import annotations

import pytest

from uma.api.runtime import UMARuntime
from uma.common.storage_metadata import normalize_fact_metadata
from uma.common.storage_metadata import normalize_chunk_metadata
from uma.common.types import Chunk, Fact, RuntimeContext, SCOPE_MODEL_VERSION


@pytest.mark.asyncio
async def test_provenance_chain_supports_fact_memory_and_wiki_artifact_expansion(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "provenance-doc.txt"
    path.write_text(
        "Kubernetes is the shared deployment platform for production workloads. "
        "The operations handbook documents rollout, rollback, and service ownership procedures.\n"
    )

    report = await memory.ingest_document(str(path), owner_type="user", owner_id="user:u1")
    chunk_store = getattr(memory.chunk_core, "store", None)
    assert chunk_store is not None
    conn = chunk_store._conn()
    try:
        rows = chunk_store._query_all(
            conn,
            """
            SELECT id
            FROM chunks
            WHERE tenant_id = ?
              AND owner_type = ?
              AND owner_id = ?
              AND doc_id = ?
            ORDER BY position ASC
            """,
            params=["default", "user", "user:u1", report.doc_id],
            log_context="test_provenance_runtime_lookup_chunks",
        )
    finally:
        conn.close()
    assert rows
    chunk_id = rows[0]["id"]

    source_chunks = await memory.chunk_core._fetch_by_ids(
        ids=[chunk_id],
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
    )
    assert source_chunks
    raw_chunk = source_chunks[0]
    now = raw_chunk.updated_at
    fact = Fact(
        id="fact_provenance_chain",
        subject="operations",
        predicate="USES",
        object="kubernetes",
        created_at=now,
        updated_at=now,
        source_ids=[chunk_id],
        confidence=0.95,
        owner_type="user",
        owner_id="user:u1",
        tenant_id="default",
        scope_model_version=SCOPE_MODEL_VERSION,
        meta=normalize_fact_metadata(
            {},
            fact_id="fact_provenance_chain",
            owner_type="user",
            owner_id="user:u1",
            created_at=now,
            updated_at=now,
            source_ids=[chunk_id],
            session_id=None,
        ),
    )
    await memory.semantic_core.upsert_fact(fact, embedding=[0.0] * int(memory.embedding_cfg.dimension))
    assert fact.meta["provenance"]["source_chunk_ids"]
    assert fact.meta["provenance"]["valid"] is True

    runtime = UMARuntime.from_memory(memory)
    context = RuntimeContext(
        tenant_id="default",
        agent_id=memory.agent_id or "agent-default",
        request_id="req-provenance",
        user_id="user:u1",
    )
    memory_result = await runtime.retrieve_memory(
        context,
        query_text="What platform do we use for production workloads?",
        memory_intent="continuity",
    )
    assert memory_result["compiled_answer"]["provenance"]["source_chunk_ids"]
    assert memory_result["compiled_memory_index"]
    assert memory_result["compiled_memory_log"]
    assert memory_result["compiled_memory_index"][0]["navigation_only"] is True

    fact_evidence = await memory.expand_evidence(fact)
    assert fact_evidence["evidence"]
    assert fact_evidence["chunk_ids"]

    answer_evidence = await memory.expand_evidence(memory_result["compiled_answer"])
    assert answer_evidence["evidence"]
    assert answer_evidence["chunk_ids"]

    wiki_chunk = Chunk(
        id=raw_chunk.id,
        doc_id=raw_chunk.doc_id,
        text=raw_chunk.text,
        page_range=raw_chunk.page_range,
        position=raw_chunk.position,
        source_path=raw_chunk.source_path,
        source_hash=raw_chunk.source_hash,
        created_at=raw_chunk.created_at,
        updated_at=raw_chunk.updated_at,
        owner_type=raw_chunk.owner_type,
        owner_id=raw_chunk.owner_id,
        tenant_id=raw_chunk.tenant_id,
        workspace_id=raw_chunk.workspace_id,
        meta=normalize_chunk_metadata(
            {
                "kind": "wiki_page",
                "kb_lane": "wiki",
                "page_slug": "operations/platform",
                "page_title": "Operations Platform",
                "category": "ops",
                "status": "active",
            },
            chunk_id=raw_chunk.id,
            doc_id=raw_chunk.doc_id,
            owner_type=raw_chunk.owner_type,
            owner_id=raw_chunk.owner_id,
            created_at=raw_chunk.created_at,
            updated_at=raw_chunk.updated_at,
            page_range=raw_chunk.page_range,
            position=raw_chunk.position,
            source_path=raw_chunk.source_path,
            source_hash=raw_chunk.source_hash,
        ),
    )
    wiki_artifact = runtime._group_memory_artifacts([wiki_chunk])[0]
    assert wiki_artifact["artifact_type"] == "compiled_memory_artifact"
    assert wiki_artifact["provenance"]["source_chunk_ids"] == [raw_chunk.id]
    assert wiki_artifact["provenance"]["valid"] is True
    assert wiki_artifact["compiled_memory_index"]["artifact_id"] == wiki_artifact["id"]
    assert wiki_artifact["compiled_memory_log"][0]["event_type"] == "wiki_artifact_created"

    wiki_evidence = await memory.expand_evidence(wiki_artifact)
    assert wiki_evidence["chunk_ids"] == [raw_chunk.id]


@pytest.mark.asyncio
async def test_compiled_memory_artifact_builds_index_log_and_transitive_raw_evidence(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "wiki-artifact-doc.txt"
    path.write_text(
        "Datadog is the monitoring platform for production operations. "
        "The runbook uses it for dashboards, alert routing, and service-level health checks.\n"
    )

    await memory.ingest_document(str(path), owner_type="user", owner_id="user:u1")
    runtime = UMARuntime.from_memory(memory)
    context = RuntimeContext(
        tenant_id="default",
        agent_id=memory.agent_id or "agent-default",
        request_id="req-compiled-artifact",
        user_id="user:u1",
    )
    memory_result = await runtime.retrieve_memory(
        context,
        query_text="What monitoring platform do we use?",
        memory_intent="continuity",
    )

    artifact = memory.compile_memory_artifact(
        artifact_id="wiki:operations/monitoring",
        title="Operations Monitoring",
        owner_type="user",
        owner_id="user:u1",
        summary="Monitoring platform used in operations.",
        topic_key="operations/monitoring",
        parent_artifacts=[memory_result["compiled_answer"]],
        related_artifact_ids=[memory_result["compiled_answer"]["id"]],
        retrieval_tags=["ops", "monitoring"],
    )

    assert artifact["artifact_type"] == "compiled_memory_artifact"
    assert artifact["provenance"]["valid"] is True
    assert artifact["compiled_memory_index"]["artifact_id"] == "wiki:operations/monitoring"
    assert artifact["compiled_memory_index"]["navigation_only"] is True
    assert artifact["compiled_memory_index"]["source_chunk_ids"] == artifact["provenance"]["source_chunk_ids"]
    assert artifact["compiled_memory_log"][0]["event_type"] == "wiki_artifact_created"
    assert artifact["compiled_memory_log"][0]["parent_artifact_ids"] == [memory_result["compiled_answer"]["id"]]

    expanded = await memory.expand_evidence(artifact)
    assert expanded["chunk_ids"] == artifact["provenance"]["source_chunk_ids"]
    assert expanded["direct_chunk_ids"] == []
    assert expanded["lineage"][0]["parent_artifact_ids"] == [memory_result["compiled_answer"]["id"]]
    assert expanded["compiled_memory_log"][0]["event_type"] == "evidence_expanded"


@pytest.mark.asyncio
async def test_compiled_memory_artifact_update_and_conflicts_remain_visible(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "wiki-conflict-doc.txt"
    path.write_text(
        "PagerDuty handles incident paging for critical services. "
        "The incident handbook records escalation policy ownership and responder routing details.\n"
    )

    report = await memory.ingest_document(str(path), owner_type="user", owner_id="user:u1")
    chunk_store = getattr(memory.chunk_core, "store", None)
    assert chunk_store is not None
    conn = chunk_store._conn()
    try:
        rows = chunk_store._query_all(
            conn,
            "SELECT id FROM chunks WHERE tenant_id=? AND owner_type=? AND owner_id=? AND doc_id=? ORDER BY position ASC",
            params=["default", "user", "user:u1", report.doc_id],
            log_context="test_compiled_memory_artifact_update_lookup_chunks",
        )
    finally:
        conn.close()
    assert rows
    chunk_id = rows[0]["id"]
    conflict = {
        "field": "pager_service",
        "status": "unresolved",
        "sides": [
            {"value": "PagerDuty", "source_chunk_ids": [chunk_id], "support_density": 1.0},
            {"value": "Opsgenie", "source_chunk_ids": [chunk_id], "support_density": 0.2},
        ],
    }

    artifact = memory.compile_memory_artifact(
        artifact_id="wiki:operations/paging",
        title="Operations Paging",
        owner_type="user",
        owner_id="user:u1",
        direct_source_chunk_ids=[chunk_id],
        direct_source_document_ids=[report.doc_id],
        conflicts=[conflict],
        operation="wiki_artifact_updated",
    )

    assert artifact["conflicts"] == [conflict]
    assert artifact["compiled_memory_index"]["has_conflicts"] is True
    assert artifact["compiled_memory_log"][0]["event_type"] == "wiki_artifact_updated"
    assert artifact["compiled_memory_log"][1]["event_type"] == "conflict_detected"

    expanded = await memory.expand_evidence(artifact)
    assert expanded["provenance"][0]["conflicts"] == [conflict]
    assert expanded["chunk_ids"] == [chunk_id]


def test_compiled_memory_artifact_without_reachable_raw_chunks_is_invalid(uma_memory) -> None:
    memory = uma_memory
    orphan_parent = memory.compile_memory_artifact(
        artifact_id="wiki:orphan-parent",
        title="Orphan Parent",
        owner_type="user",
        owner_id="user:u1",
        related_artifact_ids=["wiki:missing-raw"],
        operation="wiki_artifact_created",
        manual=False,
    )
    child = memory.compile_memory_artifact(
        artifact_id="wiki:orphan-child",
        title="Orphan Child",
        owner_type="user",
        owner_id="user:u1",
        parent_artifacts=[orphan_parent],
        operation="wiki_artifact_created",
    )

    assert child["provenance"]["valid"] is False
    assert "missing_source_chunk_ids" in child["provenance"]["invalid_reasons"]
    assert "unreachable_raw_source_chunks" in child["provenance"]["invalid_reasons"]


def test_manual_compiled_memory_artifact_keeps_audit_without_fake_chunks(uma_memory) -> None:
    artifact = uma_memory.compile_memory_artifact(
        artifact_id="wiki:manual/notes",
        title="Manual Notes",
        owner_type="user",
        owner_id="user:u1",
        manual=True,
        operation="manual_update",
    )

    assert artifact["provenance"]["manual"] is True
    assert artifact["provenance"]["source_chunk_ids"] == []
    assert artifact["provenance"]["valid"] is True
    assert artifact["compiled_memory_log"][0]["event_type"] == "manual_update"
