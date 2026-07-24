"""Memory management: management API, trust updates, provenance, wiki subsystem, manifest supersession.

Covers explain_result, lint_memory_drift, trust update with audit history,
provenance chain validation, wiki page identity and evidence links,
and manifest version supersession.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from uma.api.management import explain_result, lint_memory_drift
from uma.api.memory import UMAMemory
from uma.api.runtime import UMARuntime
from uma.common.storage_metadata import normalize_chunk_metadata, normalize_fact_metadata
from uma.common.types import Chunk, Fact, RuntimeContext, SCOPE_MODEL_VERSION
from uma.ingest.ingest_service import capture_source, curate_compiled_memory, derive_memory_artifacts
from uma.ingest.types import IngestConfig
from uma.memory import wiki as wiki_module
import pytest
import uma.api.management as management_api

# ── test_management_api ──────────────────────────────────────────





def test_memory_public_surface_keeps_animus_support_and_drops_management_methods() -> None:
    assert hasattr(UMAMemory, "load_userprofile")
    assert hasattr(UMAMemory, "load_agentprofile")
    assert hasattr(UMAMemory, "load_memory_bootstrap")
    assert hasattr(UMAMemory, "load_daily_diary_bootstrap")
    assert not hasattr(UMAMemory, "expand_evidence")
    assert not hasattr(UMAMemory, "compile_memory_artifact")
    assert not hasattr(UMAMemory, "explain_result")
    assert not hasattr(UMAMemory, "lint_memory_drift")


def test_management_module_exports_supported_operations_only() -> None:
    assert management_api.__all__ == [
        "explain_result",
        "lint_memory_drift",
        "QuarantinedRecord",
        "list_quarantined",
        "reinstate_quarantined",
        "purge_quarantined",
        "IntegrityVerificationResult",
        "verify_integrity",
        "list_retrieval_audit",
    ]


def test_animus_profile_loaders_remain_public_and_functional(uma_memory, tmp_path) -> None:
    user_profile = tmp_path / "USER.md"
    agent_profile = tmp_path / "SOUL.md"
    user_profile.write_text("# User\nlikes coffee\n", encoding="utf-8")
    agent_profile.write_text("# Agent\nprefers concise answers\n", encoding="utf-8")

    assert uma_memory.load_userprofile(str(user_profile)) is uma_memory
    assert uma_memory.load_agentprofile(str(agent_profile)) is uma_memory
    assert "likes coffee" in uma_memory.animus_profile_provider.get_user_profile_text()
    assert "prefers concise answers" in uma_memory.animus_profile_provider.get_agent_profile_text()


@pytest.mark.asyncio
async def test_management_explain_uses_canonical_provenance(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "management-doc.txt"
    path.write_text(
        "Prometheus backs metrics collection for production systems and feeds alerting workflows.\n"
    )
    await memory.ingest_document(str(path), owner_type="user", owner_id="user:u1")

    runtime = UMARuntime.from_memory(memory)
    context = RuntimeContext(
        tenant_id="default",
        agent_id=memory.agent_id,
        request_id="req-management",
        user_id="user:u1",
    )
    memory_result = await runtime.retrieve_memory(
        context,
        query_text="What handles production metrics?",
        memory_intent="continuity",
        include_debug=True,
    )

    artifact = memory_result.debug["compiled_answer"]
    explanation = await explain_result(memory, artifact, user_id="user:u1")

    assert explanation["evidence"]
    assert explanation["chunk_ids"] == artifact["provenance"]["source_chunk_ids"]
    assert explanation["compiled_memory_index"]["artifact_id"] == artifact["id"]


@pytest.mark.asyncio
async def test_management_lint_reports_invalid_parent_lineage_without_rewriting(uma_memory) -> None:
    memory = uma_memory
    manual_parent = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="wiki:manual/root",
        title="Manual Root",
        owner_type="user",
        owner_id="user:u1",
        manual=True,
    )["compiled_artifact"]
    child = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="wiki:manual/child",
        title="Manual Child",
        owner_type="user",
        owner_id="user:u1",
        parent_artifacts=[manual_parent],
    )["compiled_artifact"]

    lint_result = await lint_memory_drift(memory, [child], user_id="user:u1", stale_after_seconds=0)

    issues = {finding["issue"] for finding in lint_result["findings"]}
    assert lint_result["status"] == "issues_found"
    assert "invalid_provenance" in issues
    assert "broken_parent_lineage" in issues


# ── test_semantic_update_trust ──────────────────────────────────────────






def _build_fact(
    *,
    fact_id: str,
    tenant_id: str,
    owner_type: str,
    owner_id: str,
    trust_score: float = 0.9,
) -> Fact:
    now = datetime.now(timezone.utc)
    return Fact(
        id=fact_id,
        subject="service",
        predicate="USES",
        object="postgres",
        created_at=now,
        updated_at=now,
        source_ids=["chunk-1"],
        confidence=0.95,
        meta={},
        owner_type=owner_type,  # type: ignore[arg-type]
        owner_id=owner_id,
        tenant_id=tenant_id,
        trust_score=trust_score,
        content_hash="fact-hash",
    )


@pytest.mark.asyncio
async def test_update_trust_updates_fact_and_records_audit_history(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_once",
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)

    ctx = RuntimeContext(
        tenant_id="default",
        agent_id=uma_memory.agent_id,
        request_id="req-trust-once",
        user_id="user:u1",
    )
    await uma_memory.semantic_core.update_trust(
        fact.id,
        0.5,
        reason="operator downgrade after review",
        ctx=ctx,
    )

    updated = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
    )
    assert updated is not None
    assert updated.trust_score == pytest.approx(0.5)
    assert len(updated.meta["trust_updates"]) == 1
    entry = updated.meta["trust_updates"][0]
    assert entry["prior_score"] == pytest.approx(0.9)
    assert entry["new_score"] == pytest.approx(0.5)
    assert entry["reason"] == "operator downgrade after review"
    assert entry["timestamp"]


@pytest.mark.asyncio
async def test_update_trust_accumulates_history_in_order(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_twice",
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)
    ctx = RuntimeContext(
        tenant_id="default",
        agent_id=uma_memory.agent_id,
        request_id="req-trust-twice",
        user_id="user:u1",
    )

    await uma_memory.semantic_core.update_trust(fact.id, 0.6, reason="first review", ctx=ctx)
    await uma_memory.semantic_core.update_trust(fact.id, 0.4, reason="second review", ctx=ctx)

    updated = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
    )
    assert updated is not None
    assert updated.trust_score == pytest.approx(0.4)
    assert [entry["reason"] for entry in updated.meta["trust_updates"]] == ["first review", "second review"]
    assert [entry["prior_score"] for entry in updated.meta["trust_updates"]] == [pytest.approx(0.9), pytest.approx(0.6)]
    assert [entry["new_score"] for entry in updated.meta["trust_updates"]] == [pytest.approx(0.6), pytest.approx(0.4)]


@pytest.mark.asyncio
async def test_update_trust_rejects_out_of_range_scores_without_mutation(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_invalid_score",
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)
    ctx = RuntimeContext(
        tenant_id="default",
        agent_id=uma_memory.agent_id,
        request_id="req-trust-invalid-score",
        user_id="user:u1",
    )

    with pytest.raises(ValueError, match="new_score must be a float in \\[0.0, 1.0\\]"):
        await uma_memory.semantic_core.update_trust(fact.id, 1.1, reason="too high", ctx=ctx)
    with pytest.raises(ValueError, match="new_score must be a float in \\[0.0, 1.0\\]"):
        await uma_memory.semantic_core.update_trust(fact.id, -0.1, reason="too low", ctx=ctx)

    unchanged = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
    )
    assert unchanged is not None
    assert unchanged.trust_score == pytest.approx(0.9)
    assert unchanged.meta.get("trust_updates") is None


@pytest.mark.asyncio
async def test_update_trust_rejects_empty_reason_without_mutation(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_empty_reason",
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)
    ctx = RuntimeContext(
        tenant_id="default",
        agent_id=uma_memory.agent_id,
        request_id="req-trust-empty-reason",
        user_id="user:u1",
    )

    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        await uma_memory.semantic_core.update_trust(fact.id, 0.5, reason="   ", ctx=ctx)

    unchanged = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
    )
    assert unchanged is not None
    assert unchanged.trust_score == pytest.approx(0.9)
    assert unchanged.meta.get("trust_updates") is None


@pytest.mark.asyncio
async def test_update_trust_hides_other_agent_fact_from_unrelated_context(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_other_agent",
        tenant_id="default",
        owner_type="agent",
        owner_id="agent-other",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)
    ctx = RuntimeContext(
        tenant_id="default",
        agent_id=uma_memory.agent_id,
        request_id="req-trust-other-agent",
        user_id="user:u1",
    )

    with pytest.raises(ValueError, match="not found"):
        await uma_memory.semantic_core.update_trust(fact.id, 0.5, reason="not visible", ctx=ctx)

    unchanged = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="default",
        owner_type="agent",
        owner_id="agent-other",
    )
    assert unchanged is not None
    assert unchanged.trust_score == pytest.approx(0.9)
    assert unchanged.meta.get("trust_updates") is None


@pytest.mark.asyncio
async def test_update_trust_hides_cross_tenant_fact_from_mismatched_context(uma_memory) -> None:
    embedding = (await uma_memory.embedder.embed(["service uses postgres"]))[0]
    fact = _build_fact(
        fact_id="fact_trust_other_tenant",
        tenant_id="tenant-b",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.9,
    )
    await uma_memory.semantic_core.upsert_fact(fact, embedding)
    ctx = RuntimeContext(
        tenant_id="tenant-a",
        agent_id=uma_memory.agent_id,
        request_id="req-trust-other-tenant",
        user_id="user:u1",
    )

    with pytest.raises(ValueError, match="not found"):
        await uma_memory.semantic_core.update_trust(fact.id, 0.5, reason="wrong tenant", ctx=ctx)

    unchanged = await uma_memory.semantic_core.store.get_fact(
        fact.id,
        tenant_id="tenant-b",
        owner_type="user",
        owner_id="user:u1",
    )
    assert unchanged is not None
    assert unchanged.trust_score == pytest.approx(0.9)
    assert unchanged.meta.get("trust_updates") is None


# ── test_provenance_runtime ──────────────────────────────────────────





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
        agent_id=memory.agent_id,
        request_id="req-provenance",
        user_id="user:u1",
    )
    memory_result = await runtime.retrieve_memory(
        context,
        query_text="What platform do we use for production workloads?",
        memory_intent="continuity",
        include_debug=True,
    )
    assert memory_result.provenance_valid is True
    assert memory_result.debug["compiled_answer"]["provenance"]["source_chunk_ids"]
    assert memory_result.debug["compiled_memory_index"]
    assert memory_result.debug["compiled_memory_log"]

    fact_evidence = await explain_result(memory, fact, user_id="user:u1")
    assert fact_evidence["evidence"]
    assert fact_evidence["chunk_ids"]

    answer_evidence = await explain_result(memory, memory_result.debug["compiled_answer"], user_id="user:u1")
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

    wiki_evidence = await explain_result(memory, wiki_artifact, user_id="user:u1")
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
        agent_id=memory.agent_id,
        request_id="req-compiled-artifact",
        user_id="user:u1",
    )
    memory_result = await runtime.retrieve_memory(
        context,
        query_text="What monitoring platform do we use?",
        memory_intent="continuity",
        include_debug=True,
    )

    page = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="wiki:operations/monitoring",
        title="Operations Monitoring",
        owner_type="user",
        owner_id="user:u1",
        summary="Monitoring platform used in operations.",
        parent_artifacts=[memory_result.debug["compiled_answer"]],
        related_artifact_ids=[memory_result.debug["compiled_answer"]["id"]],
        retrieval_tags=["ops", "monitoring"],
    )
    artifact = page["compiled_artifact"]

    assert artifact["artifact_type"] == "compiled_memory_artifact"
    assert artifact["provenance"]["valid"] is True
    assert artifact["compiled_memory_index"]["artifact_id"] == "wiki:operations/monitoring"
    assert artifact["compiled_memory_index"]["source_chunk_ids"] == artifact["provenance"]["source_chunk_ids"]
    assert artifact["compiled_memory_log"][0]["event_type"] == "wiki_artifact_created"
    assert artifact["compiled_memory_log"][0]["parent_artifact_ids"] == [memory_result.debug["compiled_answer"]["id"]]

    expanded = await explain_result(memory, artifact, user_id="user:u1")
    assert expanded["chunk_ids"] == artifact["provenance"]["source_chunk_ids"]
    assert expanded["direct_chunk_ids"] == []
    assert expanded["lineage"][0]["parent_artifact_ids"] == [memory_result.debug["compiled_answer"]["id"]]
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

    original = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="wiki:operations/paging",
        title="Operations Paging",
        owner_type="user",
        owner_id="user:u1",
        direct_source_chunk_ids=[chunk_id],
        direct_source_document_ids=[report.doc_id],
    )["compiled_artifact"]
    artifact = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="wiki:operations/paging",
        title="Operations Paging",
        owner_type="user",
        owner_id="user:u1",
        direct_source_chunk_ids=[chunk_id],
        direct_source_document_ids=[report.doc_id],
        conflicts=[conflict],
        existing_page=original,
    )["compiled_artifact"]

    assert artifact["conflicts"] == [conflict]
    assert artifact["compiled_memory_index"]["has_conflicts"] is True
    assert [event["event_type"] for event in artifact["compiled_memory_log"]] == [
        "wiki_artifact_created",
        "wiki_artifact_updated",
        "conflict_detected",
    ]

    expanded = await explain_result(memory, artifact, user_id="user:u1")
    assert expanded["provenance"]["conflicts"] == [conflict]
    assert expanded["chunk_ids"] == [chunk_id]


def test_compiled_memory_artifact_without_reachable_raw_chunks_is_invalid(uma_memory) -> None:
    memory = uma_memory
    orphan_parent = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="wiki:orphan-parent",
        title="Orphan Parent",
        owner_type="user",
        owner_id="user:u1",
        related_artifact_ids=["wiki:missing-raw"],
        manual=True,
    )["compiled_artifact"]
    child = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="wiki:orphan-child",
        title="Orphan Child",
        owner_type="user",
        owner_id="user:u1",
        parent_artifacts=[orphan_parent],
    )["compiled_artifact"]

    assert child["provenance"]["valid"] is False
    assert "missing_source_chunk_ids" in child["provenance"]["invalid_reasons"]
    assert "unreachable_raw_source_chunks" in child["provenance"]["invalid_reasons"]


def test_manual_compiled_memory_artifact_keeps_audit_without_fake_chunks(uma_memory) -> None:
    artifact = wiki_module.regenerate_wiki_page(
        memory=uma_memory,
        page_key="wiki:manual/notes",
        title="Manual Notes",
        owner_type="user",
        owner_id="user:u1",
        manual=True,
    )["compiled_artifact"]

    assert artifact["provenance"]["manual"] is True
    assert artifact["provenance"]["source_chunk_ids"] == []
    assert artifact["provenance"]["valid"] is True
    assert artifact["compiled_memory_log"][0]["event_type"] == "manual_update"
    assert artifact["compiled_memory_log"][0]["manual"] is True


def test_compiled_memory_artifact_update_appends_log_event_instead_of_replacing_history(uma_memory) -> None:
    original = wiki_module.regenerate_wiki_page(
        memory=uma_memory,
        page_key="wiki:ops/platform",
        title="Ops Platform",
        owner_type="user",
        owner_id="user:u1",
        direct_source_chunk_ids=["chunk-1"],
        direct_source_document_ids=["doc-1"],
    )["compiled_artifact"]
    updated = wiki_module.regenerate_wiki_page(
        memory=uma_memory,
        page_key="wiki:ops/platform",
        title="Ops Platform",
        owner_type="user",
        owner_id="user:u1",
        direct_source_chunk_ids=["chunk-1"],
        direct_source_document_ids=["doc-1"],
        existing_page=original,
    )["compiled_artifact"]

    assert [event["event_type"] for event in updated["compiled_memory_log"]] == [
        "wiki_artifact_created",
        "wiki_artifact_updated",
    ]


@pytest.mark.asyncio
async def test_expand_evidence_is_cycle_safe_for_parent_artifact_lineage(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "cycle-safe-doc.txt"
    path.write_text(
        "Grafana dashboards summarize service health and operational trends. "
        "Teams use them alongside alerts to inspect incidents and capacity changes.\n"
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
            log_context="test_expand_evidence_cycle_safe_lookup_chunks",
        )
    finally:
        conn.close()
    assert rows
    chunk_id = rows[0]["id"]

    artifact_a = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="wiki:cycle/a",
        title="Cycle A",
        owner_type="user",
        owner_id="user:u1",
        direct_source_chunk_ids=[chunk_id],
        direct_source_document_ids=[report.doc_id],
    )["compiled_artifact"]
    artifact_b = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="wiki:cycle/b",
        title="Cycle B",
        owner_type="user",
        owner_id="user:u1",
        parent_artifacts=[artifact_a],
    )["compiled_artifact"]
    artifact_a["parent_artifacts"] = [artifact_b]
    artifact_a["parent_artifact_ids"] = [artifact_b["id"]]

    expanded = await explain_result(memory, artifact_b, user_id="user:u1")
    assert expanded["chunk_ids"] == [chunk_id]
    assert expanded["transitive_chunk_ids"] == [chunk_id]
    assert expanded["missing_chunk_ids"] == []


# ── test_wiki_subsystem ──────────────────────────────────────────






def test_wiki_page_identity_is_deterministic() -> None:
    first = wiki_module.page_identity_for_key("Operations Platform")
    second = wiki_module.page_identity_for_key("operations platform")
    prefixed = wiki_module.page_identity_for_key("wiki:operations/platform")

    assert first["slug"] == "operations-platform"
    assert first == second
    assert prefixed == {"page_id": "wiki:operations/platform", "slug": "operations/platform"}


@pytest.mark.asyncio
async def test_regenerated_wiki_page_is_canonical_record_with_evidence_links(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "wiki-canonical.txt"
    path.write_text(
        "VictoriaMetrics stores long-term metrics for platform operations and auditing. "
        "The operations handbook explains retention, service health review, and evidence-backed incident analysis.\n"
    )
    await memory.ingest_document(str(path), owner_type="user", owner_id="user:u1")

    runtime = UMARuntime.from_memory(memory)
    context = RuntimeContext(
        tenant_id="default",
        agent_id=memory.agent_id,
        request_id="req-wiki-canonical",
        user_id="user:u1",
    )
    memory_result = await runtime.retrieve_memory(
        context,
        query_text="What stores long-term metrics?",
        memory_intent="continuity",
        include_debug=True,
    )

    page = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="ops/metrics",
        title="Ops Metrics",
        owner_type="user",
        owner_id="user:u1",
        parent_artifacts=[memory_result.debug["compiled_answer"]],
        category="operations",
    )

    assert page["page_type"] == wiki_module.WIKI_PAGE_RECORD_TYPE
    assert page["page_id"] == "wiki:ops/metrics"
    assert page["compiled_artifact"]["id"] == page["page_id"]
    assert page["evidence_links"]["source_chunk_ids"]
    assert page["provenance"]["valid"] is True


def test_wiki_page_refresh_keeps_identity_when_title_changes(uma_memory) -> None:
    original = wiki_module.regenerate_wiki_page(
        memory=uma_memory,
        page_key="wiki:ops/metrics",
        title="Ops Metrics",
        owner_type="user",
        owner_id="user:u1",
        manual=True,
    )
    refreshed = wiki_module.regenerate_wiki_page(
        memory=uma_memory,
        page_key="wiki:ops/metrics",
        title="Platform Metrics",
        owner_type="user",
        owner_id="user:u1",
        existing_page=original,
        manual=True,
    )

    assert original["page_id"] == "wiki:ops/metrics"
    assert refreshed["page_id"] == original["page_id"]
    assert refreshed["slug"] == original["slug"]
    assert refreshed["title"] == "Platform Metrics"



@pytest.mark.asyncio
async def test_wiki_lint_reports_invalid_parent_lineage(uma_memory) -> None:
    manual_parent = wiki_module.regenerate_wiki_page(
        memory=uma_memory,
        page_key="manual/root",
        title="Manual Root",
        owner_type="user",
        owner_id="user:u1",
        manual=True,
    )
    child = wiki_module.regenerate_wiki_page(
        memory=uma_memory,
        page_key="manual/child",
        title="Manual Child",
        owner_type="user",
        owner_id="user:u1",
        parent_artifacts=[manual_parent["compiled_artifact"]],
    )

    lint_result = await lint_memory_drift(uma_memory, [child], user_id="user:u1")

    issues = {finding["issue"] for finding in lint_result["findings"]}
    assert lint_result["status"] == "issues_found"
    assert "invalid_provenance" in issues
    assert "broken_parent_lineage" in issues


@pytest.mark.asyncio
async def test_stage3_curation_uses_wiki_subsystem(uma_memory, tmp_path, monkeypatch) -> None:
    memory = uma_memory
    path = tmp_path / "wiki-stage3.txt"
    path.write_text(
        "OpenSearch indexes operational logs for investigation workflows. "
        "The runbook explains incident search, retention policy checks, and evidence-backed response steps.\n"
    )
    config = IngestConfig(doc_episode_enabled=False)
    capture = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        tenant_id="default",
        config=config,
        memory=memory,
    )
    derive = await derive_memory_artifacts(
        capture,
        config=config,
        memory=memory,
    )
    seen: list[str] = []
    original = wiki_module.regenerate_wiki_page

    def wrapped(**kwargs):
        seen.append(str(kwargs["page_key"]))
        return original(**kwargs)

    monkeypatch.setattr(wiki_module, "regenerate_wiki_page", wrapped)

    curated = await curate_compiled_memory(capture, derive, memory=memory)

    assert seen == [capture.parsed.doc_id]
    assert curated.compiled_artifacts[0]["id"].startswith("wiki:")



# ── test_manifest_supersession ──────────────────────────────────────────






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
