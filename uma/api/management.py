"""Developer and admin management APIs for UMA memory.

This module keeps inspection, curation, projection, and drift checks out of
the normal product-facing `UMAMemory` surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from uma.common.integrity import (
    hash_chunk_content,
    hash_episode_content,
    hash_fact_content,
    hash_skill_content,
)
from uma.common.provenance import provenance_for_artifact
from uma.common.types.types_scope import DEFAULT_TENANT_ID
from uma.memory import wiki as wiki_module

if TYPE_CHECKING:
    from .memory import UMAMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quarantine types
# ---------------------------------------------------------------------------

_LANE_STORE_KEY = {
    "semantic": "semantic",
    "episodic": "episodic",
    "procedural": "procedural",
    "raw": "chunk",
}


@dataclass
class IntegrityVerificationResult:
    record_id: str
    lane: str
    status: str                      # "verified" or "failed"
    expected_hash: Optional[str]     # stored hash; populated on failure
    actual_hash: Optional[str]       # recomputed hash; populated on failure
    quarantined: bool                # True if this call quarantined the record


@dataclass
class QuarantinedRecord:
    id: str
    lane: str
    quarantined_at: datetime
    severity: str
    matched_rules: list[str]
    owner_type: str
    owner_id: str
    content_preview: str


def _artifact_id(artifact: Any) -> str | None:
    if isinstance(artifact, Mapping):
        value = artifact.get("id") or artifact.get("artifact_id") or artifact.get("doc_id")
        if value is not None:
            return str(value)
    return None


def _primary_artifact(result: Any) -> Any:
    if isinstance(result, Mapping) and result.get("compiled_answer") is not None:
        return result["compiled_answer"]
    if isinstance(result, Mapping) and result.get("compiled_artifact") is not None:
        return result["compiled_artifact"]
    return result


async def explain_result(
    memory: "UMAMemory",
    result: Any,
    *,
    agent_id: str,
    user_id: str,
    tenant_id: str = DEFAULT_TENANT_ID,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Explain a retrieval result or compiled artifact using canonical provenance."""
    runtime_context = memory._resolve_runtime_context(
        agent_id=agent_id,
        user_id=user_id,
        tenant_id=tenant_id,
        request_id="management:explain_result",
        workspace_id=workspace_id,
        session_id=None,
    )
    artifact = _primary_artifact(result)
    page = wiki_module.wiki_page_from_record(artifact)
    expanded = await memory.runtime.expand_evidence(runtime_context, artifact)
    provenance = provenance_for_artifact(artifact)
    retrieval_path = provenance.get("retrieval_path")
    if not retrieval_path and isinstance(result, Mapping):
        retrieval_path = result.get("trace") or result.get("retrieval_path") or []
    compiled_index = None
    compiled_log = list(expanded.get("compiled_memory_log") or [])
    if isinstance(artifact, Mapping):
        compiled_index = artifact.get("compiled_memory_index")
    if isinstance(result, Mapping):
        if compiled_index is None:
            index_entries = result.get("compiled_memory_index") or []
            if isinstance(index_entries, list) and index_entries:
                compiled_index = index_entries[0]
    return {
        "status": "explained",
        "artifact_id": _artifact_id(artifact),
        "product": result.get("product") if isinstance(result, Mapping) else None,
        "provenance": provenance,
        "evidence_mode": expanded["mode"],
        "evidence": expanded["evidence"],
        "direct_chunk_ids": expanded["direct_chunk_ids"],
        "transitive_chunk_ids": expanded["transitive_chunk_ids"],
        "chunk_ids": expanded["chunk_ids"],
        "missing_chunk_ids": expanded["missing_chunk_ids"],
        "unresolved_parent_artifact_ids": expanded["unresolved_parent_artifact_ids"],
        "lineage": expanded["lineage"],
        "retrieval_path": retrieval_path or [],
        "compiled_memory_index": compiled_index,
        "compiled_memory_log": compiled_log,
        "conflicts": list(provenance.get("conflicts") or []),
        "invalid_reasons": list(provenance.get("invalid_reasons") or []),
        "wiki_page": page,
    }


_ARTIFACT_TYPE_LANE: dict[str, str] = {}  # populated lazily on first call


def _detect_lane(artifact: Any) -> Optional[str]:
    """Return the memory lane for a typed artifact, or None if it's a wiki/dict artifact."""
    # Import here to avoid circular imports; cache class references on first call
    global _ARTIFACT_TYPE_LANE
    if not _ARTIFACT_TYPE_LANE:
        try:
            from uma.common.types import Fact, Episode, Skill, Chunk
            _ARTIFACT_TYPE_LANE = {
                Fact: "semantic",
                Episode: "episodic",
                Skill: "procedural",
                Chunk: "raw",
            }
        except ImportError:
            return None
    for cls, lane in _ARTIFACT_TYPE_LANE.items():
        if isinstance(artifact, cls):
            return lane
    return None


async def lint_memory_drift(
    memory: "UMAMemory",
    artifacts: Any | Sequence[Any],
    *,
    agent_id: str,
    user_id: str,
    tenant_id: str = DEFAULT_TENANT_ID,
    workspace_id: str | None = None,
    stale_after_seconds: int | None = None,
) -> dict[str, Any]:
    """Lint compiled memory artifacts for provenance and integrity drift.

    Wiki/compiled artifacts are inspected without mutation. For typed lane
    artifacts (Fact, Episode, Skill, Chunk), this function calls
    ``verify_integrity``; a hash mismatch quarantines the record and appends a
    security audit entry. This typed-record path is intentionally mutating.
    Integrity failures are returned with category ``integrity_failure``.
    """
    items = (
        list(artifacts)
        if isinstance(artifacts, Sequence)
        and not isinstance(artifacts, (str, bytes, bytearray, Mapping))
        else [artifacts]
    )
    findings: list[dict[str, Any]] = []
    statuses: list[str] = []

    for artifact in items:
        lane = _detect_lane(artifact)
        if lane is not None:
            # Typed lane record: run integrity verification only
            record_id = getattr(artifact, "id", None)
            owner_type = getattr(artifact, "owner_type", None)
            owner_id = getattr(artifact, "owner_id", None)
            if record_id and owner_type and owner_id:
                try:
                    iv_result = await verify_integrity(
                        memory,
                        record_id=record_id,
                        lane=lane,
                        owner_type=owner_type,
                        owner_id=owner_id,
                        tenant_id=tenant_id,
                    )
                    if iv_result.status == "failed":
                        findings.append({
                            "category": "integrity_failure",
                            "record_id": record_id,
                            "lane": lane,
                            "expected_hash": iv_result.expected_hash,
                            "actual_hash": iv_result.actual_hash,
                            "quarantined": iv_result.quarantined,
                        })
                        statuses.append("integrity_failure")
                    else:
                        statuses.append("ok")
                except Exception as exc:
                    findings.append({
                        "category": "integrity_check_error",
                        "record_id": record_id,
                        "lane": lane,
                        "error": str(exc),
                    })
                    statuses.append("error")
            continue

        # Wiki/compiled artifact: run the existing provenance drift check
        lint_result = await wiki_module.lint_wiki_page(
            memory,
            artifact,
            agent_id=agent_id,
            user_id=user_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            stale_after_seconds=stale_after_seconds,
        )
        findings.extend(list(lint_result["findings"]))
        statuses.append(str(lint_result["drift_status"]))

    status = "ok" if not findings else "issues_found"
    logger.info(
        "lint_memory_drift: artifacts=%d findings=%d status=%s",
        len(items),
        len(findings),
        status,
    )
    return {
        "status": status,
        "artifacts_scanned": len(items),
        "findings": findings,
        "drift_statuses": statuses,
    }

async def verify_integrity(
    memory: "UMAMemory",
    *,
    record_id: str,
    lane: str,
    owner_type: str,
    owner_id: str,
    tenant_id: str = DEFAULT_TENANT_ID,
) -> IntegrityVerificationResult:
    """
    Re-compute the canonical hash of a stored record and compare it to the stored content_hash.

    On mismatch, the record is quarantined through the same path PR4 uses and an
    "integrity_failure" entry is appended to meta.security.audit_log.

    lane must be one of: "semantic", "episodic", "procedural", "raw".
    A missing content_hash indicates a programming error and raises RuntimeError.
    """
    store_key = _LANE_STORE_KEY.get(lane)
    if store_key is None:
        raise ValueError(f"verify_integrity: unknown lane {lane!r}")
    store = memory._stores.get(store_key)
    if store is None:
        raise RuntimeError(f"verify_integrity: store for lane={lane!r} not found")

    # Fetch the record (bypasses quarantine filter — forensics may need to verify quarantined records)
    record = None
    if lane == "semantic":
        record = await store.get_fact(record_id, tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id)
    elif lane == "episodic":
        record = await store.get_episode(record_id, tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id)
    elif lane == "procedural":
        record = await store.get_skill(record_id, tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id)
    elif lane == "raw":
        record = await store.get_chunk(record_id, tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id)

    if record is None:
        raise ValueError(f"verify_integrity: record {record_id!r} not found in lane={lane!r}")

    # Retrieve stored hash and recompute canonical hash
    if lane == "semantic":
        stored_hash = getattr(record, "content_hash", None)
        if not stored_hash:
            raise RuntimeError(
                f"verify_integrity: fact {record_id!r} has no content_hash — this is a programming error"
            )
        actual_hash = hash_fact_content(record.subject, record.predicate, record.object)
    elif lane == "episodic":
        stored_hash = getattr(record, "content_hash", None)
        if not stored_hash:
            raise RuntimeError(
                f"verify_integrity: episode {record_id!r} has no content_hash — this is a programming error"
            )
        actual_hash = hash_episode_content(record.summary)
    elif lane == "procedural":
        stored_hash = getattr(record, "content_hash", None)
        if not stored_hash:
            raise RuntimeError(
                f"verify_integrity: skill {record_id!r} has no content_hash — this is a programming error"
            )
        actual_hash = hash_skill_content(record.name, record.plan)
    elif lane == "raw":
        # Chunks store their text hash in meta["text_hash"] (PR1 invariant)
        stored_hash = (getattr(record, "meta", None) or {}).get("text_hash")
        if not stored_hash:
            raise RuntimeError(
                f"verify_integrity: chunk {record_id!r} has no meta.text_hash — this is a programming error"
            )
        actual_hash = hash_chunk_content(record.text)

    if actual_hash == stored_hash:
        logger.info(
            "verify_integrity: ok record=%s lane=%s owner=%s:%s",
            record_id, lane, owner_type, owner_id,
        )
        return IntegrityVerificationResult(
            record_id=record_id,
            lane=lane,
            status="verified",
            expected_hash=None,
            actual_hash=None,
            quarantined=False,
        )

    # Mismatch: quarantine the record
    now = datetime.now(timezone.utc)
    audit_entry = {
        "event": "integrity_failure",
        "timestamp": now.isoformat(),
        "expected_hash": stored_hash,
        "actual_hash": actual_hash,
    }
    quarantined = await store.quarantine_record(
        record_id,
        tenant_id=tenant_id,
        owner_type=owner_type,
        owner_id=owner_id,
        quarantined_at=now.isoformat(),
        audit_entry=audit_entry,
    )
    logger.warning(
        "verify_integrity: MISMATCH record=%s lane=%s owner=%s:%s quarantined=%s",
        record_id, lane, owner_type, owner_id, quarantined,
    )
    return IntegrityVerificationResult(
        record_id=record_id,
        lane=lane,
        status="failed",
        expected_hash=stored_hash,
        actual_hash=actual_hash,
        quarantined=quarantined,
    )


async def list_quarantined(
    memory: "UMAMemory",
    *,
    owner_type: str,
    owner_id: str,
    tenant_id: str = DEFAULT_TENANT_ID,
    lane: Optional[str] = None,
    limit: int = 100,
) -> list[QuarantinedRecord]:
    """
    Return quarantined records across all lanes (or a specific lane), owner-scoped.
    lane must be one of: "semantic", "episodic", "procedural", "raw" (or None for all).
    """
    lanes_to_check = [lane] if lane else list(_LANE_STORE_KEY.keys())
    results: list[QuarantinedRecord] = []

    for ln in lanes_to_check:
        store_key = _LANE_STORE_KEY.get(ln)
        if store_key is None:
            raise ValueError(f"list_quarantined: unknown lane {ln!r}")
        store = memory._stores.get(store_key)
        if store is None:
            continue

        try:
            if ln == "semantic":
                records = await store.list_facts_for_owner(
                    tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id,
                    include_quarantined=True,
                )
                for r in records:
                    if r.quarantined_at is None:
                        continue
                    scan = (r.meta or {}).get("security", {}).get("injection_scan", {})
                    results.append(QuarantinedRecord(
                        id=r.id, lane=ln,
                        quarantined_at=r.quarantined_at,
                        severity=scan.get("severity", "high"),
                        matched_rules=scan.get("matched_rules", []),
                        owner_type=r.owner_type, owner_id=r.owner_id,
                        content_preview=str(r.object)[:200] if r.object else "",
                    ))
            elif ln == "episodic":
                records = await store.list_episodes(
                    tenant_id, owner_type, owner_id, include_quarantined=True,
                )
                for r in records:
                    if r.quarantined_at is None:
                        continue
                    scan = (r.meta or {}).get("security", {}).get("injection_scan", {})
                    results.append(QuarantinedRecord(
                        id=r.id, lane=ln,
                        quarantined_at=r.quarantined_at,
                        severity=scan.get("severity", "high"),
                        matched_rules=scan.get("matched_rules", []),
                        owner_type=r.owner_type, owner_id=r.owner_id,
                        content_preview=(r.summary or "")[:200],
                    ))
            elif ln == "procedural":
                records = await store.list_skills(
                    tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id,
                    include_quarantined=True,
                )
                for r in records:
                    if r.quarantined_at is None:
                        continue
                    scan = (r.meta or {}).get("security", {}).get("injection_scan", {})
                    results.append(QuarantinedRecord(
                        id=r.id, lane=ln,
                        quarantined_at=r.quarantined_at,
                        severity=scan.get("severity", "high"),
                        matched_rules=scan.get("matched_rules", []),
                        owner_type=r.owner_type, owner_id=r.owner_id,
                        content_preview=(r.description or r.name or "")[:200],
                    ))
            elif ln == "raw":
                records = await store.list_chunks_for_owner(
                    tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id,
                    include_quarantined=True,
                )
                for r in records:
                    if r.quarantined_at is None:
                        continue
                    scan = (r.meta or {}).get("security", {}).get("injection_scan", {})
                    results.append(QuarantinedRecord(
                        id=r.id, lane=ln,
                        quarantined_at=r.quarantined_at,
                        severity=scan.get("severity", "high"),
                        matched_rules=scan.get("matched_rules", []),
                        owner_type=r.owner_type, owner_id=r.owner_id,
                        content_preview=(r.text or "")[:200],
                    ))
        except Exception:
            logger.exception(
                "list_quarantined: failed to query lane=%s owner=%s:%s",
                ln,
                owner_type,
                owner_id,
            )
            raise

    results.sort(key=lambda r: r.quarantined_at, reverse=True)
    return results[:limit]


async def reinstate_quarantined(
    memory: "UMAMemory",
    *,
    record_id: str,
    lane: str,
    owner_type: str,
    owner_id: str,
    tenant_id: str = DEFAULT_TENANT_ID,
    reason: str,
) -> bool:
    """
    Reinstate a quarantined record: clear quarantined_at and append an audit log entry.
    Returns True if the record was found and updated.
    lane must be one of: "semantic", "episodic", "procedural", "raw".
    """
    store_key = _LANE_STORE_KEY.get(lane)
    if store_key is None:
        raise ValueError(f"reinstate_quarantined: unknown lane {lane!r}")
    store = memory._stores.get(store_key)
    if store is None:
        raise RuntimeError(f"reinstate_quarantined: store for lane={lane!r} not found")

    audit_entry = {
        "action": "reinstate",
        "lane": lane,
        "reason": reason,
        "reinstated_at": datetime.now(timezone.utc).isoformat(),
    }
    updated = await store.reinstate_quarantined_record(
        record_id,
        tenant_id=tenant_id,
        owner_type=owner_type,
        owner_id=owner_id,
        audit_entry=audit_entry,
    )
    if updated:
        logger.info(
            "reinstate_quarantined: record=%s lane=%s owner=%s:%s reason=%r",
            record_id, lane, owner_type, owner_id, reason,
        )
    return updated


async def purge_quarantined(
    memory: "UMAMemory",
    *,
    record_id: str,
    lane: str,
    owner_type: str,
    owner_id: str,
    tenant_id: str = DEFAULT_TENANT_ID,
    reason: str,
) -> bool:
    """
    Permanently delete a quarantined record from SQL and vector index.
    The record must be quarantined (quarantined_at IS NOT NULL); active records are not purged.
    Returns True if the record existed and was deleted.
    lane must be one of: "semantic", "episodic", "procedural", "raw".
    """
    store_key = _LANE_STORE_KEY.get(lane)
    if store_key is None:
        raise ValueError(f"purge_quarantined: unknown lane {lane!r}")
    store = memory._stores.get(store_key)
    if store is None:
        raise RuntimeError(f"purge_quarantined: store for lane={lane!r} not found")

    # Safety check: only purge records that are actually quarantined.
    quarantined = await list_quarantined(
        memory,
        owner_type=owner_type,
        owner_id=owner_id,
        tenant_id=tenant_id,
        lane=lane,
    )
    if not any(r.id == record_id for r in quarantined):
        logger.warning(
            "purge_quarantined: record=%s lane=%s not found in quarantine for owner=%s:%s",
            record_id, lane, owner_type, owner_id,
        )
        return False

    if lane == "semantic":
        await store.delete_fact(record_id, tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id)
    elif lane == "episodic":
        await store.delete_episode(record_id, tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id)
    elif lane == "procedural":
        await store.delete_skill(record_id, tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id)
    elif lane == "raw":
        await store.delete_chunk(record_id, tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id)

    logger.info(
        "purge_quarantined: deleted record=%s lane=%s owner=%s:%s reason=%r",
        record_id, lane, owner_type, owner_id, reason,
    )
    return True


__all__ = [
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


# ---------------------------------------------------------------------------
# Retrieval audit log (CR3)
# ---------------------------------------------------------------------------


async def list_retrieval_audit(
    memory: "UMAMemory",
    *,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    severity_min: Optional[str] = None,
    limit: int = 100,
    all_tenants: bool = False,
) -> list[dict]:
    """Read recent retrieval-audit rows.

    One row per `retrieve_context` / `retrieve_memory` call (the audit
    log writes from `UMARuntime.retrieve_context`). Rows record the
    request id, scope (tenant / user / agent), a hash + 80-char preview
    of the query, the boundary-scan severity, the participating lanes,
    a result count, and whether the LLM hops actually ran.

    The full query text is never stored. The hash plus preview is
    enough to correlate log lines and inspect a suspicious pattern
    without persisting an arbitrary user payload.

    Parameters
    ----------
    memory : UMAMemory
        The UMAMemory whose runtime owns the audit store.
    tenant_id : Optional[str]
        Exact-match filter on tenant. Required unless all_tenants=True.
    user_id : Optional[str]
        Exact-match filter on user. If None, returns all users within
        the resolved tenant scope.
    severity_min : Optional[str]
        Returns rows at or above this severity tier. Values:
        "none" / "low" / "medium" / "high". None means no floor.
    limit : int
        Max rows returned (capped at 1000 internally).
    all_tenants : bool
        Explicit opt-in for a cross-tenant admin view. Raises
        ValueError if False and tenant_id is not provided — the audit
        log is the one read path in UMA where cross-tenant visibility
        is a legitimate operator capability, but it must be requested
        deliberately rather than fall out of an omitted parameter.

    Returns
    -------
    List[dict]
        Newest first. Each dict carries request_id, tenant_id, user_id,
        agent_id, query_hash, query_preview, scan_severity, lanes,
        result_count, refined_via_llm, pruned_via_llm, created_at.
        Empty list if audit is disabled or the store can't be read.

    Raises
    ------
    ValueError
        If tenant_id is None and all_tenants is not True.
    """
    runtime = getattr(memory, "runtime", None)
    if runtime is None:
        logger.debug("list_retrieval_audit: memory has no runtime attribute")
        return []
    store = runtime._get_retrieval_audit_store()
    if store is None:
        logger.debug("list_retrieval_audit: audit store unavailable (disabled or failed)")
        return []
    return await store.list_rows(
        tenant_id=tenant_id,
        user_id=user_id,
        severity_min=severity_min,
        limit=limit,
        all_tenants=all_tenants,
    )
