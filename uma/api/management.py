"""Developer and admin management APIs for UMA memory.

This module keeps inspection, curation, projection, and drift checks out of
the normal product-facing `UMAMemory` surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence

from uma.common.provenance import provenance_for_artifact
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
class QuarantinedRecord:
    id: str
    lane: str
    quarantined_at: datetime
    severity: str
    matched_rules: List[str]
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
    user_id: str,
    tenant_id: str = "default",
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Explain a retrieval result or compiled artifact using canonical provenance."""
    runtime_context = memory._resolve_runtime_context(
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


async def lint_memory_drift(
    memory: "UMAMemory",
    artifacts: Any | Sequence[Any],
    *,
    user_id: str,
    tenant_id: str = "default",
    workspace_id: str | None = None,
    stale_after_seconds: int | None = None,
) -> dict[str, Any]:
    """Lint compiled memory artifacts for provenance drift without mutating them."""
    items = (
        list(artifacts)
        if isinstance(artifacts, Sequence)
        and not isinstance(artifacts, (str, bytes, bytearray, Mapping))
        else [artifacts]
    )
    findings: list[dict[str, Any]] = []
    statuses: list[str] = []

    for artifact in items:
        lint_result = await wiki_module.lint_wiki_page(
            memory,
            artifact,
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

async def list_quarantined(
    memory: "UMAMemory",
    *,
    owner_type: str,
    owner_id: str,
    tenant_id: str = "default",
    lane: Optional[str] = None,
    limit: int = 100,
) -> List[QuarantinedRecord]:
    """
    Return quarantined records across all lanes (or a specific lane), owner-scoped.
    lane must be one of: "semantic", "episodic", "procedural", "raw" (or None for all).
    """
    lanes_to_check = [lane] if lane else list(_LANE_STORE_KEY.keys())
    results: List[QuarantinedRecord] = []

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
            logger.exception("list_quarantined: failed to query lane=%s owner=%s:%s", ln, owner_type, owner_id)

    results.sort(key=lambda r: r.quarantined_at, reverse=True)
    return results[:limit]


async def reinstate_quarantined(
    memory: "UMAMemory",
    *,
    record_id: str,
    lane: str,
    owner_type: str,
    owner_id: str,
    tenant_id: str = "default",
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
    tenant_id: str = "default",
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
]
