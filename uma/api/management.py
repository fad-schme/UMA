"""Developer and admin management APIs for UMA memory.

This module keeps inspection, curation, projection, and drift checks out of
the normal product-facing `UMAMemory` surface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from uma.common.provenance import provenance_for_artifact
from uma.memory import wiki as wiki_module

if TYPE_CHECKING:
    from .memory import UMAMemory

logger = logging.getLogger(__name__)


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

__all__ = [
    "explain_result",
    "lint_memory_drift",
]
