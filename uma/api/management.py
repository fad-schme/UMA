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


def _require_runtime_context(memory: "UMAMemory") -> Any:
    return memory._require_bound_runtime_context()


async def explain_result(
    memory: "UMAMemory",
    result: Any,
) -> dict[str, Any]:
    """Explain a retrieval result or compiled artifact using canonical provenance."""
    runtime_context = _require_runtime_context(memory)
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


def update_wiki_page(
    memory: "UMAMemory",
    *,
    artifact_id: str,
    title: str,
    owner_type: str,
    owner_id: str,
    text: str | None = None,
    summary: str | None = None,
    topic_key: str | None = None,
    direct_source_chunk_ids: Sequence[str] | None = None,
    direct_source_document_ids: Sequence[str] | None = None,
    parent_artifacts: Sequence[Any] | None = None,
    related_artifact_ids: Sequence[str] | None = None,
    retrieval_tags: Sequence[str] | None = None,
    retrieval_path: Sequence[Mapping[str, Any]] | None = None,
    support_density: float | None = None,
    confidence: float | None = None,
    conflicts: Sequence[Mapping[str, Any]] | None = None,
    existing_artifact: Any | None = None,
    manual: bool = False,
) -> dict[str, Any]:
    """Create or refresh a canonical compiled wiki artifact."""
    operation = "manual_update" if manual else ("wiki_artifact_updated" if existing_artifact is not None else "wiki_artifact_created")
    page = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key=topic_key or artifact_id,
        title=title,
        owner_type=owner_type,
        owner_id=owner_id,
        text=text,
        summary=summary,
        direct_source_chunk_ids=list(direct_source_chunk_ids or []),
        direct_source_document_ids=list(direct_source_document_ids or []),
        parent_artifacts=list(parent_artifacts or []),
        related_artifact_ids=list(related_artifact_ids or []),
        retrieval_tags=list(retrieval_tags or []),
        retrieval_path=list(retrieval_path or []),
        support_density=support_density,
        confidence=confidence,
        conflicts=list(conflicts or []),
        existing_page=existing_artifact,
        manual=manual,
    )
    artifact = page["compiled_artifact"]
    return {
        "status": "updated",
        "operation": operation,
        "page": page,
        "artifact": artifact,
        "artifact_id": artifact["id"],
        "compiled_memory_index": artifact["compiled_memory_index"],
        "compiled_memory_log": list(artifact.get("compiled_memory_log") or []),
        "provenance": artifact["provenance"],
    }


async def export_wiki_projection(
    memory: "UMAMemory",
    artifact: Mapping[str, Any],
    *,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Render a compiled wiki artifact into a rebuildable markdown projection."""
    del memory
    return wiki_module.export_wiki_projection(artifact, output_path=output_path)


async def lint_memory_drift(
    memory: "UMAMemory",
    artifacts: Any | Sequence[Any],
    *,
    stale_after_seconds: int | None = None,
) -> dict[str, Any]:
    """Lint compiled memory artifacts for provenance drift without mutating them."""
    items = list(artifacts) if isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes, bytearray, Mapping)) else [artifacts]
    findings: list[dict[str, Any]] = []
    statuses: list[str] = []
    for artifact in items:
        lint_result = await wiki_module.lint_wiki_page(
            memory,
            artifact,
            stale_after_seconds=stale_after_seconds,
        )
        findings.extend(list(lint_result["findings"]))
        statuses.append(str(lint_result["drift_status"]))
    status = "ok" if not findings else "issues_found"
    logger.info("lint_memory_drift: artifacts=%d findings=%d status=%s", len(items), len(findings), status)
    return {
        "status": status,
        "artifacts_scanned": len(items),
        "findings": findings,
        "drift_statuses": statuses,
    }


__all__ = [
    "explain_result",
    "export_wiki_projection",
    "lint_memory_drift",
    "update_wiki_page",
]
