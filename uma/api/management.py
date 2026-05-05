"""Developer and admin management APIs for UMA memory.

This module keeps inspection, curation, projection, and drift checks out of
the normal product-facing `UMAMemory` surface.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from uma.common.provenance import collect_parent_artifact_ids, provenance_for_artifact

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
    return result


def _require_runtime_context(memory: "UMAMemory") -> Any:
    return memory._require_bound_runtime_context()


def _format_projection_markdown(artifact: Mapping[str, Any]) -> str:
    title = str(artifact.get("title") or artifact.get("summary") or artifact.get("id") or "Untitled Wiki Page")
    summary = str(artifact.get("summary") or "").strip()
    text = str(artifact.get("text") or "").strip()
    provenance = provenance_for_artifact(artifact)
    lines = [
        f"# {title}",
        "",
        "<!-- projection_only: true -->",
        f"<!-- artifact_id: {_artifact_id(artifact) or ''} -->",
        f"<!-- derived_at: {provenance.get('derived_at') or ''} -->",
        f"<!-- derivation_type: {provenance.get('derivation_type') or ''} -->",
        "",
    ]
    if summary:
        lines.extend(["## Summary", "", summary, ""])
    if text:
        lines.extend(["## Content", "", text, ""])
    lines.extend(["## Evidence", ""])
    for chunk_id in provenance.get("source_chunk_ids") or []:
        lines.append(f"- chunk:{chunk_id}")
    for artifact_id in provenance.get("parent_artifact_ids") or []:
        lines.append(f"- parent_artifact:{artifact_id}")
    conflicts = provenance.get("conflicts") or []
    if conflicts:
        lines.extend(["", "## Conflicts", ""])
        for conflict in conflicts:
            claim = str((conflict or {}).get("claim") or "conflict")
            lines.append(f"- {claim}")
    return "\n".join(lines).strip() + "\n"


async def explain_result(
    memory: "UMAMemory",
    result: Any,
) -> dict[str, Any]:
    """Explain a retrieval result or compiled artifact using canonical provenance."""
    runtime_context = _require_runtime_context(memory)
    artifact = _primary_artifact(result)
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
    if not manual and not list(direct_source_chunk_ids or []) and not list(parent_artifacts or []):
        raise ValueError(
            "update_wiki_page: direct_source_chunk_ids or parent_artifacts are required unless manual=True"
        )
    if manual:
        operation = "manual_update"
    else:
        operation = "wiki_artifact_updated" if existing_artifact is not None else "wiki_artifact_created"
    artifact = memory.runtime.compile_memory_artifact(
        artifact_id=artifact_id,
        title=title,
        owner_type=owner_type,
        owner_id=owner_id,
        text=text,
        summary=summary,
        topic_key=topic_key,
        direct_source_chunk_ids=list(direct_source_chunk_ids or []),
        direct_source_document_ids=list(direct_source_document_ids or []),
        parent_artifacts=list(parent_artifacts or []),
        related_artifact_ids=list(related_artifact_ids or []),
        retrieval_tags=list(retrieval_tags or []),
        retrieval_path=list(retrieval_path or []),
        support_density=support_density,
        confidence=confidence,
        conflicts=list(conflicts or []),
        existing_artifact=existing_artifact,
        manual=manual,
        operation=operation,
    )
    return {
        "status": "updated",
        "operation": operation,
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
    del memory  # Projection is read-only over canonical compiled state.
    markdown = _format_projection_markdown(artifact)
    written_path: str | None = None
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        written_path = str(path)
    return {
        "status": "exported",
        "artifact_id": _artifact_id(artifact),
        "projection_only": True,
        "path": written_path,
        "markdown": markdown,
        "source_chunk_ids": list(provenance_for_artifact(artifact).get("source_chunk_ids") or []),
        "parent_artifact_ids": list(collect_parent_artifact_ids(artifact)),
    }


async def lint_memory_drift(
    memory: "UMAMemory",
    artifacts: Any | Sequence[Any],
    *,
    stale_after_seconds: int | None = None,
) -> dict[str, Any]:
    """Lint compiled memory artifacts for provenance drift without mutating them."""
    runtime_context = _require_runtime_context(memory)
    items = list(artifacts) if isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes, bytearray, Mapping)) else [artifacts]
    findings: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for artifact in items:
        expanded = await memory.runtime.expand_evidence(runtime_context, artifact)
        provenance = provenance_for_artifact(artifact)
        artifact_id = _artifact_id(artifact)
        if not bool(provenance.get("valid")):
            findings.append(
                {
                    "artifact_id": artifact_id,
                    "severity": "error",
                    "issue": "invalid_provenance",
                    "details": list(provenance.get("invalid_reasons") or []),
                }
            )
        if expanded["missing_chunk_ids"]:
            findings.append(
                {
                    "artifact_id": artifact_id,
                    "severity": "error",
                    "issue": "missing_chunks",
                    "details": list(expanded["missing_chunk_ids"]),
                }
            )
        if expanded["unresolved_parent_artifact_ids"]:
            findings.append(
                {
                    "artifact_id": artifact_id,
                    "severity": "error",
                    "issue": "broken_parent_lineage",
                    "details": list(expanded["unresolved_parent_artifact_ids"]),
                }
            )
        conflicts = list(provenance.get("conflicts") or [])
        if conflicts:
            findings.append(
                {
                    "artifact_id": artifact_id,
                    "severity": "warning",
                    "issue": "conflicts_present",
                    "details": conflicts,
                }
            )
        if stale_after_seconds is not None and provenance.get("derived_at"):
            try:
                derived_at = datetime.fromisoformat(str(provenance["derived_at"]).replace("Z", "+00:00"))
            except ValueError:
                derived_at = None
            if derived_at is not None and (now - derived_at).total_seconds() > stale_after_seconds:
                findings.append(
                    {
                        "artifact_id": artifact_id,
                        "severity": "warning",
                        "issue": "stale_compiled_artifact",
                        "details": {"derived_at": provenance["derived_at"], "stale_after_seconds": stale_after_seconds},
                    }
                )
    status = "ok" if not findings else "issues_found"
    logger.info("lint_memory_drift: artifacts=%d findings=%d status=%s", len(items), len(findings), status)
    return {
        "status": status,
        "artifacts_scanned": len(items),
        "findings": findings,
    }


__all__ = [
    "explain_result",
    "export_wiki_projection",
    "lint_memory_drift",
    "update_wiki_page",
]
