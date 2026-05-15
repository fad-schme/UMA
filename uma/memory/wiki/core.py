from __future__ import annotations

from datetime import datetime, timezone
import logging
import re
from typing import Any, Mapping, Sequence

from uma.common.ownership import validate_explicit_owner
from uma.common.provenance import collect_parent_artifact_ids, provenance_for_artifact

logger = logging.getLogger(__name__)

WIKI_PAGE_RECORD_TYPE = "wiki_page_record"
WIKI_ACTIVE_STATUS = "active"
WIKI_STALE_STATUS = "stale"
WIKI_INVALID_STATUS = "invalid"
WIKI_ARCHIVED_STATUS = "archived"
WIKI_MANUAL_STATUS = "manual"


def slugify_page_key(page_key: str) -> str:
    """Return a deterministic wiki slug for one page key."""
    if not isinstance(page_key, str) or not page_key.strip():
        raise ValueError("slugify_page_key: page_key must be a non-empty string")
    normalized = page_key.strip().lower()
    if normalized.startswith("wiki:"):
        normalized = normalized[5:]
    normalized = normalized.replace("::", "/").replace(":", "/").replace("_", "-").replace(" ", "-")
    normalized = re.sub(r"[^a-z0-9/\-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    normalized = re.sub(r"/{2,}", "/", normalized)
    normalized = normalized.strip("-/") or "page"
    return normalized


def page_identity_for_key(page_key: str) -> dict[str, str]:
    """Return the canonical page id and slug for a wiki page key."""
    slug = slugify_page_key(page_key)
    return {
        "page_id": f"wiki:{slug}",
        "slug": slug,
    }


def wiki_page_from_record(record: Any) -> dict[str, Any]:
    """Normalize either a wiki page record or compiled artifact into a wiki page record."""
    if isinstance(record, Mapping) and record.get("page_type") == WIKI_PAGE_RECORD_TYPE:
        page = dict(record)
        page["evidence_links"] = dict(page.get("evidence_links") or {})
        page["compiled_memory_index"] = dict(page.get("compiled_memory_index") or {})
        page["provenance"] = dict(page.get("provenance") or {})
        page["compiled_memory_log"] = [dict(item) for item in (page.get("compiled_memory_log") or []) if isinstance(item, Mapping)]
        if isinstance(page.get("compiled_artifact"), Mapping):
            page["compiled_artifact"] = dict(page["compiled_artifact"])
        return page

    artifact = dict(record) if isinstance(record, Mapping) else {}
    metadata = dict(artifact.get("metadata") or {})
    page_key = (
        metadata.get("page_slug")
        or artifact.get("topic_key")
        or artifact.get("title")
        or artifact.get("id")
        or artifact.get("doc_id")
        or "page"
    )
    identity = page_identity_for_key(str(page_key))
    provenance = provenance_for_artifact(artifact)
    manual = bool(provenance.get("manual") or artifact.get("manual"))
    status = _resolve_status(
        requested_status=metadata.get("status") or artifact.get("status"),
        manual=manual,
        provenance_valid=bool(provenance.get("valid", True)),
    )
    title = str(metadata.get("page_title") or artifact.get("title") or identity["slug"])
    text = artifact.get("text")
    summary = artifact.get("summary")
    sections = _normalize_sections(metadata.get("sections"), text=text, summary=summary)
    return {
        "page_type": WIKI_PAGE_RECORD_TYPE,
        "page_id": str(metadata.get("page_id") or identity["page_id"]),
        "slug": str(metadata.get("page_slug") or identity["slug"]),
        "title": title,
        "category": str(metadata.get("category") or "general"),
        "status": status,
        "text": text,
        "summary": summary,
        "sections": sections,
        "evidence_links": {
            "source_chunk_ids": list(provenance.get("source_chunk_ids") or []),
            "source_document_ids": list(provenance.get("source_document_ids") or []),
            "parent_artifact_ids": list(provenance.get("parent_artifact_ids") or collect_parent_artifact_ids(artifact)),
            "related_artifact_ids": list(artifact.get("related_artifact_ids") or []),
        },
        "compiled_artifact_id": str(artifact.get("id") or metadata.get("page_id") or identity["page_id"]),
        "compiled_artifact_ids": [str(artifact.get("id") or metadata.get("page_id") or identity["page_id"])],
        "derived_at": provenance.get("derived_at"),
        "updated_at": provenance.get("derived_at"),
        "provenance": provenance,
        "conflicts": list(provenance.get("conflicts") or artifact.get("conflicts") or []),
        "drift_status": str(metadata.get("drift_status") or "unchecked"),
        "manual": manual,
        "compiled_memory_index": dict(artifact.get("compiled_memory_index") or {}),
        "compiled_memory_log": [dict(item) for item in (artifact.get("compiled_memory_log") or []) if isinstance(item, Mapping)],
        "compiled_artifact": artifact,
    }


def regenerate_wiki_page(
    *,
    memory: Any,
    page_key: str,
    title: str,
    owner_type: str,
    owner_id: str,
    text: str | None = None,
    summary: str | None = None,
    category: str | None = None,
    status: str | None = None,
    sections: Sequence[Mapping[str, Any]] | None = None,
    direct_source_chunk_ids: Sequence[str] | None = None,
    direct_source_document_ids: Sequence[str] | None = None,
    parent_artifacts: Sequence[Any] | None = None,
    related_artifact_ids: Sequence[str] | None = None,
    retrieval_tags: Sequence[str] | None = None,
    conflicts: Sequence[Mapping[str, Any]] | None = None,
    existing_page: Any | None = None,
    manual: bool = False,
) -> dict[str, Any]:
    """Create or refresh one canonical wiki page record from evidence and compiled artifacts."""
    if memory is None:
        raise ValueError("regenerate_wiki_page: memory is required")
    owner = validate_explicit_owner(owner_type=owner_type, owner_id=owner_id)
    identity = page_identity_for_key(page_key)
    existing = wiki_page_from_record(existing_page) if existing_page is not None else None
    existing_artifact = dict(existing.get("compiled_artifact") or {}) if existing is not None else (
        dict(existing_page) if isinstance(existing_page, Mapping) and existing_page.get("artifact_type") == "compiled_memory_artifact" else None
    )
    resolved_title = str(title or (existing.get("title") if existing is not None else "") or identity["slug"])
    resolved_text = text if text is not None else (existing.get("text") if existing is not None else None)
    resolved_summary = summary if summary is not None else (existing.get("summary") if existing is not None else None)
    resolved_category = str(category or (existing.get("category") if existing is not None else "general") or "general")
    normalized_sections = _normalize_sections(
        sections if sections is not None else (existing.get("sections") if existing is not None else None),
        text=resolved_text,
        summary=resolved_summary,
    )
    metadata = {
        "page_id": identity["page_id"],
        "page_slug": identity["slug"],
        "page_title": resolved_title,
        "category": resolved_category,
        "sections": normalized_sections,
        "drift_status": "unchecked",
    }
    resolved_status = _resolve_status(
        requested_status=status or (existing.get("status") if existing is not None else None),
        manual=manual,
        provenance_valid=True,
    )
    operation = "manual_update" if manual else ("wiki_artifact_updated" if existing_artifact is not None else "wiki_artifact_created")
    artifact = memory.runtime.compile_memory_artifact(
        artifact_id=identity["page_id"],
        title=resolved_title,
        owner_type=str(owner["owner_type"]),
        owner_id=str(owner["owner_id"]),
        text=resolved_text,
        summary=resolved_summary,
        topic_key=identity["slug"],
        direct_source_chunk_ids=list(direct_source_chunk_ids or []),
        direct_source_document_ids=list(direct_source_document_ids or []),
        parent_artifacts=list(parent_artifacts or []),
        related_artifact_ids=list(related_artifact_ids or []),
        retrieval_tags=list(retrieval_tags or []),
        conflicts=[dict(item) for item in (conflicts or []) if isinstance(item, Mapping)],
        existing_artifact=existing_artifact,
        manual=manual,
        operation=operation,
        metadata=metadata,
        status=resolved_status,
    )
    page = wiki_page_from_record(artifact)
    if not bool(page["provenance"].get("valid")):
        page["status"] = WIKI_INVALID_STATUS
        page["compiled_artifact"]["status"] = WIKI_INVALID_STATUS
    logger.info(
        "regenerate_wiki_page: page_id=%s owner=%s:%s source_chunks=%d parent_artifacts=%d",
        page["page_id"],
        owner["owner_type"],
        owner["owner_id"],
        len(page["evidence_links"]["source_chunk_ids"]),
        len(page["evidence_links"]["parent_artifact_ids"]),
    )
    return page




async def lint_wiki_page(
    memory: Any,
    page_or_artifact: Any,
    *,
    user_id: str,
    tenant_id: str = "default",
    workspace_id: str | None = None,
    stale_after_seconds: int | None = None,
) -> dict[str, Any]:
    """Check one wiki page for structural drift without rewriting canonical state."""
    if memory is None:
        raise ValueError("lint_wiki_page: memory is required")
    
    runtime_context = memory._resolve_runtime_context(
        user_id=user_id,
        tenant_id=tenant_id,
        request_id="wiki:lint_wiki_page",
        workspace_id=workspace_id,
        session_id=None,
    )
    page = wiki_page_from_record(page_or_artifact)
    expanded = await memory.runtime.expand_evidence(runtime_context, page["compiled_artifact"])
    provenance = page["provenance"]
    findings: list[dict[str, Any]] = []
    if not bool(provenance.get("valid")):
        findings.append(
            {
                "page_id": page["page_id"],
                "severity": "error",
                "issue": "invalid_provenance",
                "details": list(provenance.get("invalid_reasons") or []),
            }
        )
    if expanded["missing_chunk_ids"]:
        findings.append(
            {
                "page_id": page["page_id"],
                "severity": "error",
                "issue": "missing_chunks",
                "details": list(expanded["missing_chunk_ids"]),
            }
        )
    if expanded["unresolved_parent_artifact_ids"]:
        findings.append(
            {
                "page_id": page["page_id"],
                "severity": "error",
                "issue": "broken_parent_lineage",
                "details": list(expanded["unresolved_parent_artifact_ids"]),
            }
        )
    if page["conflicts"]:
        findings.append(
            {
                "page_id": page["page_id"],
                "severity": "warning",
                "issue": "conflicts_present",
                "details": list(page["conflicts"]),
            }
        )
    if page["manual"] and not page["compiled_memory_log"]:
        findings.append(
            {
                "page_id": page["page_id"],
                "severity": "error",
                "issue": "manual_page_missing_audit_metadata",
                "details": [],
            }
        )
    if stale_after_seconds is not None and provenance.get("derived_at"):
        derived_at = _parse_iso_datetime(str(provenance["derived_at"]))
        if derived_at is not None and (datetime.now(timezone.utc) - derived_at).total_seconds() > stale_after_seconds:
            findings.append(
                {
                    "page_id": page["page_id"],
                    "severity": "warning",
                    "issue": "stale_compiled_artifact",
                    "details": {"derived_at": provenance["derived_at"], "stale_after_seconds": stale_after_seconds},
                }
            )
    drift_status = _drift_status_for_findings(findings)
    logger.info("lint_wiki_page: page_id=%s findings=%d drift_status=%s", page["page_id"], len(findings), drift_status)
    return {
        "status": "ok" if not findings else "issues_found",
        "page_id": page["page_id"],
        "artifact_id": page["compiled_artifact_id"],
        "drift_status": drift_status,
        "findings": findings,
    }


def _resolve_status(
    *,
    requested_status: str | None,
    manual: bool,
    provenance_valid: bool,
) -> str:
    if manual:
        return WIKI_MANUAL_STATUS
    if not provenance_valid:
        return WIKI_INVALID_STATUS
    normalized = str(requested_status or "").strip().lower()
    if normalized in {
        WIKI_ACTIVE_STATUS,
        WIKI_STALE_STATUS,
        WIKI_INVALID_STATUS,
        WIKI_ARCHIVED_STATUS,
        WIKI_MANUAL_STATUS,
    }:
        return normalized
    return WIKI_ACTIVE_STATUS


def _normalize_sections(
    sections: Sequence[Mapping[str, Any]] | None,
    *,
    text: str | None,
    summary: str | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for section in sections or []:
        if not isinstance(section, Mapping):
            continue
        heading = str(section.get("heading") or "").strip()
        body = str(section.get("body") or "").strip()
        if heading or body:
            normalized.append({"heading": heading or "Section", "body": body})
    if normalized:
        return normalized
    if summary:
        normalized.append({"heading": "Summary", "body": str(summary).strip()})
    if text:
        normalized.append({"heading": "Content", "body": str(text).strip()})
    return normalized




def _parse_iso_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _drift_status_for_findings(findings: Sequence[Mapping[str, Any]]) -> str:
    severities = {str(item.get("severity") or "") for item in findings}
    if "error" in severities:
        return WIKI_INVALID_STATUS
    if findings:
        return WIKI_STALE_STATUS
    return WIKI_ACTIVE_STATUS
