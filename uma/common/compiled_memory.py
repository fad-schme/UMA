from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from uma.common.provenance import (
    build_provenance,
    collect_parent_artifact_ids,
    collect_source_chunk_ids,
    collect_source_document_ids,
    collect_transitive_source_chunk_ids,
    collect_transitive_source_document_ids,
    provenance_for_artifact,
)
from uma.common.storage_metadata import WIKI_LANE

COMPILED_MEMORY_ARTIFACT_TYPE = "compiled_memory_artifact"
COMPILED_MEMORY_INDEX_TYPE = "compiled_memory_index_entry"
COMPILED_MEMORY_LOG_TYPE = "compiled_memory_log_event"


def build_compiled_memory_artifact(
    *,
    artifact_id: str,
    title: str,
    owner_type: str,
    owner_id: str,
    artifact_kind: str,
    text: str | None = None,
    summary: str | None = None,
    topic_key: str | None = None,
    derived_at: str | datetime | None = None,
    derivation_type: str,
    direct_source_chunk_ids: Sequence[Any] | None = None,
    direct_source_document_ids: Sequence[Any] | None = None,
    parent_artifacts: Sequence[Any] | None = None,
    parent_artifact_ids: Sequence[Any] | None = None,
    related_artifact_ids: Sequence[Any] | None = None,
    retrieval_tags: Sequence[Any] | None = None,
    retrieval_path: Sequence[Mapping[str, Any]] | None = None,
    support_density: float | None = None,
    confidence: float | None = None,
    conflicts: Sequence[Mapping[str, Any]] | None = None,
    manual: bool = False,
    status: str = "active",
    metadata: Mapping[str, Any] | None = None,
    existing_artifact: Any | None = None,
    actor: Mapping[str, Any] | None = None,
    event_type: str | None = None,
) -> dict[str, Any]:
    normalized_direct_chunk_ids = _string_list(direct_source_chunk_ids)
    normalized_direct_document_ids = _string_list(direct_source_document_ids)
    normalized_parent_artifacts = [item for item in (parent_artifacts or []) if item is not None]
    normalized_parent_artifact_ids = _ordered_unique(
        list(_string_list(parent_artifact_ids))
        + [str(_artifact_value(item, "id") or "").strip() for item in normalized_parent_artifacts if _artifact_value(item, "id")]
    )
    normalized_related_artifact_ids = _ordered_unique(
        _string_list(related_artifact_ids) + normalized_parent_artifact_ids
    )
    normalized_retrieval_tags = _ordered_unique(_string_list(retrieval_tags))
    normalized_conflicts = [dict(item) for item in (conflicts or []) if isinstance(item, Mapping)]
    transitive_source_chunk_ids = _ordered_unique(
        normalized_direct_chunk_ids
        + [chunk_id for parent in normalized_parent_artifacts for chunk_id in collect_transitive_source_chunk_ids(parent)]
    )
    transitive_source_document_ids = _ordered_unique(
        normalized_direct_document_ids
        + [doc_id for parent in normalized_parent_artifacts for doc_id in collect_transitive_source_document_ids(parent)]
    )
    inferred_support_density = support_density
    if inferred_support_density is None:
        inferred_support_density = 1.0 if transitive_source_chunk_ids else 0.0

    artifact = {
        "id": str(artifact_id),
        "artifact_type": COMPILED_MEMORY_ARTIFACT_TYPE,
        "kind": str(artifact_kind or COMPILED_MEMORY_ARTIFACT_TYPE),
        "kb_lane": WIKI_LANE,
        "title": str(title or artifact_id),
        "topic_key": str(topic_key or title or artifact_id),
        "text": text,
        "summary": summary,
        "status": str(status or "active"),
        "owner_type": str(owner_type or ""),
        "owner_id": str(owner_id or ""),
        "direct_source_chunk_ids": normalized_direct_chunk_ids,
        "direct_source_document_ids": normalized_direct_document_ids,
        "parent_artifacts": normalized_parent_artifacts,
        "parent_artifact_ids": normalized_parent_artifact_ids,
        "related_artifact_ids": normalized_related_artifact_ids,
        "retrieval_tags": normalized_retrieval_tags,
        "retrieval_path": [dict(item) for item in (retrieval_path or []) if isinstance(item, Mapping)],
        "support_density": inferred_support_density,
        "confidence": confidence,
        "conflicts": normalized_conflicts,
        "manual": bool(manual),
        "metadata": dict(metadata or {}),
        "is_terminal_truth": False,
    }
    existing_provenance = provenance_for_artifact(existing_artifact) if existing_artifact is not None else provenance_for_artifact(metadata or {})
    artifact["provenance"] = build_provenance(
        existing=existing_provenance,
        source_chunk_ids=transitive_source_chunk_ids,
        source_document_ids=transitive_source_document_ids,
        derived_at=derived_at or _utcnow_isoformat(),
        derivation_type=derivation_type,
        retrieval_path=artifact["retrieval_path"],
        parent_artifact_ids=normalized_parent_artifact_ids,
        support_density=inferred_support_density,
        confidence=confidence,
        conflicts=normalized_conflicts,
        manual=manual,
        require_source_chunks=not manual,
    )
    if normalized_parent_artifact_ids and not transitive_source_chunk_ids and not manual:
        _invalidate_provenance(artifact["provenance"], "unreachable_raw_source_chunks")
    artifact["compiled_memory_index"] = build_compiled_memory_index_entry(artifact)
    existing_log = _existing_log_events(existing_artifact)
    artifact["compiled_memory_log"] = existing_log + build_compiled_memory_log(
        artifact,
        event_type=event_type or derivation_type,
        actor=actor,
    )
    return artifact


def build_compiled_memory_index_entry(artifact: Any) -> dict[str, Any]:
    provenance = provenance_for_artifact(artifact)
    conflicts = provenance.get("conflicts") if isinstance(provenance.get("conflicts"), list) else []
    return {
        "entry_type": COMPILED_MEMORY_INDEX_TYPE,
        "artifact_id": str(_artifact_value(artifact, "id") or _artifact_value(artifact, "doc_id") or ""),
        "title": str(_artifact_value(artifact, "title") or _artifact_value(artifact, "page_title") or _artifact_value(artifact, "doc_id") or ""),
        "topic_key": str(_artifact_value(artifact, "topic_key") or _artifact_value(artifact, "page_slug") or _artifact_value(artifact, "title") or _artifact_value(artifact, "doc_id") or ""),
        "artifact_kind": str(_artifact_value(artifact, "kind") or ""),
        "updated_at": provenance.get("derived_at") or _artifact_value(artifact, "updated_at") or _artifact_value(artifact, "created_at"),
        "source_chunk_ids": collect_transitive_source_chunk_ids(artifact),
        "related_artifact_ids": _ordered_unique(
            _string_list(_artifact_value(artifact, "related_artifact_ids"))
            + collect_parent_artifact_ids(artifact)
        ),
        "retrieval_tags": _string_list(_artifact_value(artifact, "retrieval_tags")),
        "summary": _artifact_value(artifact, "summary"),
        "has_conflicts": bool(conflicts),
        "conflict_count": len(conflicts),
        "navigation_only": True,
        "provenance_valid": bool(provenance.get("valid")),
    }


def build_compiled_memory_log(
    artifact: Any,
    *,
    event_type: str,
    actor: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    event = build_compiled_memory_log_event(
        event_type=event_type,
        artifact=artifact,
        actor=actor,
    )
    events = [event]
    provenance = provenance_for_artifact(artifact)
    conflicts = provenance.get("conflicts") if isinstance(provenance.get("conflicts"), list) else []
    if conflicts:
        events.append(
            build_compiled_memory_log_event(
                event_type="conflict_detected",
                artifact=artifact,
                actor=actor,
            )
        )
    if not provenance.get("valid"):
        events.append(
            build_compiled_memory_log_event(
                event_type="provenance_invalidated",
                artifact=artifact,
                actor=actor,
            )
        )
    return events


def build_compiled_memory_log_event(
    *,
    event_type: str,
    artifact: Any | None = None,
    artifact_id: str | None = None,
    timestamp: str | datetime | None = None,
    source_chunk_ids: Sequence[Any] | None = None,
    parent_artifact_ids: Sequence[Any] | None = None,
    derivation_type: str | None = None,
    retrieval_path: Sequence[Mapping[str, Any]] | None = None,
    conflicts: Sequence[Mapping[str, Any]] | None = None,
    actor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = provenance_for_artifact(artifact) if artifact is not None else {}
    artifact_identifier = str(
        artifact_id
        or _artifact_value(artifact, "id")
        or _artifact_value(artifact, "doc_id")
        or ""
    )
    event_timestamp = _isoformat(timestamp) or provenance.get("derived_at") or _utcnow_isoformat()
    normalized_source_chunk_ids = _ordered_unique(
        _string_list(source_chunk_ids) + collect_transitive_source_chunk_ids(artifact) if artifact is not None else _string_list(source_chunk_ids)
    )
    normalized_parent_artifact_ids = _ordered_unique(
        _string_list(parent_artifact_ids) + (collect_parent_artifact_ids(artifact) if artifact is not None else [])
    )
    normalized_conflicts = [dict(item) for item in (conflicts or provenance.get("conflicts") or []) if isinstance(item, Mapping)]
    return {
        "event_type_marker": COMPILED_MEMORY_LOG_TYPE,
        "event_id": f"log:{artifact_identifier or 'artifact'}:{event_type}:{event_timestamp}",
        "event_type": str(event_type or ""),
        "timestamp": event_timestamp,
        "artifact_id": artifact_identifier or None,
        "source_chunk_ids": normalized_source_chunk_ids,
        "parent_artifact_ids": normalized_parent_artifact_ids,
        "derivation_type": str(derivation_type or provenance.get("derivation_type") or event_type or ""),
        "retrieval_path": [dict(item) for item in (retrieval_path or provenance.get("retrieval_path") or []) if isinstance(item, Mapping)],
        "conflicts": normalized_conflicts,
        "manual": bool(provenance.get("manual")),
        "provenance_valid": bool(provenance.get("valid", True)),
        "actor": dict(actor or {}),
    }


def _artifact_value(artifact: Any, field_name: str) -> Any:
    if artifact is None:
        return None
    if isinstance(artifact, dict):
        return artifact.get(field_name)
    return getattr(artifact, field_name, None)


def _invalidate_provenance(provenance: dict[str, Any], reason: str) -> None:
    invalid_reasons = list(provenance.get("invalid_reasons") or [])
    if reason not in invalid_reasons:
        invalid_reasons.append(reason)
    provenance["invalid_reasons"] = invalid_reasons
    provenance["valid"] = False


def _isoformat(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _ordered_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _string_list(values: Sequence[Any] | Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        text = str(values).strip()
        return [text] if text else []
    return _ordered_unique([str(value or "").strip() for value in values])


def _utcnow_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat()


def _existing_log_events(artifact: Any | None) -> list[dict[str, Any]]:
    if artifact is None:
        return []
    if isinstance(artifact, dict):
        raw_events = artifact.get("compiled_memory_log") or []
    else:
        raw_events = getattr(artifact, "compiled_memory_log", None) or []
    return [dict(item) for item in raw_events if isinstance(item, Mapping)]
