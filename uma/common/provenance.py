from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence


def build_provenance(
    *,
    existing: Mapping[str, Any] | None = None,
    source_chunk_ids: Sequence[Any] | None = None,
    source_document_ids: Sequence[Any] | None = None,
    derived_at: datetime | str | None = None,
    derivation_type: str | None = None,
    retrieval_path: Sequence[Mapping[str, Any]] | None = None,
    parent_artifact_ids: Sequence[Any] | None = None,
    support_density: float | None = None,
    confidence: float | None = None,
    conflicts: Sequence[Mapping[str, Any]] | None = None,
    evidence_scopes: Sequence[Mapping[str, Any]] | None = None,
    manual: bool | None = None,
    require_source_chunks: bool = False,
) -> dict[str, Any]:
    payload = dict(existing or {})

    normalized_source_chunk_ids = _string_list(
        source_chunk_ids
        if source_chunk_ids is not None
        else payload.get("source_chunk_ids") or payload.get("source_ids") or payload.get("chunk_ids")
    )
    normalized_source_document_ids = _string_list(
        source_document_ids
        if source_document_ids is not None
        else payload.get("source_document_ids") or ([payload.get("doc_id")] if payload.get("doc_id") else [])
    )
    normalized_parent_artifact_ids = _string_list(
        parent_artifact_ids if parent_artifact_ids is not None else payload.get("parent_artifact_ids")
    )
    normalized_retrieval_path = [dict(item) for item in (retrieval_path if retrieval_path is not None else payload.get("retrieval_path") or []) if isinstance(item, Mapping)]
    normalized_conflicts = [dict(item) for item in (conflicts if conflicts is not None else payload.get("conflicts") or []) if isinstance(item, Mapping)]
    normalized_evidence_scopes = [dict(item) for item in (evidence_scopes if evidence_scopes is not None else payload.get("evidence_scopes") or []) if isinstance(item, Mapping)]
    normalized_manual = bool(payload.get("manual")) if manual is None else bool(manual)

    payload["source_chunk_ids"] = normalized_source_chunk_ids
    payload["source_document_ids"] = normalized_source_document_ids
    payload["derived_at"] = _isoformat(derived_at) or payload.get("derived_at")
    payload["derivation_type"] = str(derivation_type or payload.get("derivation_type") or "").strip()
    payload["retrieval_path"] = normalized_retrieval_path
    payload["parent_artifact_ids"] = normalized_parent_artifact_ids
    payload["support_density"] = _clamp01(
        support_density if support_density is not None else payload.get("support_density")
    )
    payload["confidence"] = _clamp01(confidence if confidence is not None else payload.get("confidence"))
    payload["conflicts"] = normalized_conflicts
    payload["evidence_scopes"] = normalized_evidence_scopes
    payload["manual"] = normalized_manual

    invalid_reasons: list[str] = []
    if require_source_chunks and not normalized_manual and not normalized_source_chunk_ids:
        invalid_reasons.append("missing_source_chunk_ids")
    if payload["derivation_type"] and not payload.get("derived_at"):
        invalid_reasons.append("missing_derivation_timestamp")
    if payload["derivation_type"] and payload["support_density"] is None:
        invalid_reasons.append("missing_support_density")

    payload["valid"] = not invalid_reasons
    payload["invalid_reasons"] = invalid_reasons
    return payload


def provenance_for_artifact(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, dict):
        direct = artifact.get("provenance")
        meta = artifact.get("meta")
    else:
        direct = getattr(artifact, "provenance", None)
        meta = getattr(artifact, "meta", None)
    if isinstance(direct, Mapping):
        return dict(direct)
    if isinstance(meta, Mapping):
        provenance = meta.get("provenance")
        if isinstance(provenance, Mapping):
            return dict(provenance)
    return {}


def collect_source_chunk_ids(artifact: Any) -> list[str]:
    provenance = provenance_for_artifact(artifact)
    ids = provenance.get("source_chunk_ids") or provenance.get("source_ids") or []
    if not ids:
        if isinstance(artifact, dict):
            ids = artifact.get("chunk_ids") or artifact.get("source_ids") or []
        else:
            ids = getattr(artifact, "source_ids", None) or getattr(artifact, "chunk_ids", None) or []
    return _string_list(ids)


def collect_direct_source_chunk_ids(artifact: Any) -> list[str]:
    if isinstance(artifact, dict):
        ids = artifact.get("direct_source_chunk_ids")
    else:
        ids = getattr(artifact, "direct_source_chunk_ids", None)
    if ids is None:
        return collect_source_chunk_ids(artifact)
    return _string_list(ids)


def collect_source_document_ids(artifact: Any) -> list[str]:
    provenance = provenance_for_artifact(artifact)
    ids = provenance.get("source_document_ids") or []
    if not ids:
        if isinstance(artifact, dict):
            ids = artifact.get("doc_ids") or [artifact.get("doc_id")] if artifact.get("doc_id") else []
        else:
            doc_id = getattr(artifact, "doc_id", None)
            ids = [doc_id] if doc_id else []
    return _string_list(ids)


def collect_parent_artifact_ids(artifact: Any) -> list[str]:
    provenance = provenance_for_artifact(artifact)
    ids = provenance.get("parent_artifact_ids") or []
    if not ids and isinstance(artifact, dict):
        ids = artifact.get("parent_artifact_ids") or []
    elif not ids:
        ids = getattr(artifact, "parent_artifact_ids", None) or []
    return _string_list(ids)


def collect_parent_artifacts(artifact: Any) -> list[Any]:
    if isinstance(artifact, dict):
        parents = artifact.get("parent_artifacts") or artifact.get("supporting_artifacts") or []
    else:
        parents = (
            getattr(artifact, "parent_artifacts", None)
            or getattr(artifact, "supporting_artifacts", None)
            or []
        )
    if isinstance(parents, Sequence) and not isinstance(parents, (str, bytes)):
        return [item for item in parents if item is not None]
    return []


def collect_transitive_source_chunk_ids(artifact: Any) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()

    def _visit(item: Any) -> None:
        marker = _artifact_marker(item)
        if marker in seen:
            return
        seen.add(marker)
        for chunk_id in collect_source_chunk_ids(item):
            if chunk_id not in collected:
                collected.append(chunk_id)
        for parent in collect_parent_artifacts(item):
            _visit(parent)

    _visit(artifact)
    return collected


def collect_transitive_source_document_ids(artifact: Any) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()

    def _visit(item: Any) -> None:
        marker = _artifact_marker(item)
        if marker in seen:
            return
        seen.add(marker)
        for doc_id in collect_source_document_ids(item):
            if doc_id not in collected:
                collected.append(doc_id)
        for parent in collect_parent_artifacts(item):
            _visit(parent)

    _visit(artifact)
    return collected


def _string_list(values: Sequence[Any] | Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return [str(values)] if str(values).strip() else []
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            out.append(text)
    return out


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _clamp01(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except Exception:
        return None
    if num < 0.0:
        return 0.0
    if num > 1.0:
        return 1.0
    return num


def _artifact_marker(artifact: Any) -> str:
    if isinstance(artifact, dict):
        ident = artifact.get("id") or artifact.get("doc_id") or artifact.get("title")
    else:
        ident = getattr(artifact, "id", None) or getattr(artifact, "doc_id", None)
    if ident:
        return str(ident)
    return f"object:{id(artifact)}"
