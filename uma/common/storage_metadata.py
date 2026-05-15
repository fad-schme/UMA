"""
Canonical UMA storage taxonomy and metadata shaping helpers.

This module defines the single shared vocabulary for persisted UMA artifacts.
Kinds and lanes are explicit and stable; callers should not infer them from
ownership, storage location, or ad hoc source naming.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
from uma.common.provenance import build_provenance

RECORD_KINDS = (
    "raw_source",
    "wiki_page",
    "semantic_fact",
    "episodic_event",
    "procedural_rule",
    "profile_fact",
    "decision_trace",
    "query_artifact",
)

KB_LANES = (
    "raw",
    "wiki",
    "semantic",
    "episodic",
    "procedural",
    "profile",
    "trace",
)

SHARED_METADATA_FIELDS = (
    "kind",
    "kb_lane",
    "owner_type",
    "owner_id",
    "scope",
    "source_id",
    "source_type",
    "created_at",
    "updated_at",
    "provenance",
    "status",
)

ACTIVE_STATUS = "active"
RAW_SOURCE_KIND = "raw_source"
WIKI_PAGE_KIND = "wiki_page"
SEMANTIC_FACT_KIND = "semantic_fact"
EPISODIC_EVENT_KIND = "episodic_event"
PROCEDURAL_RULE_KIND = "procedural_rule"
PROFILE_FACT_KIND = "profile_fact"
DECISION_TRACE_KIND = "decision_trace"
QUERY_ARTIFACT_KIND = "query_artifact"

RAW_LANE = "raw"
WIKI_LANE = "wiki"
SEMANTIC_LANE = "semantic"
EPISODIC_LANE = "episodic"
PROCEDURAL_LANE = "procedural"
PROFILE_LANE = "profile"
TRACE_LANE = "trace"

_KIND_TO_LANE = {
    RAW_SOURCE_KIND: RAW_LANE,
    WIKI_PAGE_KIND: WIKI_LANE,
    SEMANTIC_FACT_KIND: SEMANTIC_LANE,
    EPISODIC_EVENT_KIND: EPISODIC_LANE,
    PROCEDURAL_RULE_KIND: PROCEDURAL_LANE,
    PROFILE_FACT_KIND: PROFILE_LANE,
    DECISION_TRACE_KIND: TRACE_LANE,
    QUERY_ARTIFACT_KIND: TRACE_LANE,
}

_LEGACY_KIND_ALIASES = {
    "manifest": RAW_SOURCE_KIND,
    "document_manifest": RAW_SOURCE_KIND,
    "skill": PROCEDURAL_RULE_KIND,
    "fact": SEMANTIC_FACT_KIND,
    "episode": EPISODIC_EVENT_KIND,
}


def lane_for_kind(kind: str) -> str:
    normalized_kind = canonical_kind_name(kind)
    lane = _KIND_TO_LANE.get(normalized_kind)
    if lane is None:
        raise ValueError(f"Unknown UMA record kind: {kind!r}")
    return lane


def canonical_kind_name(kind: Any) -> str:
    normalized_kind = _clean_string(kind)
    if not normalized_kind:
        return ""
    return _LEGACY_KIND_ALIASES.get(normalized_kind, normalized_kind)


def canonical_scope(*, owner_type: str, session_id: str | None = None) -> str:
    if _clean_string(session_id):
        return "session"
    normalized_owner_type = _clean_string(owner_type)
    if normalized_owner_type in {"agent", "user", "workspace", "system"}:
        return normalized_owner_type
    return "unknown"


def normalize_metadata(
    *,
    meta: Mapping[str, Any] | None,
    kind: str,
    owner_type: str,
    owner_id: str,
    created_at: datetime | str | None,
    updated_at: datetime | str | None,
    source_id: str | None,
    source_type: str | None,
    session_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """
    Merge existing metadata into UMA's canonical shared metadata vocabulary.

    Existing keys are preserved unless they conflict with canonical semantics.
    The returned dict is the authoritative metadata object callers should rely on.
    """
    out = dict(meta or {})
    canonical_kind = canonical_kind_name(out.get("kind")) or canonical_kind_name(kind)
    if canonical_kind not in RECORD_KINDS:
        raise ValueError(f"Unknown UMA record kind: {canonical_kind!r}")

    canonical_lane = _clean_string(out.get("kb_lane")) or lane_for_kind(canonical_kind)
    if canonical_lane not in KB_LANES:
        raise ValueError(f"Unknown UMA kb_lane: {canonical_lane!r}")

    provenance_payload = dict(out.get("provenance") or {})
    if provenance:
        provenance_payload.update(dict(provenance))

    out["kind"] = canonical_kind
    out["kb_lane"] = canonical_lane
    out["owner_type"] = _clean_string(owner_type) or str(owner_type or "")
    out["owner_id"] = str(owner_id or "")
    out["scope"] = _clean_string(out.get("scope")) or canonical_scope(
        owner_type=owner_type,
        session_id=session_id,
    )
    out["source_id"] = _clean_string(out.get("source_id")) or _clean_string(source_id)
    out["source_type"] = _clean_string(out.get("source_type")) or _clean_string(source_type)
    out["created_at"] = _isoformat(out.get("created_at")) or _isoformat(created_at)
    out["updated_at"] = _isoformat(out.get("updated_at")) or _isoformat(updated_at) or out["created_at"]
    out["provenance"] = provenance_payload
    out["status"] = _clean_string(out.get("status")) or _clean_string(status) or ACTIVE_STATUS
    return out


def shared_metadata_view(
    *,
    meta: Mapping[str, Any] | None,
    owner_type: str,
    owner_id: str,
    created_at: datetime | str | None,
    updated_at: datetime | str | None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Return the canonical metadata view exposed at retrieval/serialization boundaries.
    """
    normalized = dict(meta or {})
    return {
        "kind": normalized.get("kind"),
        "kb_lane": normalized.get("kb_lane"),
        "owner_type": _clean_string(normalized.get("owner_type")) or _clean_string(owner_type),
        "owner_id": str(normalized.get("owner_id") or owner_id or ""),
        "scope": _clean_string(normalized.get("scope")) or canonical_scope(
            owner_type=owner_type,
            session_id=session_id,
        ),
        "source_id": normalized.get("source_id"),
        "source_type": normalized.get("source_type"),
        "created_at": normalized.get("created_at") or _isoformat(created_at),
        "updated_at": normalized.get("updated_at") or _isoformat(updated_at) or _isoformat(created_at),
        "provenance": dict(normalized.get("provenance") or {}),
        "status": normalized.get("status") or ACTIVE_STATUS,
    }


def normalize_document_metadata(
    meta: Mapping[str, Any] | None,
    *,
    doc_id: str,
    owner_type: str,
    owner_id: str,
    ingested_at: datetime | str | None,
    source_path: str,
    source_hash: str,
) -> dict[str, Any]:
    existing = dict(meta or {})
    explicit_kind = canonical_kind_name(existing.get("kind"))
    explicit_source_kind = _clean_string(existing.get("source_kind"))
    kind = explicit_kind or (WIKI_PAGE_KIND if explicit_source_kind == WIKI_PAGE_KIND else RAW_SOURCE_KIND)
    kb_lane = _clean_string(existing.get("kb_lane")) or lane_for_kind(kind)
    return normalize_metadata(
        meta=existing,
        kind=kind,
        owner_type=owner_type,
        owner_id=owner_id,
        created_at=ingested_at,
        updated_at=ingested_at,
        source_id=doc_id,
        source_type=existing.get("source_type") or "document",
        provenance=build_provenance(
            existing=existing.get("provenance"),
            source_document_ids=[doc_id],
            derived_at=ingested_at,
            derivation_type="document_ingest",
            manual=(kind == WIKI_PAGE_KIND and explicit_source_kind == WIKI_PAGE_KIND),
            require_source_chunks=False,
        )
        | {
            "doc_id": doc_id,
            "source_path": source_path,
            "source_hash": source_hash,
            "immutable_source": kb_lane == RAW_LANE,
            "projection_only": False,
        },
        status=existing.get("status") or ACTIVE_STATUS,
    )


def normalize_chunk_metadata(
    meta: Mapping[str, Any] | None,
    *,
    chunk_id: str,
    doc_id: str,
    owner_type: str,
    owner_id: str,
    created_at: datetime | str | None,
    updated_at: datetime | str | None,
    page_range: tuple[int, int] | None,
    position: int | None,
    source_path: str,
    source_hash: str,
) -> dict[str, Any]:
    normalized = normalize_metadata(
        meta=meta,
        kind=RAW_SOURCE_KIND,
        owner_type=owner_type,
        owner_id=owner_id,
        created_at=created_at,
        updated_at=updated_at,
        source_id=doc_id,
        source_type="document_chunk",
        provenance=build_provenance(
            existing=(meta or {}).get("provenance") if isinstance(meta, Mapping) else None,
            source_chunk_ids=[chunk_id],
            source_document_ids=[doc_id],
            derived_at=updated_at or created_at,
            derivation_type="chunk_ingest",
            support_density=1.0,
            require_source_chunks=True,
        )
        | {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source_path": source_path,
            "source_hash": source_hash,
            "page_range": list(page_range) if page_range is not None else None,
            "position": position,
            "immutable_source": True,
        },
    )
    normalized.setdefault("domain", "kb_doc")
    return normalized


def normalize_fact_metadata(
    meta: Mapping[str, Any] | None,
    *,
    fact_id: str,
    owner_type: str,
    owner_id: str,
    created_at: datetime | str | None,
    updated_at: datetime | str | None,
    source_ids: list[str] | None,
    session_id: str | None,
) -> dict[str, Any]:
    existing = dict(meta or {})
    explicit_kind = canonical_kind_name(existing.get("kind"))
    explicit_lane = _clean_string(existing.get("kb_lane"))
    domain = _clean_string(existing.get("domain"))
    kind = explicit_kind
    if not kind:
        if domain == "user_profile":
            kind = PROFILE_FACT_KIND
        else:
            kind = SEMANTIC_FACT_KIND
    lane = explicit_lane or lane_for_kind(kind)
    normalized = normalize_metadata(
        meta={**existing, "kb_lane": lane},
        kind=kind,
        owner_type=owner_type,
        owner_id=owner_id,
        created_at=created_at,
        updated_at=updated_at,
        source_id=(source_ids[0] if source_ids else existing.get("doc_id") or fact_id),
        source_type=existing.get("source_type") or ("chunk" if source_ids else "fact"),
        session_id=session_id,
        provenance=build_provenance(
            existing=existing.get("provenance"),
            source_chunk_ids=list(source_ids or []),
            source_document_ids=([existing.get("doc_id")] if existing.get("doc_id") else []),
            derived_at=updated_at or created_at,
            derivation_type=("promotion" if (existing.get("promotion") or existing.get("promoted_from")) else "semantic_extract"),
            parent_artifact_ids=(
                [existing.get("promotion", {}).get("source_fact_id")]
                if isinstance(existing.get("promotion"), Mapping) and existing.get("promotion", {}).get("source_fact_id")
                else ([existing.get("promoted_from", {}).get("fact_id")] if isinstance(existing.get("promoted_from"), Mapping) and existing.get("promoted_from", {}).get("fact_id") else [])
            ),
            support_density=(1.0 if source_ids else 0.0),
            require_source_chunks=True,
        )
        | {
            "fact_id": fact_id,
            "source_ids": list(source_ids or []),
            "doc_id": existing.get("doc_id"),
            "source_path": existing.get("source_path"),
            "source_hash": existing.get("source_hash"),
            "derivation": existing.get("promotion") or existing.get("promoted_from"),
        },
    )
    if kind == PROFILE_FACT_KIND:
        normalized.setdefault("domain", "user_profile")
    else:
        normalized.setdefault("domain", "kb_doc")
    return normalized


def normalize_episode_metadata(
    meta: Mapping[str, Any] | None,
    *,
    episode_id: str,
    owner_type: str,
    owner_id: str,
    timestamp: datetime | str | None,
    session_id: str | None,
) -> dict[str, Any]:
    existing = dict(meta or {})
    normalized = normalize_metadata(
        meta=existing,
        kind=EPISODIC_EVENT_KIND,
        owner_type=owner_type,
        owner_id=owner_id,
        created_at=timestamp,
        updated_at=timestamp,
        source_id=episode_id,
        source_type=(existing.get("source_type") or "episode"),
        session_id=session_id,
        provenance=build_provenance(
            existing=existing.get("provenance"),
            source_document_ids=([existing.get("doc_id")] if existing.get("doc_id") else []),
            derived_at=timestamp,
            derivation_type=str(existing.get("import_mode") or "episode_event"),
            manual=(existing.get("import_mode") == "manual"),
            support_density=(1.0 if existing.get("doc_id") else None),
            require_source_chunks=False,
        )
        | {
            "episode_id": episode_id,
            "doc_id": existing.get("doc_id"),
            "source_file": existing.get("source_file"),
            "diary_date": existing.get("diary_date"),
            "turn_id": existing.get("turn_id"),
            "import_mode": existing.get("import_mode"),
        },
    )
    return normalized


def normalize_skill_metadata(
    meta: Mapping[str, Any] | None,
    *,
    skill_id: str,
    owner_type: str,
    owner_id: str,
    created_at: datetime | str | None,
    updated_at: datetime | str | None,
) -> dict[str, Any]:
    """
    Normalize procedural-rule metadata.

    Minimum provenance contract today:
    - always preserve `skill_id`
    - preserve authored/import source hints when callers provide them
      (`source`, `source_file`, `import_mode`)

    Procedural rules are often authored directly rather than extracted from raw
    evidence, so evidence-backed provenance is not implied unless the caller
    provides a real source linkage.
    """
    existing = dict(meta or {})
    normalized = normalize_metadata(
        meta=existing,
        kind=PROCEDURAL_RULE_KIND,
        owner_type=owner_type,
        owner_id=owner_id,
        created_at=created_at,
        updated_at=updated_at,
        source_id=skill_id,
        source_type=existing.get("source_type") or "skill",
        provenance=build_provenance(
            existing=existing.get("provenance"),
            derived_at=updated_at or created_at,
            derivation_type=str(existing.get("import_mode") or "manual"),
            manual=True,
            require_source_chunks=False,
        )
        | {
            "skill_id": skill_id,
            "source": existing.get("source"),
            "source_file": existing.get("source_file"),
            "import_mode": existing.get("import_mode"),
        },
    )
    normalized.setdefault("domain", "procedural")
    return normalized


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
