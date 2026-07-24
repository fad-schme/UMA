from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

from .types import (
    CaptureSourceResult,
    CurateCompiledMemoryResult,
    DeriveMemoryArtifactsResult,
    DocumentChunk,
    IngestConfig,
    IngestReport,
    ParsedDocument,
)
from .parser import parse_file
from .normalizer import normalize_document
from .chunker import chunk_sections, finalize_chunks
from .embedder import embed_chunks
from uma.memory.semantic import extractor as semantic_extractor
from .episodic_writer import write_document_episode

from uma.common.types import Chunk, Fact
from uma.stores.document_sql import DocumentRecord
from uma.common.ownership import validate_explicit_owner
from uma.common.storage_metadata import normalize_document_metadata
from uma.common.integrity import hash_fact_content
from uma.common.trust import SourceDescriptor, score_source
from uma.common.injection_scan import scan_artifact_text
from uma.retrieve.user_query_helper import build_fact_embedding_text
logger = logging.getLogger(__name__)

_INGEST_PIPELINE_VERSION = "doc_ingest_v1"
_EXTRACTOR_VERSION = "doc_fact_extract_v1"
_SPLITTER_VERSION = "doc_normalize_v1"
_CHUNKER_VERSION = "doc_chunk_v2"
_MEMORY_BOOTSTRAP_VERSION = "memory_bootstrap_v2"
_DAILY_DIARY_BOOTSTRAP_VERSION = "daily_diary_bootstrap_v1"


class _IngestRuntime(NamedTuple):
    embedder: Any
    llm: Any
    semantic_core: Any
    chunk_core: Any
    episodic_core: Any
    graph_core: Any
    document_store: Any
    embedding_cfg: Any


def _merge_manifest_meta(
    *,
    existing: dict | None,
    ingest_signature: dict,
    now: datetime,
    reingest_reason: str | None = None,
    sanitization_counts: dict | None = None,
) -> dict:
    meta = dict(existing or {})
    meta.setdefault("created_by", _INGEST_PIPELINE_VERSION)
    meta.setdefault("first_seen_at", now.isoformat())

    meta["last_seen_at"] = now.isoformat()

    prior_sig = meta.get("ingest_signature")
    if prior_sig and prior_sig != ingest_signature:
        meta["prior_ingest_signature"] = prior_sig

    meta["ingest_signature"] = ingest_signature

    if reingest_reason:
        meta["reingest_reason"] = reingest_reason

    if sanitization_counts:
        security = dict(meta.get("security") or {})
        security["sanitization"] = sanitization_counts
        meta["security"] = security

    # Keep a short capped history for auditability / migrations.
    history = meta.get("ingest_history")
    if not isinstance(history, list):
        history = []

    entry = {
        "at": now.isoformat(),
        "signature": ingest_signature,
    }
    if reingest_reason:
        entry["reason"] = reingest_reason
    history.append(entry)
    meta["ingest_history"] = history[-5:]

    return meta


def _resolve_ingest_runtime(memory: Any) -> _IngestRuntime:
    semantic_core = getattr(memory, "semantic_core", None)
    chunk_core = getattr(memory, "chunk_core", None)
    episodic_core = getattr(memory, "episodic_core", None)
    graph_core = getattr(memory, "graph_core", None)
    document_store = getattr(memory, "document_store", None)
    embedder = getattr(memory, "embedder", None)
    llm = getattr(memory, "llm", None)
    embedding_cfg = getattr(memory, "embedding_cfg", None)

    if semantic_core is None:
        raise ValueError("ingest_document: memory.semantic_core is required")
    if chunk_core is None:
        raise ValueError("ingest_document: memory.chunk_core is required")
    if episodic_core is None:
        raise ValueError("ingest_document: memory.episodic_core is required")
    if document_store is None:
        raise ValueError("ingest_document: memory.document_store is required")
    if embedder is None or not hasattr(embedder, "embed"):
        raise ValueError("ingest_document: memory.embedder with .embed() required")
    if llm is None or not hasattr(llm, "generate"):
        raise ValueError("ingest_document: memory.llm with .generate() required")
    if embedding_cfg is None:
        raise ValueError("ingest_document: memory.embedding_cfg is required")
    if not getattr(embedding_cfg, "model", None):
        raise ValueError("ingest_document: memory.embedding_cfg.model is required for idempotent ingest signatures")

    return _IngestRuntime(
        embedder=embedder,
        llm=llm,
        semantic_core=semantic_core,
        chunk_core=chunk_core,
        episodic_core=episodic_core,
        graph_core=graph_core,
        document_store=document_store,
        embedding_cfg=embedding_cfg,
    )


def _build_ingest_signature(config: IngestConfig, runtime: _IngestRuntime) -> dict:
    return {
        "pipeline_version": _INGEST_PIPELINE_VERSION,
        "splitter_version": _SPLITTER_VERSION,
        "chunker_version": _CHUNKER_VERSION,
        "extractor_version": _EXTRACTOR_VERSION,
        "chunk_size_tokens": config.chunk_size_tokens,
        "overlap_tokens": config.overlap_tokens,
        "embedding_model": runtime.embedding_cfg.model,
        "embedding_dim": getattr(runtime.embedder, "dimension", None),
    }


def _validate_text_source_path(file_path: str, *, api_name: str) -> str:
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError(f"{api_name}: file_path must be a non-empty string")

    normalized_path = os.path.abspath(file_path.strip())
    if not os.path.exists(normalized_path):
        return normalized_path
    if not os.path.isfile(normalized_path):
        raise ValueError(f"{api_name}: path is not a file: {normalized_path}")
    return normalized_path


def _build_skip_result(
    *,
    reason: str,
    path: str,
    user_id: str,
    tenant_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "skipped",
        "reason": reason,
        "path": path,
        "user_id": user_id,
        "tenant_id": tenant_id,
    }
    if extra:
        payload.update(extra)
    return payload


def _read_text_source(path: str, *, api_name: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        logger.exception("%s: failed to read file path=%s", api_name, path)
        raise RuntimeError(f"{api_name}: failed to read file: {path}") from exc




async def _embed_and_persist_facts(
    *,
    facts: List[Fact],
    embedder: Any,
    semantic_store: Any,
    warnings: List[str] | None = None,
    log_context: str,
) -> list[str]:
    if not facts:
        return []
    texts = [build_fact_embedding_text(fact) for fact in facts]
    try:
        vectors = await embedder.embed(texts)
    except Exception:
        logger.exception("%s: fact embedding failed", log_context)
        if warnings is not None:
            warnings.append("failed to embed extracted facts")
        return []

    expected_dim = getattr(embedder, "dimension", None)
    if not isinstance(expected_dim, int) or expected_dim <= 0:
        raise ValueError(f"{log_context}: embedder.dimension must be a positive integer")
    if not isinstance(vectors, list) or len(vectors) != len(facts):
        raise RuntimeError(f"{log_context}: embedding returned invalid shape for extracted facts")

    persisted_fact_ids: list[str] = []
    for fact, vector in zip(facts, vectors):
        if not isinstance(vector, list) or len(vector) != expected_dim:
            raise ValueError(
                f"{log_context}: invalid fact embedding dim for fact_id={fact.id} "
                f"(expected={expected_dim} got={len(vector) if isinstance(vector, list) else None})"
            )
        try:
            await semantic_store.upsert_fact(fact, vector)
            persisted_fact_ids.append(fact.id)
        except Exception:
            logger.exception("%s: failed to upsert extracted fact %s", log_context, fact.id)
            if warnings is not None:
                warnings.append(f"failed to persist extracted fact {fact.id}")
    return persisted_fact_ids


async def _load_existing_manifest(
    *,
    document_store: Any,
    tenant_id: str,
    owner_type: str,
    owner_id: str,
    source_hash: str,
    log_context: str,
) -> Any | None:
    if document_store is None or not hasattr(document_store, "get_by_owner_and_hash"):
        return None
    if not tenant_id:
        logger.error("%s: _load_existing_manifest requires tenant_id", log_context)
        raise ValueError(f"{log_context}: tenant_id is required for manifest lookup")
    try:
        return await document_store.get_by_owner_and_hash(
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            source_hash=source_hash,
        )
    except Exception:
        logger.exception("%s: manifest lookup failed", log_context)
        return None


async def _load_latest_manifest_for_source(
    *,
    document_store: Any,
    tenant_id: str,
    owner_type: str,
    owner_id: str,
    source_path: str,
    log_context: str,
) -> Any | None:
    if document_store is None or not hasattr(document_store, "get_latest_by_source_path"):
        return None
    if not tenant_id:
        logger.error("%s: _load_latest_manifest_for_source requires tenant_id", log_context)
        raise ValueError(f"{log_context}: tenant_id is required for manifest lookup")
    try:
        return await document_store.get_latest_by_source_path(
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            source_path=source_path,
        )
    except Exception:
        logger.exception("%s: source-identity manifest lookup failed", log_context)
        return None


def _manifest_signature_matches(*, existing_manifest: Any | None, ingest_signature: dict[str, Any]) -> bool:
    if existing_manifest is None:
        return False
    existing_sig = (getattr(existing_manifest, "meta", None) or {}).get("ingest_signature") or {}
    return existing_sig == ingest_signature


def _prepare_superseding_manifest_meta(
    *,
    existing_meta: dict[str, Any] | None,
    ingest_signature: dict[str, Any],
    now: datetime,
    prior_doc_id: str | None,
    reingest_reason: str | None = None,
    sanitization_counts: dict | None = None,
) -> dict[str, Any]:
    meta = _merge_manifest_meta(
        existing=existing_meta,
        ingest_signature=ingest_signature,
        now=now,
        reingest_reason=reingest_reason,
        sanitization_counts=sanitization_counts,
    )
    meta.pop("superseded_by", None)
    meta.pop("superseded_at", None)
    if prior_doc_id:
        meta["supersedes"] = prior_doc_id
    else:
        meta.pop("supersedes", None)
    return meta


async def _record_manifest_supersession(
    *,
    document_store: Any,
    prior_manifest: Any | None,
    new_doc_id: str,
    now: datetime,
    log_context: str,
) -> None:
    if prior_manifest is None or document_store is None:
        return

    prior_meta = dict(getattr(prior_manifest, "meta", None) or {})
    prior_meta["superseded_by"] = new_doc_id
    prior_meta["superseded_at"] = now.isoformat()

    await _upsert_source_manifest(
        document_store=document_store,
        doc_id=prior_manifest.doc_id,
        source_path=prior_manifest.source_path,
        source_hash=prior_manifest.source_hash,
        ingested_at=getattr(prior_manifest, "ingested_at", now),
        tenant_id=prior_manifest.tenant_id,
        owner_type=prior_manifest.owner_type,
        owner_id=prior_manifest.owner_id,
        workspace_id=getattr(prior_manifest, "workspace_id", None),
        origin_agent_id=getattr(prior_manifest, "origin_agent_id", None),
        origin_user_id=getattr(prior_manifest, "origin_user_id", None),
        origin_session_id=getattr(prior_manifest, "origin_session_id", None),
        meta=prior_meta,
        log_context=log_context,
    )


async def _upsert_source_manifest(
    *,
    document_store: Any,
    doc_id: str,
    source_path: str,
    source_hash: str,
    ingested_at: datetime,
    tenant_id: str,
    owner_type: str,
    owner_id: str,
    workspace_id: str | None,
    origin_agent_id: str | None,
    origin_user_id: str | None,
    origin_session_id: str | None,
    meta: dict[str, Any],
    log_context: str,
) -> None:
    if document_store is None or not hasattr(document_store, "upsert_document"):
        return
    try:
        await document_store.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                source_path=source_path,
                source_hash=source_hash,
                ingested_at=ingested_at,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                workspace_id=workspace_id,
                origin_agent_id=origin_agent_id,
                origin_user_id=origin_user_id,
                origin_session_id=origin_session_id,
                scope_model_version="v2",
                meta=meta,
            )
        )
    except Exception:
        logger.exception("%s: failed to persist manifest doc_id=%s", log_context, doc_id)


def _build_bootstrap_manifest_meta(
    *,
    source_kind: str,
    source_type: str,
    ingest_signature: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "source_kind": source_kind,
        "import_mode": "bootstrap",
        "ingest_signature": ingest_signature,
        "source_type": source_type,
    }
    if extra:
        meta.update(extra)
    return meta


async def _capture_bootstrap_source(
    *,
    file_path: str,
    runtime_context: Any,
    document_store: Any,
    api_name: str,
    source_kind: str,
    signature_builder: Any,
    entry_extractor: Any,
    config: Any | None = None,
) -> tuple[str, str, list[str], dict[str, Any], str, Dict[str, Any] | None]:
    normalized_user_id = runtime_context.user_id
    normalized_tenant_id = runtime_context.tenant_id
    normalized_path = _validate_text_source_path(file_path, api_name=api_name)

    if not normalized_user_id:
        raise ValueError(f"{api_name}: bound runtime_context.user_id is required")
    if not os.path.exists(normalized_path):
        return normalized_path, "", [], {}, "", _build_skip_result(
            reason="missing_file",
            path=normalized_path,
            user_id=normalized_user_id,
            tenant_id=normalized_tenant_id,
        )

    # H2 defense: bound the file size before reading. Bootstrap files are
    # typically tiny (MEMORY.md, daily diary entries), so the cap rarely
    # matters in practice but provides defense against a caller pointing at
    # an unintended large file.
    if config is not None:
        from uma.ingest.mime_check import enforce_file_size_limit
        max_bytes = int(getattr(config, "max_file_bytes", 0) or 0)
        if max_bytes > 0:
            enforce_file_size_limit(normalized_path, max_bytes)

    raw_text = _read_text_source(
        normalized_path,
        api_name=api_name,
    )
    if not raw_text.strip():
        return normalized_path, raw_text, [], {}, "", _build_skip_result(
            reason="empty_file",
            path=normalized_path,
            user_id=normalized_user_id,
            tenant_id=normalized_tenant_id,
        )

    entries = list(entry_extractor(raw_text))
    if not entries:
        return normalized_path, raw_text, [], {}, "", _build_skip_result(
            reason="no_entries",
            path=normalized_path,
            user_id=normalized_user_id,
            tenant_id=normalized_tenant_id,
        )

    ingest_signature = dict(signature_builder(raw_text=raw_text))
    source_hash = str(ingest_signature.get("content_hash") or "")
    existing_manifest = await _load_existing_manifest(
        document_store=document_store,
        tenant_id=normalized_tenant_id,
        owner_type="user",
        owner_id=normalized_user_id,
        source_hash=source_hash,
        log_context=api_name,
    )
    if _manifest_signature_matches(existing_manifest=existing_manifest, ingest_signature=ingest_signature):
        return normalized_path, raw_text, entries, ingest_signature, source_hash, _build_skip_result(
            reason="idempotent",
            path=normalized_path,
            user_id=normalized_user_id,
            tenant_id=normalized_tenant_id,
            extra={"entries_found": len(entries), "source_kind": source_kind},
        )

    return normalized_path, raw_text, entries, ingest_signature, source_hash, None


async def _run_manifest_gate(
    *,
    parsed: ParsedDocument,
    config: IngestConfig,
    runtime: _IngestRuntime,
    owner_type: str,
    owner_id: str,
    tenant_id: str,
    workspace_id: str | None,
    warnings: List[str],
) -> tuple[dict, Any | None, IngestReport | None]:
    ingest_signature = _build_ingest_signature(config, runtime)
    exact_manifest = await _load_existing_manifest(
        document_store=runtime.document_store,
        tenant_id=tenant_id,
        owner_type=owner_type,
        owner_id=owner_id,
        source_hash=parsed.source_hash,
        log_context="ingest_document",
    )

    if exact_manifest is not None and _manifest_signature_matches(
        existing_manifest=exact_manifest,
        ingest_signature=ingest_signature,
    ):
        now_refresh = datetime.now(timezone.utc)
        await _upsert_source_manifest(
            document_store=runtime.document_store,
            doc_id=exact_manifest.doc_id,
            source_path=parsed.source_path,
            source_hash=parsed.source_hash,
            ingested_at=now_refresh,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            workspace_id=workspace_id,
            origin_agent_id=getattr(exact_manifest, "origin_agent_id", None),
            origin_user_id=getattr(exact_manifest, "origin_user_id", None),
            origin_session_id=getattr(exact_manifest, "origin_session_id", None),
            meta=_merge_manifest_meta(
                existing=getattr(exact_manifest, "meta", None) or {},
                ingest_signature=ingest_signature,
                now=now_refresh,
            ),
            log_context="ingest_document",
        )

        warnings.append(f"skipped ingest (idempotent): owner={owner_type}:{owner_id} hash={parsed.source_hash}")
        return (
            ingest_signature,
            exact_manifest,
            IngestReport(
                doc_id=exact_manifest.doc_id,
                chunks_created=0,
                facts_created=0,
                graph_edges_created=0,
                warnings=warnings,
            ),
        )

    prior_manifest = exact_manifest
    if prior_manifest is None:
        prior_manifest = await _load_latest_manifest_for_source(
            document_store=runtime.document_store,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            source_path=parsed.source_path,
            log_context="ingest_document",
        )
        if prior_manifest is None:
            return ingest_signature, None, None

    warnings.append(f"re-ingesting existing manifest doc_id={prior_manifest.doc_id} (signature changed)")
    return ingest_signature, prior_manifest, None


def _prepare_document_chunks(
    *,
    parsed: ParsedDocument,
    config: IngestConfig,
    warnings: List[str],
) -> tuple[List[DocumentChunk], IngestReport | None]:
    sections = normalize_document(parsed)
    if not sections:
        warnings.append("no sections after normalization")

    raw_chunks = chunk_sections(
        sections,
        chunk_size_tokens=config.chunk_size_tokens,
        overlap_tokens=config.overlap_tokens,
    )
    logger.info("DOC_CHUNK_RAW count=%d", len(raw_chunks))
    if not raw_chunks:
        warnings.append("no chunks created")

    try:
        final_chunks = finalize_chunks(raw_chunks)
    except Exception as exc:
        logger.exception("ingest_document: strict chunk validation failed; refusing persistence")
        raise ValueError(f"ingest_document: strict chunk validation failed: {exc}") from exc

    logger.info("DOC_CHUNK_FINAL count=%d", len(final_chunks))
    if final_chunks:
        return final_chunks, None

    logger.warning("No final_chunks produced; skipping embedding/persistence.")
    return (
        final_chunks,
        IngestReport(
            doc_id=parsed.doc_id,
            chunks_created=0,
            facts_created=0,
            graph_edges_created=0,
            warnings=warnings + ["no final chunks produced"],
        ),
    )


def _chunk_records_to_inputs(chunks: List[Chunk]) -> List[DocumentChunk]:
    out: List[DocumentChunk] = []
    for chunk in chunks or []:
        meta = dict(getattr(chunk, "meta", None) or {})
        out.append(
            DocumentChunk(
                chunk_id=str(getattr(chunk, "id", "") or ""),
                doc_id=str(getattr(chunk, "doc_id", "") or ""),
                text=str(getattr(chunk, "text", "") or ""),
                page_range=getattr(chunk, "page_range", (1, 1)),
                position=int(getattr(chunk, "position", 0) or 0),
                paragraph_index_start=meta.get("paragraph_index_start"),
                paragraph_index_end=meta.get("paragraph_index_end"),
            )
        )
    return out


async def _fetch_existing_chunks_for_doc(
    *,
    runtime: _IngestRuntime,
    tenant_id: str,
    owner_type: str,
    owner_id: str,
    doc_id: str,
) -> List[Chunk]:
    chunk_store = getattr(runtime.chunk_core, "store", None)
    if chunk_store is None:
        return []
    conn = chunk_store._conn()
    try:
        rows = chunk_store._query_all(
            conn,
            """
            SELECT *
            FROM chunks
            WHERE tenant_id = ?
              AND owner_type = ?
              AND owner_id = ?
              AND doc_id = ?
            ORDER BY position ASC
            """,
            params=[tenant_id, owner_type, owner_id, doc_id],
            log_context="ingest_capture_existing_chunks",
        )
        return [chunk_store._row_to_object(row) for row in (rows or [])]
    finally:
        conn.close()


async def _persist_chunks(
    *,
    parsed: ParsedDocument,
    final_chunks: List[DocumentChunk],
    config: IngestConfig,
    runtime: _IngestRuntime,
    ingest_signature: dict,
    existing_manifest: Any | None,
    owner_type: str,
    owner_id: str,
    tenant_id: str,
    workspace_id: str | None,
    warnings: List[str],
) -> List[Chunk]:
    is_supersession = existing_manifest is not None and getattr(existing_manifest, "doc_id", None) != parsed.doc_id
    chunk_rows = _build_chunk_rows(
        parsed=parsed,
        final_chunks=final_chunks,
        config=config,
        tenant_id=tenant_id,
        owner_type=owner_type,
        owner_id=owner_id,
        workspace_id=workspace_id,
    )

    # Embed first; if this raises, no manifest is written so re-ingest remains safe.
    created_chunks = await _embed_and_upsert_chunks(
        final_chunks=final_chunks,
        chunk_rows=chunk_rows,
        config=config,
        runtime=runtime,
        warnings=warnings,
    )

    # Write manifest only after embedding succeeds to avoid stale manifest blocking re-ingest.
    now_persist = datetime.now(timezone.utc)
    reingest_reason = "signature_changed" if existing_manifest is not None else None
    await _upsert_source_manifest(
        document_store=runtime.document_store,
        doc_id=parsed.doc_id,
        source_path=parsed.source_path,
        source_hash=parsed.source_hash,
        ingested_at=parsed.extracted_at,
        tenant_id=tenant_id,
        owner_type=owner_type,
        owner_id=owner_id,
        workspace_id=workspace_id,
        origin_agent_id=getattr(existing_manifest, "origin_agent_id", None) if existing_manifest is not None else None,
        origin_user_id=getattr(existing_manifest, "origin_user_id", None) if existing_manifest is not None else None,
        origin_session_id=getattr(existing_manifest, "origin_session_id", None) if existing_manifest is not None else None,
        meta=_prepare_superseding_manifest_meta(
            existing_meta=(getattr(existing_manifest, "meta", None) or {}) if existing_manifest is not None else {},
            ingest_signature=ingest_signature,
            now=now_persist,
            prior_doc_id=getattr(existing_manifest, "doc_id", None) if is_supersession else None,
            reingest_reason=reingest_reason,
            sanitization_counts=getattr(parsed, "sanitization_counts", None) or None,
        ),
        log_context="ingest_document",
    )
    await _record_manifest_supersession(
        document_store=runtime.document_store,
        prior_manifest=existing_manifest if is_supersession else None,
        new_doc_id=parsed.doc_id,
        now=now_persist,
        log_context="ingest_document",
    )
    return created_chunks


def _build_chunk_rows(
    *,
    parsed: ParsedDocument,
    final_chunks: List[DocumentChunk],
    config: IngestConfig,
    tenant_id: str,
    owner_type: str,
    owner_id: str,
    workspace_id: str | None,
) -> Dict[str, Chunk]:
    chunk_rows: Dict[str, Chunk] = {}
    now = datetime.now(timezone.utc)
    for chunk in final_chunks:
        text_hash = hashlib.sha256((chunk.text or "").encode("utf-8")).hexdigest()
        chunk_trust = score_source(SourceDescriptor(kind="document"))
        chunk_meta: Dict[str, Any] = {
            "text_hash": text_hash,
            "chunk_size_tokens": config.chunk_size_tokens,
            "overlap_tokens": config.overlap_tokens,
            "chunker_version": _CHUNKER_VERSION,
            "paragraph_index_scope": "page_range",
            "paragraph_index_start": chunk.paragraph_index_start,
            "paragraph_index_end": chunk.paragraph_index_end,
        }
        chunk_trust, chunk_meta, chunk_quarantined_at = scan_artifact_text(
            chunk.text or "",
            chunk_trust,
            chunk_meta,
            log_context=f"ingest/{chunk.doc_id}",
            now=now,
        )
        chunk_rows[chunk.chunk_id] = Chunk(
            id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            page_range=chunk.page_range,
            position=chunk.position,
            source_path=parsed.source_path,
            source_hash=parsed.source_hash,
            created_at=now,
            updated_at=now,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            workspace_id=workspace_id,
            trust_score=chunk_trust,
            quarantined_at=chunk_quarantined_at,
            meta=chunk_meta,
        )
    return chunk_rows


async def _embed_and_upsert_chunks(
    *,
    final_chunks: List[DocumentChunk],
    chunk_rows: Dict[str, Chunk],
    config: IngestConfig,
    runtime: _IngestRuntime,
    warnings: List[str],
) -> List[Chunk]:
    created_chunks: List[Chunk] = []
    logger.info(
        "DOC_CHUNK_EMBED_AND_PERSIST count=%d sample_ids=%s",
        len(final_chunks),
        [chunk.chunk_id for chunk in final_chunks[:3]],
    )
    expected_dim = getattr(runtime.embedder, "dimension", None)
    chunk_embeddings = await embed_chunks(
        final_chunks,
        embedder=runtime.embedder,
        batch_size=config.embed_batch_size,
        expected_dim=expected_dim if isinstance(expected_dim, int) and expected_dim > 0 else None,
        max_attempts=config.embed_max_retries,
        initial_delay=config.embed_initial_delay_s,
        backoff_factor=config.embed_backoff_factor,
        max_delay=config.embed_max_delay_s,
        strict=True,
    )
    if set(chunk_embeddings.keys()) != {chunk.chunk_id for chunk in final_chunks}:
        raise RuntimeError("Embedding id mismatch; refusing to persist inconsistent chunk set.")

    for chunk_id, embedding in chunk_embeddings.items():
        row = chunk_rows.get(chunk_id)
        if row is None:
            continue
        try:
            await runtime.chunk_core.upsert_chunk(row, embedding)
            created_chunks.append(row)
        except Exception:
            warnings.append(f"failed to persist chunk {chunk_id}")
            logger.exception("ingest_document: failed to upsert chunk %s", chunk_id)
    return created_chunks


async def _extract_facts_and_update_graph(
    *,
    parsed: ParsedDocument,
    final_chunks: List[DocumentChunk],
    persisted_chunks: List[Chunk],
    config: IngestConfig,
    runtime: _IngestRuntime,
    owner_type: str,
    owner_id: str,
    tenant_id: str,
    workspace_id: str | None,
    warnings: List[str],
) -> tuple[List[Fact], int, int, List[str]]:
    # H3 defense: drop quarantined chunks before fact extraction.
    # Quarantine means "do not use this artifact for anything." Sending a
    # quarantined chunk to the LLM is a use — it pays for the call, may
    # leak the payload into provider logs, and produces facts that have
    # to be quarantined anyway. Filtering upstream is cheaper and stricter.
    #
    # Pairing rule: captured_chunk_inputs[i] corresponds to captured_chunks[i]
    # by chunk_id. We build a set of quarantined chunk_ids from the persisted
    # Chunk records and drop the matching DocumentChunk inputs.
    quarantined_ids: set[str] = set()
    for ch in persisted_chunks or []:
        if getattr(ch, "quarantined_at", None) is not None:
            cid = getattr(ch, "id", None)
            if isinstance(cid, str) and cid:
                quarantined_ids.add(cid)

    if quarantined_ids:
        safe_inputs = [c for c in final_chunks if c.chunk_id not in quarantined_ids]
        dropped = len(final_chunks) - len(safe_inputs)
        if dropped:
            logger.info(
                "ingest_document: skipped %d quarantined chunk(s) before fact extraction doc_id=%s",
                dropped,
                parsed.doc_id,
            )
            warnings.append(
                f"fact extraction: {dropped} quarantined chunk(s) skipped"
            )
        final_chunks = safe_inputs

    extract_chunks = final_chunks
    if config.extract_max_chunks is not None:
        extract_chunks = semantic_extractor.FactExtractor.select_chunks_for_fact_extraction(
            final_chunks,
            max_chunks=int(config.extract_max_chunks),
        )

    if not extract_chunks:
        logger.info(
            "ingest_document: no eligible chunks for fact extraction doc_id=%s",
            parsed.doc_id,
        )
        return [], 0, 0, []

    fact_extractor = semantic_extractor.FactExtractor(llm=runtime.llm)
    extracted_fact_records, llm_batch_failures = await fact_extractor.extract_chunk_facts_batch(
        extract_chunks,
        owner_type=owner_type,
        owner_id=owner_id,
        source_path=parsed.source_path,
        source_hash=parsed.source_hash,
        doc_id=parsed.doc_id,
        min_fact_words=int(config.doc_min_fact_words),
        batch_size_chunks=int(config.fact_extraction_batch_size_chunks),
        max_chars=int(config.fact_extraction_batch_max_chars),
    )
    if llm_batch_failures > 0:
        msg = (
            f"fact extraction: {llm_batch_failures} LLM batch(es) failed for doc_id={parsed.doc_id}; "
            "deterministic fallback was used — consider switching to a more capable LLM"
        )
        warnings.append(msg)
        logger.warning("ingest_document: %s", msg)

    for fact in extracted_fact_records:
        if fact.owner_type != owner_type:
            fact.owner_type = owner_type
        if fact.owner_id != owner_id:
            fact.owner_id = owner_id
        fact.tenant_id = tenant_id
        fact.workspace_id = workspace_id
        fact.trust_score = score_source(SourceDescriptor(kind="document"))
        fact.meta = dict(fact.meta or {})
        fact.meta.setdefault("doc_id", parsed.doc_id)
        fact.meta.setdefault("source_path", parsed.source_path)
        fact.meta.setdefault("source_hash", parsed.source_hash)
        fact.meta.setdefault("fact_text", fact.object)
        fact.meta.setdefault("fact_type", "summary" if fact.predicate == "SUMMARY" else "claim")
        fact.meta["ingest_pipeline_version"] = _INGEST_PIPELINE_VERSION
        fact.meta["extractor_version"] = _EXTRACTOR_VERSION
        fact.meta["chunker_version"] = _CHUNKER_VERSION

    persisted_fact_ids: List[str] = []
    if extracted_fact_records:
        persisted_fact_ids = await _embed_and_persist_facts(
            facts=extracted_fact_records,
            embedder=runtime.embedder,
            semantic_store=runtime.semantic_core,
            warnings=warnings,
            log_context="ingest_document",
        )

    graph_edges = 0
    if runtime.graph_core is not None and extracted_fact_records:
        graph_edges = await runtime.graph_core.add_facts_batch(extracted_fact_records)
    return extracted_fact_records, len(persisted_fact_ids), graph_edges, persisted_fact_ids


async def capture_source(
    file_path: str,
    *,
    owner_type: str | None = None,
    owner_id: str | None = None,
    tenant_id: str | None = None,
    workspace_id: str | None = None,
    config: IngestConfig | None = None,
    memory: Any,
) -> CaptureSourceResult:
    """Stage 1: capture raw input into normalized source records and terminal chunks.

    tenant_id is required for durable artifacts (DAT invariant). It MUST be
    passed explicitly by the caller; falling back to DEFAULT_TENANT_ID silently
    is a tenancy break and is no longer allowed at this layer.
    """
    warnings: List[str] = []
    if memory is None:
        raise ValueError("capture_source: memory is required")

    config = config or IngestConfig()
    runtime = _resolve_ingest_runtime(memory)

    if not owner_type or not owner_id:
        raise ValueError("capture_source: owner_type and owner_id are required")
    if not tenant_id or not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError("capture_source: tenant_id is required")
    owner = validate_explicit_owner(
        tenant_id=tenant_id,
        owner_type=owner_type,
        owner_id=owner_id,
        workspace_id=workspace_id,
    )
    if owner["owner_type"] not in {"agent", "user", "workspace"}:
        raise ValueError("owner_type must be one of: agent, user, workspace")
    tenant_id = str(owner["tenant_id"])
    owner_type = str(owner["owner_type"])
    owner_id = str(owner["owner_id"])
    workspace_id = owner["workspace_id"]

    if not file_path or not isinstance(file_path, str):
        raise ValueError("capture_source: file_path must be a non-empty string")

    # H2 defense: bound on-disk file size and PDF page count before any
    # parser is invoked. enforce_mime_consistency checks size first when
    # max_bytes is positive, so an oversized file never gets its bytes
    # opened beyond the stat() call.
    from uma.ingest.mime_check import enforce_mime_consistency
    enforce_mime_consistency(file_path, max_bytes=config.max_file_bytes)

    parsed = parse_file(file_path, pdf_max_pages=config.pdf_max_pages)
    if not parsed.pages:
        if config.allow_empty_pages:
            warnings.append("document has no extractable pages")
        else:
            raise ValueError("capture_source: document has no extractable pages")

    ingest_signature, existing_manifest, early_report = await _run_manifest_gate(
        parsed=parsed,
        config=config,
        runtime=runtime,
        owner_type=owner_type,
        owner_id=owner_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        warnings=warnings,
    )
    if early_report is not None:
        source_chunks = await _fetch_existing_chunks_for_doc(
            runtime=runtime,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            doc_id=parsed.doc_id,
        )
        return CaptureSourceResult(
            parsed=parsed,
            ingest_signature=ingest_signature,
            captured_chunk_inputs=_chunk_records_to_inputs(source_chunks),
            captured_chunks=source_chunks,
            owner_type=owner_type,
            owner_id=owner_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            warnings=list(warnings),
            skipped=True,
            early_report=early_report,
        )

    source_chunk_inputs, early_report = _prepare_document_chunks(
        parsed=parsed,
        config=config,
        warnings=warnings,
    )
    if early_report is not None:
        return CaptureSourceResult(
            parsed=parsed,
            ingest_signature=ingest_signature,
            captured_chunk_inputs=source_chunk_inputs,
            captured_chunks=[],
            owner_type=owner_type,
            owner_id=owner_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            warnings=list(warnings),
            early_report=early_report,
        )

    source_chunks = await _persist_chunks(
        parsed=parsed,
        final_chunks=source_chunk_inputs,
        config=config,
        runtime=runtime,
        ingest_signature=ingest_signature,
        existing_manifest=existing_manifest,
        owner_type=owner_type,
        owner_id=owner_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        warnings=warnings,
    )
    return CaptureSourceResult(
        parsed=parsed,
        ingest_signature=ingest_signature,
        captured_chunk_inputs=source_chunk_inputs,
        captured_chunks=source_chunks,
        owner_type=owner_type,
        owner_id=owner_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        warnings=list(warnings),
    )


async def derive_memory_artifacts(
    capture: CaptureSourceResult,
    *,
    config: IngestConfig | None = None,
    memory: Any,
) -> DeriveMemoryArtifactsResult:
    """Stage 2: derive facts, graph edges, and episodic artifacts from captured chunks."""
    if memory is None:
        raise ValueError("derive_memory_artifacts: memory is required")
    config = config or IngestConfig()
    runtime = _resolve_ingest_runtime(memory)
    warnings = list(capture.warnings)
    derived_facts, facts_created, graph_edges, fact_ids = await _extract_facts_and_update_graph(
        parsed=capture.parsed,
        final_chunks=capture.captured_chunk_inputs,
        persisted_chunks=capture.captured_chunks,
        config=config,
        runtime=runtime,
        owner_type=capture.owner_type,
        owner_id=capture.owner_id,
        tenant_id=capture.tenant_id,
        workspace_id=capture.workspace_id,
        warnings=warnings,
    )

    if config.doc_episode_enabled:
        await write_document_episode(
            doc_id=capture.parsed.doc_id,
            summary_text=f"Document ingested: {capture.parsed.source_path}",
            owner_type=capture.owner_type,
            owner_id=capture.owner_id,
            tenant_id=capture.tenant_id,
            workspace_id=capture.workspace_id,
            user_id=capture.owner_id if capture.owner_type == "user" else None,
            embedder=runtime.embedder,
            episodic_core=runtime.episodic_core,
        )
    return DeriveMemoryArtifactsResult(
        parsed=capture.parsed,
        captured_chunk_inputs=capture.captured_chunk_inputs,
        captured_chunks=capture.captured_chunks,
        derived_facts=derived_facts,
        facts_created=facts_created,
        graph_edges_created=graph_edges,
        owner_type=capture.owner_type,
        owner_id=capture.owner_id,
        tenant_id=capture.tenant_id,
        workspace_id=capture.workspace_id,
        warnings=warnings,
        fact_ids=fact_ids,
    )


async def curate_compiled_memory(
    capture: CaptureSourceResult,
    derive: DeriveMemoryArtifactsResult,
    *,
    memory: Any,
) -> CurateCompiledMemoryResult:
    """Stage 3: curate compiled-memory/wiki artifacts from derived artifacts and evidence."""
    from uma.memory import wiki as wiki_module

    if memory is None:
        raise ValueError("curate_compiled_memory: memory is required")
    warnings = list(derive.warnings)
    if not capture.captured_chunks:
        return CurateCompiledMemoryResult(
            parsed=capture.parsed,
            compiled_artifacts=[],
            index_entries=[],
            log_events=[],
            owner_type=capture.owner_type,
            owner_id=capture.owner_id,
            tenant_id=capture.tenant_id,
            workspace_id=capture.workspace_id,
            warnings=warnings,
        )

    summary_fact = next((fact for fact in derive.derived_facts if getattr(fact, "predicate", None) == "SUMMARY"), None)
    summary_text = str(getattr(summary_fact, "object", "") or "").strip() if summary_fact is not None else None
    summary_text = summary_text or None
    retrieval_tags = [
        item
        for item in dict.fromkeys(
            [
                capture.parsed.source_path.rsplit("/", 1)[-1],
                capture.parsed.doc_id,
                *[
                    str(getattr(fact, "subject", "") or "").strip()
                    for fact in derive.derived_facts[:5]
                    if getattr(fact, "subject", None)
                ],
            ]
        )
        if item
    ]
    wiki_page = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key=capture.parsed.doc_id,
        title=capture.parsed.source_path.rsplit("/", 1)[-1] or capture.parsed.doc_id,
        owner_type=capture.owner_type,
        owner_id=capture.owner_id,
        summary=summary_text,
        category="ingest",
        direct_source_chunk_ids=[chunk.id for chunk in capture.captured_chunks if getattr(chunk, "id", None)],
        direct_source_document_ids=[capture.parsed.doc_id],
        related_artifact_ids=[fact.id for fact in derive.derived_facts if getattr(fact, "id", None)],
        retrieval_tags=retrieval_tags,
    )
    compiled_artifact = wiki_page["compiled_artifact"]
    return CurateCompiledMemoryResult(
        parsed=capture.parsed,
        compiled_artifacts=[compiled_artifact],
        index_entries=[compiled_artifact["compiled_memory_index"]],
        log_events=list(compiled_artifact.get("compiled_memory_log") or []),
        owner_type=capture.owner_type,
        owner_id=capture.owner_id,
        tenant_id=capture.tenant_id,
        workspace_id=capture.workspace_id,
        warnings=warnings,
    )


def _extract_daily_diary_entries(raw_text: str) -> list[str]:
    """Extract one diary entry per markdown bullet."""
    entries: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("- "):
            continue
        entry = stripped[2:].strip()
        if entry:
            entries.append(entry)
    return entries


def _build_daily_diary_bootstrap_signature(*, raw_text: str) -> dict[str, Any]:
    """Build a stable manifest signature for daily diary bootstrap ingest."""
    return {
        "pipeline_version": "daily_diary_bootstrap_v1",
        "content_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    }


def _build_memory_bootstrap_signature(*, raw_text: str) -> dict[str, Any]:
    """Build a stable manifest signature for MEMORY.md bootstrap ingest."""
    return {
        "pipeline_version": _MEMORY_BOOTSTRAP_VERSION,
        "content_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    }



async def ingest_memory_bootstrap(
    file_path: str,
    *,
    memory: Any,
    runtime_context: Any,
    config: Any | None = None,
) -> Dict[str, Any]:
    """Ingest MEMORY.md with one chunk + one fact per bullet for clean per-entry provenance.

    Bypasses the LLM extraction pipeline (which skips chunks < 300 chars) to ensure
    short MEMORY.md bullets are stored with full chunk-backed fact provenance.

    config is forwarded to the capture step for H2 file-size enforcement. If
    omitted, a default IngestConfig is used.
    """
    resolved_config = config if config is not None else IngestConfig()

    runtime = _resolve_ingest_runtime(memory)
    user_id = runtime_context.user_id
    tenant_id = runtime_context.tenant_id
    workspace_id = runtime_context.workspace_id

    # Pre-flight: missing file, empty file, no entries, idempotent.
    normalized_path, _raw_text, entries, ingest_signature, source_hash, skip = await _capture_bootstrap_source(
        file_path=file_path,
        runtime_context=runtime_context,
        document_store=runtime.document_store,
        api_name="load_memory_bootstrap",
        source_kind="memory_bootstrap",
        signature_builder=_build_memory_bootstrap_signature,
        entry_extractor=_extract_daily_diary_entries,
        config=resolved_config,
    )
    if skip is not None:
        return skip

    # Build one Chunk + one Fact per bullet, bypassing the LLM extraction pipeline.
    doc_id = f"memory-bootstrap:{source_hash[:24]}"
    now = datetime.now(timezone.utc)
    warnings: List[str] = []

    final_chunks: List[DocumentChunk] = []
    chunk_rows: Dict[str, Any] = {}
    # Per-chunk scan verdicts are computed at the boundary (when bullet text
    # enters UMA) and inherited by the derived fact. Option A discipline:
    # scan at the boundary; everything derived inherits.
    chunk_scan_verdicts: Dict[str, tuple[float, Dict[str, Any], "datetime | None"]] = {}

    for i, entry in enumerate(entries):
        chunk_id = hashlib.sha256(
            f"memory_chunk:{user_id}:{source_hash}:{i}".encode("utf-8")
        ).hexdigest()
        text_hash = hashlib.sha256(entry.encode("utf-8")).hexdigest()
        doc_chunk = DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=entry,
            page_range=(1, 1),
            position=i,
        )
        final_chunks.append(doc_chunk)

        chunk_trust = score_source(SourceDescriptor(kind="bootstrap_memory"))
        chunk_meta: Dict[str, Any] = {
            "text_hash": text_hash,
            "source_kind": "memory_bootstrap",
            "chunker_version": _MEMORY_BOOTSTRAP_VERSION,
        }
        chunk_trust, chunk_meta, chunk_quarantined_at = scan_artifact_text(
            entry,
            chunk_trust,
            chunk_meta,
            log_context=f"memory_bootstrap/{user_id}/{i}",
            now=now,
        )
        chunk_scan_verdicts[chunk_id] = (chunk_trust, chunk_meta, chunk_quarantined_at)

        chunk_rows[chunk_id] = Chunk(
            id=chunk_id,
            doc_id=doc_id,
            text=entry,
            page_range=(1, 1),
            position=i,
            source_path=normalized_path,
            source_hash=source_hash,
            created_at=now,
            updated_at=now,
            tenant_id=tenant_id,
            owner_type="user",
            owner_id=user_id,
            workspace_id=workspace_id,
            trust_score=chunk_trust,
            quarantined_at=chunk_quarantined_at,
            meta=chunk_meta,
        )

    created_chunks = await _embed_and_upsert_chunks(
        final_chunks=final_chunks,
        chunk_rows=chunk_rows,
        config=IngestConfig(),
        runtime=runtime,
        warnings=warnings,
    )

    # One fact per entry, citing its chunk for provenance.
    # Facts inherit the scan verdict from their source chunk (Option A): the
    # boundary scan happened at chunk-construction time above; nothing new
    # enters UMA at this step. Inheriting keeps trust scores consistent and
    # ensures a quarantined chunk produces a quarantined fact.
    facts: List[Fact] = []
    for doc_chunk in final_chunks:
        fact_id = "fact_" + hashlib.sha256(
            f"memory_fact:{user_id}:{source_hash}:{doc_chunk.chunk_id}".encode("utf-8")
        ).hexdigest()
        inherited_trust, chunk_meta_for_fact, inherited_quarantined_at = chunk_scan_verdicts[doc_chunk.chunk_id]
        fact_meta: Dict[str, Any] = {
            "doc_id": doc_id,
            "source_path": normalized_path,
            "source_hash": source_hash,
            "fact_text": doc_chunk.text,
            "fact_type": "claim",
            "ingest_pipeline_version": _MEMORY_BOOTSTRAP_VERSION,
        }
        # Propagate the chunk's security record so retrieval and audit can see
        # the same scan verdict on both artifacts.
        chunk_security = (chunk_meta_for_fact or {}).get("security")
        if chunk_security:
            fact_meta["security"] = dict(chunk_security)
        f = Fact(
            id=fact_id,
            subject=user_id,
            predicate="STATES",
            object=doc_chunk.text,
            created_at=now,
            updated_at=now,
            source_ids=[doc_chunk.chunk_id],
            confidence=1.0,
            owner_type="user",
            owner_id=user_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            trust_score=inherited_trust,
            quarantined_at=inherited_quarantined_at,
            content_hash=hash_fact_content(user_id, "STATES", doc_chunk.text),
            meta=fact_meta,
        )
        facts.append(f)

    persisted_fact_ids = await _embed_and_persist_facts(
        facts=facts,
        embedder=runtime.embedder,
        semantic_store=runtime.semantic_core,
        warnings=warnings,
        log_context="ingest_memory_bootstrap",
    )

    # Write manifest only after all persists succeed.
    ingested_at = datetime.now(timezone.utc)
    await _upsert_source_manifest(
        document_store=runtime.document_store,
        doc_id=doc_id,
        source_path=normalized_path,
        source_hash=source_hash,
        ingested_at=ingested_at,
        tenant_id=tenant_id,
        owner_type="user",
        owner_id=user_id,
        workspace_id=workspace_id,
        origin_agent_id=runtime_context.agent_id,
        origin_user_id=runtime_context.user_id,
        origin_session_id=runtime_context.session_id,
        meta=normalize_document_metadata(
            _build_bootstrap_manifest_meta(
                source_kind="memory_bootstrap",
                source_type="memory_bootstrap",
                ingest_signature=ingest_signature,
                extra={"entries_found": len(entries)},
            ),
            doc_id=doc_id,
            owner_type="user",
            owner_id=user_id,
            ingested_at=ingested_at,
            source_path=normalized_path,
            source_hash=source_hash,
        ),
        log_context="load_memory_bootstrap",
    )

    logger.info(
        "ingest_memory_bootstrap: path=%s entries=%d chunks=%d facts=%d",
        normalized_path, len(entries), len(created_chunks), len(persisted_fact_ids),
    )
    return {
        "status": "ingested",
        "path": normalized_path,
        "entries_found": len(entries),
        "facts_created": len(persisted_fact_ids),
        "fact_ids": persisted_fact_ids,
    }


async def ingest_daily_diary_bootstrap(
    file_path: str,
    *,
    memory: Any,
    runtime_context: Any,
    config: Any | None = None,
) -> Dict[str, Any]:
    """Ingest a daily diary file.

    Runs the full document pipeline (chunker + LLM fact extraction) for the
    chunk/fact/wiki lanes, then writes one episode per diary entry for the
    episodic lane. The two manifest gates are independent so each lane is
    idempotent on re-ingest.

    config is forwarded to both the capture step (H2 file-size limit) and
    the document pipeline (chunk sizing, embedding retries, page cap, etc.).
    If omitted, a default IngestConfig is used.
    """
    resolved_config = config if config is not None else IngestConfig()

    runtime = _resolve_ingest_runtime(memory)
    normalized_user_id = runtime_context.user_id
    normalized_tenant_id = runtime_context.tenant_id
    workspace_id = runtime_context.workspace_id

    # Pre-flight: check for missing file, empty file, no entries, and idempotency.
    # This must happen before calling ingest_document to return proper skip shapes.
    normalized_path, _raw_text, entries, ingest_signature, source_hash, skip = await _capture_bootstrap_source(
        file_path=file_path,
        runtime_context=runtime_context,
        document_store=runtime.document_store,
        api_name="load_daily_diary_bootstrap",
        source_kind="daily_diary",
        signature_builder=_build_daily_diary_bootstrap_signature,
        entry_extractor=_extract_daily_diary_entries,
        config=resolved_config,
    )

    if skip is not None:
        return skip

    diary_date = None
    try:
        diary_date = Path(normalized_path).stem
    except Exception:  # nosec B110 — optional field; None is the safe fallback
        pass

    # --- Chunk / fact / wiki lane ---
    ingest_report = await ingest_document(
        normalized_path,
        owner_type="user",
        owner_id=normalized_user_id,
        tenant_id=normalized_tenant_id,
        workspace_id=workspace_id,
        config=resolved_config,
        memory=memory,
    )
    logger.info(
        "ingest_daily_diary_bootstrap: document pipeline path=%s chunks=%d facts=%d",
        normalized_path, ingest_report.chunks_created, ingest_report.facts_created,
    )

    # --- Episodic lane ---
    from uma.ingest.episodic_writer import write_daily_diary_episodes

    logger.info(
        "ingest_daily_diary_bootstrap: episodic lane path=%s entries=%d",
        normalized_path, len(entries),
    )
    episode_ids = await write_daily_diary_episodes(
        file_path=normalized_path,
        diary_date=diary_date,
        entries=entries,
        owner_type="user",
        owner_id=normalized_user_id,
        tenant_id=normalized_tenant_id,
        workspace_id=workspace_id,
        user_id=normalized_user_id,
        embedder=runtime.embedder,
        episodic_core=runtime.episodic_core,
    )

    if episode_ids:
        ingested_at = datetime.now(timezone.utc)
        await _upsert_source_manifest(
            document_store=runtime.document_store,
            doc_id=f"daily-diary:{source_hash}",
            source_path=normalized_path,
            source_hash=source_hash,
            ingested_at=ingested_at,
            tenant_id=normalized_tenant_id,
            owner_type="user",
            owner_id=normalized_user_id,
            workspace_id=workspace_id,
            origin_agent_id=runtime_context.agent_id,
            origin_user_id=runtime_context.user_id,
            origin_session_id=runtime_context.session_id,
            meta=normalize_document_metadata(
                _build_bootstrap_manifest_meta(
                    source_kind="daily_diary",
                    source_type="daily_diary",
                    ingest_signature=ingest_signature,
                    extra={
                        "diary_date": diary_date,
                        "entries_found": len(entries),
                        "episodes_created": len(episode_ids),
                    },
                ),
                doc_id=f"daily-diary:{source_hash}",
                owner_type="user",
                owner_id=normalized_user_id,
                ingested_at=ingested_at,
                source_path=normalized_path,
                source_hash=source_hash,
            ),
            log_context="load_daily_diary_bootstrap",
        )

    return {
        "status": "ingested",
        "path": normalized_path,
        "diary_date": diary_date,
        "entries_found": len(entries),
        "chunks_created": ingest_report.chunks_created,
        "facts_created": ingest_report.facts_created,
        "episodes_created": len(episode_ids),
        "episode_ids": episode_ids,
    }


async def ingest_document(
    file_path: str,
    *,
    owner_type: str | None = None,
    owner_id: str | None = None,
    tenant_id: str | None = None,
    workspace_id: str | None = None,
    config: IngestConfig | None = None,
    memory: Any,
) -> IngestReport:
    """
    End-to-end ingestion of an unstructured document.

    tenant_id is required for durable artifacts (DAT invariant). Callers must
    pass it explicitly; no implicit fallback to the default tenant is provided
    at this layer.

    Returns IngestReport with counts + warnings.
    """
    config = config or IngestConfig()
    capture = await capture_source(
        file_path,
        owner_type=owner_type,
        owner_id=owner_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        config=config,
        memory=memory,
    )
    if capture.early_report is not None and capture.skipped:
        return capture.early_report
    if capture.early_report is not None and not capture.captured_chunks:
        return capture.early_report

    derive = await derive_memory_artifacts(
        capture,
        config=config,
        memory=memory,
    )
    await curate_compiled_memory(
        capture,
        derive,
        memory=memory,
    )

    chunks_failed = len(capture.captured_chunk_inputs) - len(capture.captured_chunks)
    if chunks_failed > 0:
        logger.warning(
            "ingest_document: %d chunk(s) failed to persist doc_id=%s",
            chunks_failed,
            capture.parsed.doc_id,
        )

    return IngestReport(
        doc_id=capture.parsed.doc_id,
        chunks_created=len(capture.captured_chunks),
        chunks_failed=chunks_failed,
        facts_created=derive.facts_created,
        graph_edges_created=derive.graph_edges_created,
        warnings=derive.warnings,
        fact_ids=derive.fact_ids,
    )
