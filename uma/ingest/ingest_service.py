from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple

from .types import DocumentChunk, IngestConfig, IngestReport, ParsedDocument
from .parser import parse_file
from .normalizer import normalize_document
from .chunker import chunk_sections, finalize_chunks
from .embedder import embed_chunks
from uma.memory.semantic import extractor as semantic_extractor
from .graph_updater import update_graph
from .episodic_writer import write_document_episode
from .consolidation_trigger import maybe_trigger_consolidation

from uma.common.types import Chunk, Fact, TargetOwner
from uma.stores.document_sql import DocumentRecord
from uma.common.ownership import resolve_target_owner
from uma.common.storage_metadata import normalize_chunk_metadata, normalize_fact_metadata

logger = logging.getLogger(__name__)

_INGEST_PIPELINE_VERSION = "doc_ingest_v1"
_EXTRACTOR_VERSION = "doc_fact_extract_v1"
_SPLITTER_VERSION = "doc_normalize_v1"
_CHUNKER_VERSION = "doc_chunk_v2"


class _IngestRuntime(NamedTuple):
    embedder: Any
    llm: Any
    semantic_core: Any
    chunk_core: Any
    episodic_core: Any
    graph_core: Any
    document_store: Any
    embedding_cfg: Any


def _coerce_ingest_target_owner(owner_type: str, owner_id: str) -> TargetOwner:
    return resolve_target_owner(
        owner_type=owner_type,
        owner_id=owner_id,
        allowed_owner_types=("agent", "user", "workspace"),
    )


def _merge_manifest_meta(
    *,
    existing: dict | None,
    ingest_signature: dict,
    source_path: str,
    now: datetime,
    reingest_reason: str | None = None,
) -> dict:
    meta = dict(existing or {})
    meta.setdefault("created_by", _INGEST_PIPELINE_VERSION)
    meta.setdefault("first_seen_at", now.isoformat())

    meta["last_seen_at"] = now.isoformat()
    meta["source_path"] = source_path

    prior_sig = meta.get("ingest_signature")
    if prior_sig and prior_sig != ingest_signature:
        meta["prior_ingest_signature"] = prior_sig

    meta["ingest_signature"] = ingest_signature

    if reingest_reason:
        meta["reingest_reason"] = reingest_reason

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


def _resolve_ingest_config(config: IngestConfig | None) -> IngestConfig:
    return config or IngestConfig()


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


async def _run_manifest_gate(
    *,
    file_path: str,
    config: IngestConfig,
    runtime: _IngestRuntime,
    owner_type: str,
    owner_id: str,
    tenant_id: str,
    workspace_id: str | None,
    warnings: List[str],
) -> tuple[ParsedDocument, dict, Any | None, IngestReport | None]:
    parsed = parse_file(file_path)
    if not parsed.pages:
        if config.allow_empty_pages:
            warnings.append("document has no extractable pages")
        else:
            raise ValueError("ingest_document: document has no extractable pages")

    ingest_signature = _build_ingest_signature(config, runtime)
    existing_manifest = None
    try:
        if hasattr(runtime.document_store, "get_by_owner_and_hash"):
            existing_manifest = await runtime.document_store.get_by_owner_and_hash(
                owner_type=owner_type,
                owner_id=owner_id,
                source_hash=parsed.source_hash,
            )
    except Exception:
        logger.exception("ingest_document: manifest lookup failed; continuing with ingest")

    if existing_manifest is None:
        return parsed, ingest_signature, None, None

    existing_sig = (getattr(existing_manifest, "meta", None) or {}).get("ingest_signature") or {}
    if existing_sig == ingest_signature:
        try:
            now_refresh = datetime.now(timezone.utc)
            refreshed_meta = _merge_manifest_meta(
                existing=getattr(existing_manifest, "meta", None) or {},
                ingest_signature=ingest_signature,
                source_path=parsed.source_path,
                now=now_refresh,
            )
            await runtime.document_store.upsert_document(
                DocumentRecord(
                    doc_id=existing_manifest.doc_id,
                    source_path=parsed.source_path,
                    source_hash=parsed.source_hash,
                    ingested_at=now_refresh,
                    tenant_id=tenant_id,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    workspace_id=workspace_id,
                    meta=refreshed_meta,
                )
            )
        except Exception:
            logger.exception(
                "ingest_document: failed to refresh existing manifest doc_id=%s",
                existing_manifest.doc_id,
            )
            warnings.append(f"failed to refresh existing manifest {existing_manifest.doc_id}")

        warnings.append(f"skipped ingest (idempotent): owner={owner_type}:{owner_id} hash={parsed.source_hash}")
        return (
            parsed,
            ingest_signature,
            existing_manifest,
            IngestReport(
                doc_id=existing_manifest.doc_id,
                chunks_created=0,
                facts_created=0,
                graph_edges_created=0,
                warnings=warnings,
            ),
        )

    try:
        now_refresh = datetime.now(timezone.utc)
        refreshed_meta = _merge_manifest_meta(
            existing=getattr(existing_manifest, "meta", None) or {},
            ingest_signature=ingest_signature,
            source_path=parsed.source_path,
            now=now_refresh,
            reingest_reason="signature_changed",
        )
        await runtime.document_store.upsert_document(
            DocumentRecord(
                doc_id=existing_manifest.doc_id,
                source_path=parsed.source_path,
                source_hash=parsed.source_hash,
                ingested_at=now_refresh,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                workspace_id=workspace_id,
                meta=refreshed_meta,
            )
        )
        warnings.append(f"re-ingesting existing manifest doc_id={existing_manifest.doc_id} (signature changed)")
    except Exception:
        logger.exception(
            "ingest_document: failed to refresh manifest for re-ingest doc_id=%s",
            existing_manifest.doc_id,
        )
        warnings.append(f"failed to refresh manifest for re-ingest {existing_manifest.doc_id}")

    return parsed, ingest_signature, existing_manifest, None


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
    try:
        merged_meta = _merge_manifest_meta(
            existing=(getattr(existing_manifest, "meta", None) or {}) if existing_manifest is not None else {},
            ingest_signature=ingest_signature,
            source_path=parsed.source_path,
            now=datetime.now(timezone.utc),
        )
        await runtime.document_store.upsert_document(
            DocumentRecord(
                doc_id=parsed.doc_id,
                source_path=parsed.source_path,
                source_hash=parsed.source_hash,
                ingested_at=parsed.extracted_at,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                workspace_id=workspace_id,
                meta=merged_meta,
            )
        )
    except Exception:
        warnings.append(f"failed to persist document manifest {parsed.doc_id}")
        logger.exception("ingest_document: document manifest upsert failed")

    created_chunks: List[Chunk] = []
    chunk_rows: Dict[str, Chunk] = {}
    now = datetime.now(timezone.utc)
    for chunk in final_chunks:
        text_hash = hashlib.sha256((chunk.text or "").encode("utf-8")).hexdigest()
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
            meta=normalize_chunk_metadata(
                {
                    "text_hash": text_hash,
                    "chunk_size_tokens": config.chunk_size_tokens,
                    "overlap_tokens": config.overlap_tokens,
                    "chunker_version": _CHUNKER_VERSION,
                    "paragraph_index_scope": "page_range",
                    "paragraph_index_start": chunk.paragraph_index_start,
                    "paragraph_index_end": chunk.paragraph_index_end,
                },
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                owner_type=owner_type,
                owner_id=owner_id,
                created_at=now,
                updated_at=now,
                page_range=chunk.page_range,
                position=chunk.position,
                source_path=parsed.source_path,
                source_hash=parsed.source_hash,
            ),
        )

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
    config: IngestConfig,
    runtime: _IngestRuntime,
    owner_type: str,
    owner_id: str,
    tenant_id: str,
    workspace_id: str | None,
    warnings: List[str],
) -> tuple[int, int]:
    extract_chunks = final_chunks
    if config.extract_max_chunks is not None:
        extract_chunks = semantic_extractor.FactExtractor.select_chunks_for_fact_extraction(
            final_chunks,
            max_chunks=int(config.extract_max_chunks),
        )

    fact_extractor = semantic_extractor.FactExtractor(llm=runtime.llm)
    extracted_fact_records: List[Fact] = await fact_extractor.extract_chunk_facts_batch(
        extract_chunks,
        owner_type=owner_type,
        owner_id=owner_id,
        source_path=parsed.source_path,
        source_hash=parsed.source_hash,
        doc_id=parsed.doc_id,
        min_fact_words=int(config.doc_min_fact_words),
        batch_size_chunks=4,
        max_chars=12000,
    )

    for fact in extracted_fact_records:
        if fact.owner_type != owner_type:
            fact.owner_type = owner_type
        if fact.owner_id != owner_id:
            fact.owner_id = owner_id
        fact.tenant_id = tenant_id
        fact.workspace_id = workspace_id
        if fact.meta is None:
            fact.meta = {}
        fact.meta = normalize_fact_metadata(
            {
                **fact.meta,
                "doc_id": parsed.doc_id,
                "source_path": parsed.source_path,
                "source_hash": parsed.source_hash,
                "ingest_pipeline_version": _INGEST_PIPELINE_VERSION,
                "extractor_version": _EXTRACTOR_VERSION,
                "chunker_version": _CHUNKER_VERSION,
                "fact_text": fact.object,
                "fact_type": "summary" if fact.predicate == "SUMMARY" else "claim",
            },
            fact_id=fact.id,
            owner_type=owner_type,
            owner_id=owner_id,
            created_at=fact.created_at,
            updated_at=fact.updated_at,
            source_ids=list(fact.source_ids or []),
            session_id=getattr(fact, "session_id", None),
        )

    facts_created = 0
    if extracted_fact_records:
        from uma.retrieve.user_query_helper import build_fact_embedding_text

        texts = [build_fact_embedding_text(fact) for fact in extracted_fact_records]
        try:
            vectors = await runtime.embedder.embed(texts)
        except Exception:
            vectors = []
            logger.exception("ingest_document: fact embedding failed")
            warnings.append("failed to embed extracted facts")

        expected_dim = getattr(runtime.embedder, "dimension", None)
        if not isinstance(expected_dim, int) or expected_dim <= 0:
            raise ValueError("ingest_document: embedder.dimension must be a positive integer")

        if vectors and isinstance(vectors, list) and len(vectors) == len(extracted_fact_records):
            for fact, vector in zip(extracted_fact_records, vectors):
                if not isinstance(vector, list) or len(vector) != expected_dim:
                    raise ValueError(
                        f"ingest_document: invalid fact embedding dim for fact_id={fact.id} "
                        f"(expected={expected_dim} got={len(vector) if isinstance(vector, list) else None})"
                    )
                try:
                    await runtime.semantic_core.upsert_fact(fact, vector)
                    facts_created += 1
                except Exception:
                    logger.exception("ingest_document: failed to upsert extracted fact %s", fact.id)
                    warnings.append(f"failed to persist extracted fact {fact.id}")
        else:
            warnings.append("embedding returned invalid shape for extracted facts")

    graph_edges = await update_graph(
        extracted_fact_records,
        graph_core=runtime.graph_core,
        concurrency=getattr(config, "graph_update_concurrency", 8),
    )
    return facts_created, graph_edges


async def _run_post_ingest_hooks(
    *,
    parsed: ParsedDocument,
    config: IngestConfig,
    runtime: _IngestRuntime,
    memory: Any,
    owner_type: str,
    owner_id: str,
) -> None:
    if getattr(config, "doc_episode_enabled", True):
        await write_document_episode(
            doc_id=parsed.doc_id,
            summary_text=f"Document ingested: {parsed.source_path}",
            owner_type=owner_type,
            owner_id=owner_id,
            user_id=owner_id if owner_type == "user" else None,
            embedder=runtime.embedder,
            episodic_core=runtime.episodic_core,
        )

    await maybe_trigger_consolidation(
        memory=memory,
        user_id=owner_id if owner_type == "user" else None,
        enabled=False,
    )


async def ingest_document(
    file_path: str,
    *,
    target_owner: TargetOwner | None = None,
    owner_type: str | None = None,
    owner_id: str | None = None,
    config: IngestConfig | None = None,
    memory: Any,
) -> IngestReport:
    """
    End-to-end ingestion of an unstructured document.

    Returns IngestReport with counts + warnings.
    """
    warnings: List[str] = []
    if memory is None:
        raise ValueError("ingest_document: memory is required")

    config = _resolve_ingest_config(config)
    runtime = _resolve_ingest_runtime(memory)

    if target_owner is None:
        if not owner_type or not owner_id:
            raise ValueError("ingest_document: target_owner or owner_type/owner_id is required")
        target_owner = _coerce_ingest_target_owner(owner_type, owner_id)
    else:
        target_owner = resolve_target_owner(
            target_owner=target_owner,
            allowed_owner_types=("agent", "user", "workspace"),
        )
    owner_type = target_owner.owner_type
    owner_id = target_owner.owner_id
    tenant_id = target_owner.tenant_id
    workspace_id = target_owner.workspace_id

    if not file_path or not isinstance(file_path, str):
        raise ValueError("ingest_document: file_path must be a non-empty string")

    parsed, ingest_signature, existing_manifest, early_report = await _run_manifest_gate(
        file_path=file_path,
        config=config,
        runtime=runtime,
        owner_type=owner_type,
        owner_id=owner_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        warnings=warnings,
    )
    if early_report is not None:
        return early_report

    final_chunks, early_report = _prepare_document_chunks(
        parsed=parsed,
        config=config,
        warnings=warnings,
    )
    if early_report is not None:
        return early_report

    created_chunks = await _persist_chunks(
        parsed=parsed,
        final_chunks=final_chunks,
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

    facts_created, graph_edges = await _extract_facts_and_update_graph(
        parsed=parsed,
        final_chunks=final_chunks,
        config=config,
        runtime=runtime,
        owner_type=owner_type,
        owner_id=owner_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        warnings=warnings,
    )

    await _run_post_ingest_hooks(
        parsed=parsed,
        config=config,
        runtime=runtime,
        memory=memory,
        owner_type=owner_type,
        owner_id=owner_id,
    )

    return IngestReport(
        doc_id=parsed.doc_id,
        chunks_created=len(created_chunks),
        facts_created=facts_created,
        graph_edges_created=graph_edges,
        warnings=warnings,
    )
