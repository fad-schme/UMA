from __future__ import annotations

import logging
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from .types import IngestConfig, IngestReport
from .parser import parse_file
from .normalizer import normalize_document
from .chunker import chunk_sections, finalize_chunks
from .embedder import embed_chunks
from ..semantic import extractor as semantic_extractor
from .graph_updater import update_graph
from .episodic_writer import write_document_episode
from .consolidation_trigger import maybe_trigger_consolidation

from ...types import Chunk, Fact, TargetOwner
from ...stores.document_sql import DocumentRecord
from ..utils.ownership import resolve_target_owner

logger = logging.getLogger(__name__)

_INGEST_PIPELINE_VERSION = "doc_ingest_v1"
_EXTRACTOR_VERSION = "doc_fact_extract_v1"
_SPLITTER_VERSION = "doc_normalize_v1"
_CHUNKER_VERSION = "doc_chunk_v2"


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


async def ingest_document(
    pdf_path: str,
    *,
    target_owner: TargetOwner | None = None,
    owner_type: str | None = None,
    owner_id: str | None = None,
    config: IngestConfig | None = None,
    memory: Any | None = None,
    embedder: Any | None = None,
    llm: Any | None = None,
    semantic_core: Any | None = None,
    chunk_core: Any | None = None,
    episodic_core: Any | None = None,
    graph_core: Any | None = None,
    document_store: Any | None = None,
) -> IngestReport:
    """
    End-to-end ingestion of an unstructured document.

    Returns IngestReport with counts + warnings.
    """
    warnings: List[str] = []
    if config is None:
        config = IngestConfig()
        if memory is not None:
            semantic_cfg = None
            try:
                raw_cfg = getattr(memory, "raw_config", None)
                semantic_cfg = raw_cfg.get("semantic") if raw_cfg else None
            except Exception:
                semantic_cfg = None
            if isinstance(semantic_cfg, dict):
                if "doc_min_fact_words" in semantic_cfg:
                    config = IngestConfig(
                        chunk_size_tokens=config.chunk_size_tokens,
                        overlap_tokens=config.overlap_tokens,
                        embed_batch_size=config.embed_batch_size,
                        embed_max_retries=config.embed_max_retries,
                        embed_initial_delay_s=config.embed_initial_delay_s,
                        embed_backoff_factor=config.embed_backoff_factor,
                        embed_max_delay_s=config.embed_max_delay_s,
                        extract_max_chunks=config.extract_max_chunks,
                        allow_empty_pages=config.allow_empty_pages,
                        doc_min_fact_words=int(semantic_cfg.get("doc_min_fact_words", config.doc_min_fact_words)),
                        doc_summary_enabled=bool(semantic_cfg.get("doc_summary_enabled", config.doc_summary_enabled)),
                        doc_summary_max_facts=int(semantic_cfg.get("doc_summary_max_facts", config.doc_summary_max_facts)),
                    )

    # If memory is provided, resolve dependencies from it.
    if memory is not None:
        embedder = embedder or getattr(memory, "embedder", None)
        llm = llm or getattr(memory, "llm", None)
        semantic_core = semantic_core or getattr(memory, "semantic_core", None)
        chunk_core = chunk_core or getattr(memory, "chunk_core", None)
        episodic_core = episodic_core or getattr(memory, "episodic_core", None)
        document_store = document_store or getattr(memory, "document_store", None)
        graph_core = graph_core or getattr(memory, "graph_core", None)

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

    if not pdf_path or not isinstance(pdf_path, str):
        raise ValueError("ingest_document: pdf_path must be a non-empty string")
    if semantic_core is None:
        raise ValueError("ingest_document: semantic_core is required")
    if chunk_core is None:
        raise ValueError("ingest_document: chunk_core is required")
    if episodic_core is None:
        raise ValueError("ingest_document: episodic_core is required")
    if document_store is None:
        raise ValueError("ingest_document: document_store is required")
    if embedder is None or not hasattr(embedder, "embed"):
        raise ValueError("ingest_document: embedder with .embed() required")
    if llm is None or not hasattr(llm, "generate"):
        raise ValueError("ingest_document: llm with .generate() required")

    # 1) Parse
    parsed = parse_file(pdf_path)
    if not parsed.pages:
        if config.allow_empty_pages:
            warnings.append("document has no extractable pages")
        else:
            raise ValueError("ingest_document: document has no extractable pages")

    # --------------------------------------------------------------
    # 1b) Manifest gate: idempotent ingest by (owner_type, owner_id, source_hash)
    # --------------------------------------------------------------
    embedding_cfg = getattr(memory, "embedding_cfg", None) if memory is not None else None
    if memory is not None:
        if embedding_cfg is None:
            raise ValueError("ingest_document: memory.embedding_cfg is required when memory is provided")
        if not getattr(embedding_cfg, "model", None):
            raise ValueError("ingest_document: embedding_cfg.model is required for idempotent ingest signatures")

    ingest_signature = {
        "pipeline_version": _INGEST_PIPELINE_VERSION,
        "splitter_version": _SPLITTER_VERSION,
        "chunker_version": _CHUNKER_VERSION,
        "extractor_version": _EXTRACTOR_VERSION,
        "chunk_size_tokens": config.chunk_size_tokens,
        "overlap_tokens": config.overlap_tokens,
        "embedding_model": embedding_cfg.model if embedding_cfg is not None else None,
        "embedding_dim": getattr(embedder, "dimension", None),
    }

    existing_manifest = None
    try:
        if hasattr(document_store, "get_by_owner_and_hash"):
            existing_manifest = await document_store.get_by_owner_and_hash(
                owner_type=owner_type,
                owner_id=owner_id,
                source_hash=parsed.source_hash,
            )
    except Exception:
        existing_manifest = None
        logger.exception("ingest_document: manifest lookup failed; continuing with ingest")

    if existing_manifest is not None:
        existing_sig = (getattr(existing_manifest, "meta", None) or {}).get("ingest_signature") or {}
        if existing_sig == ingest_signature:
            # Relink/refresh only: update manifest timestamps/path/meta, but do not regenerate artifacts.
            try:
                now_refresh = datetime.now(timezone.utc)
                refreshed_meta = _merge_manifest_meta(
                    existing=getattr(existing_manifest, "meta", None) or {},
                    ingest_signature=ingest_signature,
                    source_path=parsed.source_path,
                    now=now_refresh,
                )
                await document_store.upsert_document(
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
                logger.exception("ingest_document: failed to refresh existing manifest doc_id=%s", existing_manifest.doc_id)
                warnings.append(f"failed to refresh existing manifest {existing_manifest.doc_id}")

            warnings.append(f"skipped ingest (idempotent): owner={owner_type}:{owner_id} hash={parsed.source_hash}")
            return IngestReport(
                doc_id=existing_manifest.doc_id,
                chunks_created=0,
                facts_created=0,
                graph_edges_created=0,
                warnings=warnings,
            )
        else:
            # Manifest exists but signature differs; refresh manifest to reflect the new ingest signature
            # and proceed with re-processing to regenerate derived artifacts.
            try:
                now_refresh = datetime.now(timezone.utc)
                refreshed_meta = _merge_manifest_meta(
                    existing=getattr(existing_manifest, "meta", None) or {},
                    ingest_signature=ingest_signature,
                    source_path=parsed.source_path,
                    now=now_refresh,
                    reingest_reason="signature_changed",
                )
                await document_store.upsert_document(
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
                warnings.append(
                    f"re-ingesting existing manifest doc_id={existing_manifest.doc_id} (signature changed)"
                )
            except Exception:
                logger.exception(
                    "ingest_document: failed to refresh manifest for re-ingest doc_id=%s",
                    existing_manifest.doc_id,
                )
                warnings.append(f"failed to refresh manifest for re-ingest {existing_manifest.doc_id}")

    # 2) Normalize
    sections = normalize_document(parsed)
    if not sections:
        warnings.append("no sections after normalization")

    # 3) Chunk (raw emission; not yet strict-finalized)
    raw_chunks = chunk_sections(
        sections,
        chunk_size_tokens=config.chunk_size_tokens,
        overlap_tokens=config.overlap_tokens,
    )
    logger.info("DOC_CHUNK_RAW count=%d", len(raw_chunks))
    if not raw_chunks:
        warnings.append("no chunks created")

    # 3a) Strict chunker gate (MUST run before any persistence)
    try:
        # finalize_chunks re-ids/repositions deterministically if merges occur
        final_chunks = finalize_chunks(raw_chunks)
    except Exception as exc:
        logger.exception("ingest_document: strict chunk validation failed; refusing persistence")
        raise ValueError(f"ingest_document: strict chunk validation failed: {exc}") from exc
    logger.info("DOC_CHUNK_FINAL count=%d", len(final_chunks))

    if not final_chunks:
        logger.warning("No final_chunks produced; skipping embedding/persistence.")
        return IngestReport(
            doc_id=parsed.doc_id,
            chunks_created=0,
            facts_created=0,
            graph_edges_created=0,
            warnings=warnings + ["no final chunks produced"],
        )

    # 3b) Document manifest (authoritative)
    try:
        now_manifest = datetime.now(timezone.utc)
        existing_meta = (getattr(existing_manifest, "meta", None) or {}) if existing_manifest is not None else {}
        merged_meta = _merge_manifest_meta(
            existing=existing_meta,
            ingest_signature=ingest_signature,
            source_path=parsed.source_path,
            now=now_manifest,
        )
        await document_store.upsert_document(
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

    # 4) Persist chunks into chunk store (authoritative)
    created_chunks: List[Chunk] = []

    now = datetime.now(timezone.utc)
    chunk_rows: Dict[str, Chunk] = {}
    for chunk in final_chunks:
        text_hash = hashlib.sha256((chunk.text or "").encode("utf-8")).hexdigest()
        row = Chunk(
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
            meta={
                "source_type": "pdf",
                "domain": "kb_doc",
                "text_hash": text_hash,
                "chunk_size_tokens": config.chunk_size_tokens,
                "overlap_tokens": config.overlap_tokens,
                "chunker_version": _CHUNKER_VERSION,
                # Paragraph indices are scoped to the originating section/page_range (not doc-global).
                # Any future paragraph-based expansion MUST use (doc_id, page_range, paragraph_index_*) together.
                "paragraph_index_scope": "page_range",
                "paragraph_index_start": chunk.paragraph_index_start,
                "paragraph_index_end": chunk.paragraph_index_end,
            },
        )
        chunk_rows[chunk.chunk_id] = row

    # Embed + upsert chunk facts (authoritative SQL + vector)
    expected_dim = getattr(embedder, "dimension", None)
    logger.info(
        "DOC_CHUNK_EMBED_AND_PERSIST count=%d sample_ids=%s",
        len(final_chunks),
        [c.chunk_id for c in final_chunks[:3]],
    )

    chunk_embeddings = await embed_chunks(
        final_chunks,
        embedder=embedder,
        batch_size=config.embed_batch_size,
        expected_dim=expected_dim if isinstance(expected_dim, int) and expected_dim > 0 else None,
        max_attempts=config.embed_max_retries,
        initial_delay=config.embed_initial_delay_s,
        backoff_factor=config.embed_backoff_factor,
        max_delay=config.embed_max_delay_s,
        strict=True,
    )

    if set(chunk_embeddings.keys()) != {c.chunk_id for c in final_chunks}:
        raise RuntimeError("Embedding id mismatch; refusing to persist inconsistent chunk set.")

    for chunk_id, emb in chunk_embeddings.items():
        row = chunk_rows.get(chunk_id)
        if not row:
            continue
        try:
            await chunk_core.upsert_chunk(row, emb)
            created_chunks.append(row)
        except Exception:
            warnings.append(f"failed to persist chunk {chunk_id}")
            logger.exception("ingest_document: failed to upsert chunk %s", chunk_id)

    # 5) Semantic extraction from chunks (optionally limit chunk count)
    extract_chunks = final_chunks
    if config.extract_max_chunks is not None:
        extract_chunks = semantic_extractor.FactExtractor.select_chunks_for_fact_extraction(
            final_chunks,
            max_chunks=int(config.extract_max_chunks),
        )

    # Core behavior: always use batched semantic extraction for performance and predictable cost.
    # Keep payload bounded to preserve JSON/schema compliance.
    fact_extractor = semantic_extractor.FactExtractor(llm=llm)
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

    # Ensure all extracted facts carry ingest metadata expected downstream.
    for f in extracted_fact_records:
        if f.owner_type != owner_type:
            f.owner_type = owner_type
        if f.owner_id != owner_id:
            f.owner_id = owner_id
        f.tenant_id = tenant_id
        f.workspace_id = workspace_id
        if f.meta is None:
            f.meta = {}
        f.meta.setdefault("domain", "kb_doc")
        f.meta.setdefault("source_type", "pdf")
        f.meta.setdefault("doc_id", parsed.doc_id)
        f.meta.setdefault("source_path", parsed.source_path)
        f.meta.setdefault("source_hash", parsed.source_hash)
        f.meta.setdefault("ingest_pipeline_version", _INGEST_PIPELINE_VERSION)
        f.meta.setdefault("extractor_version", _EXTRACTOR_VERSION)
        f.meta.setdefault("chunker_version", _CHUNKER_VERSION)
        f.meta.setdefault("fact_text", f.object)
        f.meta.setdefault("fact_type", "summary" if f.predicate == "SUMMARY" else "claim")

    # Embed + upsert extracted facts using core helper
    facts_created = 0
    if extracted_fact_records:
        from ..utils.user_query_helper import build_fact_embedding_text
        texts = [build_fact_embedding_text(f) for f in extracted_fact_records]
        try:
            vectors = await embedder.embed(texts)
        except Exception:
            vectors = []
            logger.exception("ingest_document: fact embedding failed")
            warnings.append("failed to embed extracted facts")

        expected_dim = getattr(embedder, "dimension", None)
        if not isinstance(expected_dim, int) or expected_dim <= 0:
            raise ValueError("ingest_document: embedder.dimension must be a positive integer")

        if vectors and isinstance(vectors, list) and len(vectors) == len(extracted_fact_records):
            for fact, vec in zip(extracted_fact_records, vectors):
                if not isinstance(vec, list) or len(vec) != expected_dim:
                    raise ValueError(
                        f"ingest_document: invalid fact embedding dim for fact_id={fact.id} "
                        f"(expected={expected_dim} got={len(vec) if isinstance(vec, list) else None})"
                    )
                try:
                    await semantic_core.upsert_fact(fact, vec)
                    facts_created += 1
                except Exception:
                    logger.exception("ingest_document: failed to upsert extracted fact %s", fact.id)
                    warnings.append(f"failed to persist extracted fact {fact.id}")
        else:
            if extracted_fact_records:
                warnings.append("embedding returned invalid shape for extracted facts")

    # 7) Graph update (derived store; bounded concurrency for latency)
    graph_edges = await update_graph(
        extracted_fact_records,
        graph_core=graph_core,
        concurrency=getattr(config, "graph_update_concurrency", 8),
    )

    # 8) Episodic entry for document ingest (optional)
    if getattr(config, "doc_episode_enabled", True):
        summary_text = f"Document ingested: {parsed.source_path}"
        await write_document_episode(
            doc_id=parsed.doc_id,
            summary_text=summary_text,
            owner_type=owner_type,
            owner_id=owner_id,
            user_id=owner_id if owner_type == "user" else None,
            embedder=embedder,
            episodic_core=episodic_core,
        )

    # 9) Optional consolidation trigger
    await maybe_trigger_consolidation(
        memory=memory,
        user_id=owner_id if owner_type == "user" else None,
        enabled=False,
    )

    return IngestReport(
        doc_id=parsed.doc_id,
        chunks_created=len(created_chunks),
        facts_created=facts_created,
        graph_edges_created=graph_edges,
        warnings=warnings,
    )
