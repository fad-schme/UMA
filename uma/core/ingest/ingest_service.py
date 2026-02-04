from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from .types import IngestConfig, IngestReport
from .parser import parse_file
from .normalizer import normalize_document
from .chunker import chunk_sections
from .embedder import embed_chunks
from .semantic_extractor import extract_facts
from .graph_updater import update_graph
from .episodic_writer import write_document_episode
from .consolidation_trigger import maybe_trigger_consolidation

from ...types_fact import Fact
from ...types_chunk import Chunk
from ...stores.document_sql import DocumentRecord
from ..utils.identity import ensure_user_subject

logger = logging.getLogger(__name__)


def _validate_owner(owner_type: str, owner_id: str) -> tuple[str, str]:
    if owner_type not in ("user", "project", "agent"):
        raise ValueError(f"Invalid owner_type={owner_type!r}")
    if not owner_id or not isinstance(owner_id, str):
        raise ValueError("owner_id must be a non-empty string")
    if owner_type == "user":
        return "user", ensure_user_subject(owner_id)
    return owner_type, owner_id


async def ingest_document(
    pdf_path: str,
    *,
    owner_type: str,
    owner_id: str,
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

    # If memory is provided, resolve dependencies from it.
    if memory is not None:
        embedder = embedder or getattr(memory, "embedder", None)
        llm = llm or getattr(memory, "llm", None)
        semantic_core = semantic_core or getattr(memory, "semantic_core", None)
        chunk_core = chunk_core or getattr(memory, "chunk_core", None)
        episodic_core = episodic_core or getattr(memory, "episodic_core", None)
        document_store = document_store or getattr(memory, "document_store", None)
        graph_core = graph_core or getattr(memory, "graph_core", None)

    owner_type, owner_id = _validate_owner(owner_type, owner_id)

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

    # 2) Normalize
    sections = normalize_document(parsed)
    if not sections:
        warnings.append("no sections after normalization")

    # 3) Chunk
    chunks = chunk_sections(
        sections,
        chunk_size_tokens=config.chunk_size_tokens,
        overlap_tokens=config.overlap_tokens,
    )
    if not chunks:
        warnings.append("no chunks created")

    # 3b) Document manifest (authoritative)
    try:
        await document_store.upsert_document(
            DocumentRecord(
                doc_id=parsed.doc_id,
                source_path=parsed.source_path,
                source_hash=parsed.source_hash,
                ingested_at=parsed.extracted_at,
                owner_type=owner_type,
                owner_id=owner_id,
                meta={},
            )
        )
    except Exception:
        warnings.append(f"failed to persist document manifest {parsed.doc_id}")
        logger.exception("ingest_document: document manifest upsert failed")

    # 4) Persist chunks into chunk store (authoritative)
    created_chunks: List[Chunk] = []

    now = datetime.now(timezone.utc)
    chunk_rows: Dict[str, Chunk] = {}
    for chunk in chunks:
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
            owner_type=owner_type,
            owner_id=owner_id,
            meta={
                "source_type": "pdf",
            },
        )
        chunk_rows[chunk.chunk_id] = row

    # Embed + upsert chunk facts (authoritative SQL + vector)
    expected_dim = getattr(embedder, "dimension", None)
    chunk_embeddings = await embed_chunks(
        chunks,
        embedder=embedder,
        batch_size=config.embed_batch_size,
        expected_dim=expected_dim if isinstance(expected_dim, int) and expected_dim > 0 else None,
        max_attempts=config.embed_max_retries,
        initial_delay=config.embed_initial_delay_s,
        backoff_factor=config.embed_backoff_factor,
        max_delay=config.embed_max_delay_s,
    )

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
    extract_chunks = chunks
    if config.extract_max_chunks is not None:
        extract_chunks = chunks[: int(config.extract_max_chunks)]
    extracted = await extract_facts(extract_chunks, llm=llm)

    # 6) Store extracted facts as semantic facts
    extracted_fact_records: List[Fact] = []
    for ef in extracted:
        fact = Fact(
            id=f"fact_{uuid4().hex}",
            subject=ef.subject,
            predicate=ef.predicate,
            object=ef.object,
            created_at=now,
            updated_at=now,
            source_ids=[ef.source_chunk_id],
            confidence=ef.confidence,
            meta={"source_chunk_id": ef.source_chunk_id},
            salience=ef.salience,
            owner_type=owner_type,
            owner_id=owner_id,
        )
        extracted_fact_records.append(fact)

    # Embed + upsert extracted facts using core helper
    facts_created = 0
    if extracted_fact_records:
        try:
            from ..core.utils.user_query_helper import build_fact_embedding_text
        except Exception:
            build_fact_embedding_text = None

        texts = [
            build_fact_embedding_text(f) if build_fact_embedding_text else f"{f.subject} {f.predicate} {f.object}"
            for f in extracted_fact_records
        ]
        try:
            vectors = await embedder.embed(texts)
        except Exception:
            vectors = []
            logger.exception("ingest_document: fact embedding failed")
            warnings.append("failed to embed extracted facts")

        if vectors and isinstance(vectors, list) and len(vectors) == len(extracted_fact_records):
            for fact, vec in zip(extracted_fact_records, vectors):
                try:
                    await semantic_core.upsert_fact(fact, vec)
                    facts_created += 1
                except Exception:
                    logger.exception("ingest_document: failed to upsert extracted fact %s", fact.id)
                    warnings.append(f"failed to persist extracted fact {fact.id}")
        else:
            if extracted_fact_records:
                warnings.append("embedding returned invalid shape for extracted facts")

    # 7) Graph update
    graph_edges = await update_graph(extracted_fact_records, graph_core=graph_core)

    # 8) Episodic entry for document ingest
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
