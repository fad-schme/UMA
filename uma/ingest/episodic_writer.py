from __future__ import annotations

import logging
from typing import Any, Optional, Sequence
from datetime import datetime, timezone
from uuid import uuid4

from uma.common.types import Episode
from uma.common.storage_metadata import normalize_episode_metadata
from uma.common.integrity import hash_episode_content
from uma.common.trust import SourceDescriptor, score_source
from uma.common.injection_scan import scan_content, apply_scan, quarantine_enabled, scan_artifact_text

logger = logging.getLogger(__name__)


async def write_document_episode(
    *,
    doc_id: str,
    summary_text: str,
    owner_type: str,
    owner_id: str,
    tenant_id: str,
    workspace_id: str | None,
    user_id: str | None,
    embedder: Any,
    episodic_core: Any | None = None,
) -> Optional[str]:
    """
    Persist a document ingestion episode.

    tenant_id is required (DAT invariant). The Episode dataclass defaults
    tenant_id to "default", but every ingestion
    path MUST stamp the correct value before storage.

    Returns episode_id if written.
    """
    if episodic_core is None:
        logger.warning("write_document_episode: episodic_core missing")
        return None
    if embedder is None:
        logger.warning("write_document_episode: embedder missing")
        return None

    if not doc_id or not summary_text:
        logger.warning("write_document_episode: missing doc_id or summary_text")
        return None
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        logger.error("write_document_episode: tenant_id is required")
        raise ValueError("write_document_episode: tenant_id is required")

    try:
        expected_dim = getattr(embedder, "dimension", None)
        if not isinstance(expected_dim, int) or expected_dim <= 0:
            raise ValueError("write_document_episode: embedder.dimension must be a positive integer")
        vectors = await embedder.embed([summary_text])
        if not vectors or not vectors[0]:
            raise ValueError("empty embedding")
        vec0 = vectors[0]
        if not isinstance(vec0, list) or len(vec0) != expected_dim:
            raise ValueError(
                f"write_document_episode: invalid embedding dim (expected={expected_dim} got={len(vec0) if isinstance(vec0, list) else None})"
            )
        embedding = [float(x) for x in vec0]
    except Exception:
        logger.exception("write_document_episode: embedding failed")
        return None

    episode_id = str(uuid4())
    episode_timestamp = datetime.now(timezone.utc)
    doc_trust = score_source(SourceDescriptor(kind="document"))
    doc_meta = normalize_episode_metadata(
        {"doc_id": doc_id, "source_type": "document_ingest"},
        episode_id=episode_id,
        owner_type=owner_type,
        owner_id=owner_id,
        timestamp=episode_timestamp,
        session_id=None,
    )
    doc_trust, doc_meta, doc_quarantined_at = scan_artifact_text(
        summary_text,
        doc_trust,
        doc_meta,
        log_context=f"episodic_writer/doc:{doc_id}",
        now=episode_timestamp,
    )
    ep = Episode(
        id=episode_id,
        timestamp=episode_timestamp,
        summary=summary_text,
        user_id=str(user_id or owner_id),
        raw=f"Document ingested: {doc_id}",
        tags=["document_ingest"],
        trust_score=doc_trust,
        content_hash=hash_episode_content(summary_text),
        quarantined_at=doc_quarantined_at,
        meta=doc_meta,
        owner_type=owner_type,
        owner_id=owner_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
    )

    try:
        await episodic_core.add_episode(ep, embedding)
        return ep.id
    except Exception:
        logger.exception("write_document_episode: add_episode failed")
        return None


async def write_daily_diary_episodes(
    *,
    file_path: str,
    diary_date: str | None,
    entries: Sequence[str],
    owner_type: str,
    owner_id: str,
    tenant_id: str,
    workspace_id: str | None,
    user_id: str | None,
    embedder: Any,
    episodic_core: Any | None = None,
) -> list[str]:
    """
    Persist one daily diary entry per episode.

    Phase-1 OpenClaw diary bootstrap keeps the logic intentionally simple:
    - each bullet entry becomes one episode
    - headings and file structure are handled upstream
    - the diary file is imported once, then UMA becomes the source of truth

    tenant_id is required (DAT invariant). Episodes are stamped with the
    tenant and workspace at construction time; EpisodicCore.add_episode
    does NOT assign tenant_id, so every caller must pass it explicitly.

    Returns the list of created episode ids.
    """
    if episodic_core is None:
        logger.warning("write_daily_diary_episodes: episodic_core missing")
        return []
    if embedder is None:
        logger.warning("write_daily_diary_episodes: embedder missing")
        return []
    if not isinstance(file_path, str) or not file_path.strip():
        logger.warning("write_daily_diary_episodes: missing file_path")
        return []
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        logger.error("write_daily_diary_episodes: tenant_id is required")
        raise ValueError("write_daily_diary_episodes: tenant_id is required")

    cleaned_entries = [
        entry.strip()
        for entry in entries
        if isinstance(entry, str) and entry.strip()
    ]
    if not cleaned_entries:
        logger.info("write_daily_diary_episodes: no diary entries to persist path=%s", file_path)
        return []

    try:
        expected_dim = getattr(embedder, "dimension", None)
        if not isinstance(expected_dim, int) or expected_dim <= 0:
            raise ValueError("write_daily_diary_episodes: embedder.dimension must be a positive integer")

        vectors = await embedder.embed(cleaned_entries)
        if not isinstance(vectors, list) or len(vectors) != len(cleaned_entries):
            raise ValueError("write_daily_diary_episodes: embedding count mismatch")
    except Exception:
        logger.exception("write_daily_diary_episodes: embedding failed path=%s", file_path)
        return []

    created_episode_ids: list[str] = []

    for entry_text, vector in zip(cleaned_entries, vectors):
        if not isinstance(vector, list) or len(vector) != expected_dim:
            logger.warning(
                "write_daily_diary_episodes: skipping entry with invalid embedding dim path=%s",
                file_path,
            )
            continue

        episode_id = str(uuid4())
        episode_timestamp = datetime.now(timezone.utc)
        diary_trust = score_source(SourceDescriptor(kind="bootstrap_diary"))
        diary_meta = normalize_episode_metadata(
            {
                "source_kind": "daily_diary",
                "source_file": file_path,
                "diary_date": diary_date,
                "import_mode": "bootstrap",
                "source_type": "daily_diary",
            },
            episode_id=episode_id,
            owner_type=owner_type,
            owner_id=owner_id,
            timestamp=episode_timestamp,
            session_id=None,
        )
        diary_trust, diary_meta, diary_quarantined_at = scan_artifact_text(
            entry_text,
            diary_trust,
            diary_meta,
            log_context=f"episodic_writer/diary:{file_path}",
            now=episode_timestamp,
        )
        episode = Episode(
            id=episode_id,
            timestamp=episode_timestamp,
            summary=entry_text,
            user_id=str(user_id or owner_id),
            raw=f"Daily diary import: {file_path}",
            tags=["daily_diary"],
            trust_score=diary_trust,
            content_hash=hash_episode_content(entry_text),
            quarantined_at=diary_quarantined_at,
            meta=diary_meta,
            owner_type=owner_type,
            owner_id=owner_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

        try:
            await episodic_core.add_episode(episode, [float(x) for x in vector])
            created_episode_ids.append(episode.id)
        except Exception:
            logger.exception(
                "write_daily_diary_episodes: add_episode failed path=%s episode_id=%s",
                file_path,
                episode.id,
            )

    return created_episode_ids