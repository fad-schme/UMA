from __future__ import annotations

import logging
from typing import Any, Optional
from datetime import datetime, timezone
from uuid import uuid4

from ...types_episode import Episode

logger = logging.getLogger(__name__)


async def write_document_episode(
    *,
    doc_id: str,
    summary_text: str,
    owner_type: str,
    owner_id: str,
    user_id: str | None,
    embedder: Any,
    episodic_core: Any | None = None,
) -> Optional[str]:
    """
    Persist a document ingestion episode.

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

    ep = Episode(
        id=str(uuid4()),
        timestamp=datetime.now(timezone.utc),
        summary=summary_text,
        user_id=str(user_id or owner_id),
        raw=f"Document ingested: {doc_id}",
        tags=["document_ingest"],
        meta={"doc_id": doc_id},
        owner_type=owner_type,
        owner_id=owner_id,
    )

    try:
        await episodic_core.add_episode(ep, embedding)
        return ep.id
    except Exception:
        logger.exception("write_document_episode: add_episode failed")
        return None
