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
    episodic_store: Any,
    embedder: Any,
) -> Optional[str]:
    """
    Persist a document ingestion episode.

    Returns episode_id if written.
    """
    if episodic_store is None or embedder is None:
        logger.warning("write_document_episode: episodic_store or embedder missing")
        return None

    if not doc_id or not summary_text:
        logger.warning("write_document_episode: missing doc_id or summary_text")
        return None

    uid = user_id or owner_id

    try:
        vectors = await embedder.embed([summary_text])
        if not vectors or not vectors[0]:
            raise ValueError("empty embedding")
        embedding = [float(x) for x in vectors[0]]
    except Exception:
        logger.exception("write_document_episode: embedding failed")
        return None

    ep = Episode(
        id=str(uuid4()),
        user_id=str(uid),
        timestamp=datetime.now(timezone.utc),
        summary=summary_text,
        raw=f"Document ingested: {doc_id}",
        tags=["document_ingest"],
        meta={"doc_id": doc_id},
        owner_type=owner_type,
        owner_id=owner_id,
    )

    try:
        await episodic_store.add_episode(ep, embedding)
        return ep.id
    except Exception:
        logger.exception("write_document_episode: add_episode failed")
        return None
