"""
episodic/archive.py
===================

EpisodicArchive
---------------

Manages episode deletion, TTL expiration, bulk archival, and cleanup.

Coding Agent Instructions
-------------------------
- Never delete episodes unless instructed by a policy.
- Keep this module side-effect free except for DB operations.
- All operations MUST be fully logged and MUST NOT crash callers.
"""

from __future__ import annotations
import logging
from typing import Any, List

logger = logging.getLogger(__name__)


class EpisodicArchive:
    """
    Episodic memory archival + deletion service.
    """

    def __init__(self, episodic_store: Any):
        self.store = episodic_store
        logger.debug("EpisodicArchive initialized.")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def delete_many(
        self,
        episode_ids: List[str],
        *,
        owner_type: str,
        owner_id: str,
    ):
        """
        Bulk-delete episodes from the store.

        Parameters
        ----------
        episode_ids : List[str]
        """
        for eid in episode_ids:
            try:
                await self.store.delete_episode(
                    eid,
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
            except Exception:
                logger.exception(
                    "Failed to delete episode id=%s owner=%s:%s",
                    eid,
                    owner_type,
                    owner_id,
                )
