"""
episodic/core.py
================

EpisodicCore – orchestrator for the episodic memory subsystem.

Responsibilities
----------------
- Convert working memory + conversation turn into Episode objects
- Persist episodes via EpisodicSQLStore
- Manage lifecycle via EpisodicArchive + EpisodicRetentionPolicy

This version is fully aligned with the UMA Pipeline, EpisodeIndexer,
and EpisodeMapper. It handles:
    user_id
    user_message
    assistant_reply
    working_memory_context (WMEntry objects)
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from .indexer import EpisodeIndexer
from .archive import EpisodicArchive
from .mapper import EpisodeMapper
from .policies import EpisodicRetentionPolicy
from ...types_episode import Episode

logger = logging.getLogger(__name__)


class EpisodicCore:
    """
    High-level episodic memory subsystem.
    """

    def __init__(
        self,
        episodic_store: Any,
        episode_indexer: EpisodeIndexer,
        retention_policy: Optional[EpisodicRetentionPolicy] = None,
    ) -> None:

        self.store = episodic_store
        self.indexer = episode_indexer
        self.mapper = EpisodeMapper()   # ← MUST remain
        self.archive = EpisodicArchive(episodic_store)
        self.policy = retention_policy or EpisodicRetentionPolicy()

        logger.info("EpisodicCore initialized (policy=%s)", type(self.policy).__name__)

    # ------------------------------------------------------------------
    # PUBLIC API — used by MemoryPipeline
    # ------------------------------------------------------------------

    async def store_episode(
        self,
        user_id: str,
        user_message: str,
        assistant_reply: str,
        working_memory_context: List[Any],
    ) -> Optional[Episode]:
        """
        Build and store an Episode from the conversation turn.

        Parameters
        ----------
        user_id : str
        user_message : str
        assistant_reply : str
        working_memory_context : List[WorkingMemoryMessage]

        Returns
        -------
        Episode or None
        """

        try:
            # ------------------------------
            # 1. Build transcript entries
            # ------------------------------
            mapped_wm = self.mapper.map_entries(working_memory_context)

            turn_entries = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_reply},
            ]

            all_entries = mapped_wm + turn_entries

            # ------------------------------
            # 2. Build episode via LLM + embedder
            # ------------------------------
            episode, embedding = await self.indexer.build_episode(
                user_id=user_id,
                wm_entries=all_entries,
            )

            # ------------------------------
            # 3. Store in episodic DB
            # ------------------------------
            await self.store.add_episode(episode, embedding)
            logger.debug("EpisodicCore: stored episode id=%s", episode.id)

            # ------------------------------
            # 4. Apply retention policy
            # ------------------------------
            await self.cleanup(user_id)

            return episode

        except Exception:
            logger.exception("EpisodicCore.store_episode failed for user_id=%s", user_id)
            return None

    # ------------------------------------------------------------------
    # Retention / Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self, user_id: str) -> None:
        """
        Apply retention policy and prune old episodes.
        """
        try:
            episodes = await self.store.list_episodes(user_id)
            prunable = self.policy.select_prunable(episodes)

            if prunable:
                await self.archive.delete_many([ep.id for ep in prunable])
                logger.debug(
                    "EpisodicCore: pruned %d episode(s) for user_id=%s",
                    len(prunable),
                    user_id,
                )

        except Exception:
            logger.exception("EpisodicCore.cleanup failed for user=%s", user_id)

    async def list_recent(self, user_id: str, n: int = 5):
        """
        Return the N most recent episodes for user_id.
        """
        try:
            return await self.store.list_recent(user_id, n)
        except Exception:
            logger.exception("EpisodicCore.list_recent failed.")
            return []