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
from ..utils.dedupe import dedupe_by_id

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
        owner_type: str,
        owner_id: str,
        user_message: str,
        assistant_reply: str,
        working_memory_context: List[Any],
    ) -> Optional[Episode]:
        """
        Build and store an Episode from the conversation turn.

        Parameters
        ----------
        owner_type : str
        owner_id : str
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
                owner_type=owner_type,
                owner_id=owner_id,
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
            await self.cleanup(owner_type, owner_id)

            return episode

        except Exception:
            logger.exception(
                "EpisodicCore.store_episode failed for owner=%s:%s",
                owner_type,
                owner_id,
            )
            return None

    async def add_episode(self, episode: Episode, embedding: List[float]) -> bool:
        """
        Persist a pre-built Episode + embedding.
        """
        if self.store is None:
            return False
        try:
            await self.store.add_episode(episode, embedding)
            return True
        except Exception:
            logger.exception("EpisodicCore.add_episode failed id=%s", getattr(episode, "id", None))
            return False

    # ------------------------------------------------------------------
    # Retention / Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self, owner_type: str, owner_id: str) -> None:
        """
        Apply retention policy and prune old episodes.
        """
        try:
            episodes = await self.store.list_episodes(owner_type, owner_id)
            prunable = self.policy.select_prunable(episodes)

            if prunable:
                await self.archive.delete_many([ep.id for ep in prunable])
                logger.debug(
                    "EpisodicCore: pruned %d episode(s) for owner=%s:%s",
                    len(prunable),
                    owner_type,
                    owner_id,
                )

        except Exception:
            logger.exception(
                "EpisodicCore.cleanup failed for owner=%s:%s",
                owner_type,
                owner_id,
            )

    
    async def list_recent(self, owner_type: str, owner_id: str, n: int = 5):
        """
        Return the N most recent episodes for an owner.
        """
        try:
            return await self.store.list_recent(owner_type, owner_id, n)
        except Exception:
            logger.exception("EpisodicCore.list_recent failed.")
            return []

    async def delete_episode(self, episode_id: str) -> bool:
        if self.store is None or not episode_id:
            return False
        try:
            await self.store.delete_episode(episode_id)
            return True
        except Exception:
            logger.exception("EpisodicCore.delete_episode failed id=%s", episode_id)
            return False

    async def list_episodes(self, owner_type: str, owner_id: str) -> List[Episode]:
        if self.store is None:
            return []
        try:
            return await self.store.list_episodes(owner_type, owner_id)
        except Exception:
            logger.exception("EpisodicCore.list_episodes failed.")
            return []

    async def upsert_cluster_summary(
        self,
        user_id: str,
        episode_ids: List[str],
        summary: str,
        latest_timestamp: str,
    ) -> bool:
        if self.store is None or not hasattr(self.store, "upsert_cluster_summary"):
            return False
        try:
            await self.store.upsert_cluster_summary(
                user_id=user_id,
                episode_ids=episode_ids,
                summary=summary,
                latest_timestamp=latest_timestamp,
            )
            return True
        except Exception:
            logger.exception("EpisodicCore.upsert_cluster_summary failed user=%s", user_id)
            return False

    def vector_index(self) -> Any:
        return getattr(self.store, "vector_index", None) if self.store is not None else None
        

    async def list_cluster_summaries(
        self,
        owner_type: str,
        owner_id: str,
        k: int = 5,
        max_episodes: int = 50,
        time_range: Optional[dict] = None,
    ):
        """
        Return episodic cluster summaries for a user.

        NOTE:
        - Clustering logic is owned by the EpisodicSQLStore.
        - EpisodicCore acts as the public façade for higher layers (pipeline / RLM).
        """
        try:
            if not hasattr(self.store, "list_cluster_summaries"):
                logger.warning(
                    "EpisodicCore.list_cluster_summaries: store does not support clustering"
                )
                return []

            return await self.store.list_cluster_summaries(
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(k),
                max_episodes=max_episodes,
                time_range=time_range,
            )
        except Exception:
            logger.exception(
                "EpisodicCore.list_cluster_summaries failed for owner=%s:%s",
                owner_type,
                owner_id,
            )
            return []

    # ------------------------------------------------------------------
    # RETRIEVAL API
    # ------------------------------------------------------------------

    async def search(
        self,
        user_id: str,
        query_embedding: List[float],
        *,
        owner_type: str,
        owner_id: str,
        k: int = 20,
        offset: int = 0,
    ) -> List[Episode]:
        if self.store is None:
            return []
        try:
            user_subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("EpisodicCore.search: invalid subject=%r", user_id)
            return []

        episodes: List[Episode] = []
        try:
            try:
                found = await self.store.search(
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(k),
                    offset=int(offset),
                )
            except TypeError:
                found = await self.store.search(
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(k),
                )
            if found:
                episodes.extend(found)
        except Exception:
            logger.exception(
                "EpisodicCore.search failed owner=%s:%s",
                owner_type,
                owner_id,
            )
        return dedupe_by_id(episodes)

    async def list_cluster_summaries(
        self,
        user_id: str,
        *,
        owner_type: str,
        owner_id: str,
        k: int = 5,
        max_episodes: int = 50,
        time_range: Optional[dict] = None,
    ) -> List[Any]:
        if self.store is None or not hasattr(self.store, "list_cluster_summaries"):
            logger.warning(
                "EpisodicCore.list_cluster_summaries: store does not support clustering"
            )
            return []
        try:
            user_subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("EpisodicCore.list_cluster_summaries: invalid subject=%r", user_id)
            return []

        clusters: List[Any] = []
        try:
            found = await self.store.list_cluster_summaries(
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(k),
                max_episodes=max_episodes,
                time_range=time_range,
            )
            if found:
                clusters.extend(found)
        except Exception:
            logger.exception(
                "EpisodicCore.list_cluster_summaries failed owner=%s:%s",
                owner_type,
                owner_id,
            )
        return dedupe_by_id(clusters)

    async def search(
        self,
        owner_type: str,
        owner_id: str,
        query_embedding: List[float],
        *,
        k: int = 20,
        offset: int = 0,
    ) -> List[Episode]:
        if self.store is None:
            return []
        try:
            try:
                return await self.store.search(
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(k),
                    offset=int(offset),
                )
            except TypeError:
                return await self.store.search(
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(k),
                )
        except Exception:
            logger.exception(
                "EpisodicCore.search failed owner=%s:%s",
                owner_type,
                owner_id,
            )
            return []

    async def fetch_summaries(self, ids: List[str]) -> List[dict]:
        if self.store is None or not hasattr(self.store, "fetch_summaries"):
            return []
        try:
            return await self.store.fetch_summaries(ids)
        except Exception:
            logger.exception("EpisodicCore.fetch_summaries failed")
            return []

    async def fetch_transcripts(self, ids: List[str]) -> List[dict]:
        if self.store is None or not hasattr(self.store, "fetch_transcripts"):
            return []
        try:
            return await self.store.fetch_transcripts(ids)
        except Exception:
            logger.exception("EpisodicCore.fetch_transcripts failed")
            return []
