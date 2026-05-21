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

Note for maintainers:
- Episodic retrieval should go through `search`.
- Clustering access is centralized in `list_cluster_summaries`.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from .indexer import EpisodeIndexer
from .archive import EpisodicArchive
from .mapper import EpisodeMapper
from .policies import EpisodicRetentionPolicy
from uma.common.types import Episode
from uma.common.types import RuntimeContext, SCOPE_MODEL_VERSION
from uma.common.dedupe import dedupe_by_id
from uma.common.integrity import hash_episode_content
from uma.common.trust import SourceDescriptor, score_source
from uma.common.injection_scan import scan_content, apply_scan, quarantine_enabled

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
        """
        Initialize the episodic core with a store, indexer, and retention policy.
        """

        self.store = episodic_store
        self.indexer = episode_indexer
        self.mapper = EpisodeMapper()   # ← MUST remain
        self.archive = EpisodicArchive(episodic_store)
        self.policy = retention_policy or EpisodicRetentionPolicy()

        logger.debug("EpisodicCore initialized (policy=%s)", type(self.policy).__name__)

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
        turn_context: RuntimeContext,
    ) -> Optional[Episode]:
        """
        Build and persist an Episode from a conversation turn.
        This is the main ingestion entry point used by the memory pipeline.
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
            episode.tenant_id = turn_context.tenant_id
            episode.workspace_id = turn_context.workspace_id
            episode.session_id = turn_context.session_id
            episode.origin_agent_id = turn_context.agent_id
            episode.origin_user_id = turn_context.user_id
            episode.origin_session_id = turn_context.session_id
            episode.scope_model_version = SCOPE_MODEL_VERSION
            episode.trust_score = score_source(SourceDescriptor(kind="turn_assistant", session_id=turn_context.session_id))
            episode.content_hash = hash_episode_content(episode.summary)

            # Scan the raw assistant_reply (the actual input being stored, not the LLM-summarized output).
            scan_result = scan_content(assistant_reply or "")
            episode.trust_score, episode.meta = apply_scan(
                episode.trust_score,
                episode.meta or {},
                scan_result,
                log_context=f"episodic/{owner_type}:{owner_id}",
            )
            if scan_result.severity == "high" and quarantine_enabled():
                from datetime import datetime as _dt, timezone
                episode.quarantined_at = _dt.now(timezone.utc)

            # ------------------------------
            # 3. Store in episodic DB
            # ------------------------------
            await self.store.add_episode(episode, embedding)
            logger.debug("EpisodicCore: stored episode id=%s", episode.id)

            # ------------------------------
            # 4. Apply retention policy
            # ------------------------------
            await self.cleanup(turn_context.tenant_id, owner_type, owner_id)

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
        Persist a pre-built Episode + embedding (bypasses indexing pipeline).
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

    async def cleanup(self, tenant_id: str = DEFAULT_TENANT_ID, owner_type: str = "", owner_id: str = "") -> None:
        """
        Apply retention policy and prune old episodes for the owner scope.
        """
        try:
            episodes = await self.store.list_episodes(tenant_id, owner_type, owner_id)
            prunable = self.policy.select_prunable(episodes)

            if prunable:
                await self.archive.delete_many(
                    [ep.id for ep in prunable],
                    tenant_id=tenant_id,
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
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

    
    async def list_recent(self, tenant_id: str = DEFAULT_TENANT_ID, owner_type: str = "", owner_id: str = "", n: int = 5):
        """
        Return the N most recent episodes for an owner scope.
        """
        try:
            return await self.store.list_recent(tenant_id, owner_type, owner_id, n)
        except Exception:
            logger.exception("EpisodicCore.list_recent failed.")
            return []

    async def delete_episode(
        self,
        episode_id: str,
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
    ) -> bool:
        """
        Delete a single episode by ID.
        """
        if self.store is None or not episode_id:
            return False
        try:
            await self.store.delete_episode(
                episode_id,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
            return True
        except Exception:
            logger.exception("EpisodicCore.delete_episode failed id=%s", episode_id)
            return False

    async def list_episodes(self, tenant_id: str = DEFAULT_TENANT_ID, owner_type: str = "", owner_id: str = "") -> List[Episode]:
        """
        Return all episodes for the given owner scope.
        """
        if self.store is None:
            return []
        try:
            return await self.store.list_episodes(tenant_id, owner_type, owner_id)
        except Exception:
            logger.exception("EpisodicCore.list_episodes failed.")
            return []

    async def upsert_cluster_summary(
        self,
        user_id: str,
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        episode_ids: List[str],
        summary: str,
        latest_timestamp: str,
    ) -> bool:
        """
        Upsert a cluster summary record into the episodic store.
        """
        if self.store is None or not hasattr(self.store, "upsert_cluster_summary"):
            return False
        try:
            await self.store.upsert_cluster_summary(
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
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
        """
        Expose the backing vector index (if present) for diagnostics.
        """
        return getattr(self.store, "vector_index", None) if self.store is not None else None
        

    # ------------------------------------------------------------------
    # RETRIEVAL API
    # ------------------------------------------------------------------

    async def search(
        self,
        user_id: str,
        query_embedding: List[float],
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        k: int = 20,
        offset: int = 0,
    ) -> List[Episode]:
        """
        Vector search over episodic summaries for the given owner scope.
        Applies subject validation and deduplicates results.
        """
        if self.store is None:
            return []
        # try:
        #     normalize_user_id(user_id)
        # except Exception:
        #     logger.exception("EpisodicCore.search: invalid subject=%r", user_id)
        #     return []
        if not owner_type or not owner_id:
            logger.error("EpisodicCore.search requires owner_type and owner_id")
            return []

        episodes: List[Episode] = []
        try:
            found = await self.store.search(
                query_embedding=query_embedding,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(k),
                offset=int(offset),
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
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        k: int = 5,
        max_episodes: int = 50,
        time_range: Optional[dict] = None,
    ) -> List[Any]:
        """
        Return episodic cluster summaries for a user and owner scope.
        Clustering logic is store-owned; core provides a safe façade.
        """
        if self.store is None or not hasattr(self.store, "list_cluster_summaries"):
            logger.warning(
                "EpisodicCore.list_cluster_summaries: store does not support clustering"
            )
            return []
        # try:
        #     normalize_user_id(user_id)
        # except Exception:
        #     logger.exception("EpisodicCore.list_cluster_summaries: invalid subject=%r", user_id)
        #     return []
        if not owner_type or not owner_id:
            logger.error("EpisodicCore.list_cluster_summaries requires owner_type and owner_id")
            return []

        clusters: List[Any] = []
        try:
            found = await self.store.list_cluster_summaries(
                tenant_id=tenant_id,
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
    

    async def fetch_summaries(self, ids: List[str], *, tenant_id: str = DEFAULT_TENANT_ID, owner_type: str, owner_id: str) -> List[dict]:
        """
        Fetch summary payloads for a list of episodic IDs.
        """
        if self.store is None or not hasattr(self.store, "fetch_summaries"):
            return []
        if not owner_type or not owner_id:
            logger.error("EpisodicCore.fetch_summaries requires owner_type and owner_id")
            return []
        try:
            return await self.store.fetch_summaries(
                ids,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("EpisodicCore.fetch_summaries failed")
            return []

    async def fetch_transcripts(self, ids: List[str], *, tenant_id: str = DEFAULT_TENANT_ID, owner_type: str, owner_id: str) -> List[dict]:
        """
        Fetch transcript payloads for a list of episodic IDs.
        """
        if self.store is None or not hasattr(self.store, "fetch_transcripts"):
            return []
        if not owner_type or not owner_id:
            logger.error("EpisodicCore.fetch_transcripts requires owner_type and owner_id")
            return []
        try:
            return await self.store.fetch_transcripts(
                ids,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("EpisodicCore.fetch_transcripts failed")
            return []
