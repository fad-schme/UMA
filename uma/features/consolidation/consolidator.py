"""
UMA Consolidator (Sleep Cycle)
================================

The consolidator merges episodic memories into semantic knowledge:

1. Fetch recent episodes for a user.
2. Cluster similar episodes (EpisodeClusterer).
3. Summarize clusters (ConsolidationSummarizer).
4. Extract high-salience facts (FactExtractor + SalienceScorer).
5. Upsert facts into SemanticSQLStore.
6. Prune old/low-value memory (Pruner).

This subsystem is OPTIONAL and enabled via ConsolidationFeature.

Production Guarantees:
----------------------
• No dependency on vector search for episodic retrieval.
• Deterministic ordering by timestamp.
• Batched embeddings.
• Deduplication of extracted facts.
• Structured logging.
• Graceful degradation if LLM or embedder errors occur.
• No mutation of unrelated UMA subsystems.
"""

from __future__ import annotations

import logging
from typing import List

from ...types_episode import Episode
from ...types_fact import Fact
from ...adapters.llm.base import LLMInterface, EmbeddingInterface
from ...stores.semantic_sql import SemanticSQLStore
from ...stores.episodic_sql import EpisodicSQLStore

from ...core.semantic.extractor import FactExtractor
from ...core.semantic.scorer import SalienceScorer

from .clusterer import EpisodeClusterer
from .summarizer import ConsolidationSummarizer
from .pruner import Pruner

logger = logging.getLogger(__name__)


class Consolidator:
    """Main consolidation workflow engine (sleep cycle)."""

    def __init__(
        self,
        episodic_store: EpisodicSQLStore,
        semantic_store: SemanticSQLStore,
        llm: LLMInterface,
        embedder: EmbeddingInterface,
        cluster_similarity: float = 0.75,
        max_episodes_per_cycle: int = 200,
        pruner: Pruner | None = None,
    ) -> None:

        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.clusterer = EpisodeClusterer(cluster_similarity)
        self.summarizer = ConsolidationSummarizer(llm)
        self.extractor = FactExtractor(llm, SalienceScorer())
        self.embedder = embedder
        self.max_episodes = max_episodes_per_cycle
        self.pruner = pruner or Pruner()

        logger.info(
            "Consolidator initialized (cluster_similarity=%.2f, max_episodes=%d)",
            cluster_similarity,
            max_episodes_per_cycle,
        )

    # ------------------------------------------------------------------
    # PUBLIC ENTRYPOINT
    # ------------------------------------------------------------------

    async def run_once(self, user_id: str) -> List[Fact]:
        """
        Run one full consolidation cycle:
            1. Fetch episodes
            2. Cluster
            3. Summarize
            4. Extract facts
            5. Upsert facts
            6. Prune episodes
        """

        # STEP 1: Fetch recent episodes
        episodes = await self._fetch_recent_episodes(user_id)
        if not episodes:
            logger.info("Consolidator: no episodes for user=%s", user_id)
            return []

        # STEP 2: Cluster similar episodes
        try:
            clusters = self.clusterer.cluster(episodes)
        except Exception:
            logger.exception("Consolidator: clustering failed for user=%s", user_id)
            return []

        distilled_facts: List[Fact] = []

        # STEP 3: Summarize each cluster & extract facts
        for cluster in clusters:
            cluster_text = self._cluster_text(cluster)

            try:
                summary = await self.summarizer.summarize_cluster(cluster_text)
            except Exception:
                logger.exception("Consolidator: summarization failed for user=%s", user_id)
                continue

            if not summary:
                continue

            # Store precomputed cluster summary for retrieval (read-only env use).
            try:
                episode_ids = [ep.id for ep in cluster if getattr(ep, "id", None)]
                latest_ts = max(ep.timestamp for ep in cluster if getattr(ep, "timestamp", None))
                await self.episodic_store.upsert_cluster_summary(
                    user_id=user_id,
                    episode_ids=episode_ids,
                    summary=summary,
                    latest_timestamp=latest_ts.isoformat(),
                )
            except Exception:
                logger.exception("Consolidator: failed to upsert cluster summary user=%s", user_id)

            try:
                new_facts = await self.extractor.extract_from_text(user_id, summary)
                distilled_facts.extend(new_facts)
            except Exception:
                logger.exception("Consolidator: fact extraction failed for user=%s", user_id)

        # STEP 4: Persist distilled facts
        await self._persist_facts(distilled_facts)

        # STEP 5: Prune obsolete / redundant episodes
        await self._prune(user_id)

        logger.info(
            "Consolidator: completed cycle for user=%s → %d distilled facts",
            user_id,
            len(distilled_facts),
        )

        return distilled_facts

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    async def _fetch_recent_episodes(self, user_id: str) -> List[Episode]:
        """
        Fetch recent episodes deterministically using timestamp ordering.
        Replaces any legacy embedding-based episode search.
        """
        try:
            episodes = await self.episodic_store.list_recent(
                user_id=user_id,
                n=self.max_episodes,
            )
            return episodes
        except Exception:
            logger.exception(
                "Consolidator: failed to fetch recent episodes for user=%s",
                user_id,
            )
            return []

    def _cluster_text(self, cluster: List[Episode]) -> List[str]:
        """
        Build text input for cluster summarization:
            • Prefer episode.summary
            • Fall back to episode.raw
        """
        summaries = [ep.summary for ep in cluster if ep.summary]
        raws = [ep.raw for ep in cluster if ep.raw]
        return summaries or raws

    # ------------------------------------------------------------------
    # Persist distilled facts (production-grade)
    # ------------------------------------------------------------------
    async def _persist_facts(self, facts: List[Fact]) -> None:
        """
        Embed & upsert facts in a production-grade way.

        Features:
        - Deduplication
        - Batch embedding
        - Robust per-batch error handling
        - Structured debug logging
        """

        if not facts:
            logger.debug("Consolidator._persist_facts: no facts to persist.")
            return

        # ---- 1. Deduplicate by (subject, predicate, object) ----
        unique: dict[tuple[str, str, str], Fact] = {}
        for f in facts:
            key = (f.subject, f.predicate, f.object)
            if key not in unique or f.confidence > unique[key].confidence:
                unique[key] = f

        deduped_facts = list(unique.values())
        logger.info(
            "Consolidator._persist_facts: %d facts → %d deduped.",
            len(facts),
            len(deduped_facts),
        )

        # ---- 2. Embed in batches ----
        BATCH_SIZE = 16

        for i in range(0, len(deduped_facts), BATCH_SIZE):
            batch = deduped_facts[i : i + BATCH_SIZE]

            # Build semantic text lines for embedding
            texts = [f"{f.subject} {f.predicate} {f.object}" for f in batch]

            try:
                vectors = await self.embedder.embed(texts)
            except Exception:
                logger.exception(
                    "Consolidator._persist_facts: embedding failure on batch [%d:%d]",
                    i,
                    i + BATCH_SIZE,
                )
                continue

            # ---- 3. Upsert each fact ----
            for fact, emb in zip(batch, vectors):
                try:
                    await self.semantic_store.upsert_fact(fact, emb)
                    logger.debug(
                        "Consolidator.upsert: id=%s subj=%s pred=%s obj=%s",
                        fact.id,
                        fact.subject,
                        fact.predicate,
                        fact.object,
                    )
                except Exception:
                    logger.exception(
                        "Consolidator: failed to upsert fact id=%s",
                        fact.id,
                    )

        logger.info(
            "Consolidator: completed upsert for %d deduped facts.",
            len(deduped_facts),
        )

        # ------------------------------------------------------------------
    # Episodic + Semantic Pruning
    # ------------------------------------------------------------------
    async def _prune(self, user_id: str) -> None:
        """
        Prune episodic and semantic memory in a deterministic, safe way.

        Steps:
        1. Load recent episodic memories
        2. Filter via Pruner.filter_episodes()
        3. Remove pruned episodes from EpisodicSQLStore
        4. Load semantic facts
        5. Filter via Pruner.filter_facts()
        6. Remove pruned facts from SemanticSQLStore
        """

        # ---------------------------------------------------------------
        # 1) EPISODIC PRUNING
        # ---------------------------------------------------------------
        try:
            episodes = await self.episodic_store.list_recent(
                user_id=user_id,
                n=self.max_episodes,
            )
        except Exception:
            logger.exception(
                "Consolidator: episodic prune fetch failed for user=%s",
                user_id,
            )
            episodes = []

        if not episodes:
            logger.debug("Consolidator: no episodic entries to prune (user=%s).", user_id)
        else:
            try:
                kept = self.pruner.filter_episodes(episodes)
            except Exception:
                logger.exception(
                    "Consolidator: prune filter failed for episodes (user=%s).",
                    user_id,
                )
                kept = episodes

            # Delete pruned episodic items
            to_remove = {ep.id for ep in episodes} - {ep.id for ep in kept}
            if to_remove:
                logger.info(
                    "Consolidator: pruning %d episodes for user=%s",
                    len(to_remove),
                    user_id,
                )
                try:
                    for ep_id in to_remove:
                        await self.episodic_store.delete_episode(ep_id)
                except Exception:
                    logger.exception(
                        "Consolidator: failed deleting episodic items (user=%s).",
                        user_id,
                    )

        # ---------------------------------------------------------------
        # 2) SEMANTIC FACT PRUNING
        # ---------------------------------------------------------------
        try:
            # NOTE: fetch all facts for subject=user
            semantic_facts = await self.semantic_store.list_facts_for_subject(user_id)
        except Exception:
            logger.exception(
                "Consolidator: semantic prune fetch failed for user=%s",
                user_id,
            )
            semantic_facts = []

        if not semantic_facts:
            logger.debug("Consolidator: no semantic facts to prune (user=%s).", user_id)
            return

        try:
            kept_facts = self.pruner.filter_facts(semantic_facts)
        except Exception:
            logger.exception(
                "Consolidator: prune filter failed for semantic facts (user=%s).",
                user_id,
            )
            kept_facts = semantic_facts

        # Determine which facts to delete
        to_delete = {f.id for f in semantic_facts} - {f.id for f in kept_facts}

        if to_delete:
            logger.info(
                "Consolidator: pruning %d semantic facts for user=%s",
                len(to_delete),
                user_id,
            )
            try:
                for fact_id in to_delete:
                    await self.semantic_store.delete_fact(fact_id)
            except Exception:
                logger.exception(
                    "Consolidator: failed deleting semantic facts (user=%s).",
                    user_id,
                )
