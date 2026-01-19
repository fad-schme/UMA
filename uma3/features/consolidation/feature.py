"""
ConsolidationFeature
====================

Optional UMA-3 plugin that exposes memory consolidation as:

    await memory_client.run_consolidation(user_id)

This feature does not run automatically; you must call it from:
- A scheduler
- A nightly batch job
- A pipeline hook (“after N turns”)

Coding Agent Instructions
-------------------------
- Must NOT modify UMA3Memory internals beyond adding the public method.
- Must fail gracefully; never raise exceptions that break agents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from ...core.registry import UMA3Feature
from .consolidator import Consolidator

if TYPE_CHECKING:
    from ...core.uma3_memory import UMA3Memory
    from ...stores.semantic_sql import SemanticSQLStore
    from ...stores.episodic_sql import EpisodicSQLStore
    from ...adapters.llm.base import LLMInterface, EmbeddingInterface

logger = logging.getLogger(__name__)


class ConsolidationFeature(UMA3Feature):
    """Optional consolidation plugin."""
    name = "consolidation"

    def __init__(
        self,
        episodic_store: "EpisodicSQLStore",
        semantic_store: "SemanticSQLStore",
        llm: "LLMInterface",
        embedder: "EmbeddingInterface",
        cluster_similarity: float = 0.75,
        max_episodes_per_cycle: int = 200,
    ) -> None:

        self.consolidator = Consolidator(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            llm=llm,
            embedder=embedder,
            cluster_similarity=cluster_similarity,
            max_episodes_per_cycle=max_episodes_per_cycle,
        )

        logger.info(
            "ConsolidationFeature initialized "
            "(cluster_similarity=%.2f, max_episodes=%d)",
            cluster_similarity,
            max_episodes_per_cycle,
        )

    def attach(self, memory_client: "UMA3Memory") -> None:
        """
        Attach run_consolidation(user_id) to UMA3Memory.

        This method is a safe "thin wrapper" that defers all logic to Consolidator.
        """
        memory_client.features[self.name] = self

        async def run_consolidation(user_id: str) -> List:
            try:
                return await self.consolidator.run_once(user_id)
            except Exception:
                logger.exception("ConsolidationFeature.run_consolidation failed.")
                return []

        memory_client.run_consolidation = run_consolidation  # type: ignore[attr-defined]
        logger.info("ConsolidationFeature attached to UMA3Memory.")