"""
ConsolidationFeature
====================

Optional UMA plugin. Builds and holds the `Consolidator` for a `UMAMemory`
instance under `memory_client.features["consolidation"]`. The public entry
point is `uma.api.management.consolidate(memory, ...)`, called from the CLI
(`uma maintenance consolidate`) or wired into a caller's own scheduler — this
feature does not run automatically.

Coding Agent Instructions
-------------------------
- Must NOT modify UMAMemory internals beyond registering under `features`.
- Must fail gracefully; never raise exceptions that break agents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from uma.common.registry import FeatureContext, FeatureHandle, UMAFeature
from .consolidator import Consolidator

if TYPE_CHECKING:
    from uma.memory.episodic.core import EpisodicCore
    from uma.memory.semantic.core import SemanticCore
    from uma.adapters.llm.base import LLMInterface, EmbeddingInterface

logger = logging.getLogger(__name__)


class ConsolidationFeature(UMAFeature):
    """Optional consolidation plugin."""
    name = "consolidation"

    def __init__(
        self,
        episodic_core: "EpisodicCore",
        semantic_core: "SemanticCore",
        llm: "LLMInterface",
        embedder: "EmbeddingInterface",
        cluster_similarity: float = 0.75,
        max_episodes_per_cycle: int = 200,
    ) -> None:

        self._episodic_core = episodic_core
        self._semantic_core = semantic_core
        self._llm = llm
        self._embedder = embedder

        self.consolidator = Consolidator(
            llm=llm,
            embedder=embedder,
            cluster_similarity=cluster_similarity,
            max_episodes_per_cycle=max_episodes_per_cycle,
            episodic_core=episodic_core,
            semantic_core=semantic_core,
        )

        logger.info(
            "ConsolidationFeature initialized "
            "(cluster_similarity=%.2f, max_episodes=%d)",
            cluster_similarity,
            max_episodes_per_cycle,
        )

    def attach(self, context: FeatureContext) -> FeatureHandle:
        """
        Register this feature (and its `Consolidator`) under
        `memory_client.features["consolidation"]`. Exposes no methods on
        `UMAMemory` — `uma.api.management.consolidate` reads the consolidator
        from `features` directly.
        """
        memory_client = context.memory

        missing = []
        if self.consolidator is None:
            missing.append("consolidator")
        if self._episodic_core is None:
            missing.append("episodic_core")
        if self._semantic_core is None:
            missing.append("semantic_core")
        if self._embedder is None:
            missing.append("embedder")
        if self._llm is None:
            missing.append("llm")

        if missing:
            logger.error(
                "ConsolidationFeature.attach: missing dependencies: %s",
                ", ".join(missing),
            )
            return FeatureHandle(name=self.name, methods=())

        memory_client.features[self.name] = self
        logger.info("ConsolidationFeature attached to UMAMemory.")
        return FeatureHandle(name=self.name, methods=())

    @classmethod
    def validate_config(cls, config: dict) -> None:
        if "cluster_similarity" in config:
            cs = config["cluster_similarity"]
            if not isinstance(cs, (int, float)) or not (0 < cs <= 1):
                raise ValueError("'cluster_similarity' must be 0 < x <= 1")
        if "max_episodes_per_cycle" in config:
            max_eps = config["max_episodes_per_cycle"]
            if not isinstance(max_eps, int) or max_eps <= 0:
                raise ValueError("'max_episodes_per_cycle' must be a positive integer")
