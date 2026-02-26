"""
ConsolidationFeature
====================

Optional UMA plugin that exposes memory consolidation as:

    await memory_client.consolidation_run(user_id)

This feature does not run automatically; you must call it from:
- A scheduler
- A nightly batch job
- A pipeline hook (“after N turns”)

Coding Agent Instructions
-------------------------
- Must NOT modify UMAMemory internals beyond adding the public method.
- Must fail gracefully; never raise exceptions that break agents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...core.utils.registry import FeatureContext, FeatureHandle, FeatureResult, UMAFeature
from .consolidator import Consolidator

if TYPE_CHECKING:
    from ...core.episodic.core import EpisodicCore
    from ...core.semantic.core import SemanticCore
    from ...adapters.llm.base import LLMInterface, EmbeddingInterface

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
        Attach consolidation_run(user_id) to UMAMemory.

        This method is a safe "thin wrapper" that defers all logic to Consolidator.
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

        def _health() -> FeatureResult:
            missing_deps = []
            if self.consolidator is None:
                missing_deps.append("consolidator")
            if self._episodic_core is None:
                missing_deps.append("episodic_core")
            if self._semantic_core is None:
                missing_deps.append("semantic_core")
            if self._embedder is None:
                missing_deps.append("embedder")
            if self._llm is None:
                missing_deps.append("llm")
            if missing_deps:
                return FeatureResult.failure([f"missing: {', '.join(missing_deps)}"])
            return FeatureResult.success()

        async def consolidation_run(user_id: str) -> FeatureResult:
            """
            Run consolidation for a user.

            Returns
            -------
            FeatureResult
                data = {"facts": List[Fact], "fact_count": int}
            """
            if not user_id or not isinstance(user_id, str):
                logger.warning(
                    "ConsolidationFeature.consolidation_run: invalid user_id=%r.",
                    user_id,
                )
                return FeatureResult.failure(["invalid user_id"], data={"facts": [], "fact_count": 0})
            try:
                facts = await self.consolidator.run_once(user_id)
                return FeatureResult.success(
                    {"facts": facts, "fact_count": len(facts)}
                )
            except Exception as exc:
                logger.exception(
                    "ConsolidationFeature.consolidation_run failed (user_id=%s).",
                    user_id,
                )
                return FeatureResult.failure(
                    [str(exc)],
                    data={"facts": [], "fact_count": 0},
                )

        try:
            memory_client.register_methods(
                self.name,
                {
                    "consolidation_health": _health,
                    "consolidation_run": consolidation_run,
                },
            )
        except Exception:
            logger.exception("ConsolidationFeature.attach: method registration failed.")
            return FeatureHandle(name=self.name, methods=())
        memory_client.features[self.name] = self
        logger.info("ConsolidationFeature attached to UMAMemory.")
        return FeatureHandle(
            name=self.name,
            methods=("consolidation_health", "consolidation_run"),
        )

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
