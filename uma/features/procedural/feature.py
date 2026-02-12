"""
ProceduralFeature: UMA plugin for skill-based procedural memory.

After attaching to UMAMemory, the agent gains:

- procedural_health() -> FeatureResult
- procedural_add_skill(skill: Skill, embedding: List[float]) -> FeatureResult
- procedural_find_skills(query_text: str, k: int = 5) -> FeatureResult
- procedural_get_skill(skill_id: str) -> FeatureResult

The feature is OPTIONAL and built on top of the core UMA memory system.

Coding agent instructions
-------------------------
- This is a plugin, not a core subsystem. It MUST subclass UMAFeature.
- Attach this feature directly to UMAMemory.
- Do NOT modify UMAMemory internals directly; only attach well-defined methods.
- Ensure that all async calls (DB, embedding) are awaited and errors are logged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from ...core.utils.registry import FeatureContext, FeatureHandle, FeatureResult, UMAFeature
from ...types import Skill
from .matcher import SkillMatcher

if TYPE_CHECKING:
    from ...core.procedural.core import ProceduralCore
    from ...adapters.llm.base import EmbeddingInterface
    from ...core.uma_memory import UMAMemory

logger = logging.getLogger(__name__)


class ProceduralFeature(UMAFeature):
    """
    UMA Procedural Memory Plugin.

    Provides a hybrid semantic + rule-based skill lookup mechanism.

    Design
    ------
    - Skills are stored via ProceduralCore with embeddings.
    - Semantic candidates are retrieved with vector search.
    - SkillMatcher applies trigger phrases + regex patterns on top.
    """

    name = "procedural"

    def __init__(
        self,
        procedural_core: "ProceduralCore",
        embedder: "EmbeddingInterface",
        matcher: Optional[SkillMatcher] = None,
        max_k: int = 50,
    ) -> None:
        """
        Parameters
        ----------
        procedural_core : ProceduralCore
            Core interface for procedural skills.
        embedder : EmbeddingInterface
            Embedder used to embed queries and skills.
        matcher : SkillMatcher, optional
            Hybrid matcher; if omitted, a default instance is created.
        """
        self.core = procedural_core
        self.embedder = embedder
        self.matcher = matcher or SkillMatcher()
        self.max_k = max_k

        logger.debug("ProceduralFeature initialized with core=%s embedder=%s",
                    type(procedural_core).__name__, type(embedder).__name__)

    # ------------------------------------------------------------------
    # UMAFeature API
    # ------------------------------------------------------------------

    def attach(self, context: FeatureContext) -> FeatureHandle:
        """
        Attach procedural memory functions to UMAMemory.

        This method:

        - Registers the feature under memory_client.features["procedural"] on success.
        - Adds async methods onto memory_client that return FeatureResult.
        """
        memory_client = context.memory

        if self.core is None or self.embedder is None:
            logger.error("ProceduralFeature.attach: missing core or embedder.")
            return FeatureHandle(name=self.name, methods=())

        def _health() -> FeatureResult:
            ok = self.core is not None and self.embedder is not None
            return FeatureResult.success() if ok else FeatureResult.failure(["missing dependencies"])

        async def procedural_add_skill(skill: Skill, embedding: List[float]) -> FeatureResult:
            """
            Store a new procedural skill via ProceduralCore.
            """
            if skill is None:
                logger.warning("ProceduralFeature.add_skill: missing skill.")
                return FeatureResult.failure(["missing skill"])
            if not isinstance(embedding, list) or not embedding:
                logger.warning("ProceduralFeature.add_skill: invalid embedding.")
                return FeatureResult.failure(["invalid embedding"])
            try:
                await self.core.add_skill(skill, embedding)
                logger.info("ProceduralFeature.add_skill: stored skill id=%s", skill.id)
                return FeatureResult.success()
            except Exception as exc:
                logger.exception("ProceduralFeature.add_skill: failed for id=%s", skill.id)
                return FeatureResult.failure([str(exc)])

        async def procedural_find_skills(query_text: str, k: int = 5) -> FeatureResult:
            """
            Retrieve relevant skills for a natural language query.

            Steps
            -----
            1. Embed query_text (semantic representation).
            2. Use ProceduralCore.search(...) to obtain candidates.
            3. Use SkillMatcher.match_skills(...) to apply rule-based checks.

            Notes
            -----
            - k is clamped to max_k to bound search cost.
            """
            query_text_clean = (query_text or "").strip()
            if not query_text_clean:
                logger.debug("ProceduralFeature.find_skills: empty query_text.")
                return FeatureResult.success([])
            user_id = getattr(memory_client, "user_id", None)
            if not user_id:
                logger.error("ProceduralFeature.find_skills: missing memory_client.user_id.")
                return FeatureResult.failure(["missing user_id"], data=[])

            try:
                vectors = await self.embedder.embed([query_text_clean])
                if not vectors:
                    logger.error("ProceduralFeature.find_skills: empty embedding result.")
                    return FeatureResult.failure(["empty embedding result"], data=[])

                query_emb = vectors[0]
            except Exception as exc:
                logger.exception("ProceduralFeature.find_skills: embed failed.")
                return FeatureResult.failure([str(exc)], data=[])

            try:
                k_int = int(k)
            except Exception:
                logger.warning("ProceduralFeature.find_skills: invalid k=%r.", k)
                return FeatureResult.failure(["invalid k"], data=[])

            if k_int <= 0:
                logger.warning("ProceduralFeature.find_skills: non-positive k=%r.", k)
                return FeatureResult.failure(["invalid k"], data=[])

            k_clamped = min(k_int, self.max_k)
            try:
                candidates = await self.core.search(
                    user_id=user_id,
                    query_embedding=query_emb,
                    owner_type="user",
                    owner_id=user_id,
                    k=k_clamped,
                )
            except Exception as exc:
                logger.exception("ProceduralFeature.find_skills: core.search failed.")
                return FeatureResult.failure([str(exc)], data=[])

            try:
                matched = self.matcher.match_skills(query_text_clean, candidates)
                logger.info(
                    "ProceduralFeature.find_skills: %d matched skills (k=%d).",
                    len(matched),
                    k,
                )
                return FeatureResult.success(matched)
            except Exception as exc:
                logger.exception("ProceduralFeature.find_skills: matcher failed.")
                return FeatureResult.failure([str(exc)], data=[])

        async def procedural_get_skill(skill_id: str) -> FeatureResult:
            """
            Retrieve a single skill by ID.

            Returns
            -------
            FeatureResult
                data = Skill | None
            """
            if not skill_id:
                logger.debug("ProceduralFeature.get_skill: empty skill_id.")
                return FeatureResult.failure(["empty skill_id"], data=None)
            user_id = getattr(memory_client, "user_id", None)
            if not user_id:
                logger.error("ProceduralFeature.get_skill: missing memory_client.user_id.")
                return FeatureResult.failure(["missing user_id"], data=None)

            try:
                skill = await self.core.get_skill(
                    skill_id,
                    owner_type="user",
                    owner_id=user_id,
                )
                if skill is None:
                    logger.info("ProceduralFeature.get_skill: no skill for id=%s", skill_id)
                else:
                    logger.debug("ProceduralFeature.get_skill: loaded id=%s", skill_id)
                return FeatureResult.success(skill)
            except Exception as exc:
                logger.exception("ProceduralFeature.get_skill: failed for id=%s", skill_id)
                return FeatureResult.failure([str(exc)], data=None)

        # Attach methods to UMAMemory (thin, well-defined surface)
        try:
            memory_client.register_methods(
                self.name,
                {
                    "procedural_health": _health,
                    "procedural_add_skill": procedural_add_skill,
                    "procedural_find_skills": procedural_find_skills,
                    "procedural_get_skill": procedural_get_skill,
                },
            )
        except Exception:
            logger.exception("ProceduralFeature.attach: method registration failed.")
            return FeatureHandle(name=self.name, methods=())

        memory_client.features[self.name] = self
        logger.info("ProceduralFeature attached to UMAMemory.")
        return FeatureHandle(
            name=self.name,
            methods=(
                "procedural_health",
                "procedural_add_skill",
                "procedural_find_skills",
                "procedural_get_skill",
            ),
        )

    @classmethod
    def validate_config(cls, config: dict) -> None:
        max_k = config.get("max_k")
        if max_k is not None and (not isinstance(max_k, int) or max_k <= 0):
            raise ValueError("'max_k' must be a positive integer")
