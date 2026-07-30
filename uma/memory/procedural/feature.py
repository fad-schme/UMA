"""
ProceduralFeature: UMA plugin for skill-based procedural memory.

After attaching to UMAMemory, the agent gains:

- procedural_health() -> FeatureResult
- procedural_add_skill(skill: Skill, embedding: List[float], *, owner_type: str | None = None, owner_id: str | None = None) -> FeatureResult
- procedural_find_skills(query_text: str, *, user_id: str | None = None, owner_type: str | None = None, owner_id: str | None = None, k: int = 5) -> FeatureResult
- procedural_get_skill(skill_id: str, *, user_id: str | None = None, owner_type: str | None = None, owner_id: str | None = None) -> FeatureResult

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
from typing import TYPE_CHECKING, Optional

from uma.common.ownership import validate_explicit_owner
from uma.common.registry import FeatureContext, FeatureHandle, FeatureResult, UMAFeature
from uma.common.types import Skill
from .matcher import SkillMatcher

if TYPE_CHECKING:
    from ...core.procedural.core import ProceduralCore
    from uma.adapters.llm.base import EmbeddingInterface

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

        # Attach methods to UMAMemory (thin, well-defined surface)
        try:
            memory_client.register_methods(
                self.name,
                {
                    "procedural_health": self._health,
                    "procedural_add_skill": self._add_skill,
                    "procedural_find_skills": self._find_skills,
                    "procedural_get_skill": self._get_skill,
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

    def _health(self) -> FeatureResult:
        ok = self.core is not None and self.embedder is not None
        return FeatureResult.success() if ok else FeatureResult.failure(["missing dependencies"])

    async def _add_skill(
        self,
        skill: Skill,
        embedding: list[float],
        *,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> FeatureResult:
        """Store a procedural skill."""
        if skill is None:
            logger.warning("ProceduralFeature.add_skill: missing skill.")
            return FeatureResult.failure(["missing skill"])
        if not isinstance(embedding, list) or not embedding:
            logger.warning("ProceduralFeature.add_skill: invalid embedding.")
            return FeatureResult.failure(["invalid embedding"])
        try:
            if owner_type is not None or owner_id is not None:
                if not owner_type or not owner_id:
                    raise ValueError("owner_type and owner_id are required together")
                await self.core.add_skill_for_owner(
                    skill,
                    embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
            else:
                await self.core.add_skill(skill, embedding)
            logger.info("ProceduralFeature.add_skill: stored skill id=%s", skill.id)
            return FeatureResult.success()
        except Exception as exc:
            logger.exception("ProceduralFeature.add_skill: failed for id=%s", skill.id)
            return FeatureResult.failure([str(exc)])

    @staticmethod
    def _normalize_owner(
        *,
        user_id: str | None,
        owner_type: str | None,
        owner_id: str | None,
    ) -> dict:
        if user_id and user_id.strip():
            if owner_type or owner_id:
                raise ValueError("provide either user_id or owner_type/owner_id, not both")
            normalized_owner = validate_explicit_owner(owner_type="user", owner_id=user_id)
        else:
            if not owner_type or not owner_id:
                raise ValueError("missing scoped owner")
            normalized_owner = validate_explicit_owner(owner_type=owner_type, owner_id=owner_id)
        if normalized_owner["owner_type"] not in {"agent", "user", "workspace"}:
            raise ValueError("owner_type must be one of: agent, user, workspace")
        return normalized_owner

    async def _find_skills(
        self,
        query_text: str,
        *,
        user_id: str | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
        k: int = 5,
    ) -> FeatureResult:
        """Retrieve and rule-match procedural skills for a scoped owner."""
        query_text_clean = (query_text or "").strip()
        if not query_text_clean:
            logger.debug("ProceduralFeature.find_skills: empty query_text.")
            return FeatureResult.success([])
        try:
            normalized_owner = self._normalize_owner(
                user_id=user_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception as exc:
            logger.error("ProceduralFeature.find_skills: invalid owner scope.")
            message = "missing user_id" if str(exc) == "missing scoped owner" else str(exc)
            return FeatureResult.failure([message], data=[])

        try:
            vectors = await self.embedder.embed([query_text_clean])
            if not vectors:
                logger.error("ProceduralFeature.find_skills: empty embedding result.")
                return FeatureResult.failure(["empty embedding result"], data=[])
            query_embedding = vectors[0]
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

        try:
            candidates = await self.core.search(
                query_embedding=query_embedding,
                tenant_id=str(normalized_owner["tenant_id"]),
                owner_type=str(normalized_owner["owner_type"]),
                owner_id=str(normalized_owner["owner_id"]),
                k=min(k_int, self.max_k),
            )
        except Exception as exc:
            logger.exception("ProceduralFeature.find_skills: core.search failed.")
            return FeatureResult.failure([str(exc)], data=[])

        try:
            matched = self.matcher.match_skills(query_text_clean, candidates)
            logger.info("ProceduralFeature.find_skills: %d matched skills (k=%d).", len(matched), k)
            return FeatureResult.success(matched)
        except Exception as exc:
            logger.exception("ProceduralFeature.find_skills: matcher failed.")
            return FeatureResult.failure([str(exc)], data=[])

    async def _get_skill(
        self,
        skill_id: str,
        *,
        user_id: str | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> FeatureResult:
        """Retrieve a single procedural skill for a scoped owner."""
        if not skill_id:
            logger.debug("ProceduralFeature.get_skill: empty skill_id.")
            return FeatureResult.failure(["empty skill_id"], data=None)
        try:
            normalized_owner = self._normalize_owner(
                user_id=user_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception as exc:
            logger.error("ProceduralFeature.get_skill: invalid owner scope.")
            message = "missing user_id" if str(exc) == "missing scoped owner" else str(exc)
            return FeatureResult.failure([message], data=None)

        try:
            skill = await self.core.get_skill(
                skill_id,
                tenant_id=str(normalized_owner["tenant_id"]),
                owner_type=str(normalized_owner["owner_type"]),
                owner_id=str(normalized_owner["owner_id"]),
            )
            if skill is None:
                logger.info("ProceduralFeature.get_skill: no skill for id=%s", skill_id)
            else:
                logger.debug("ProceduralFeature.get_skill: loaded id=%s", skill_id)
            return FeatureResult.success(skill)
        except Exception as exc:
            logger.exception("ProceduralFeature.get_skill: failed for id=%s", skill_id)
            return FeatureResult.failure([str(exc)], data=None)

    @classmethod
    def validate_config(cls, config: dict) -> None:
        max_k = config.get("max_k")
        if max_k is not None and (not isinstance(max_k, int) or max_k <= 0):
            raise ValueError("'max_k' must be a positive integer")
