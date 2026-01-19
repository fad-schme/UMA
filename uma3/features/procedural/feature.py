"""
ProceduralFeature: UMA-3 plugin for skill-based procedural memory.

After attaching to UMA3Memory, the agent gains:

- add_skill(skill: Skill, embedding: List[float])
- find_skills(query_text: str, k: int = 5) -> List[Skill]
- get_skill(skill_id: str) -> Optional[Skill]

The feature is OPTIONAL and built on top of the core UMA-3 memory system.

Coding agent instructions
-------------------------
- This is a plugin, not a core subsystem. It MUST subclass UMA3Feature.
- Attach this feature to UMA3Memory via the FeatureRegistry or bootstrap logic.
- Do NOT modify UMA3Memory internals directly; only attach well-defined methods.
- Ensure that all async calls (DB, embedding) are awaited and errors are logged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from ...core.registry import UMA3Feature
from ...types_skill import Skill
from .matcher import SkillMatcher

if TYPE_CHECKING:
    from ...stores.procedural_sql import ProceduralSQLStore
    from ...adapters.llm.base import EmbeddingInterface
    from ...core.uma3_memory import UMA3Memory

logger = logging.getLogger(__name__)


class ProceduralFeature(UMA3Feature):
    """
    UMA-3 Procedural Memory Plugin.

    Provides a hybrid semantic + rule-based skill lookup mechanism.

    Design
    ------
    - Skills are stored in ProceduralSQLStore with embeddings.
    - Semantic candidates are retrieved with vector search.
    - SkillMatcher applies trigger phrases + regex patterns on top.
    """

    name = "procedural"

    def __init__(
        self,
        store: "ProceduralSQLStore",
        embedder: "EmbeddingInterface",
        matcher: Optional[SkillMatcher] = None,
    ) -> None:
        """
        Parameters
        ----------
        store : ProceduralSQLStore
            Backing store for skills + vector search.
        embedder : EmbeddingInterface
            Embedder used to embed queries and skills.
        matcher : SkillMatcher, optional
            Hybrid matcher; if omitted, a default instance is created.
        """
        self.store = store
        self.embedder = embedder
        self.matcher = matcher or SkillMatcher()

        logger.info("ProceduralFeature initialized with store=%s embedder=%s",
                    type(store).__name__, type(embedder).__name__)

    # ------------------------------------------------------------------
    # UMA3Feature API
    # ------------------------------------------------------------------

    def attach(self, memory_client: "UMA3Memory") -> None:
        """
        Attach procedural memory functions to UMA3Memory.

        This method:

        - Registers the feature under memory_client.features["procedural"].
        - Adds three async methods onto memory_client:

            add_skill(skill: Skill, embedding: List[float]) -> None
            find_skills(query_text: str, k: int = 5) -> List[Skill]
            get_skill(skill_id: str) -> Optional[Skill]

        All attached methods are thin wrappers over this feature.
        """
        memory_client.features[self.name] = self

        async def add_skill(skill: Skill, embedding: List[float]) -> None:
            """
            Store a new procedural skill into the procedural store.
            """
            try:
                await self.store.add_skill(skill, embedding)
                logger.info("ProceduralFeature.add_skill: stored skill id=%s", skill.id)
            except Exception:
                logger.exception("ProceduralFeature.add_skill: failed for id=%s", skill.id)
                raise

        async def find_skills(query_text: str, k: int = 5) -> List[Skill]:
            """
            Retrieve relevant skills for a natural language query.

            Steps
            -----
            1. Embed query_text (semantic representation).
            2. Use ProceduralSQLStore.search(...) to obtain candidates.
            3. Use SkillMatcher.match_skills(...) to apply rule-based checks.
            """
            query_text_clean = (query_text or "").strip()
            if not query_text_clean:
                logger.debug("ProceduralFeature.find_skills: empty query_text.")
                return []

            try:
                vectors = await self.embedder.embed([query_text_clean])
                if not vectors:
                    logger.error("ProceduralFeature.find_skills: empty embedding result.")
                    return []

                query_emb = vectors[0]
            except Exception:
                logger.exception("ProceduralFeature.find_skills: embed failed.")
                return []

            try:
                candidates = await self.store.search(query_emb, k=k)
            except Exception:
                logger.exception("ProceduralFeature.find_skills: store.search failed.")
                return []

            try:
                matched = self.matcher.match_skills(query_text_clean, candidates)
                logger.info(
                    "ProceduralFeature.find_skills: %d matched skills (k=%d).",
                    len(matched),
                    k,
                )
                return matched
            except Exception:
                logger.exception("ProceduralFeature.find_skills: matcher failed.")
                return []

        async def get_skill(skill_id: str) -> Optional[Skill]:
            """
            Retrieve a single skill by ID, or None if not found.
            """
            if not skill_id:
                logger.debug("ProceduralFeature.get_skill: empty skill_id.")
                return None

            try:
                skill = await self.store.get_skill(skill_id)
                if skill is None:
                    logger.info("ProceduralFeature.get_skill: no skill for id=%s", skill_id)
                else:
                    logger.debug("ProceduralFeature.get_skill: loaded id=%s", skill_id)
                return skill
            except Exception:
                logger.exception("ProceduralFeature.get_skill: failed for id=%s", skill_id)
                return None

        # Attach methods to UMA3Memory (thin, well-defined surface)
        memory_client.add_skill = add_skill        # type: ignore[attr-defined]
        memory_client.find_skills = find_skills    # type: ignore[attr-defined]
        memory_client.get_skill = get_skill        # type: ignore[attr-defined]

        logger.info("ProceduralFeature attached to UMA3Memory.")