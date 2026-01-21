"""
SkillIndexer for UMA Procedural Memory.

This class converts skill definitions into:

- A Skill model instance
- A semantic embedding vector (for vector search)

Future extensions may:
- Auto-generate skills from successful episodes via LLM analysis
- Cluster related skills
- Maintain versioned skill definitions

Coding agent instructions
-------------------------
- Keep this class focused on building skill objects + embeddings.
- Do NOT perform database writes here; that's ProceduralSQLStore's job.
- Ensure all embed calls are awaited and errors are logged.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Tuple, Any

from ...adapters.llm.base import LLMInterface, EmbeddingInterface
from ...types_skill import Skill

logger = logging.getLogger(__name__)


class SkillIndexer:
    """
    Builds and embeds skills for procedural memory.

    Parameters
    ----------
    llm : LLMInterface
        Currently unused, but reserved for future skill-generation logic.
    embedder : EmbeddingInterface
        Used to embed the textual representation of skill definitions.
    """

    def __init__(self, llm: LLMInterface, embedder: EmbeddingInterface) -> None:
        self.llm = llm
        self.embedder = embedder
        logger.info("SkillIndexer initialized (llm=%s, embedder=%s).",
                    type(llm).__name__, type(embedder).__name__)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def build_skill_from_definition(
        self,
        name: str,
        trigger_phrases: List[str],
        trigger_patterns: List[str],
        plan: Dict[str, Any],
        tools: List[str],
        example: str,
        meta: Dict[str, Any],
    ) -> Tuple[Skill, List[float]]:
        """
        Build and embed a Skill from explicit definitions.

        Parameters
        ----------
        name : str
            Human-readable name.
        trigger_phrases : List[str]
            String phrases that should match user queries.
        trigger_patterns : List[str]
            Regex patterns for more advanced matching.
        plan : Dict[str, Any]
            Workflow plan (steps, conditions, etc.).
        tools : List[str]
            Names/identifiers of tools this skill uses.
        example : str
            Example user query or usage narrative.
        meta : Dict[str, Any]
            Arbitrary free-form metadata.

        Returns
        -------
        (skill, embedding) : Tuple[Skill, List[float]]
        """
        skill = Skill(
            id=str(uuid.uuid4()),
            name=name,
            trigger_phrases=trigger_phrases,
            trigger_patterns=trigger_patterns,
            plan=plan,
            tools=tools,
            example=example,
            meta=meta,
        )

        embed_text = self._build_embedding_text(
            name=name,
            trigger_phrases=trigger_phrases,
            trigger_patterns=trigger_patterns,
            plan=plan,
            tools=tools,
            example=example,
        )

        try:
            vectors = await self.embedder.embed([embed_text])
            if not vectors:
                raise RuntimeError("SkillIndexer: embedder returned empty list.")
            emb = vectors[0]
        except Exception:
            logger.exception("SkillIndexer: embedding generation failed.")
            raise

        logger.info("SkillIndexer: built skill id=%s name=%s", skill.id, skill.name)
        return skill, emb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_embedding_text(
        self,
        name: str,
        trigger_phrases: List[str],
        trigger_patterns: List[str],
        plan: Dict[str, Any],
        tools: List[str],
        example: str,
    ) -> str:
        """
        Construct the text used for computing the skill embedding.

        Includes:
            - name
            - triggers
            - patterns
            - example
            - plan
            - tools
        """
        return (
            f"name: {name}\n"
            f"phrases: {trigger_phrases}\n"
            f"patterns: {trigger_patterns}\n"
            f"example: {example}\n"
            f"plan: {plan}\n"
            f"tools: {tools}\n"
        )