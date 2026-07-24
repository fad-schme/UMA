"""
SkillIndexer for UMA Procedural Memory.

This class converts skill definitions into:

- A Skill model instance
- A semantic embedding vector (for vector search)

Future additions may:
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
from typing import Dict, List, Any

from uma.adapters.llm.base import LLMInterface, EmbeddingInterface

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
        logger.debug("SkillIndexer initialized (llm=%s, embedder=%s).",
                    type(llm).__name__, type(embedder).__name__)

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
