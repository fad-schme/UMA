"""
semantic/core.py
================

SemanticCore — The unified interface for UMA semantic memory.

Exposes:
    - extract(subject_or_user_id, text)
    - ingest(subject_or_user_id, text)

Subject convention (v1)
-----------------------
UMA-RLM standardizes semantic subjects as:
    "user:<id>"

Callers MAY pass either:
- raw user_id: "123"
- canonical subject: "user:123"

SemanticCore will normalize automatically.

Coding Agent Instructions
-------------------------
- Keep this interface simple and stable.
- This is the ONLY class MemoryPipeline should call for semantic ingestion.
"""

from __future__ import annotations

import logging
from typing import Any, List

from ...types_fact import Fact
from ..utils.identity import ensure_user_subject
from .ingestor import SemanticIngestor

logger = logging.getLogger(__name__)


class SemanticCore:
    """
    High-level interface for UMA semantic memory.

    Wraps:
        - FactExtractor
        - SalienceScorer
        - SemanticIngestor
        - SemanticSQLStore

    This class enforces subject normalization but delegates all business logic
    to SemanticIngestor.
    """

    def __init__(
        self,
        llm: Any,
        embedder: Any,
        semantic_store: Any,
        salience_threshold: float = 0.3,
    ) -> None:
        self.ingestor = SemanticIngestor(
            llm=llm,
            embedder=embedder,
            semantic_store=semantic_store,
            salience_threshold=salience_threshold,
        )
        logger.info("SemanticCore initialized.")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def extract(self, subject: str, text: str) -> List[Fact]:
        """
        Extract semantic facts (not persisted).

        Parameters
        ----------
        subject : str
            Raw user_id or canonical subject ("user:<id>").
        text : str
            Text to extract facts from (typically assistant reply).

        Returns
        -------
        List[Fact]
            Extracted facts (not persisted).
        """
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.extract: invalid subject=%r", subject)
            return []

        return await self.ingestor.extract(subj, text)

    async def ingest(self, subject: str, text: str) -> List[Fact]:
        """
        Extract + ingest facts into the semantic store.

        Parameters
        ----------
        subject : str
            Raw user_id or canonical subject ("user:<id>").
        text : str
            Text to ingest facts from.

        Returns
        -------
        List[Fact]
            Persisted facts.
        """
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.ingest: invalid subject=%r", subject)
            return []

        return await self.ingestor.ingest(subj, text)