"""
semantic/core.py
================

SemanticCore — The unified interface for UMA-3 semantic memory.

Exposes:
    - extract(subject, text)
    - ingest(subject, text)

Note:
    •	subject is typically "user:<id>" or similar.
	•	text should be relatively concise (e.g., a single turn or reply), not entire logs.
Used by UMA3Memory.initialize() and MemoryPipeline.

Coding Agent Instructions
-------------------------
- Keep this interface simple and stable.
- This is the ONLY class MemoryPipeline should call for semantic ingestion.
"""

from __future__ import annotations
import logging
from typing import Any, List

from ...types_fact import Fact
from .ingestor import SemanticIngestor

logger = logging.getLogger(__name__)


class SemanticCore:
    """
    High-level interface for UMA-3 semantic memory.

    Wraps:
        - FactExtractor
        - SalienceScorer
        - SemanticIngestor
        - SemanticSQLStore

    Does NOT contain business rules; it delegates to SemanticIngestor.
    """

    def __init__(
        self,
        llm: Any,
        embedder: Any,
        semantic_store: Any,
        salience_threshold: float = 0.3,
    ):
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
        """
        return await self.ingestor.extract(subject, text)

    async def ingest(self, subject: str, text: str) -> List[Fact]:
        """
        Extract + ingest facts into the semantic store.
        """
        return await self.ingestor.ingest(subject, text)