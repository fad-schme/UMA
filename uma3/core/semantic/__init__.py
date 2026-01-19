"""
UMA-3 Semantic Memory Subsystem (Core)
=====================================

This package implements:
    - FactExtractor       : LLM-based fact extraction engine
    - SalienceScorer     : Scores extracted facts by importance
    - SemanticIngestor   : Embeds + upserts facts into SemanticSQLStore
    - SemanticCore       : High-level public interface used by UMA3Memory & Pipeline
"""

from .extractor import FactExtractor
from .scorer import SalienceScorer
from .ingestor import SemanticIngestor
from .core import SemanticCore

__all__ = [
    "FactExtractor",
    "SalienceScorer",
    "SemanticIngestor",
    "SemanticCore"
]