"""
uma.core.retrieval
===================

Core retrieval subsystem for UMA.

This package provides deterministic, single-shot retrieval (RetrievalService)
and its supporting components (MemorySelector).

RLM (recursive retrieval) lives under:
    uma.core.retrieval.rlm

Public exports
--------------
- RetrievalService: developer-facing retrieval API (single-shot)
- MemorySelector: deterministic ranking + truncation
"""

from .service import RetrievalService
from .selector import MemorySelector

__all__ = [
    "RetrievalService",
    "MemorySelector",
]
