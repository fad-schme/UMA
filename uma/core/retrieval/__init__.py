"""
uma3.core.retrieval
===================

Core retrieval subsystem for UMA-3.

This package provides deterministic, single-shot retrieval (RetrievalService)
and its supporting components (MultiStoreRetriever + MemorySelector).

RLM (recursive retrieval) lives under:
    uma3.core.retrieval.rlm

Public exports
--------------
- RetrievalService: developer-facing retrieval API (single-shot)
- MultiStoreRetriever: raw multi-store retrieval engine
- MemorySelector: deterministic ranking + truncation
"""

from .service import RetrievalService
from .retrieval import MultiStoreRetriever
from .selector import MemorySelector

__all__ = [
    "RetrievalService",
    "MultiStoreRetriever",
    "MemorySelector",
]