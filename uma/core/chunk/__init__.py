"""
Chunk Memory Subsystem (UMA Core)
=================================

Provides a core interface for document chunks:
    - Ingest / upsert via ChunkCore
    - Retrieval via ChunkCore.search_chunks / lexical_search
"""

from .core import ChunkCore

__all__ = [
    "ChunkCore",
]
