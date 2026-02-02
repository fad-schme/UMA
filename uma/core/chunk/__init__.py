"""
Chunk Memory Subsystem (UMA Core)
=================================

Provides a core interface for document chunks:
    - Ingest / upsert via ChunkCore
    - Retrieval via ChunkCore.search / search_text
"""

from .core import ChunkCore

__all__ = [
    "ChunkCore",
]
