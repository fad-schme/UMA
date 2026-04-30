"""
Procedural Memory Subsystem (UMA Core)
======================================

Provides a core interface for procedural skills:
    - Ingest / CRUD via ProceduralCore
    - Retrieval via ProceduralCore.search
"""

from .core import ProceduralCore

__all__ = [
    "ProceduralCore",
]
