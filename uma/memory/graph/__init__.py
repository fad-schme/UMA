"""
UMA Graph Subsystem
===================

Temporal knowledge graph memory for UMA.

Modules
-------
- core.py     : GraphCore — adapter lifecycle, reads, safety gating
- updater.py  : GraphUpdater — all graph writes (Cypher construction)

Adapters live under:
    uma.adapters.graph.*
"""

from .core import GraphCore
from .updater import GraphUpdater

__all__ = [
    "GraphCore",
    "GraphUpdater",
]
