"""
UMA-3 Temporal Graph Subsystem
==============================

This package contains the temporal graph memory subsystem for UMA-3.

Responsibilities
----------------
- Represent users, episodes, facts, and entities as graph nodes
- Maintain temporal and semantic relationships
- Provide a backend-agnostic abstraction over graph databases

Identity Convention (v1)
------------------------
ALL User nodes are keyed using the canonical form:

    User.id = "user:<id>"

This is enforced by the graph updater via ensure_user_subject().

Modules
-------
- core.py     : TemporalGraphCore (high-level interface)
- updater.py  : GraphUpdater (graph mapping & Cypher construction)

Adapters live under:
    uma3.adapters.graph.*
"""

from .core import TemporalGraphCore
from .updater import GraphUpdater

__all__ = [
    "TemporalGraphCore",
    "GraphUpdater",
]