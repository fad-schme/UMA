"""
UMA-3 Temporal Graph Subsystem
==============================

Modules:
    adapters.graph.base        – GraphAdapter base & abstract interface
    adapters.graph.neo4j_adapter – Neo4jAdapter backend
    core.py                    – TemporalGraphCore (high-level interface)
    updater.py                 – Graph update logic (Cypher construction)

Coding Agent Instructions
-------------------------
- Keep all temporal-graph logic here.
- UMA3Memory.initialize() should instantiate TemporalGraphCore.
- MemoryPipeline should call graph_core.add_episode(), etc.
"""

from ...adapters.graph.base import GraphAdapter
from ...adapters.graph.neo4j_adapter import Neo4jAdapter
from ...adapters.graph.memgraph_adapter import MemgraphAdapter
from .updater import GraphUpdater
from .core import TemporalGraphCore

__all__ = [
    "GraphAdapter",
    "Neo4jAdapter",
    "GraphUpdater",
    "MemgraphAdapter",
    "TemporalGraphCore",
]