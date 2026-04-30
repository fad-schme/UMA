"""
UMA Temporal Graph Subsystem
==============================

Exports:
- TemporalGraphCore : The unified graph engine
- GraphAdapter      : Abstract interface for concrete backends
"""

from uma.adapters.graph.base import GraphAdapter

__all__ = ["GraphAdapter"]