"""
UMA Temporal Graph Subsystem
==============================

Exports:
- TemporalGraphCore : The unified graph engine
- GraphAdapter      : Abstract interface for concrete backends
"""

from .core import TemporalGraphCore
from ...adapters.graph.base import GraphAdapter

__all__ = ["TemporalGraphCore", "GraphAdapter"]