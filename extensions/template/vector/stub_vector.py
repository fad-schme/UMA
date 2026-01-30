"""
Template vector adapter.

Implement a factory that returns a VectorIndex.
"""

from __future__ import annotations

from typing import Any

from uma.adapters.vector.base import VectorIndex


def make_index(dim: int, **config: Any) -> VectorIndex:
    """
    Factory for custom vector index.

    Parameters
    ----------
    dim : int
        Vector dimension.
    config : dict
        Adapter-specific configuration from storage.vector_config.
    """
    raise NotImplementedError("Implement make_index() and return a VectorIndex")
