from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .base import VectorIndex


class QdrantIndex(VectorIndex):
    """
    Placeholder public Qdrant adapter surface for the container profile.

    The public repo keeps the container profile declarative, but does not ship
    a live Qdrant implementation in this package boundary.
    """

    def __init__(self, dim: int, **_: object) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("QdrantIndex: dim must be a positive integer.")
        raise RuntimeError(
            "QdrantIndex is not included in this public package. "
            "The public container profile remains declarative only."
        )

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        raise RuntimeError("QdrantIndex is not available in this public package.")

    def query(
        self,
        vector: List[float],
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        raise RuntimeError("QdrantIndex is not available in this public package.")

    def delete(self, ids: List[str]) -> None:
        raise RuntimeError("QdrantIndex is not available in this public package.")
