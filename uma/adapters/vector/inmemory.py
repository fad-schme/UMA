"""
InMemoryVectorIndex
====================

A lightweight, dependency-free vector index implementation meant for:

    • local development
    • notebooks
    • CI environments
    • testing without FAISS or cloud vector DBs

NOT FOR PRODUCTION.
-------------------
This backend performs O(N) similarity search using Python lists.
It is memory-bound and should not be used for large vector sets.

Implements the VectorIndex API so it can plug into UMA-3 seamlessly.
"""


from __future__ import annotations

import logging
import math
from typing import List, Dict, Optional, Tuple

from .base import VectorIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WARNING: InMemoryVectorIndex is NOT for production use.
# ---------------------------------------------------------------------------
INMEMORY_WARNING = (
    "InMemoryVectorIndex activated as fallback. This index is NOT suitable for production data volumes."
)
logger.warning(INMEMORY_WARNING)


class InMemoryVectorIndex(VectorIndex):
    def __init__(self, dim: int):
        """
        Parameters
        ----------
        dim : int
            Embedding dimension to enforce.
        """
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"InMemoryVectorIndex: invalid dimension {dim}")

        self.dim = dim
        self.dimension = dim   # UMA-3 expects this attribute
        self.index = self      # UMA-3 expects vector_index.index.* API

        # In-memory storage
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict] = {}

        logger.info(
            "InMemoryVectorIndex initialized (dim=%d). Intended for development, CI, or fallback only.",
            dim,
        )

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) + 1e-8
        nb = math.sqrt(sum(y * y for y in b)) + 1e-8
        return dot / (na * nb)

    # ---------------------------------------------------------
    # API Methods
    # ---------------------------------------------------------

    def upsert(self, ids: List[str], vectors: List[List[float]], metadata: Optional[List[Dict]] = None):
        for i, v in zip(ids, vectors):
            if len(v) != self.dim:
                raise ValueError(
                    f"InMemoryVectorIndex: expected vector dim={self.dim}, got={len(v)}"
                )
            self._vectors[i] = v
        if metadata:
            for i, m in zip(ids, metadata):
                self._metadata[i] = m

    def delete(self, ids: List[str]) -> None:
        for _id in ids:
            self._vectors.pop(_id, None)
            self._metadata.pop(_id, None)

    def query(
        self,
        vector: List[float],
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Returns:
            List of (id, score) pairs sorted by cosine similarity DESC.
        """
        if len(vector) != self.dim:
            raise ValueError(
                f"InMemoryVectorIndex.query: expected dim={self.dim}, got={len(vector)}"
            )

        results = []

        for key, vec in self._vectors.items():
            try:
                # Apply filters (if any)
                if filters:
                    meta = self._metadata.get(key, {})
                    if any(meta.get(fk) != fv for fk, fv in filters.items()):
                        continue

                score = self._cosine(vector, vec)
                results.append((key, score))
            except Exception:
                logger.exception(
                    "InMemoryVectorIndex.query: failed computing similarity for id=%s", key
                )
                continue

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


    @classmethod
    def fallback_if_faiss_unavailable(cls, dim: int):
        """
        Factory method used by UMA-3 when FAISSIndex import fails.
        Ensures consistent logging and safe instantiation.
        """
        logger.warning(
            "FAISS unavailable. Falling back to InMemoryVectorIndex (dim=%d).", dim
        )
        return cls(dim)

# ---------------------------------------------------------------------------
# Explicit warning: do not use in production
# ---------------------------------------------------------------------------
logger.debug(
    "InMemoryVectorIndex loaded. Use FAISS, Pinecone, or Weaviate for production.")
