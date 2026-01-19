"""
FAISS-based VectorIndex implementation for UMA-3.

This module wraps a simple FAISS inner-product index and keeps Python-side
mappings for IDs and metadata.

Coding agent instructions
-------------------------
- For production, consider:
  - Persisting FAISS index to disk.
  - Sharding / multi-index support.
  - Using GPU indices if needed.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import VectorIndex

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    faiss = None
    logger.error("Failed to import faiss: %s", exc)


class FaissIndex(VectorIndex):
    """
    Simple FAISS index using inner product similarity.

    Notes
    -----
    - Keeps all vectors in memory.
    - Stores metadata in an in-memory dict keyed by id.
    """

    def __init__(self, dim: int) -> None:
        if faiss is None:
            raise RuntimeError(
                "faiss is not installed. Install `faiss-cpu` or `faiss-gpu`."
            )
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._ids: List[str] = []
        self._meta: Dict[str, Dict] = {}
        logger.info("Initialized FaissIndex with dimension=%d", dim)

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        if len(ids) != len(vectors):
            raise ValueError("FaissIndex.upsert: ids and vectors length mismatch")

        if not vectors:
            logger.debug("FaissIndex.upsert called with empty vectors; no-op.")
            return

        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"FaissIndex.upsert: expected vectors with dim={self.dim}, got {arr.shape}"
            )

        self.index.add(arr)
        self._ids.extend(ids)

        metadata = metadata or [{} for _ in ids]
        for i, meta in zip(ids, metadata):
            self._meta[i] = meta or {}

        logger.info(
            "FaissIndex.upsert: added %d vectors; total size=%d",
            len(ids),
            len(self._ids),
        )

    def query(
        self,
        vector: List[float],
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        if not self._ids:
            logger.debug("FaissIndex.query: index empty; returning [].")
            return []

        arr = np.asarray([vector], dtype="float32")
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"FaissIndex.query: expected query vector dim={self.dim}, got {arr.shape[1]}"
            )

        scores, idxs = self.index.search(arr, k)
        scores = scores[0]
        idxs = idxs[0]
        results: List[Tuple[str, float]] = []

        for idx, score in zip(idxs, scores):
            if idx < 0 or idx >= len(self._ids):
                continue
            id_ = self._ids[idx]
            meta = self._meta.get(id_, {})
            if filters:
                # Simple AND filter
                if not all(meta.get(fk) == fv for fk, fv in filters.items()):
                    continue
            results.append((id_, float(score)))

        logger.debug("FaissIndex.query: returning %d results", len(results))
        return results