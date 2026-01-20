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
        base = faiss.IndexFlatIP(dim)
        # ID map allows delete/update by integer IDs.
        self.index = faiss.IndexIDMap2(base)
        self._id_map: Dict[str, int] = {}
        self._rev_map: Dict[int, str] = {}
        self._next_id = 1
        self._meta: Dict[str, Dict] = {}
        logger.info("Initialized FaissIndex with dimension=%d", dim)

    def _get_or_create_id(self, sid: str) -> int:
        existing = self._id_map.get(sid)
        if existing is not None:
            return existing
        new_id = self._next_id
        self._next_id += 1
        self._id_map[sid] = new_id
        self._rev_map[new_id] = sid
        return new_id

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

        int_ids = []
        to_remove = []
        for sid in ids:
            if sid in self._id_map:
                to_remove.append(self._id_map[sid])
            int_ids.append(self._get_or_create_id(sid))

        if to_remove:
            removed = self.index.remove_ids(np.asarray(to_remove, dtype="int64"))
            logger.debug("FaissIndex.upsert: removed %d existing ids", int(removed))

        self.index.add_with_ids(arr, np.asarray(int_ids, dtype="int64"))

        metadata = metadata or [{} for _ in ids]
        for i, meta in zip(ids, metadata):
            self._meta[i] = meta or {}

        logger.info(
            "FaissIndex.upsert: added %d vectors; total size=%d",
            len(ids),
            int(self.index.ntotal),
        )

    def query(
        self,
        vector: List[float],
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:
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
            if idx < 0:
                continue
            id_ = self._rev_map.get(int(idx))
            if not id_:
                continue
            meta = self._meta.get(id_, {})
            if filters:
                # Simple AND filter
                if not all(meta.get(fk) == fv for fk, fv in filters.items()):
                    continue
            results.append((id_, float(score)))

        logger.debug("FaissIndex.query: returning %d results", len(results))
        return results

    def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        to_remove = []
        for sid in ids:
            int_id = self._id_map.pop(sid, None)
            if int_id is None:
                continue
            self._rev_map.pop(int_id, None)
            self._meta.pop(sid, None)
            to_remove.append(int_id)

        if to_remove:
            removed = self.index.remove_ids(np.asarray(to_remove, dtype="int64"))
            logger.debug("FaissIndex.delete: removed %d ids", int(removed))
