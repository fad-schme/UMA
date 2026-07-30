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

Implements the VectorIndex API so it can plug into UMA seamlessly.
"""


from __future__ import annotations

import logging
import math
from typing import Any, Optional

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
        self.dimension = dim   # UMA expects this attribute
        self.index = self      # UMA expects vector_index.index.* API

        # C1: storage now keeps isolation fields as first-class entries.
        # extra metadata stays in a parallel dict for non-isolation keys.
        self._vectors: dict[str, list[float]] = {}
        self._scopes: dict[str, tuple[str, str, str]] = {}  # id -> (tenant_id, owner_type, owner_id)
        self._extra: dict[str, dict[str, Any]] = {}

        logger.info(
            "InMemoryVectorIndex initialized (dim=%d). Intended for development, CI, or fallback only.",
            dim,
        )

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) + 1e-8
        nb = math.sqrt(sum(y * y for y in b)) + 1e-8
        return dot / (na * nb)

    # ---------------------------------------------------------
    # API Methods
    # ---------------------------------------------------------

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        *,
        tenant_ids: list[str],
        owner_types: list[str],
        owner_ids: list[str],
        extra_metadata: Optional[list[dict]] = None,
    ) -> None:
        """Insert or update vectors in the in-memory store. Intended for testing and CI only."""
        n = len(ids)
        if len(tenant_ids) != n:
            raise ValueError(
                f"InMemoryVectorIndex.upsert: tenant_ids length ({len(tenant_ids)}) "
                f"does not match ids length ({n})."
            )
        if len(owner_types) != n:
            raise ValueError(
                f"InMemoryVectorIndex.upsert: owner_types length ({len(owner_types)}) "
                f"does not match ids length ({n})."
            )
        if len(owner_ids) != n:
            raise ValueError(
                f"InMemoryVectorIndex.upsert: owner_ids length ({len(owner_ids)}) "
                f"does not match ids length ({n})."
            )
        extra_list = extra_metadata or [{} for _ in ids]
        if len(extra_list) != n:
            raise ValueError(
                f"InMemoryVectorIndex.upsert: extra_metadata length ({len(extra_list)}) "
                f"does not match ids length ({n})."
            )

        # C1: validate ALL rows before any state mutation. Per-row writes
        # interleaved with per-row validation would leak partial state on
        # a bad input — earlier rows in the batch would be visible but
        # later ones would not, leaving an inconsistent index.
        prepared: list[tuple] = []  # (sid, vec, (tid, ot, oid), extras_dict)
        for sid, vec, tid, ot, oid, extra in zip(
            ids, vectors, tenant_ids, owner_types, owner_ids, extra_list,
        ):
            if len(vec) != self.dim:
                raise ValueError(
                    f"InMemoryVectorIndex: expected vector dim={self.dim}, got={len(vec)}"
                )
            if not isinstance(tid, str) or not tid.strip():
                raise ValueError(
                    f"InMemoryVectorIndex.upsert: tenant_id must be a non-empty string (id={sid!r})."
                )
            if not isinstance(ot, str) or not ot.strip():
                raise ValueError(
                    f"InMemoryVectorIndex.upsert: owner_type must be a non-empty string (id={sid!r})."
                )
            if not isinstance(oid, str) or not oid.strip():
                raise ValueError(
                    f"InMemoryVectorIndex.upsert: owner_id must be a non-empty string (id={sid!r})."
                )
            extra = extra or {}
            if not isinstance(extra, dict):
                raise ValueError("InMemoryVectorIndex.upsert: extra_metadata items must be dicts.")
            for reserved in ("tenant_id", "owner_type", "owner_id"):
                if reserved in extra:
                    raise ValueError(
                        f"InMemoryVectorIndex.upsert: extra_metadata must not contain "
                        f"reserved isolation key {reserved!r}; pass via the explicit "
                        f"parallel-list parameter instead (id={sid!r})."
                    )
            prepared.append((sid, vec, (tid.strip(), ot.strip(), oid.strip()), dict(extra)))

        # All validation passed. Commit.
        for sid, vec, scope, extra in prepared:
            self._vectors[sid] = vec
            self._scopes[sid] = scope
            self._extra[sid] = extra

    def delete(self, ids: list[str]) -> None:
        """Remove vectors from the in-memory store."""
        for _id in ids:
            self._vectors.pop(_id, None)
            self._scopes.pop(_id, None)
            self._extra.pop(_id, None)

    def query(
        self,
        vector: list[float],
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        extra_filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[str, float]]:
        """
        Returns:
            List of (id, score) pairs sorted by cosine similarity DESC,
            restricted to the (tenant_id, owner_type, owner_id) scope.
        """
        if len(vector) != self.dim:
            raise ValueError(
                f"InMemoryVectorIndex.query: expected dim={self.dim}, got={len(vector)}"
            )
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("InMemoryVectorIndex.query: tenant_id must be a non-empty string.")
        if not isinstance(owner_type, str) or not owner_type.strip():
            raise ValueError("InMemoryVectorIndex.query: owner_type must be a non-empty string.")
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("InMemoryVectorIndex.query: owner_id must be a non-empty string.")

        scope_key = (tenant_id.strip(), owner_type.strip(), owner_id.strip())
        results = []

        for key, vec in self._vectors.items():
            # C1: isolation filter is the first thing checked. No
            # cross-scope row can reach the cosine computation or the
            # extra_filters check.
            if self._scopes.get(key) != scope_key:
                continue

            try:
                if extra_filters:
                    meta = self._extra.get(key, {})
                    if any(meta.get(fk) != fv for fk, fv in extra_filters.items()):
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
        Factory method used by UMA when FAISSIndex import fails.
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
