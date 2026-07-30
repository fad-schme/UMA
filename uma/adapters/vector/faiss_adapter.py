"""
FAISS-based VectorIndex implementation for UMA.

This module wraps a simple FAISS inner-product index and keeps Python-side
mappings for IDs and metadata.

Isolation note (C1)
-------------------
FAISS does not support metadata predicates pushed into the search. The
isolation filter for (tenant_id, owner_type, owner_id) is applied in
Python AFTER FAISS returns the top-k candidates. This means under heavy
cross-tenant load FAISS can still suffer recall loss — the LanceDB
adapter is preferred for multi-tenant deployments because it pushes the
filter into the database engine.

This adapter compensates partially by oversampling FAISS (searching for
k * `_oversample_multiplier` candidates internally) so a moderate
cross-tenant occupancy of the top-k can still surface enough scope-
local candidates. This is a heuristic, not a guarantee.

Coding agent instructions
-------------------------
- For production multi-tenant deployments, use the LanceDB adapter.
- For single-tenant or smaller deployments, FAISS is fine.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

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
    - Isolation (tenant_id, owner_type, owner_id) is stored alongside
      each id and applied as a Python post-filter; FAISS itself does
      not support pushed-down predicates.
    """

    # How many extra candidates to fetch internally so the post-filter
    # has headroom to find scope-local results. 4× the requested k is a
    # reasonable balance — enough to survive moderate cross-tenant
    # occupancy without bloating per-query cost.
    _oversample_multiplier: int = 4

    def __init__(self, dim: int) -> None:
        if faiss is None:
            raise RuntimeError(
                "faiss is not installed. Install `faiss-cpu` or `faiss-gpu`."
            )
        self.dim = dim
        base = faiss.IndexFlatIP(dim)
        # ID map allows delete/update by integer IDs.
        self.index = faiss.IndexIDMap2(base)
        self._id_map: dict[str, int] = {}
        self._rev_map: dict[int, str] = {}
        self._next_id = 1
        # C1: isolation kept separate from extras so the filter is
        # explicit and side-channel-free.
        self._scopes: dict[str, tuple[str, str, str]] = {}
        self._extra: dict[str, dict[str, Any]] = {}
        logger.debug("Initialized FaissIndex with dimension=%d", dim)

    def _get_or_create_id(self, sid: str) -> int:
        existing = self._id_map.get(sid)
        if existing is not None:
            return existing
        new_id = self._next_id
        self._next_id += 1
        self._id_map[sid] = new_id
        self._rev_map[new_id] = sid
        return new_id

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms

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
        """
        Insert or update vectors in the FAISS index.

        All rows are validated before any state mutation (C1 atomicity contract).
        Isolation fields are stored in a parallel ``_scopes`` dict and applied as a
        Python post-filter after FAISS returns candidates. For multi-tenant deployments
        prefer the LanceDB adapter, which pushes the filter before the k-cap.
        """
        if len(ids) != len(vectors):
            raise ValueError("FaissIndex.upsert: ids and vectors length mismatch")

        if not vectors:
            logger.debug("FaissIndex.upsert called with empty vectors; no-op.")
            return

        n = len(ids)
        if len(tenant_ids) != n:
            raise ValueError(
                f"FaissIndex.upsert: tenant_ids length ({len(tenant_ids)}) "
                f"does not match ids length ({n})."
            )
        if len(owner_types) != n:
            raise ValueError(
                f"FaissIndex.upsert: owner_types length ({len(owner_types)}) "
                f"does not match ids length ({n})."
            )
        if len(owner_ids) != n:
            raise ValueError(
                f"FaissIndex.upsert: owner_ids length ({len(owner_ids)}) "
                f"does not match ids length ({n})."
            )
        extra_list = extra_metadata or [{} for _ in ids]
        if len(extra_list) != n:
            raise ValueError(
                f"FaissIndex.upsert: extra_metadata length ({len(extra_list)}) "
                f"does not match ids length ({n})."
            )

        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"FaissIndex.upsert: expected vectors with dim={self.dim}, got {arr.shape}"
            )
        arr = self._normalize(arr)

        # C1: validate ALL per-row isolation fields and extra_metadata
        # contents BEFORE we touch the FAISS index. Validating after
        # `add_with_ids` would leave the index in a corrupted state on a
        # bad input — vector present, scope dict missing — which would
        # then surface as cross-scope leaks or silent retrieval misses.
        prepared_scopes: list[tuple[str, str, str]] = []
        prepared_extras: list[dict[str, Any]] = []
        for sid, tid, ot, oid, extra in zip(
            ids, tenant_ids, owner_types, owner_ids, extra_list,
        ):
            if not isinstance(tid, str) or not tid.strip():
                raise ValueError(
                    f"FaissIndex.upsert: tenant_id must be a non-empty string (id={sid!r})."
                )
            if not isinstance(ot, str) or not ot.strip():
                raise ValueError(
                    f"FaissIndex.upsert: owner_type must be a non-empty string (id={sid!r})."
                )
            if not isinstance(oid, str) or not oid.strip():
                raise ValueError(
                    f"FaissIndex.upsert: owner_id must be a non-empty string (id={sid!r})."
                )
            extra = extra or {}
            if not isinstance(extra, dict):
                raise ValueError("FaissIndex.upsert: extra_metadata items must be dicts.")
            for reserved in ("tenant_id", "owner_type", "owner_id"):
                if reserved in extra:
                    raise ValueError(
                        f"FaissIndex.upsert: extra_metadata must not contain "
                        f"reserved isolation key {reserved!r}; pass via the "
                        f"explicit parallel-list parameter instead (id={sid!r})."
                    )
            prepared_scopes.append((tid.strip(), ot.strip(), oid.strip()))
            prepared_extras.append(dict(extra))

        # All validation passed. Now safe to mutate FAISS + scope/extra dicts.
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

        for sid, scope, extra in zip(ids, prepared_scopes, prepared_extras):
            self._scopes[sid] = scope
            self._extra[sid] = extra

        logger.info(
            "FaissIndex.upsert: added %d vectors; total size=%d",
            len(ids),
            int(self.index.ntotal),
        )

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
        Nearest-neighbour search scoped to ``(tenant_id, owner_type, owner_id)``.

        FAISS does not support pushed-down predicates, so this adapter oversamples by
        ``_oversample_multiplier`` (default 4×) and post-filters in Python. Under heavy
        cross-tenant load recall may degrade — this is a documented heuristic, not a
        guarantee. Use LanceDB for production multi-tenant deployments.
        """
        if self.index.ntotal == 0:
            logger.debug("FaissIndex.query: index empty; returning [].")
            return []

        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("FaissIndex.query: tenant_id must be a non-empty string.")
        if not isinstance(owner_type, str) or not owner_type.strip():
            raise ValueError("FaissIndex.query: owner_type must be a non-empty string.")
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("FaissIndex.query: owner_id must be a non-empty string.")

        arr = np.asarray([vector], dtype="float32")
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"FaissIndex.query: expected query vector dim={self.dim}, got {arr.shape[1]}"
            )
        arr = self._normalize(arr)

        # C1: oversample candidates from FAISS so the post-filter has
        # headroom. FAISS doesn't support pushed-down predicates, so
        # this is the best we can do without changing the backend.
        oversample_k = min(
            int(self.index.ntotal),
            max(k * self._oversample_multiplier, k),
        )
        scores, idxs = self.index.search(arr, oversample_k)
        scores = scores[0]
        idxs = idxs[0]
        scope_key = (tenant_id.strip(), owner_type.strip(), owner_id.strip())
        results: list[tuple[str, float]] = []

        for idx, score in zip(idxs, scores):
            if idx < 0:
                continue
            id_ = self._rev_map.get(int(idx))
            if not id_:
                continue
            # C1: isolation filter applied FIRST. Cross-scope rows
            # cannot reach the extra_filters step.
            if self._scopes.get(id_) != scope_key:
                continue
            if extra_filters:
                meta = self._extra.get(id_, {})
                if not all(meta.get(fk) == fv for fk, fv in extra_filters.items()):
                    continue
            results.append((id_, float(score)))
            if len(results) >= k:
                break

        logger.debug("FaissIndex.query: returning %d results", len(results))
        return results

    def delete(self, ids: list[str]) -> None:
        """Remove vectors from the FAISS index and clear their scope and metadata entries."""
        if not ids:
            return
        to_remove = []
        for sid in ids:
            int_id = self._id_map.pop(sid, None)
            if int_id is None:
                continue
            self._rev_map.pop(int_id, None)
            self._scopes.pop(sid, None)
            self._extra.pop(sid, None)
            to_remove.append(int_id)

        if to_remove:
            removed = self.index.remove_ids(np.asarray(to_remove, dtype="int64"))
            logger.debug("FaissIndex.delete: removed %d ids", int(removed))
