"""
Qdrant-based VectorIndex implementation for UMA.

This adapter implements UMA's `VectorIndex` interface using Qdrant (remote or local).
It is designed to be lean, production-safe, and explicit about failure modes.

Key design points
-----------------
- Creates the target collection if it does not exist (idempotent).
- Uses string point IDs (compatible with UMA IDs).
- Supports exact-match metadata filters (AND semantics) via Qdrant Filter/Must.
- Uses Qdrant cosine distance by default (recommended for embeddings).
- Performs basic input validation and logs important events.

Dependencies
------------
- `qdrant-client` must be installed:
    pip install qdrant-client

Coding best practices
---------------------
- No hidden globals; explicit configuration in `__init__`.
- Proper logging and error handling around network calls.
- Avoids unnecessary complexity (no retries here; handle at higher level if desired).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from uma.adapters.vector.base import VectorIndex

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models as qmodels  # type: ignore
except Exception as exc:  # pragma: no cover
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore
    logger.error("Failed to import qdrant-client: %s", exc)


class QdrantIndex(VectorIndex):
    """
    Qdrant-backed vector index.

    Parameters
    ----------
    dim : int
        Vector dimension.
    collection : str
        Qdrant collection name.
    url : Optional[str]
        Qdrant URL, e.g. "http://localhost:6333". If None, uses local Qdrant if `path` is set.
    api_key : Optional[str]
        Qdrant API key (for Qdrant Cloud or secured deployments).
    path : Optional[str]
        Local Qdrant storage path (embedded/local mode). Use either `url` or `path`.
    prefer_grpc : bool
        Use gRPC if available (often faster). Defaults to False for maximum compatibility.
    timeout_s : Optional[float]
        Client timeout in seconds.
    distance : str
        One of {"cosine","dot","euclid"}. Defaults to "cosine".
    on_disk_payload : bool
        Whether to store payload on disk (Qdrant option); useful for large payloads.
    """

    def __init__(
        self,
        dim: int,
        *,
        collection: str,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        prefer_grpc: bool = False,
        timeout_s: Optional[float] = 10.0,
        distance: str = "cosine",
        on_disk_payload: bool = False,
    ) -> None:
        if QdrantClient is None or qmodels is None:
            raise RuntimeError(
                "qdrant-client is not installed. Install it with `pip install qdrant-client`."
            )

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("QdrantIndex: dim must be a positive integer.")
        if not collection or not isinstance(collection, str):
            raise ValueError("QdrantIndex: collection must be a non-empty string.")

        if url and path:
            raise ValueError("QdrantIndex: provide either `url` or `path`, not both.")

        self.dim = dim
        self.collection = collection

        self._distance = self._parse_distance(distance)

        try:
            # QdrantClient supports either remote `url` or local `path`
            if url:
                self._client = QdrantClient(
                    url=url,
                    api_key=api_key,
                    prefer_grpc=prefer_grpc,
                    timeout=timeout_s,
                )
                logger.info("Initialized QdrantIndex remote client url=%s collection=%s", url, collection)
            else:
                # Local mode (embedded). Path is required for persistence.
                if not path:
                    raise ValueError("QdrantIndex: local mode requires `path`.")
                self._client = QdrantClient(
                    path=path,
                    prefer_grpc=False,  # local mode uses HTTP internally; keep simple
                    timeout=timeout_s,
                )
                logger.info("Initialized QdrantIndex local client path=%s collection=%s", path, collection)
        except Exception as exc:
            logger.exception("QdrantIndex: failed to initialize Qdrant client.")
            raise RuntimeError(f"QdrantIndex: failed to initialize client: {exc}") from exc

        # Ensure collection exists (idempotent)
        self._ensure_collection(on_disk_payload=on_disk_payload)

    # ---------------------------------------------------------------------
    # VectorIndex interface
    # ---------------------------------------------------------------------

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """
        Insert or update points in Qdrant.

        Notes
        -----
        - Uses string IDs as provided.
        - Payload = metadata (exact-match filters depend on payload fields).
        """
        self._validate_ids_vectors(ids, vectors)

        if not vectors:
            logger.debug("QdrantIndex.upsert called with empty vectors; no-op.")
            return

        meta_list = metadata or [{} for _ in ids]
        if len(meta_list) != len(ids):
            raise ValueError("QdrantIndex.upsert: metadata length mismatch with ids.")

        points: List[qmodels.PointStruct] = []
        for pid, vec, meta in zip(ids, vectors, meta_list):
            self._validate_vector(vec)
            payload = dict(meta or {})
            payload.setdefault("uma_id", str(pid))
            qid = self._normalize_id(pid)
            points.append(qmodels.PointStruct(id=qid, vector=vec, payload=payload))

        try:
            self._client.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )
            logger.info("QdrantIndex.upsert: upserted %d points into %s", len(points), self.collection)
        except Exception as exc:
            logger.exception("QdrantIndex.upsert failed collection=%s", self.collection)
            raise RuntimeError(f"QdrantIndex.upsert failed: {exc}") from exc

    def query(
        self,
        vector: List[float],
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Perform a nearest-neighbor search.

        Parameters
        ----------
        vector : List[float]
            Query embedding
        k : int
            Number of results
        filters : Optional[Dict]
            Exact-match AND filter on payload fields (e.g. {"user_id":"u1","owner_scope":"agent"}).

        Returns
        -------
        List[Tuple[id, score]]
            Qdrant returns scores aligned with distance type:
            - cosine/dot: higher is more similar
            - euclid: lower distance; Qdrant still returns a "score" but interpretation depends on backend.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("QdrantIndex.query: k must be a positive integer.")
        self._validate_vector(vector)

        qfilter = self._build_filter(filters)

        try:
            hits = self._client.search(
                collection_name=self.collection,
                query_vector=vector,
                query_filter=qfilter,
                limit=k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.exception("QdrantIndex.query failed collection=%s", self.collection)
            raise RuntimeError(f"QdrantIndex.query failed: {exc}") from exc

        results: List[Tuple[str, float]] = []
        for h in hits or []:
            payload = getattr(h, "payload", None) or {}
            uma_id = payload.get("uma_id") if isinstance(payload, dict) else None
            if uma_id:
                results.append((str(uma_id), float(h.score)))
            else:
                results.append((str(h.id), float(h.score)))

        logger.debug("QdrantIndex.query: returning %d results (k=%d) filters=%s", len(results), k, filters)
        return results

    def delete(self, ids: List[str]) -> None:
        """Delete points from Qdrant by ID."""
        if not ids:
            return

        try:
            selector = qmodels.PointIdsList(points=[self._normalize_id(i) for i in ids])
            self._client.delete(
                collection_name=self.collection,
                points_selector=selector,
                wait=True,
            )
            logger.info("QdrantIndex.delete: deleted %d points from %s", len(ids), self.collection)
        except Exception as exc:
            logger.exception("QdrantIndex.delete failed collection=%s", self.collection)
            raise RuntimeError(f"QdrantIndex.delete failed: {exc}") from exc

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _ensure_collection(self, *, on_disk_payload: bool) -> None:
        """Create the collection if missing, otherwise no-op."""
        try:
            exists = self._client.collection_exists(self.collection)
        except Exception as exc:
            logger.exception("QdrantIndex: failed to check collection existence.")
            raise RuntimeError(f"QdrantIndex: cannot check collection existence: {exc}") from exc

        if exists:
            logger.debug("QdrantIndex: collection exists: %s", self.collection)
            return

        try:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.dim,
                    distance=self._distance,
                    on_disk=on_disk_payload,
                ),
            )
            logger.info(
                "QdrantIndex: created collection=%s dim=%d distance=%s on_disk_payload=%s",
                self.collection,
                self.dim,
                self._distance,
                on_disk_payload,
            )
        except Exception as exc:
            logger.exception("QdrantIndex: failed to create collection=%s", self.collection)
            raise RuntimeError(f"QdrantIndex: cannot create collection: {exc}") from exc

    def _build_filter(self, filters: Optional[Dict]) -> Optional[qmodels.Filter]:
        """
        Build a Qdrant Filter for exact-match AND conditions.

        UMA's `VectorIndex.query(..., filters={...})` is defined as a simple AND filter.
        """
        if not filters:
            return None
        if not isinstance(filters, dict):
            raise ValueError("QdrantIndex.query: filters must be a dict or None.")

        must: List[qmodels.FieldCondition] = []
        for key, val in filters.items():
            if key is None or key == "":
                continue
            # Exact match: keyword match
            must.append(
                qmodels.FieldCondition(
                    key=str(key),
                    match=qmodels.MatchValue(value=val),
                )
            )

        if not must:
            return None

        return qmodels.Filter(must=must)

    @staticmethod
    def _normalize_id(pid: Any):
        """
        Qdrant IDs must be uint or UUID. Use UUID5 for stable mapping of strings.
        """
        if isinstance(pid, int):
            return pid
        pid_str = str(pid)
        try:
            return str(uuid.UUID(pid_str))
        except Exception:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, pid_str))

    def _validate_ids_vectors(self, ids: List[str], vectors: List[List[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("QdrantIndex.upsert: ids and vectors length mismatch.")
        for sid in ids:
            if not isinstance(sid, str) or not sid.strip():
                raise ValueError("QdrantIndex.upsert: all ids must be non-empty strings.")

    def _validate_vector(self, vec: List[float]) -> None:
        if not isinstance(vec, list) or not vec:
            raise ValueError("QdrantIndex: vector must be a non-empty list of floats.")
        if len(vec) != self.dim:
            raise ValueError(f"QdrantIndex: expected vector dim={self.dim}, got {len(vec)}.")
        # Validate numeric values (avoid NaNs / non-numbers)
        for i, v in enumerate(vec):
            if not isinstance(v, (int, float)):
                raise ValueError(f"QdrantIndex: vector[{i}] must be numeric, got {type(v)}.")

    def _parse_distance(self, distance: str) -> Any:
        d = (distance or "").strip().lower()
        if d in ("cosine", "cos"):
            return qmodels.Distance.COSINE
        if d in ("dot", "ip", "inner_product"):
            return qmodels.Distance.DOT
        if d in ("euclid", "l2"):
            return qmodels.Distance.EUCLID
        raise ValueError(
            "QdrantIndex: unsupported distance. Use one of {'cosine','dot','euclid'}."
        )
