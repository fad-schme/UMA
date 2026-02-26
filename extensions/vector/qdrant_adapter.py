"""
Qdrant-based VectorIndex implementation for UMA (production-ready, Qdrant-native hybrid).

This adapter implements UMA's `VectorIndex` interface using Qdrant and performs
the best available search strategy for Qdrant hybrid retrieval:

- Dense semantic search over a named dense vector (default: "dense")
- Sparse BM25 keyword search over a named sparse vector (default: "sparse")
- Reciprocal Rank Fusion (RRF) to combine both result sets into one ranked list

Interface constraints & reserved keys
-------------------------------------
UMA's VectorIndex.query signature provides only a dense query vector. For BM25 hybrid,
the adapter also needs raw query text. We pass it through `filters` using a reserved key:

- filters["__query_text"] : str  (used for BM25 sparse query)
- metadata["__text"]      : str  (used at upsert time to build BM25 sparse vectors)

These reserved keys are removed from payload filters and payload storage. This keeps
Qdrant payload minimal (no chunk text duplication).

Dependencies
------------
- qdrant-client
- fastembed (optional; enables BM25 sparse vectors)

    pip install qdrant-client
    pip install fastembed

Design goals
------------
- UMA VectorIndex compliance (upsert/query/delete)
- Hybrid retrieval when possible, dense-only fallback when query text absent
- Server-side RRF when available; safe Python-side RRF fallback otherwise
- Minimal payload storage
- Clear logging and explicit error handling

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

try:
    from fastembed import SparseTextEmbedding  # type: ignore
except Exception as exc:  # pragma: no cover
    SparseTextEmbedding = None  # type: ignore
    logger.warning("fastembed not installed; BM25 sparse vectors will be unavailable: %s", exc)


class QdrantIndex(VectorIndex):
    """
    Qdrant-backed hybrid index (dense + BM25 sparse).

    Parameters
    ----------
    dim:
        Dense vector dimension.
    collection:
        Qdrant collection name.
    url:
        Remote Qdrant URL (e.g., "http://localhost:6333"). Use either `url` or `path`.
    api_key:
        Qdrant API key (Qdrant Cloud / secured deployments).
    path:
        Local Qdrant storage path (embedded/local mode). Use either `url` or `path`.
    prefer_grpc:
        Prefer gRPC (often faster). Requires server support and client configuration.
    timeout_s:
        Client timeout in seconds.
    distance:
        Dense vector distance: "cosine", "dot", or "euclid". Default "cosine".
    on_disk_payload:
        Whether to store payload on disk (Qdrant option).
    dense_vector_name:
        Name for dense vector in a multi-vector collection.
    sparse_vector_name:
        Name for sparse BM25 vector in a multi-vector collection.
    bm25_enabled:
        Whether to enable BM25 sparse vectors (default True). If fastembed is not installed, BM25 will be auto-disabled.
    bm25_model_name:
        FastEmbed BM25 model name (default "Qdrant/bm25").
    metadata_text_key:
        Reserved metadata key containing document text (default "__text").
    query_text_filter_key:
        Reserved filters key containing query text (default "__query_text").
    prefetch_multiplier:
        Prefetch size multiplier relative to k when fusing (default 4).
    min_prefetch:
        Minimum per-modality prefetch size for fusion (default 50).
    rrf_k:
        Reciprocal Rank Fusion constant k (default 60 in classic IR; smaller emphasizes top ranks).
        We default to 60 for stability.
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
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
        bm25_enabled: bool = True,
        bm25_model_name: str = "Qdrant/bm25",
        metadata_text_key: str = "__text",
        query_text_filter_key: str = "__query_text",
        prefetch_multiplier: int = 4,
        min_prefetch: int = 50,
        rrf_k: int = 60,
        **_: Any,
    ) -> None:
        if QdrantClient is None or qmodels is None:
            raise RuntimeError("qdrant-client is not installed. Install it with `pip install qdrant-client`.")
        if bm25_enabled and SparseTextEmbedding is None:
            logger.warning(
                "fastembed is not installed; BM25 hybrid search will be disabled and QdrantIndex will run dense-only. "
                "Install with `pip install fastembed` to enable BM25 sparse vectors."
            )
            bm25_enabled = False

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("QdrantIndex: dim must be a positive integer.")
        if not collection or not isinstance(collection, str):
            raise ValueError("QdrantIndex: collection must be a non-empty string.")
        if url and path:
            raise ValueError("QdrantIndex: provide either `url` or `path`, not both.")
        if not dense_vector_name or not sparse_vector_name:
            raise ValueError("QdrantIndex: dense_vector_name and sparse_vector_name must be non-empty.")
        if dense_vector_name == sparse_vector_name:
            raise ValueError("QdrantIndex: dense_vector_name and sparse_vector_name must be different.")
        if not isinstance(prefetch_multiplier, int) or prefetch_multiplier < 1:
            raise ValueError("QdrantIndex: prefetch_multiplier must be integer >= 1.")
        if not isinstance(min_prefetch, int) or min_prefetch < 1:
            raise ValueError("QdrantIndex: min_prefetch must be integer >= 1.")
        if not isinstance(rrf_k, int) or rrf_k < 1:
            raise ValueError("QdrantIndex: rrf_k must be integer >= 1.")

        self.dim = dim
        self.collection = collection
        self._distance = self._parse_distance(distance)
        self._dense_name = dense_vector_name
        self._sparse_name = sparse_vector_name
        self._metadata_text_key = metadata_text_key
        self._query_text_filter_key = query_text_filter_key
        self._prefetch_multiplier = prefetch_multiplier
        self._min_prefetch = min_prefetch
        self._rrf_k = rrf_k

        # Sparse BM25 embedding generator (optional)
        self._bm25_enabled = bool(bm25_enabled)
        self._bm25 = SparseTextEmbedding(model_name=bm25_model_name) if self._bm25_enabled else None

        try:
            if url:
                self._client = QdrantClient(
                    url=url,
                    api_key=api_key,
                    prefer_grpc=prefer_grpc,
                    timeout=timeout_s,
                )
                logger.info("Initialized QdrantIndex remote client url=%s collection=%s", url, collection)
            else:
                if not path:
                    raise ValueError("QdrantIndex: local mode requires `path`.")
                self._client = QdrantClient(
                    path=path,
                    prefer_grpc=False,
                    timeout=timeout_s,
                )
                logger.info("Initialized QdrantIndex local client path=%s collection=%s", path, collection)
        except Exception as exc:
            logger.exception("QdrantIndex: failed to initialize Qdrant client.")
            raise RuntimeError(f"QdrantIndex: failed to initialize client: {exc}") from exc

        self._ensure_collection(on_disk_payload=on_disk_payload)
        self._validate_collection_schema()

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
        Insert or update points in Qdrant with BOTH dense and BM25 sparse vectors.

        Requirements
        -----------
        - `vectors` must be dense embeddings of length `dim`.
        - `metadata[i][metadata_text_key]` MAY be provided as a non-empty string
          so BM25 sparse vectors can be computed. If omitted, sparse vectors are
          stored as empty (dense-only behavior).

        Reserved keys behavior
        ----------------------
        - metadata_text_key (default "__text") is used for sparse embedding generation
          and then REMOVED from payload (not stored in Qdrant payload).
        - "uma_id" is always stored in payload for stable id mapping.
        """
        self._validate_ids_vectors(ids, vectors)

        if not vectors:
            logger.debug("QdrantIndex.upsert called with empty vectors; no-op.")
            return

        meta_list = metadata or [{} for _ in ids]
        if len(meta_list) != len(ids):
            raise ValueError("QdrantIndex.upsert: metadata length mismatch with ids.")

        payloads: List[Dict[str, Any]] = []
        sparse_vectors: List[qmodels.SparseVector] = []
        texts_for_bm25: List[str] = []
        text_positions: List[int] = []

        for i, meta in enumerate(meta_list):
            if not isinstance(meta, dict):
                raise ValueError(f"QdrantIndex.upsert: metadata[{i}] must be a dict.")

            txt = meta.get(self._metadata_text_key)
            if isinstance(txt, str) and txt.strip():
                texts_for_bm25.append(txt.strip())
                text_positions.append(i)

            payload = dict(meta)
            # Never store full text in payload (SQL is canonical for chunk text).
            payload.pop(self._metadata_text_key, None)
            payload.pop("text", None)
            payload.setdefault("uma_id", str(ids[i]))
            payloads.append(payload)
            sparse_vectors.append(qmodels.SparseVector(indices=[], values=[]))

        if texts_for_bm25 and self._bm25 is not None:
            try:
                sparse_embeddings = list(self._bm25.embed(texts_for_bm25))
            except Exception as exc:
                logger.exception("QdrantIndex.upsert: BM25 sparse embedding generation failed.")
                raise RuntimeError(f"QdrantIndex.upsert: BM25 sparse embedding failed: {exc}") from exc

            if len(sparse_embeddings) != len(texts_for_bm25):
                raise RuntimeError(
                    "QdrantIndex.upsert: BM25 embed returned "
                    f"{len(sparse_embeddings)} vectors for {len(texts_for_bm25)} texts."
                )

            for sparse_emb, pos in zip(sparse_embeddings, text_positions):
                sparse_vectors[pos] = self._to_sparse_vector(sparse_emb)
        elif texts_for_bm25 and self._bm25 is None:
            logger.debug(
                "QdrantIndex.upsert: text provided for BM25 but BM25 is disabled; storing empty sparse vectors (dense-only mode)."
            )

        points: List[qmodels.PointStruct] = []
        for pid, dense_vec, payload, sparse_vec in zip(ids, vectors, payloads, sparse_vectors):
            self._validate_dense_vector(dense_vec)

            points.append(
                qmodels.PointStruct(
                    id=self._normalize_id(pid),
                    vector={self._dense_name: dense_vec, self._sparse_name: sparse_vec},
                    payload=payload,
                )
            )

        try:
            self._client.upsert(collection_name=self.collection, points=points, wait=True)
            logger.info(
                "QdrantIndex.upsert: upserted %d points into %s (dense=%s sparse=%s)",
                len(points),
                self.collection,
                self._dense_name,
                self._sparse_name,
            )
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
        Perform best-available search for Qdrant:

        - If filters include query text under query_text_filter_key (default "__query_text"):
            run hybrid dense + BM25 sparse + RRF fusion (server-side if supported, else Python-side).
        - Otherwise:
            run dense-only search.

        Filters
        -------
        Payload filters are exact-match AND conditions on payload fields.
        The reserved query-text key is removed before building payload filters.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("QdrantIndex.query: k must be a positive integer.")
        self._validate_dense_vector(vector)

        query_text: Optional[str] = None
        payload_filters: Optional[Dict[str, Any]] = None

        if filters is not None:
            if not isinstance(filters, dict):
                raise ValueError("QdrantIndex.query: filters must be a dict or None.")
            payload_filters = dict(filters)
            qt = payload_filters.pop(self._query_text_filter_key, None)
            if qt is not None:
                if not isinstance(qt, str) or not qt.strip():
                    raise ValueError(f"QdrantIndex.query: '{self._query_text_filter_key}' must be a non-empty string.")
                query_text = qt.strip()

        qfilter = self._build_filter(payload_filters)
        prefetch_k = max(self._prefetch_multiplier * k, self._min_prefetch)

        if query_text is None:
            return self._dense_search(vector=vector, k=k, qfilter=qfilter)

        if self._bm25 is None:
            logger.debug(
                "QdrantIndex.query: query text provided but BM25 is disabled; falling back to dense-only."
            )
            return self._dense_search(vector=vector, k=k, qfilter=qfilter)

        # Build sparse query vector (BM25). If it fails, fallback to dense-only.
        try:
            sparse_query_emb = next(iter(self._bm25.embed([query_text])))
            sparse_query_vec = self._to_sparse_vector(sparse_query_emb)
        except Exception as exc:
            logger.exception("QdrantIndex.query: BM25 sparse query embedding failed. Falling back to dense-only.",
                exc,)
            return self._dense_search(vector=vector, k=k, qfilter=qfilter)

        # 1) Try server-side RRF fusion (best case).
        try:
            results = self._hybrid_search_server_rrf(
                dense_vector=vector,
                sparse_vector=sparse_query_vec,
                k=k,
                prefetch_k=prefetch_k,
                qfilter=qfilter,
            )
            return results
        except Exception as exc:
            logger.warning(
                "QdrantIndex.query: server-side hybrid RRF unavailable/failed (%s). Falling back to client-side RRF.",
                exc,
            )

        # 2) Robust fallback: run both searches independently and RRF fuse in Python.
        dense = self._dense_search(vector=vector, k=prefetch_k, qfilter=qfilter)
        sparse = self._sparse_search(sparse_vector=sparse_query_vec, k=prefetch_k, qfilter=qfilter)
        fused = self._rrf_fuse(dense=dense, sparse=sparse, k=k)

        logger.debug(
            "QdrantIndex.query: returning %d results (k=%d) hybrid client-side RRF filters=%s",
            len(fused),
            k,
            payload_filters,
        )
        return fused

    def delete(self, ids: List[str]) -> None:
        """Delete points from Qdrant by ID."""
        if not ids:
            return

        try:
            selector = qmodels.PointIdsList(points=[self._normalize_id(i) for i in ids])
            self._client.delete(collection_name=self.collection, points_selector=selector, wait=True)
            logger.info("QdrantIndex.delete: deleted %d points from %s", len(ids), self.collection)
        except Exception as exc:
            logger.exception("QdrantIndex.delete failed collection=%s", self.collection)
            raise RuntimeError(f"QdrantIndex.delete failed: {exc}") from exc

    # ---------------------------------------------------------------------
    # Hybrid/dense/sparse query implementations
    # ---------------------------------------------------------------------

    def _hybrid_search_server_rrf(
        self,
        *,
        dense_vector: List[float],
        sparse_vector: qmodels.SparseVector,
        k: int,
        prefetch_k: int,
        qfilter: Optional[qmodels.Filter],
    ) -> List[Tuple[str, float]]:
        """
        Server-side hybrid fusion via query_points + prefetch + RrfQuery.
        Raises if the server/client does not support required APIs.
        """
        if not hasattr(self._client, "query_points"):
            raise RuntimeError("qdrant-client does not support query_points.")

        # These model classes may not exist depending on client version.
        if not hasattr(qmodels, "Prefetch") or not hasattr(qmodels, "RrfQuery") or not hasattr(qmodels, "Rrf"):
            raise RuntimeError("qdrant-client models missing Prefetch/RrfQuery/Rrf.")

        resp = self._client.query_points(
            collection_name=self.collection,
            prefetch=[
                qmodels.Prefetch(query=sparse_vector, using=self._sparse_name, limit=prefetch_k),
                qmodels.Prefetch(query=dense_vector, using=self._dense_name, limit=prefetch_k),
            ],
            query=qmodels.RrfQuery(rrf=qmodels.Rrf(k=self._rrf_k)),
            query_filter=qfilter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        points = getattr(resp, "points", None)
        if points is None:
            points = resp

        return self._extract_id_score(points)

    def _dense_search(
        self,
        *,
        vector: List[float],
        k: int,
        qfilter: Optional[qmodels.Filter],
    ) -> List[Tuple[str, float]]:
        """
        Dense-only search, using query_points if available, else falling back to search/search_points.
        """
        # Prefer query_points for named vectors
        if hasattr(self._client, "query_points"):
            try:
                resp = self._client.query_points(
                    collection_name=self.collection,
                    query=vector,
                    using=self._dense_name,
                    query_filter=qfilter,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                )
                points = getattr(resp, "points", None)
                if points is None:
                    points = resp
                return self._extract_id_score(points)
            except Exception as exc:
                logger.warning("QdrantIndex._dense_search: query_points failed (%s). Trying search fallback.", exc)

        # Fallback: search/search_points with named vectors may not work across all versions,
        # but we try a conservative call.
        hits = self._search_points_compat(query_vector=vector, query_filter=qfilter, limit=k)
        return self._extract_id_score(hits)

    def _sparse_search(
        self,
        *,
        sparse_vector: qmodels.SparseVector,
        k: int,
        qfilter: Optional[qmodels.Filter],
    ) -> List[Tuple[str, float]]:
        """
        Sparse-only search over BM25 sparse vector.
        Requires query_points (recommended). If unavailable, raises.
        """
        if not hasattr(self._client, "query_points"):
            raise RuntimeError("qdrant-client does not support query_points required for sparse search.")

        resp = self._client.query_points(
            collection_name=self.collection,
            query=sparse_vector,
            using=self._sparse_name,
            query_filter=qfilter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(resp, "points", None)
        if points is None:
            points = resp
        return self._extract_id_score(points)

    @staticmethod
    def _rrf_fuse_lists(
        *,
        dense: List[Tuple[str, float]],
        sparse: List[Tuple[str, float]],
        k: int,
        rrf_k: int = 60,
    ) -> List[Tuple[str, float]]:
        """
        _rrf_fuse_lists: Reciprocal Rank Fusion over two ranked lists.
        Returns (id, fused_score) sorted descending, truncated to k.

        We ignore modality raw scores and fuse by rank position (classic RRF).
        """
        # Use provided rrf_k for stable behavior.
        # Note: rrf_k here is the constant in 1/(rrf_k + rank).
        scores: Dict[str, float] = {}

        for rank, (pid, _) in enumerate(dense, start=1):
            scores[pid] = scores.get(pid, 0.0) + (1.0 / (rrf_k + rank))
        for rank, (pid, _) in enumerate(sparse, start=1):
            scores[pid] = scores.get(pid, 0.0) + (1.0 / (rrf_k + rank))

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:k]

    def _rrf_fuse(self, *, dense: List[Tuple[str, float]], sparse: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
        return self.__class__._rrf_fuse_lists(dense=dense, sparse=sparse, k=k, rrf_k=self._rrf_k)

    # ---------------------------------------------------------------------
    # Collection/schema helpers
    # ---------------------------------------------------------------------

    def _ensure_collection(self, *, on_disk_payload: bool) -> None:
        """Create the collection if missing, otherwise no-op (dense + sparse configs)."""
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
                vectors_config={
                    self._dense_name: qmodels.VectorParams(size=self.dim, distance=self._distance),
                },
                sparse_vectors_config={
                    self._sparse_name: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF),
                },
                on_disk_payload=on_disk_payload,
            )
            logger.info(
                "QdrantIndex: created collection=%s dense_name=%s sparse_name=%s dim=%d distance=%s on_disk_payload=%s",
                self.collection,
                self._dense_name,
                self._sparse_name,
                self.dim,
                self._distance,
                on_disk_payload,
            )
        except Exception as exc:
            logger.exception("QdrantIndex: failed to create collection=%s", self.collection)
            raise RuntimeError(f"QdrantIndex: cannot create collection: {exc}") from exc

    def _validate_collection_schema(self) -> None:
        """
        Validate that the collection has the required named dense and sparse vector configs.
        Fail fast with a clear error if the collection exists but is misconfigured.
        """
        try:
            info = self._client.get_collection(self.collection)
        except Exception as exc:
            logger.exception("QdrantIndex: failed to fetch collection info=%s", self.collection)
            raise RuntimeError(f"QdrantIndex: cannot read collection info: {exc}") from exc

        config = getattr(info, "config", None)
        params = getattr(config, "params", None) if config is not None else None

        vectors = getattr(params, "vectors", None) if params is not None else None
        sparse_vectors = getattr(params, "sparse_vectors", None) if params is not None else None

        # Dense vectors config can be stored in different shapes depending on client/server versions.
        dense_ok = False
        if isinstance(vectors, dict):
            dense_ok = self._dense_name in vectors
        else:
            # Some versions wrap vectors config in an object; try attribute access
            try:
                dense_map = getattr(vectors, "vectors", None)
                if isinstance(dense_map, dict):
                    dense_ok = self._dense_name in dense_map
            except Exception:
                dense_ok = False

        sparse_ok = False
        if isinstance(sparse_vectors, dict):
            sparse_ok = self._sparse_name in sparse_vectors
        else:
            try:
                sparse_map = getattr(sparse_vectors, "vectors", None)
                if isinstance(sparse_map, dict):
                    sparse_ok = self._sparse_name in sparse_map
            except Exception:
                sparse_ok = False

        if not dense_ok or not sparse_ok:
            raise RuntimeError(
                "QdrantIndex: collection schema mismatch. "
                f"Expected named dense vector '{self._dense_name}' and sparse vector '{self._sparse_name}'. "
                "Create a new collection with the correct schema or migrate the existing one."
            )

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[qmodels.Filter]:
        """Exact-match AND filter over payload fields."""
        if not filters:
            return None

        must: List[qmodels.FieldCondition] = []
        for key, val in filters.items():
            if key is None or key == "":
                continue
            must.append(qmodels.FieldCondition(key=str(key), match=qmodels.MatchValue(value=val)))

        if not must:
            return None
        return qmodels.Filter(must=must)

    @staticmethod
    def _extract_id_score(points: Any) -> List[Tuple[str, float]]:
        """Extract UMA id + score from Qdrant points/hits, using payload['uma_id'] when present."""
        results: List[Tuple[str, float]] = []
        for p in points or []:
            payload = getattr(p, "payload", None) or {}
            uma_id = payload.get("uma_id") if isinstance(payload, dict) else None
            pid = str(uma_id) if uma_id else str(getattr(p, "id", ""))
            score = float(getattr(p, "score", 0.0))
            if pid:
                results.append((pid, score))
        return results

    def _search_points_compat(self, **kwargs: Any) -> Any:
        """
        Compatibility shim for qdrant-client search APIs.
        Attempts `search`, then `search_points`, then `query_points` with a simplified signature.
        """
        if hasattr(self._client, "search"):
            return self._client.search(collection_name=self.collection, **kwargs)

        if hasattr(self._client, "search_points"):
            res = self._client.search_points(collection_name=self.collection, **kwargs)
            return getattr(res, "points", res)

        if hasattr(self._client, "query_points"):
            # For older clients that do not support query_points with our kwargs, we attempt minimal usage.
            res = self._client.query_points(collection_name=self.collection, **kwargs)
            return getattr(res, "points", res)

        raise AttributeError("Qdrant client does not support search/query methods.")

    @staticmethod
    def _normalize_id(pid: Any) -> str:
        """Qdrant IDs must be uint or UUID; use UUID5 for stable mapping of strings."""
        if isinstance(pid, int):
            return str(pid)
        pid_str = str(pid)
        try:
            return str(uuid.UUID(pid_str))
        except Exception:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, pid_str))

    @staticmethod
    def _to_sparse_vector(sparse_embedding: Any) -> qmodels.SparseVector:
        """Convert FastEmbed BM25 sparse embedding to Qdrant SparseVector."""
        indices = list(getattr(sparse_embedding, "indices", []))
        values = list(getattr(sparse_embedding, "values", []))

        if len(indices) != len(values):
            raise ValueError("BM25 sparse embedding indices/values length mismatch.")

        if not indices:
            return qmodels.SparseVector(indices=[], values=[])

        try:
            idx = [int(i) for i in indices]
            val = [float(v) for v in values]
        except Exception as exc:
            raise ValueError(f"BM25 sparse embedding has non-numeric indices/values: {exc}") from exc

        return qmodels.SparseVector(indices=idx, values=val)

    def _validate_ids_vectors(self, ids: List[str], vectors: List[List[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("QdrantIndex.upsert: ids and vectors length mismatch.")
        for sid in ids:
            if not isinstance(sid, str) or not sid.strip():
                raise ValueError("QdrantIndex.upsert: all ids must be non-empty strings.")

    def _validate_dense_vector(self, vec: List[float]) -> None:
        if not isinstance(vec, list) or not vec:
            raise ValueError("QdrantIndex: vector must be a non-empty list of floats.")
        if len(vec) != self.dim:
            raise ValueError(f"QdrantIndex: expected vector dim={self.dim}, got {len(vec)}.")
        for i, v in enumerate(vec):
            if not isinstance(v, (int, float)):
                raise ValueError(f"QdrantIndex: vector[{i}] must be numeric, got {type(v)}.")

    @staticmethod
    def _parse_distance(distance: str) -> Any:
        d = (distance or "").strip().lower()
        if d in ("cosine", "cos"):
            return qmodels.Distance.COSINE
        if d in ("dot", "ip", "inner_product"):
            return qmodels.Distance.DOT
        if d in ("euclid", "l2"):
            return qmodels.Distance.EUCLID
        raise ValueError("QdrantIndex: unsupported distance. Use one of {'cosine','dot','euclid'}.")
