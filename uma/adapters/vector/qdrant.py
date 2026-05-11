from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .base import VectorIndex

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models as qmodels  # type: ignore
except Exception as exc:  # pragma: no cover
    QdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]
    logger.error("Failed to import qdrant-client: %s", exc)


class QdrantIndex(VectorIndex):
    """Dense-only Qdrant-backed vector index for UMA's container profile."""

    def __init__(
        self,
        dim: int,
        *,
        collection: str,
        url: str,
        api_key: Optional[str] = None,
        distance: str = "cosine",
        **_: Any,
    ) -> None:
        if QdrantClient is None or qmodels is None:
            raise RuntimeError(
                "qdrant-client is not installed. Install the public vector extras, for "
                "example `pip install -e '.[vector]'`."
            )
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("QdrantIndex: dim must be a positive integer.")
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError("QdrantIndex: collection must be a non-empty string.")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("QdrantIndex: url must be a non-empty string.")

        self.dim = dim
        self.collection = collection.strip()
        self.url = url.strip()
        self._distance = self._parse_distance(distance)
        self._client = QdrantClient(url=self.url, api_key=api_key)

        self._ensure_collection()
        logger.info(
            "Initialized QdrantIndex url=%s collection=%s dim=%d",
            self.url,
            self.collection,
            self.dim,
        )

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        self._validate_upsert_inputs(ids, vectors)
        if not vectors:
            return

        metadata_list = metadata or [{} for _ in ids]
        if len(metadata_list) != len(ids):
            raise ValueError("QdrantIndex.upsert: metadata length mismatch with ids.")

        points = []
        for item_id, vector, meta in zip(ids, vectors, metadata_list):
            if meta is None:
                meta = {}
            if not isinstance(meta, dict):
                raise ValueError("QdrantIndex.upsert: metadata items must be dicts.")
            payload = dict(meta)
            payload["uma_id"] = item_id
            points.append(
                qmodels.PointStruct(
                    id=self._point_id(item_id),
                    vector=[float(value) for value in vector],
                    payload=payload,
                )
            )

        try:
            self._client.upsert(self.collection, points=points, wait=True)
        except Exception:
            logger.exception("QdrantIndex.upsert failed collection=%s", self.collection)
            raise

    def query(
        self,
        vector: List[float],
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("QdrantIndex.query: k must be a positive integer.")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("QdrantIndex.query: filters must be a dict or None.")
        query_vector = self._coerce_vector(vector)
        query_filter = self._build_filter(filters)

        try:
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self.collection,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                )
                points = getattr(response, "points", response)
            else:
                points = self._client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                )
        except Exception:
            logger.exception("QdrantIndex.query failed collection=%s", self.collection)
            raise

        results: List[Tuple[str, float]] = []
        for point in points or []:
            payload = getattr(point, "payload", None) or {}
            item_id = payload.get("uma_id") if isinstance(payload, dict) else None
            resolved_id = str(item_id) if item_id else str(getattr(point, "id", ""))
            if not resolved_id:
                continue
            results.append((resolved_id, float(getattr(point, "score", 0.0))))
        return results

    def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        point_ids = [self._point_id(item_id) for item_id in ids]
        selector = qmodels.PointIdsList(points=point_ids)
        try:
            self._client.delete(
                collection_name=self.collection,
                points_selector=selector,
                wait=True,
            )
        except Exception:
            logger.exception("QdrantIndex.delete failed collection=%s", self.collection)
            raise

    def _ensure_collection(self) -> None:
        exists = False
        try:
            if hasattr(self._client, "collection_exists"):
                exists = bool(self._client.collection_exists(self.collection))
            else:
                self._client.get_collection(self.collection)
                exists = True
        except Exception:
            exists = False

        if exists:
            return

        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(
                size=self.dim,
                distance=self._distance,
            ),
        )
        logger.info("Created Qdrant collection=%s dim=%d", self.collection, self.dim)

    def _coerce_vector(self, vector: List[float]) -> List[float]:
        if not isinstance(vector, list) or len(vector) != self.dim:
            raise ValueError(
                f"QdrantIndex: expected vector dim={self.dim}, got="
                f"{len(vector) if isinstance(vector, list) else 'invalid'}."
            )
        coerced = [float(value) for value in vector]
        for value in coerced:
            if not isinstance(value, float):
                raise ValueError("QdrantIndex: vectors must contain numeric values.")
        return coerced

    def _validate_upsert_inputs(self, ids: List[str], vectors: List[List[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("QdrantIndex.upsert: ids and vectors length mismatch.")
        for item_id in ids:
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("QdrantIndex.upsert: all ids must be non-empty strings.")
        for vector in vectors:
            self._coerce_vector(vector)

    @staticmethod
    def _point_id(value: str) -> str:
        try:
            return str(uuid.UUID(str(value)))
        except Exception:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(value)))

    @staticmethod
    def _build_filter(filters: Optional[Dict[str, Any]]):
        if not filters:
            return None
        must = [
            qmodels.FieldCondition(
                key=str(key),
                match=qmodels.MatchValue(value=value),
            )
            for key, value in filters.items()
            if key is not None and str(key) != ""
        ]
        return qmodels.Filter(must=must) if must else None

    @staticmethod
    def _parse_distance(distance: str):
        normalized = (distance or "").strip().lower()
        if normalized in {"cosine", "cos"}:
            return qmodels.Distance.COSINE
        if normalized in {"dot", "ip", "inner_product"}:
            return qmodels.Distance.DOT
        if normalized in {"euclid", "l2"}:
            return qmodels.Distance.EUCLID
        raise ValueError(
            "QdrantIndex: unsupported distance. Use one of {'cosine', 'dot', 'euclid'}."
        )
