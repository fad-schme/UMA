from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Optional

from .base import VectorIndex

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models as qmodels  # type: ignore
except Exception as exc:  # pragma: no cover
    QdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]
    logger.debug("qdrant-client is unavailable: %s", exc)


class QdrantIndex(VectorIndex):
    """Dense-only Qdrant-backed vector index for UMA's container profile."""

    def __init__(
        self,
        dim: int,
        *,
        collection: Optional[str] = None,
        table_name: str = "uma_vectors",
        url: str,
        api_key: Optional[str] = None,
        distance: str = "cosine",
        **_: Any,
    ) -> None:
        if QdrantClient is None or qmodels is None:
            raise RuntimeError(
                "QdrantIndex requires qdrant-client. Install it with "
                "`python -m pip install 'uma-mem[qdrant]'`."
            )
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("QdrantIndex: dim must be a positive integer.")
        resolved_collection = collection or table_name
        if (
            not isinstance(resolved_collection, str)
            or not resolved_collection.strip()
        ):
            raise ValueError("QdrantIndex: collection must be a non-empty string.")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("QdrantIndex: url must be a non-empty string.")

        self.dim = dim
        self.dimension = dim
        self.index = self
        self.collection = resolved_collection.strip()
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
        ids: list[str],
        vectors: list[list[float]],
        *,
        tenant_ids: list[str],
        owner_types: list[str],
        owner_ids: list[str],
        extra_metadata: Optional[list[dict]] = None,
    ) -> None:
        prepared = self._prepare_upsert(
            ids,
            vectors,
            tenant_ids=tenant_ids,
            owner_types=owner_types,
            owner_ids=owner_ids,
            extra_metadata=extra_metadata,
        )
        if not vectors:
            return

        points = []
        for item_id, vector, scope, metadata in prepared:
            tenant_id, owner_type, owner_id = scope
            payload = {
                **metadata,
                "tenant_id": tenant_id,
                "owner_type": owner_type,
                "owner_id": owner_id,
            }
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
        vector: list[float],
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        extra_filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[str, float]]:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("QdrantIndex.query: k must be a positive integer.")
        scope = self._validate_scope(tenant_id, owner_type, owner_id)
        if extra_filters is not None and not isinstance(extra_filters, dict):
            raise ValueError(
                "QdrantIndex.query: extra_filters must be a dict or None."
            )
        if extra_filters:
            self._reject_reserved_metadata(extra_filters)
        query_vector = self._coerce_vector(vector)
        query_filter = self._build_filter(
            tenant_id=scope[0],
            owner_type=scope[1],
            owner_id=scope[2],
            extra_filters=extra_filters,
        )

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

        results: list[tuple[str, float]] = []
        for point in points or []:
            payload = getattr(point, "payload", None) or {}
            item_id = payload.get("uma_id") if isinstance(payload, dict) else None
            resolved_id = str(item_id) if item_id else str(getattr(point, "id", ""))
            if not resolved_id:
                continue
            score = float(getattr(point, "score", 0.0))
            results.append((resolved_id, score))
        return results

    def delete(self, ids: list[str]) -> None:
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
        if hasattr(self._client, "collection_exists"):
            exists = bool(self._client.collection_exists(self.collection))
        else:
            try:
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

    def _coerce_vector(self, vector: list[float]) -> list[float]:
        if not isinstance(vector, list) or len(vector) != self.dim:
            raise ValueError(
                f"QdrantIndex: expected vector dim={self.dim}, got="
                f"{len(vector) if isinstance(vector, list) else 'invalid'}."
            )
        try:
            coerced = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "QdrantIndex: vectors must contain numeric values."
            ) from exc
        if not all(math.isfinite(value) for value in coerced):
            raise ValueError("QdrantIndex: vectors must contain finite values.")
        return coerced

    def _prepare_upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        *,
        tenant_ids: list[str],
        owner_types: list[str],
        owner_ids: list[str],
        extra_metadata: Optional[list[dict]],
    ) -> list[tuple[str, list[float], tuple[str, str, str], dict[str, Any]]]:
        if len(ids) != len(vectors):
            raise ValueError("QdrantIndex.upsert: ids and vectors length mismatch.")
        count = len(ids)
        for name, values in (
            ("tenant_ids", tenant_ids),
            ("owner_types", owner_types),
            ("owner_ids", owner_ids),
        ):
            if len(values) != count:
                raise ValueError(
                    f"QdrantIndex.upsert: {name} length "
                    f"({len(values)}) does not match ids length ({count})."
                )
        metadata_items = (
            [{} for _ in ids]
            if extra_metadata is None
            else extra_metadata
        )
        if len(metadata_items) != count:
            raise ValueError(
                "QdrantIndex.upsert: extra_metadata length "
                f"({len(metadata_items)}) does not match ids length ({count})."
            )

        prepared = []
        for item_id, vector, tenant_id, owner_type, owner_id, metadata in zip(
            ids,
            vectors,
            tenant_ids,
            owner_types,
            owner_ids,
            metadata_items,
        ):
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(
                    "QdrantIndex.upsert: all ids must be non-empty strings."
                )
            if not isinstance(metadata, dict):
                raise ValueError(
                    "QdrantIndex.upsert: extra_metadata items must be dicts."
                )
            self._reject_reserved_metadata(metadata)
            prepared.append(
                (
                    item_id,
                    self._coerce_vector(vector),
                    self._validate_scope(
                        tenant_id,
                        owner_type,
                        owner_id,
                    ),
                    dict(metadata),
                )
            )
        return prepared

    @staticmethod
    def _validate_scope(
        tenant_id: str,
        owner_type: str,
        owner_id: str,
    ) -> tuple[str, str, str]:
        values = {
            "tenant_id": tenant_id,
            "owner_type": owner_type,
            "owner_id": owner_id,
        }
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"QdrantIndex: {name} must be a non-empty string."
                )
        return (
            tenant_id.strip(),
            owner_type.strip(),
            owner_id.strip(),
        )

    @staticmethod
    def _reject_reserved_metadata(metadata: dict[str, Any]) -> None:
        for key in ("tenant_id", "owner_type", "owner_id"):
            if key in metadata:
                raise ValueError(
                    "QdrantIndex: extra metadata must not contain reserved "
                    f"isolation key {key!r}."
                )

    @staticmethod
    def _point_id(value: str) -> str:
        try:
            return str(uuid.UUID(str(value)))
        except (ValueError, AttributeError, TypeError):
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(value)))

    @staticmethod
    def _build_filter(
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
        extra_filters: Optional[dict[str, Any]],
    ):
        filters = {
            "tenant_id": tenant_id,
            "owner_type": owner_type,
            "owner_id": owner_id,
            **(extra_filters or {}),
        }
        must = [
            qmodels.FieldCondition(
                key=str(key),
                match=qmodels.MatchValue(value=value),
            )
            for key, value in filters.items()
            if key is not None and str(key) != ""
        ]
        return qmodels.Filter(must=must)

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
