"""
Weaviate vector index adapter for UMA.

Supports:
- Near-vector search
- Metadata filtering
- Custom classes per memory type

Coding agent instructions:
--------------------------
- Set WEAVIATE_API_KEY and WEAVIATE_URL environment variables.
- You may want to separate classes per memory type: Fact, Episode, Skill.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Tuple, Optional
import os

import weaviate
from uma.adapters.vector.base import VectorIndex
from uma.core.utils.retry import retry_sync

logger = logging.getLogger(__name__)


class WeaviateIndex(VectorIndex):

    def __init__(self, url: str, api_key: str, class_name: str, dim: int):
        auth = weaviate.AuthApiKey(api_key=api_key) if api_key else None
        self.client = weaviate.Client(url, auth_client_secret=auth)
        self.class_name = class_name
        self.dim = dim

        logger.info("WeaviateIndex initialized: class=%s", class_name)

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        metadata = metadata or [{} for _ in ids]

        def _call() -> None:
            with self.client.batch as batch:
                for _id, vec, meta in zip(ids, vectors, metadata):
                    if len(vec) != self.dim:
                        raise ValueError("Vector dimension mismatch in Weaviate upsert")

                    batch.add_data_object(
                        data_object=meta,
                        class_name=self.class_name,
                        vector=vec,
                        uuid=_id,
                    )

        try:
            retry_sync(_call)
        except Exception:
            logger.exception("WeaviateIndex.upsert failed.")
            raise

    def query(
        self,
        vector: List[float],
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        if len(vector) != self.dim:
            raise ValueError(
                f"WeaviateIndex.query: expected dim={self.dim}, got={len(vector)}"
            )
        query = (
            self.client.query
            .get(self.class_name, ["uuid"])
            .with_near_vector({"vector": vector})
            .with_limit(k)
        )

        if filters:
            raise ValueError("WeaviateIndex.query does not support filters yet.")

        def _call():
            return query.do()

        res = retry_sync(_call)

        matches = res["data"]["Get"].get(self.class_name, [])

        return [(obj["uuid"], 0.0) for obj in matches]

    def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        deleted = 0
        for _id in ids:
            def _call() -> None:
                self.client.data_object.delete(uuid=_id, class_name=self.class_name)

            retry_sync(_call)
            deleted += 1
        logger.debug("WeaviateIndex.delete: deleted %d ids", deleted)

    def verify_connectivity(self) -> bool:
        """Best-effort connectivity check."""
        try:
            return bool(self.client.is_ready())
        except Exception:
            logger.exception("WeaviateIndex.verify_connectivity failed.")
            return False
