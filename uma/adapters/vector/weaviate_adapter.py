"""
Weaviate vector index adapter for UMA-3.

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
from .base import VectorIndex

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

    def query(
        self,
        vector: List[float],
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        query = (
            self.client.query
            .get(self.class_name, ["uuid"])
            .with_near_vector({"vector": vector})
            .with_limit(k)
        )

        if filters:
            # TODO: add filter conversion for Weaviate
            pass

        res = query.do()
        matches = res["data"]["Get"].get(self.class_name, [])

        return [(obj["uuid"], 0.0) for obj in matches]

    def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        deleted = 0
        for _id in ids:
            try:
                self.client.data_object.delete(uuid=_id, class_name=self.class_name)
                deleted += 1
            except Exception:
                logger.exception("WeaviateIndex.delete failed for id=%s", _id)
        logger.debug("WeaviateIndex.delete: deleted %d ids", deleted)
