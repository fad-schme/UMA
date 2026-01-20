"""
Pinecone vector index adapter for UMA-3.

This backend is used when Pinecone is preferred for:
- High-scale vector search
- Filters
- Highly parallel workloads

Coding agent instructions:
--------------------------
- Set PINECONE_API_KEY in environment.
- Use the same embedding dimension configured in UMAConfig.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Optional

import os
import pinecone

from .base import VectorIndex

logger = logging.getLogger(__name__)


class PineconeIndex(VectorIndex):

    def __init__(self, index_name: str, dim: int):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set.")

        pinecone.init(api_key=api_key)

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=dim, metric="cosine")

        self.index = pinecone.Index(index_name)
        self.dim = dim

        logger.info("PineconeIndex initialized for %s", index_name)

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        items = []
        metadata = metadata or [{} for _ in ids]

        for _id, vec, meta in zip(ids, vectors, metadata):
            if len(vec) != self.dim:
                raise ValueError(
                    f"Vector dim mismatch: expected {self.dim}, got {len(vec)}"
                )

            items.append({"id": _id, "values": vec, "metadata": meta})

        try:
            self.index.upsert(items)
        except Exception:
            logger.exception("PineconeIndex: upsert failed.")
            raise

    def query(
        self,
        vector: List[float],
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        try:
            results = self.index.query(vector=vector, top_k=k, filter=filters)
        except Exception:
            logger.exception("PineconeIndex.query failed.")
            return []

        return [(match["id"], match["score"]) for match in results["matches"]]

    def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        try:
            self.index.delete(ids=ids)
            logger.debug("PineconeIndex.delete: deleted %d ids", len(ids))
        except Exception:
            logger.exception("PineconeIndex.delete failed.")
