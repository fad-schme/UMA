"""
Vector index abstraction for UMA-3.

This provides a backend-agnostic interface for vector indices, so UMA-3
can support FAISS, Pinecone, Weaviate, etc., via the same contract.

Coding agent instructions
-------------------------
- This interface is used by SemanticSQLStore and EpisodicStore.
- Implement backend adapters (e.g., FaissIndex) conforming to this interface.
- Ensure implementations are safe under concurrent access in your context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class VectorIndex(ABC):
    """
    Abstract base class for vector indices.

    Implementations must provide:

    - upsert(ids, vectors, metadata)
    - query(vector, k, filters)
    """

    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """
        Insert or update vectors in the index.

        Parameters
        ----------
        ids:
            Unique identifiers for each vector.
        vectors:
            List of dense numeric vectors.
        metadata:
            Optional list of metadata dicts per vector.
        """
        raise NotImplementedError

    @abstractmethod
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
        vector:
            Query embedding.
        k:
            Max number of results.
        filters:
            Optional metadata filter: dict of {key: value} to match exactly.

        Returns
        -------
        List[Tuple[id, score]]
        """
        raise NotImplementedError