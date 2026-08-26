"""
Vector index abstraction for UMA.

This provides a backend-agnostic interface for vector indices, so UMA
can support FAISS, Pinecone, Weaviate, etc., via the same contract.

Isolation contract (C1)
-----------------------
Every UMA artifact carries explicit ownership: tenant_id, owner_type,
owner_id. The vector-index contract makes these mandatory at every
read and write so isolation is enforced **by construction at the
storage layer** rather than as an application-layer Python filter
applied after the backend's k-nearest cap.

- `upsert` requires parallel `tenant_ids`, `owner_types`, `owner_ids`
  lists matching the length of `ids` / `vectors`. Adapters store these
  as first-class fields the backend can index.
- `query` requires `tenant_id`, `owner_type`, `owner_id` as keyword
  arguments. Adapters push these into the backend's native predicate
  language BEFORE the candidate cap is applied. Cross-tenant rows
  cannot leak past this boundary regardless of the cap or any client
  bug.
- `extra_filters` (on query) and `extra_metadata` (on upsert) carry
  any non-isolation keys callers still need (e.g. `doc_id`, `kind`,
  `kb_lane`). Adapters apply `extra_filters` after the native
  isolation predicate runs.

Coding agent instructions
-------------------------
- This interface is used by SemanticSQLStore, EpisodicStore,
  ProceduralSQLStore, and ChunkSQLStore.
- Implement backend adapters (e.g., FaissIndex) conforming to this
  interface. Adapters MAY ignore `extra_filters` and let the caller
  post-filter, but the isolation keys (`tenant_id`, `owner_type`,
  `owner_id`) MUST be respected — they are not optional.
- Ensure implementations are safe under concurrent access in your
  context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class VectorIndex(ABC):
    """
    Abstract base class for vector indices.

    Implementations must provide:

    - upsert(ids, vectors, tenant_ids, owner_types, owner_ids, extra_metadata)
    - query(vector, tenant_id, owner_type, owner_id, k, extra_filters)
    - delete(ids)
    """

    @abstractmethod
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
        Insert or update vectors in the index.

        Parameters
        ----------
        ids:
            Unique identifiers for each vector.
        vectors:
            List of dense numeric vectors.
        tenant_ids:
            Tenant identifier for each vector. Length must match `ids`.
            Stored as a first-class indexable field, NOT inside metadata.
        owner_types:
            Owner-type for each vector ("agent" | "user" | "workspace" |
            "system"). Length must match `ids`.
        owner_ids:
            Owner identifier for each vector. Length must match `ids`.
        extra_metadata:
            Optional non-isolation metadata per vector. Adapters may
            persist this verbatim (e.g. as JSON) for lane-specific
            fields like `kind`, `kb_lane`, `doc_id`. Must NOT contain
            `tenant_id`, `owner_type`, or `owner_id` — those are the
            explicit parallel-list parameters.
        """
        raise NotImplementedError

    @abstractmethod
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
        Perform a nearest-neighbor search within the isolation scope.

        Parameters
        ----------
        vector:
            Query embedding.
        tenant_id, owner_type, owner_id:
            Required isolation scope. Adapters push these into the
            backend's native filter so the candidate cap (e.g.
            LanceDB's `limit`) is applied AFTER scoping — no cross-
            tenant rows can leak past this boundary.
        k:
            Max number of results within the isolation scope.
        extra_filters:
            Optional non-isolation predicates (e.g. `{"doc_id": "..."}`).
            Adapters apply these after the isolation filter.

        Returns
        -------
        List[Tuple[id, score]]
        """
        raise NotImplementedError

    def get_vectors(
        self,
        ids: list[str],
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
    ) -> dict[str, list[float]]:
        """
        Return stored embedding vectors for the given ids, keyed by id.

        Not abstract: retrieval-ranking-gap ticket 07 (MMR/diversity-aware
        chunk selection) is the only caller, and it must degrade to plain
        top-k selection when vectors aren't available rather than force
        every backend to support this. The default implementation returns
        an empty dict, which callers treat as "vectors unavailable" for
        every id, not as "these ids don't exist" — never partially fill in
        results for ids you can't verify are in scope.

        Parameters
        ----------
        ids:
            Vector ids to look up. Only ids already known to be in the
            caller's isolation scope should be passed (e.g. ids returned by
            a prior `query()` call under the same tenant/owner) — this
            method does not itself guarantee isolation for arbitrary ids.
        tenant_id, owner_type, owner_id:
            Isolation scope, required for backends whose lookup needs it
            (e.g. LanceDB re-queries by id within this scope) and as a
            defensive check for backends that can verify it (e.g. in-memory).

        Returns
        -------
        Dict[id, vector]. Ids the backend can't cheaply provide a vector for
        (or that fall outside the given scope) are simply absent from the
        dict — never a placeholder or zero vector.
        """
        return {}

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """
        Delete vectors from the index.

        Parameters
        ----------
        ids:
            List of vector IDs to remove.

        Note: scoping deletes by tenant is unnecessary — UMA generates
        unique ids across all callers via SQL primary keys, so cross-
        tenant id collision is impossible by construction. The
        signature remains id-only.
        """
        raise NotImplementedError
