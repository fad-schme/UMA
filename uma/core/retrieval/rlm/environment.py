# uma/core/retrieval/rlm/environment.py

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Literal, Optional, Union

from ...utils.identity import ensure_user_subject

logger = logging.getLogger(__name__)

NumericVector = List[Union[int, float]]


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result




class UMAMemoryEnvironment:
    """
    Production retrieval environment (read-only).

    This is the ONLY interface the RLM controller uses. It is designed to:
    - enforce scoping
    - enforce bounds
    - provide a stable, core-driven retrieval surface
    - avoid leaking DB/adapters into the controller loop
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory
        self._agent_id = getattr(memory, "agent_id", None)

        self._wm = getattr(memory, "working_memory", None)
        self._semantic_core = getattr(memory, "semantic_core", None)
        self._chunk_core = getattr(memory, "chunk_core", None)
        self._episodic_core = getattr(memory, "episodic_core", None)
        self._procedural_core = getattr(memory, "procedural_core", None)
        self._graph_core = getattr(memory, "graph_core", None)
        self._embedder = getattr(memory, "embedder", None)

        if self._embedder is None:
            raise ValueError("UMAMemoryEnvironment requires an embedder to operate")

        # Log missing subsystems. Not fatal for environment (controller can still run partially).
        if self._wm is None:
            logger.warning("UMAMemoryEnvironment: working_memory missing")
        if self._semantic_core is None:
            logger.warning("UMAMemoryEnvironment: semantic_core missing")
        if self._chunk_core is None:
            logger.warning("UMAMemoryEnvironment: chunk_core missing")
        if self._episodic_core is None:
            logger.warning("UMAMemoryEnvironment: episodic_core missing")
        if self._procedural_core is None:
            logger.warning("UMAMemoryEnvironment: procedural_core missing")
        if self._graph_core is None:
            logger.warning("UMAMemoryEnvironment: graph_core missing")

    # ------------------------------------------------------------------
    # Internal helpers (bounds, sanitation)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_k(name: str, k: int, hard_cap: int = 500) -> int:
        """Clamp k to a safe bound to prevent runaway retrieval."""
        try:
            k_int = int(k)
        except Exception as exc:
            raise ValueError(f"{name}: k must be int-like") from exc
        if k_int <= 0:
            raise ValueError(f"{name}: k must be >= 1")
        return min(k_int, hard_cap)

    @staticmethod
    def _safe_depth(depth: Any, max_depth: int = 3) -> int:
        """Clamp graph depth to a small bounded number."""
        try:
            d = int(depth)
        except Exception:
            d = 1
        return max(1, min(d, max_depth))

    @staticmethod
    def _safe_offset(offset: Any) -> int:
        """Parse an optional offset safely (non-negative)."""
        if offset is None:
            return 0
        try:
            off = int(offset)
        except Exception:
            return 0
        return max(0, min(off, 100000))

    @staticmethod
    def _sanitize_time_range(time_range: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(time_range, dict):
            return None

        sanitized: Dict[str, Any] = {}
        start = time_range.get("start")
        end = time_range.get("end")
        offset = time_range.get("offset")

        if start is not None:
            sanitized["start"] = start
        if end is not None and (start is None or end >= start):
            sanitized["end"] = end
        if offset is not None:
            sanitized["offset"] = max(0, int(float(offset)))

        return sanitized or None

    @staticmethod
    def _limit_fact_ids(fact_ids: List[str], limit: int = 50) -> List[str]:
        out: List[str] = []
        seen = set()
        for fid in fact_ids or []:
            if not isinstance(fid, str):
                continue
            fid = fid.strip()
            if not fid or fid in seen:
                continue
            seen.add(fid)
            out.append(fid)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _filter_time_range(episodes: List[Any], time_range: Optional[Dict[str, Any]]) -> List[Any]:
        if not time_range:
            return episodes
        start = time_range.get("start") if isinstance(time_range, dict) else None
        end = time_range.get("end") if isinstance(time_range, dict) else None
        if start is None and end is None:
            return episodes

        filtered: List[Any] = []
        for ep in episodes:
            ts = getattr(ep, "timestamp", None)
            if ts is None:
                continue
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                continue
            filtered.append(ep)
        return filtered

    @staticmethod
    def _max_predicate_scope() -> int:
        return 20


    # ------------------------------------------------------------------
    # Semantic
    # ------------------------------------------------------------------

    async def get_query_embedding(self, query_text: str) -> NumericVector:
        """
        Convert query text to embedding using the configured embedder.
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("Environment.get_query_embedding: query_text must be non-empty")
        try:
            expected_dim = getattr(self._embedder, "dimension", None)
            if not isinstance(expected_dim, int) or expected_dim <= 0:
                raise ValueError("Environment.get_query_embedding: embedder.dimension must be a positive integer")
            vectors = await self._embedder.embed([query_text])
            if not vectors or not isinstance(vectors, list) or not vectors[0]:
                raise ValueError("Embedder returned empty embedding.")
            vec0 = vectors[0]
            if not isinstance(vec0, list) or len(vec0) != expected_dim:
                raise ValueError(f"Embedder returned invalid dim (expected={expected_dim} got={len(vec0) if isinstance(vec0, list) else None}).")
            return [float(x) for x in vec0]
        except Exception as exc:
            logger.exception("Environment.get_query_embedding failed")
            raise ValueError("Failed to embed query text.") from exc

    async def fetch_chunks(
        self,
        user_id: str,
        *,
        ids: List[str],
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Fetch chunks by IDs (bounded, owner-scoped).
        """
        if self._chunk_core is None:
            return []
        if not isinstance(ids, list) or not ids:
            return []
        if len(ids) > 50:
            ids = ids[:50]
        try:
            user_subject = ensure_user_subject(user_id)
            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    return []
            else:
                resolved_owner_id = owner_id or user_subject

            return await self._chunk_core._fetch_by_ids(
                ids=[str(x) for x in ids if x],
                owner_type=owner_type,
                owner_id=resolved_owner_id,
                log_context="Environment.fetch_chunks",
            )
        except Exception:
            logger.exception("Environment.fetch_chunks failed")
            return []

    async def fetch_facts_by_ids(self, user_id: str, ids: List[str]) -> List[Any]:
        """
        Fetch semantic facts by IDs (bounded by the controller).

        NOTE:
        - Store should enforce ownership if IDs are global.
        - Environment still scopes user_id at call boundary.
        """
        if self._semantic_core is None or not ids:
            return []
        try:
            _ = ensure_user_subject(user_id)  # ensures caller isn't passing garbage
            facts = await self._semantic_core.fetch_by_ids(ids)
            logger.debug("Environment.fetch_facts_by_ids: returned %d", len(facts or []))
            return facts or []
        except Exception:
            logger.exception("Environment.fetch_facts_by_ids failed")
            return []


    async def fetch_more_facts(
        self,
        user_id: str,
        predicate: str,
        k: int,
        offset: int = 0,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        if self._semantic_core is None:
            return []
        try:
            user_subject = ensure_user_subject(user_id)
            k = self._validate_k("Environment.fetch_more_facts", k)
            offset = self._safe_offset(offset)

            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    return []
            elif owner_type == "user":
                resolved_owner_id = owner_id or user_subject
            else:
                logger.warning("Environment.fetch_more_facts: invalid owner_type=%r", owner_type)
                return []

            subject: Optional[str] = user_subject if owner_type == "user" else None
            return await self._semantic_core.fetch_more_facts(
                subject=subject,
                predicate=predicate,
                owner_type=owner_type,
                owner_id=resolved_owner_id,
                k=int(k),
                offset=int(offset),
            )
        except Exception:
            logger.exception("Environment.fetch_more_facts failed")
            return []

    # ------------------------------------------------------------------
    # Episodic
    # ------------------------------------------------------------------

    async def episodic_cluster_summaries(
        self,
        user_id: str,
        k: int = 5,
        max_episodes: int = 50,
        time_range: Optional[Dict[str, Any]] = None,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Retrieve episodic clusters via EpisodicCore (NOT the SQL store).

        This method intentionally routes through the core layer because:
        - clustering logic is not a store responsibility
        - it may require summarization / aggregation / policies
        """
        episodic_core = getattr(self._memory, "episodic_core", None)
        if episodic_core is None:
            logger.warning("Environment.episodic_cluster_summaries: episodic_core missing")
            return []

        k = self._validate_k("Environment.episodic_cluster_summaries", k)

        try:
            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    return []
            else:
                resolved_owner_id = owner_id or user_id
            clusters = await episodic_core.list_cluster_summaries(
                user_id=user_id,
                owner_type=owner_type,
                owner_id=resolved_owner_id,
                k=int(k),
                max_episodes=int(max_episodes),
                time_range=time_range,
            )
            logger.debug("Environment.episodic_cluster_summaries: returned %d", len(clusters))
            return clusters
        except Exception:
            logger.exception("Environment.episodic_cluster_summaries failed")
            return []

    async def fetch_episode_clusters(
        self,
        user_id: str,
        k: int = 5,
        max_episodes: int = 50,
        time_range: Optional[Dict[str, Any]] = None,
        min_salience: Optional[float] = None,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Fetch episodic clusters with optional salience filtering.
        """
        k = self._validate_k("Environment.fetch_episode_clusters.k", k)
        max_episodes = self._validate_k(
            "Environment.fetch_episode_clusters.max_episodes", max_episodes
        )
        sanitized_time_range = self._sanitize_time_range(time_range)
        clusters = await self.episodic_cluster_summaries(
            user_id=user_id,
            k=k,
            max_episodes=max_episodes,
            time_range=sanitized_time_range,
            owner_type=owner_type,
            owner_id=owner_id,
        )
        if min_salience is None:
            logger.debug("Environment.fetch_episode_clusters: returned %d", len(clusters))
            return clusters

        min_salience = max(0.0, min(1.0, float(min_salience)))
        filtered: List[Dict[str, Any]] = []
        for cluster in clusters:
            sal = self._cluster_salience(cluster)
            if sal is None or sal >= min_salience:
                filtered.append(cluster)
        logger.debug("Environment.fetch_episode_clusters: returned %d", len(filtered))
        return filtered

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    async def graph_neighbors(
        self,
        user_id: str,
        node_id: str,
        predicate_scope: Optional[List[str]] = None,
        depth: int = 1,
        k: int = 10,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Safe bounded graph expansion (DAT-safe).

        This function enforces:
        - ownership scoping (owner_type/owner_id)
        - bounded depth and result size
        - no raw Cypher exposure
        """
        if self._graph_core is None:
            return []

        k = self._validate_k("Environment.graph_neighbors", k)
        depth_i = self._safe_depth(depth)

        try:
            user_subject = ensure_user_subject(user_id)
            if not node_id:
                return []

            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    return []
            else:
                resolved_owner_id = owner_id or user_subject
            results = await _maybe_await(
                self._graph_core.neighbors(
                user_id=user_subject,
                node_id=node_id,
                predicate_scope=predicate_scope,
                depth=depth_i,
                k=k,
                owner_type=owner_type,
                owner_id=resolved_owner_id,
                )
            )
            logger.debug("Environment.graph_neighbors: returned %d", len(results or []))
            return results or []

        except Exception:
            logger.exception("Environment.graph_neighbors failed")
            return []
        

    async def expand_graph(
        self,
        user_id: str,
        subject: str,
        predicate: Optional[str] = None,
        hops: int = 1,
        direction: Optional[Literal["inbound", "outbound", "both"]] = None,
        k: int = 10,
    ) -> List[Any]:
        if self._graph_core is None:
            return []

        k = self._validate_k("Environment.expand_graph.k", k)
        depth = self._safe_depth(hops)

        dir_val = None
        if direction:
            normalized = str(direction).lower()
            if normalized in {"inbound", "outbound", "both"}:
                dir_val = normalized
            else:
                logger.debug(
                    "Environment.expand_graph: dropping invalid direction=%s", direction
                )

        predicate_scope = []
        if predicate and isinstance(predicate, str):
            predicate_scope.append(predicate.upper())
        if predicate_scope:
            predicate_scope = predicate_scope[:self._max_predicate_scope()]
        else:
            predicate_scope = None

        if not subject:
            return []

        try:
            user_subject = ensure_user_subject(user_id)
            if dir_val and dir_val != "both":
                logger.debug(
                    "Environment.expand_graph: direction=%s currently treated as both", dir_val
                )

            return await self.graph_neighbors(
                user_id=user_subject,
                node_id=subject,
                predicate_scope=predicate_scope,
                depth=depth,
                k=k,
            )
        except Exception:
            logger.exception("Environment.expand_graph failed")
            return []

    def _cluster_salience(self, cluster: Dict[str, Any]) -> Optional[float]:
        if not isinstance(cluster, dict):
            return None

        candidates = [
            cluster.get("salience"),
            (cluster.get("meta") or {}).get("salience"),
            cluster.get("score"),
            cluster.get("salience_score"),
            cluster.get("avg_salience"),
        ]

        for val in candidates:
            try:
                if val is None:
                    continue
                return float(val)
            except Exception:
                continue
        return None
