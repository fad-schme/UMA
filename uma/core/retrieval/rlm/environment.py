# uma/core/retrieval/rlm/environment.py

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Literal, Optional, Union

from ...utils.identity import normalize_user_id

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

        if getattr(memory, "embedder", None) is None:
            raise ValueError("UMAMemoryEnvironment requires an embedder to operate")

        # Missing subsystems are expected before ingestion warmup; avoid noisy logs here.

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
            embedder = getattr(self._memory, "embedder", None)
            expected_dim = getattr(embedder, "dimension", None)
            if not isinstance(expected_dim, int) or expected_dim <= 0:
                raise ValueError("Environment.get_query_embedding: embedder.dimension must be a positive integer")
            vectors = await embedder.embed([query_text])
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
        chunk_core = getattr(self._memory, "chunk_core", None)
        if chunk_core is None:
            logger.error("Environment.fetch_chunks: chunk_core is None")
            raise RuntimeError("Environment.fetch_chunks: chunk_core is None")
        if not isinstance(ids, list) or not ids:
            return []
        if len(ids) > 50:
            ids = ids[:50]
        try:
            normalized_user_id = normalize_user_id(user_id)
            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    raise RuntimeError("Environment.fetch_chunks: missing agent_id for agent scope")
            else:
                resolved_owner_id = owner_id or normalized_user_id
            if not owner_type or not resolved_owner_id:
                logger.error("Environment.fetch_chunks requires owner_type and owner_id")
                raise ValueError("Environment.fetch_chunks requires owner_type and owner_id")
            clean_ids = [str(x) for x in ids if x]
            logger.debug(
                "Environment.fetch_chunks: ids_count=%d owner=%s:%s",
                len(clean_ids),
                owner_type,
                resolved_owner_id,
            )
            chunks = await chunk_core._fetch_by_ids(
                ids=clean_ids,
                owner_type=owner_type,
                owner_id=resolved_owner_id,
                log_context="Environment.fetch_chunks",
            )
            if clean_ids and not chunks:
                logger.warning(
                    "Environment.fetch_chunks: fetched 0 for ids_count=%d owner=%s:%s",
                    len(clean_ids),
                    owner_type,
                    resolved_owner_id,
                )
            return chunks
        except Exception:
            logger.exception("Environment.fetch_chunks failed")
            raise

    async def fetch_facts_by_ids(
        self,
        user_id: str,
        ids: List[str],
        *,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Fetch semantic facts by IDs (bounded by the controller).

        NOTE:
        - Store should enforce ownership if IDs are global.
        - Environment still scopes user_id at call boundary.
        """
        semantic_core = getattr(self._memory, "semantic_core", None)
        if semantic_core is None:
            logger.error("Environment.fetch_facts_by_ids: semantic_core is None")
            raise RuntimeError("Environment.fetch_facts_by_ids: semantic_core is None")
        if not ids:
            return []
        try:
            normalized_user_id = normalize_user_id(user_id)  # ensures caller isn't passing garbage
            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    raise RuntimeError("Environment.fetch_facts_by_ids: missing agent_id for agent scope")
            elif owner_type == "user":
                resolved_owner_id = owner_id or normalized_user_id
            else:
                raise ValueError(f"Environment.fetch_facts_by_ids: invalid owner_type={owner_type!r}")
            facts = await semantic_core.fetch_by_ids(
                ids,
                owner_type=owner_type,
                owner_id=resolved_owner_id,
            )
            logger.debug("Environment.fetch_facts_by_ids: returned %d", len(facts or []))
            return facts or []
        except Exception:
            logger.exception("Environment.fetch_facts_by_ids failed")
            raise


    async def fetch_more_facts(
        self,
        user_id: str,
        predicate: str,
        k: int,
        offset: int = 0,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        semantic_core = getattr(self._memory, "semantic_core", None)
        if semantic_core is None:
            logger.error("Environment.fetch_more_facts: semantic_core is None")
            raise RuntimeError("Environment.fetch_more_facts: semantic_core is None")
        try:
            normalized_user_id = normalize_user_id(user_id)
            k = self._validate_k("Environment.fetch_more_facts", k)
            offset = self._safe_offset(offset)

            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    raise RuntimeError("Environment.fetch_more_facts: missing agent_id for agent scope")
            elif owner_type == "user":
                resolved_owner_id = owner_id or normalized_user_id
            else:
                raise ValueError(f"Environment.fetch_more_facts: invalid owner_type={owner_type!r}")

            # Ownership-only: subject must not be used for semantic retrieval.
            return await semantic_core.fetch_more_facts(
                predicate=predicate,
                owner_type=owner_type,
                owner_id=resolved_owner_id,
                k=int(k),
                offset=int(offset),
            )
        except Exception:
            logger.exception("Environment.fetch_more_facts failed")
            raise

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
            logger.error("Environment.episodic_cluster_summaries: episodic_core is None")
            raise RuntimeError("Environment.episodic_cluster_summaries: episodic_core is None")

        k = self._validate_k("Environment.episodic_cluster_summaries", k)

        try:
            normalized_user_id = normalize_user_id(user_id)
            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    raise RuntimeError("Environment.episodic_cluster_summaries: missing agent_id for agent scope")
            else:
                resolved_owner_id = owner_id or normalized_user_id
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
            raise

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

    async def graph_resolve_nodes(
        self,
        user_id: str,
        *,
        names: List[str],
        domain_scope: Optional[List[str]] = None,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
        limit: int = 8,
    ) -> List[str]:
        """
        Best-effort node resolution: map human strings -> graph node ids.

        Bounded and ownership-scoped:
        - Only returns nodes that have at least one relationship with (owner_type, owner_id)
        - Matches against common string properties (id/name/value/text)
        """
        graph_core = getattr(self._memory, "graph_core", None)
        if graph_core is None:
            logger.error("Environment.graph_resolve_nodes: graph_core is None")
            raise RuntimeError("Environment.graph_resolve_nodes: graph_core is None")

        limit_i = max(1, min(50, int(limit)))
        normalized_user_id = normalize_user_id(user_id)

        if owner_type == "agent":
            resolved_owner_id = owner_id or self._agent_id
            if not resolved_owner_id:
                raise RuntimeError("Environment.graph_resolve_nodes: missing agent_id for agent scope")
        else:
            resolved_owner_id = owner_id or normalized_user_id

        cleaned: List[str] = []
        seen = set()
        for n in names or []:
            s = str(n or "").strip().lower()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
            if len(cleaned) >= 20:
                break
        if not cleaned:
            return []

        adapter = getattr(graph_core, "adapter", None)
        run_query = getattr(adapter, "run_query", None)
        if not callable(run_query):
            return []

        domains = None
        if domain_scope:
            try:
                domains = [str(d).strip().lower() for d in (domain_scope or []) if d]
                domains = [d for d in domains if d]
            except Exception:
                domains = None

        try:
            rows = run_query(
                """
                MATCH (n)-[r]-()
                WHERE r.owner_type = $owner_type AND r.owner_id = $owner_id
                  AND ($domains IS NULL OR toLower(coalesce(r.domain, "")) IN $domains)
                  AND (
                    toLower(coalesce(n.name, "")) IN $names
                    OR toLower(coalesce(n.id, "")) IN $names
                    OR toLower(coalesce(n.value, "")) IN $names
                    OR toLower(coalesce(n.text, "")) IN $names
                  )
                RETURN DISTINCT n.id AS node_id
                LIMIT $limit
                """,
                params={
                    "names": cleaned,
                    "domains": domains,
                    "owner_type": owner_type,
                    "owner_id": resolved_owner_id,
                    "limit": limit_i,
                },
            )
        except Exception:
            logger.exception("Environment.graph_resolve_nodes failed")
            raise

        out: List[str] = []
        seen_ids = set()
        for row in rows or []:
            try:
                node_id = row.get("node_id") if isinstance(row, dict) else None
                s = str(node_id or "").strip()
                if not s:
                    continue
                if s in seen_ids:
                    continue
                seen_ids.add(s)
                out.append(s)
                if len(out) >= limit_i:
                    break
            except Exception:
                continue
        return out

    async def graph_neighbors(
        self,
        user_id: str,
        node_id: str,
        predicate_scope: Optional[List[str]] = None,
        domain_scope: Optional[List[str]] = None,
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
        graph_core = getattr(self._memory, "graph_core", None)
        if graph_core is None:
            logger.error("Environment.graph_neighbors: graph_core is None")
            raise RuntimeError("Environment.graph_neighbors: graph_core is None")

        k = self._validate_k("Environment.graph_neighbors", k)
        depth_i = self._safe_depth(depth)

        try:
            normalized_user_id = normalize_user_id(user_id)
            if not node_id:
                raise ValueError("Environment.graph_neighbors: node_id must be non-empty")

            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    raise RuntimeError("Environment.graph_neighbors: missing agent_id for agent scope")
            else:
                resolved_owner_id = owner_id or normalized_user_id
            results = await _maybe_await(
                graph_core.neighbors(
                    user_id=normalized_user_id,
                    node_id=node_id,
                    predicate_scope=predicate_scope,
                    domain_scope=domain_scope,
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
            raise
        

    async def expand_graph(
        self,
        user_id: str,
        subject: str,
        predicate: Optional[str] = None,
        hops: int = 1,
        direction: Optional[Literal["inbound", "outbound", "both"]] = None,
        k: int = 10,
        *,
        domain_scope: Optional[List[str]] = None,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        graph_core = getattr(self._memory, "graph_core", None)
        if graph_core is None:
            logger.error("Environment.expand_graph: graph_core is None")
            raise RuntimeError("Environment.expand_graph: graph_core is None")

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
            normalized_user_id = normalize_user_id(user_id)
            if dir_val and dir_val != "both":
                logger.debug(
                    "Environment.expand_graph: direction=%s currently treated as both", dir_val
                )

            # Best-effort: treat subject as a "name" and resolve to node ids first.
            resolved = await self.graph_resolve_nodes(
                user_id=normalized_user_id,
                names=[subject],
                domain_scope=domain_scope,
                owner_type=owner_type,
                owner_id=owner_id,
                limit=4,
            )
            node_ids = resolved or [subject]

            merged: List[Any] = []
            seen = set()
            for node_id in node_ids[:4]:
                items = await self.graph_neighbors(
                    user_id=normalized_user_id,
                    node_id=node_id,
                    predicate_scope=predicate_scope,
                    domain_scope=domain_scope,
                    depth=depth,
                    k=k,
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
                for it in items or []:
                    try:
                        if isinstance(it, dict):
                            props = it.get("properties") or {}
                            key = str(props.get("id") or it.get("id") or "") or str(it)
                        else:
                            key = str(it)
                        if key in seen:
                            continue
                        seen.add(key)
                        merged.append(it)
                    except Exception:
                        merged.append(it)

            return merged
        except Exception:
            logger.exception("Environment.expand_graph failed")
            raise

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

    async def execute_action(
        self,
        *,
        user_subject: str,
        action: Any,
        query_embedding: NumericVector,
        query_text: Optional[str],
        owner_type: str,
        owner_id: Optional[str],
        default_k: int,
        trace_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Execute a single RetrievalAction via the environment.

        This keeps the controller thin while maintaining the architectural rule:
        - chunk logic lives in ChunkCore
        - semantic logic lives in SemanticCore
        - episodic logic lives in EpisodicCore
        - environment handles scoping/bounds and calls cores
        """
        try:
            k = int(getattr(action, "k", None) or default_k)
        except Exception:
            k = int(default_k)

        lane_owner_type = getattr(action, "owner_type", None) or owner_type
        lane_owner_id = owner_id
        if lane_owner_type == "agent":
            lane_owner_id = self._agent_id if lane_owner_id is None else lane_owner_id

        a = getattr(action, "action", None)

        if a == "search_semantic":
            semantic_core = getattr(self._memory, "semantic_core", None)
            if semantic_core is None:
                logger.error("Environment.execute_action: semantic_core is None")
                raise RuntimeError("Environment.execute_action: semantic_core is None")
            filters = getattr(action, "filters", None)
            return await semantic_core.search(
                query_embedding=[float(x) for x in query_embedding],
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
                k=k,
                offset=0,
                filters=filters,
                query_text=query_text,
            )

        if a == "fetch_more_facts":
            offset = 0
            filters = getattr(action, "filters", None)
            if isinstance(filters, dict):
                try:
                    offset = int(filters.get("offset", 0) or 0)
                except Exception:
                    offset = 0
            return await self.fetch_more_facts(
                user_id=user_subject,
                predicate=getattr(action, "predicate", None),
                k=k,
                offset=offset,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )

        if a == "fetch_facts":
            return await self.fetch_facts_by_ids(
                user_id=user_subject,
                ids=getattr(action, "ids", None) or [],
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )

        if a == "fetch_chunks":
            return await self.fetch_chunks(
                user_id=user_subject,
                ids=getattr(action, "ids", None) or [],
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )

        if a == "search_chunks":
            chunk_core = getattr(self._memory, "chunk_core", None)
            if chunk_core is None:
                logger.error("Environment.execute_action: chunk_core is None")
                raise RuntimeError("Environment.execute_action: chunk_core is None")
            return await chunk_core.search_chunks_for_rlm(
                query_embedding=[float(x) for x in query_embedding],
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
                k=k,
                query_text=query_text,
            )

        if a == "search_episodic":
            episodic_core = getattr(self._memory, "episodic_core", None)
            if episodic_core is None:
                logger.error("Environment.execute_action: episodic_core is None")
                raise RuntimeError("Environment.execute_action: episodic_core is None")
            return await episodic_core.search(
                user_id=user_subject,
                query_embedding=[float(x) for x in query_embedding],
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
                k=k,
            )

        if a == "episodic_clusters":
            return await self.episodic_cluster_summaries(
                user_id=user_subject,
                k=k,
                max_episodes=int(default_k),
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )

        if a == "fetch_episode_clusters":
            return await self.fetch_episode_clusters(
                user_id=user_subject,
                k=k,
                max_episodes=int(default_k),
                time_range=getattr(action, "time_range", None),
                min_salience=getattr(action, "min_salience", None),
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )

        if a == "graph_neighbors":
            return await self.graph_neighbors(
                user_id=user_subject,
                node_id=getattr(action, "node_id", None),
                predicate_scope=getattr(action, "predicate_scope", None),
                depth=int(getattr(action, "depth", 1) or 1),
                k=k,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )

        if a == "expand_graph":
            return await self.expand_graph(
                user_id=user_subject,
                subject=getattr(action, "subject", None),
                predicate=getattr(action, "predicate", None),
                hops=int(getattr(action, "hops", 1) or 1),
                direction=getattr(action, "direction", None),
                k=k,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )

        if a == "search_procedural":
            procedural_core = getattr(self._memory, "procedural_core", None)
            if procedural_core is None:
                logger.error("Environment.execute_action: procedural_core is None")
                raise RuntimeError("Environment.execute_action: procedural_core is None")
            return await procedural_core.search(
                user_id=None,
                query_embedding=[float(x) for x in query_embedding],
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
                k=k,
            )

        raise ValueError(f"Environment.execute_action: unknown action={a!r}")
