# uma/core/retrieval/rlm/environment.py

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Literal, Optional, Union

from ...utils.identity import ensure_user_subject
from ...utils.user_query_helper import extract_query_terms, expand_query_terms

logger = logging.getLogger(__name__)

NumericVector = List[Union[int, float]]


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def _extract_terms(query_text: str) -> List[str]:
    terms = expand_query_terms(query_text) or extract_query_terms(query_text)
    cleaned: List[str] = []
    for t in terms or []:
        if not isinstance(t, str):
            continue
        t = t.strip().lower()
        if not t or " " in t:
            continue
        if len(t) < 3:
            continue
        cleaned.append(t)
    if cleaned:
        return cleaned
    # Fallback: naive tokenization
    return [t.lower() for t in extract_query_terms(query_text) if isinstance(t, str) and t.strip()]


def _filter_chunks_by_terms(chunks: List[Any], terms: List[str]) -> List[Any]:
    filtered: List[Any] = []
    for ch in chunks:
        text = ""
        if isinstance(ch, dict):
            text = str(ch.get("text") or "")
        else:
            text = str(getattr(ch, "text", "") or "")
        if not text:
            continue
        lower = text.lower()
        if any(t in lower for t in terms):
            filtered.append(ch)
    return filtered


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
        self._project_id = getattr(memory, "project_id", None)

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
            vectors = await self._embedder.embed([query_text])
            if not vectors or not isinstance(vectors, list) or not vectors[0]:
                raise ValueError("Embedder returned empty embedding.")
            return [float(x) for x in vectors[0]]
        except Exception as exc:
            logger.exception("Environment.get_query_embedding failed")
            raise ValueError("Failed to embed query text.") from exc

    async def search_semantic(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Vector search over semantic facts.

        filters (optional):
        - topic: str
        - subject: str (must equal ensure_user_subject(user_id) or ignored)
        - offset: int
        """
        if self._semantic_core is None:
            return []

        k = self._validate_k("Environment.search_semantic", k)
        offset = self._safe_offset(filters.get("offset") if isinstance(filters, dict) else None)

        try:
            user_subject = ensure_user_subject(user_id)

            # Hard enforce subject scoping:
            # - if filters include subject, it must match the user's subject (or be ignored).
            subject = user_subject

            if isinstance(filters, dict) and "subject" in filters:
                provided = filters.get("subject")
                # Strict: do not allow cross-user subjects.
                try:
                    provided = ensure_user_subject(str(filters.get("subject")))
                    if provided == user_subject:
                        subject = provided
                except Exception:
                    pass

            retrieval_cfg = getattr(self._memory, "retrieval_cfg", None)
            ctx_cfg = getattr(retrieval_cfg, "context", None) if retrieval_cfg else None
            allowed_topics = getattr(ctx_cfg, "allowed_topics", None) if ctx_cfg else None
            if isinstance(allowed_topics, list):
                allowed_topics = [t for t in allowed_topics if isinstance(t, str) and t.strip()]
            else:
                allowed_topics = None

            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    return []
            else:
                resolved_owner_id = owner_id or user_subject
            facts = await self._semantic_core.search(
                subject=subject,
                query_embedding=list(query_embedding),
                owner_type=owner_type,
                owner_id=resolved_owner_id,
                k=int(k),
                offset=int(offset),
                filters=filters,
                query_text=query_text,
                allowed_topics=allowed_topics,
            )

            logger.debug("Environment.search_semantic: returned %d", len(facts))
            return facts
        except Exception:
            logger.exception("Environment.search_semantic failed")
            return []

    async def search_chunks(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> List[Any]:
        """
        Vector search over document chunks.
        """
        if self._chunk_core is None:
            return []
        k = self._validate_k("Environment.search_chunks", k)
        try:
            user_subject = ensure_user_subject(user_id)

            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    return []
            else:
                resolved_owner_id = owner_id or user_subject
            chunks = await self._chunk_core.search(
                user_id=user_subject,
                query_embedding=list(query_embedding),
                owner_type=owner_type,
                owner_id=resolved_owner_id,
                k=int(k),
            )
            logger.debug(
                "Environment.search_chunks: vector returned=%d owner_type=%s owner_id=%s",
                len(chunks or []),
                owner_type,
                resolved_owner_id,
            )
            if query_text and chunks:
                terms = _extract_terms(query_text)
                if terms:
                    logger.debug(
                        "Environment.search_chunks: filtering with terms=%s",
                        terms,
                    )
                    chunks = _filter_chunks_by_terms(chunks, terms)
                    logger.debug(
                        "Environment.search_chunks: filtered count=%d",
                        len(chunks or []),
                    )
            if (not chunks) and query_text:
                logger.debug("Environment.search_chunks: lexical fallback triggered")
                chunks = await self._chunk_core.search_text(
                    query_text=query_text,
                    owner_type=owner_type,
                    owner_id=resolved_owner_id,
                    k=int(k),
                )
                logger.debug(
                    "Environment.search_chunks: lexical returned=%d",
                    len(chunks or []),
                )
            logger.debug("Environment.search_chunks: returned %d", len(chunks))
            return chunks
        except Exception:
            logger.exception("Environment.search_chunks failed")
            return []

    async def search_procedural(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Vector search over procedural skills.
        """
        if self._procedural_core is None:
            return []
        k = self._validate_k("Environment.search_procedural", k)
        try:
            user_subject = ensure_user_subject(user_id)

            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    return []
            else:
                resolved_owner_id = owner_id or user_subject
            skills = await self._procedural_core.search(
                user_id=user_subject,
                query_embedding=list(query_embedding),
                owner_type=owner_type,
                owner_id=resolved_owner_id,
                k=int(k),
            )
            logger.debug("Environment.search_procedural: returned %d", len(skills))
            return skills
        except Exception:
            logger.exception("Environment.search_procedural failed")
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
            else:
                resolved_owner_id = owner_id or user_subject
            return await self._semantic_core.fetch_more_facts(
                subject=user_subject,
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

    async def search_episodic(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        time_range: Optional[Dict[str, Any]] = None,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Vector search over episodic summaries.

        time_range (optional):
        - start: comparable timestamp (store-defined)
        - end: comparable timestamp (store-defined)
        - offset: int
        """
        if self._episodic_core is None:
            return []
        k = self._validate_k("Environment.search_episodic", k)
        offset = self._safe_offset(time_range.get("offset") if isinstance(time_range, dict) else None)

        try:
            user_subject = ensure_user_subject(user_id)

            if owner_type == "agent":
                resolved_owner_id = owner_id or self._agent_id
                if not resolved_owner_id:
                    return []
            else:
                resolved_owner_id = owner_id or user_subject
            episodes = await self._episodic_core.search(
                user_id=user_subject,
                query_embedding=list(query_embedding),
                owner_type=owner_type,
                owner_id=resolved_owner_id,
                k=int(k),
                offset=int(offset),
            )

            episodes = self._filter_time_range(episodes or [], time_range)
            logger.debug("Environment.search_episodic: returned %d", len(episodes))
            return episodes
        except Exception:
            logger.exception("Environment.search_episodic failed")
            return []

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
