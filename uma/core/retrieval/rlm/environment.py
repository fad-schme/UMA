# uma/core/retrieval/rlm/environment.py

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from ...utils.identity import ensure_user_subject
from ...utils.user_query_helper import extract_query_terms, expand_query_terms, build_fact_embedding_text

logger = logging.getLogger(__name__)

NumericVector = List[Union[int, float]]


class MemoryEnvironment(Protocol):
    """
    Safe, read-only environment exposed to the RLM controller.

    Design contract
    ---------------
    - No raw DB access (controller never touches adapters/stores directly)
    - No arbitrary queries (only bounded, pre-defined methods)
    - All calls are user-scoped
    - All calls enforce limits (k, depth, etc.)
    """

    async def get_working_memory(self, user_id: str, window: Optional[int] = None) -> List[Any]: ...
    async def get_query_embedding(self, query_text: str) -> NumericVector: ...

    async def search_semantic(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

    async def search_chunks(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
    ) -> List[Dict[str, Any]]: ...

    async def fetch_facts_by_ids(self, user_id: str, ids: List[str]) -> List[Dict[str, Any]]: ...

    async def search_episodic(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        time_range: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: ...

    async def fetch_episode_summaries(self, user_id: str, ids: List[str]) -> List[Dict[str, Any]]: ...
    async def fetch_episode_transcripts(self, user_id: str, ids: List[str]) -> List[Dict[str, Any]]: ...

    async def episodic_cluster_summaries(
        self,
        user_id: str,
        k: int = 5,
        max_episodes: int = 50,
        time_range: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: ...

    async def fetch_episode_clusters(
        self,
        user_id: str,
        k: int = 5,
        max_episodes: int = 50,
        time_range: Optional[Dict[str, Any]] = None,
        min_salience: Optional[float] = None,
    ) -> List[Dict[str, Any]]: ...

    async def search_procedural(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: ...

    async def fetch_skills_by_ids(self, user_id: str, ids: List[str]) -> List[Dict[str, Any]]: ...

    async def graph_neighbors(
        self,
        user_id: str,
        node_id: str,
        predicate_scope: Optional[List[str]] = None,
        depth: int = 1,
        k: int = 10,
    ) -> List[Dict[str, Any]]: ...

    async def expand_graph(
        self,
        user_id: str,
        subject: str,
        predicate: Optional[str] = None,
        hops: int = 1,
        direction: Optional[Literal["inbound", "outbound", "both"]] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]: ...

    async def resolve_conflicts(self, user_id: str, fact_ids: List[str]) -> List[Dict[str, Any]]: ...


class UMAMemoryEnvironment:
    """
    Production implementation of MemoryEnvironment (read-only).

    This is the ONLY interface the RLM controller uses. It is designed to:
    - enforce scoping
    - enforce bounds
    - provide stable data shapes (dict snippets) regardless of store internals
    - avoid leaking DB/adapters into the controller loop
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory
        self._agent_id = getattr(memory, "agent_id", None)
        self._project_id = getattr(memory, "project_id", None)

        self._wm = getattr(memory, "working_memory", None)
        self._semantic_store = getattr(memory, "semantic_store", None)
        self._chunk_store = getattr(memory, "chunk_store", None)
        self._episodic_store = getattr(memory, "episodic_store", None)
        self._procedural_store = getattr(memory, "procedural_store", None)
        self._graph_core = getattr(memory, "graph_core", None)
        self._embedder = getattr(memory, "embedder", None)

        if self._embedder is None:
            raise ValueError("UMAMemoryEnvironment requires an embedder to operate")

        # Config-driven optional topic guardrails (if present)
        self._allowed_topics: Optional[List[str]] = None
        try:
            retrieval_cfg = getattr(memory, "retrieval_cfg", None)
            ctx_cfg = getattr(retrieval_cfg, "context", None) if retrieval_cfg else None
            allowed = getattr(ctx_cfg, "allowed_topics", None) if ctx_cfg else None
            if isinstance(allowed, list):
                self._allowed_topics = [t for t in allowed if isinstance(t, str) and t.strip()]
        except Exception:
            logger.exception("UMAMemoryEnvironment: failed to load allowed_topics")

        # Log missing subsystems. Not fatal for environment (controller can still run partially).
        if self._wm is None:
            logger.warning("UMAMemoryEnvironment: working_memory missing")
        if self._semantic_store is None:
            logger.warning("UMAMemoryEnvironment: semantic_store missing")
        if self._chunk_store is None:
            logger.warning("UMAMemoryEnvironment: chunk_store missing")
        if self._episodic_store is None:
            logger.warning("UMAMemoryEnvironment: episodic_store missing")
        if self._procedural_store is None:
            logger.warning("UMAMemoryEnvironment: procedural_store missing")
        if self._graph_core is None:
            logger.warning("UMAMemoryEnvironment: graph_core missing")

    @staticmethod
    def _iter_owner_filters(
        *,
        user_subject: str,
        agent_id: Optional[str],
        project_id: Optional[str],
    ) -> List[tuple[str, str]]:
        filters: List[tuple[str, str]] = [("user", user_subject)]
        if agent_id:
            filters.append(("agent", agent_id))
        if project_id:
            filters.append(("project", f"{user_subject}:{project_id}"))
        return filters

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
    # Embeddings + Working Memory
    # ------------------------------------------------------------------

    async def get_working_memory(self, user_id: str, window: Optional[int] = None) -> List[Any]:
        """
        Return last-N WM items for user. Never raises.

        Note:
        - WM is not vector-searched here; it is a sliding window buffer.
        """
        try:
            user_subject = ensure_user_subject(user_id)
            if not self._wm:
                return []
            results = self._wm.get_context(user_subject, last_n=window) or []
            logger.debug("Environment.get_working_memory: returned %d", len(results))
            return results
        except Exception:
            logger.exception("Environment.get_working_memory failed")
            return []

    async def get_query_embedding(self, query_text: str) -> NumericVector:
        """
        Embed a query string using configured embedder.

        Returns:
        - Numeric vector (list of floats) on success
        - [] on failure
        """
        if not isinstance(query_text, str) or not query_text.strip():
            return []
        try:
            # Standardize embedder interface: embed(List[str]) -> List[List[float]]
            vectors = await self._embedder.embed([query_text])
            if not vectors or not isinstance(vectors, list) or not vectors[0]:
                logger.error("Environment.get_query_embedding: empty embedding result")
                return []
            vec = [float(x) for x in vectors[0]]
            logger.debug("Environment.get_query_embedding: dim=%d", len(vec))
            return vec
        except Exception:
            logger.exception("Environment.get_query_embedding failed")
            return []

    # ------------------------------------------------------------------
    # Semantic
    # ------------------------------------------------------------------

    async def search_semantic(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector search over semantic facts.

        filters (optional):
        - topic: str
        - subject: str (must equal ensure_user_subject(user_id) or ignored)
        - offset: int
        """
        if self._semantic_store is None:
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

            requested_topic = filters.get("topic") if isinstance(filters, dict) else None

            facts: List[Any] = []
            # Best-effort offset support + multi-scope owner filters
            for owner_type, owner_id in self._iter_owner_filters(
                user_subject=subject,
                agent_id=self._agent_id,
                project_id=self._project_id,
            ):
                try:
                    try:
                        found = await self._semantic_store.search(
                            query_embedding=list(query_embedding),
                            subject=subject,
                            owner_type=owner_type,
                            owner_id=owner_id,
                            k=int(k),
                            offset=int(offset),
                        )
                    except TypeError:
                        found = await self._semantic_store.search(
                            query_embedding=list(query_embedding),
                            subject=subject,
                            owner_type=owner_type,
                            owner_id=owner_id,
                            k=int(k),
                        )
                    if found:
                        facts.extend(found)
                except Exception:
                    logger.exception(
                        "Environment.search_semantic: owner=%s:%s failed",
                        owner_type,
                        owner_id,
                    )

            # Optional topic filtering (soft)
            if requested_topic:
                filtered = [
                    f for f in facts
                    if requested_topic in _fact_topics(f)
                ]
                if filtered:
                    facts = filtered
            if self._allowed_topics:
                filtered = [
                    f for f in facts
                    if any(t in self._allowed_topics for t in _fact_topics(f))
                ]
                if filtered:
                    facts = filtered

            requested_predicate = filters.get("predicate") if isinstance(filters, dict) else None
            if requested_predicate:
                requested_predicate = str(requested_predicate).upper()
                facts = [
                    f for f in facts
                    if getattr(f, "predicate", "").upper() == requested_predicate
                ]

            if query_text:
                terms = expand_query_terms(query_text) or extract_query_terms(query_text)
                if terms:
                    lowered_terms = [t.lower() for t in terms]
                    original_count = len(facts)
                    filtered = []
                    for fact in facts:
                        text = build_fact_embedding_text(fact).lower()
                        if any(t in text for t in lowered_terms):
                            filtered.append(fact)
                    if filtered:
                        facts = filtered
                        logger.debug(
                            "Environment.search_semantic: lexical filter kept %d/%d",
                            len(facts),
                            original_count,
                        )
                    else:
                        try:
                            fallback: List[Any] = []
                            for owner_type, owner_id in self._iter_owner_filters(
                                user_subject=subject,
                                agent_id=self._agent_id,
                                project_id=self._project_id,
                            ):
                                try:
                                    found = await self._semantic_store.search_text(
                                        query=query_text,
                                        subject=subject,
                                        limit=int(k),
                                        owner_type=owner_type,
                                        owner_id=owner_id,
                                    )
                                    if found:
                                        fallback.extend(found)
                                except Exception:
                                    logger.exception(
                                        "Environment.search_semantic: lexical fallback owner=%s:%s failed",
                                        owner_type,
                                        owner_id,
                                    )
                            if fallback:
                                facts = fallback
                                logger.debug(
                                    "Environment.search_semantic: lexical fallback returned %d",
                                    len(facts),
                                )
                        except Exception:
                            logger.exception("Environment.search_semantic: lexical fallback failed")

            snippets = [self._fact_snippet(f) for f in facts]
            logger.debug("Environment.search_semantic: returned %d", len(snippets))
            return snippets
        except Exception:
            logger.exception("Environment.search_semantic failed")
            return []

    async def fetch_facts_by_ids(self, user_id: str, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch semantic facts by IDs (bounded by the controller).

        NOTE:
        - Store should enforce ownership if IDs are global.
        - Environment still scopes user_id at call boundary.
        """
        if self._semantic_store is None or not ids:
            return []
        try:
            _ = ensure_user_subject(user_id)  # ensures caller isn't passing garbage
            facts = await self._semantic_store.fetch_facts_by_ids(ids)
            snippets = [self._fact_snippet(f) for f in (facts or [])]
            logger.debug("Environment.fetch_facts_by_ids: returned %d", len(snippets))
            return snippets
        except Exception:
            logger.exception("Environment.fetch_facts_by_ids failed")
            return []


    async def fetch_more_facts(
        self,
        user_id: str,
        predicate: str,
        k: int,
        offset: int = 0,
        owner_scope: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self._semantic_store is None:
            return []
        try:
            user_subject = ensure_user_subject(user_id)
            k = self._validate_k("Environment.fetch_more_facts", k)
            offset = self._safe_offset(offset)

            scope = (owner_scope or "").lower()
            if scope and scope not in {"user", "agent", "project"}:
                scope = ""

            owner_filters: List[tuple[str, str]] = []
            if scope:
                if scope == "user":
                    owner_filters = [("user", user_subject)]
                elif scope == "agent" and self._agent_id:
                    owner_filters = [("agent", self._agent_id)]
                elif scope == "project" and self._project_id:
                    owner_filters = [("project", f"{user_subject}:{self._project_id}")]
                else:
                    return []
            else:
                owner_filters = self._iter_owner_filters(
                    user_subject=user_subject,
                    agent_id=self._agent_id,
                    project_id=self._project_id,
                )

            facts: List[Any] = []
            for owner_type, owner_id in owner_filters:
                # First, try store.search with offset (works for stores that support offset)
                try:
                    try:
                        found = await self._semantic_store.search(
                            query_embedding=[],  # empty embedding intended for predicate-only retrieval
                            subject=user_subject,
                            owner_type=owner_type,
                            owner_id=owner_id,
                            k=int(k),
                            offset=int(offset),
                        )
                    except TypeError:
                        found = await self._semantic_store.search(
                            query_embedding=[],
                            subject=user_subject,
                            owner_type=owner_type,
                            owner_id=owner_id,
                            k=int(k),
                        )
                    if found:
                        facts.extend(found)
                except Exception:
                    # fallback: if store provides predicate_scan/fetch_by_predicate, use it
                    if hasattr(self._semantic_store, "fetch_by_predicate"):
                        try:
                            found = await self._semantic_store.fetch_by_predicate(
                                subject=user_subject,
                                predicate=predicate,
                                limit=int(k),
                                offset=int(offset),
                                owner_type=owner_type,
                                owner_id=owner_id,
                            )
                            if found:
                                facts.extend(found)
                        except Exception:
                            logger.exception(
                                "Environment.fetch_more_facts: predicate fetch failed owner=%s:%s",
                                owner_type,
                                owner_id,
                            )
                    else:
                        logger.exception(
                            "Environment.fetch_more_facts: owner=%s:%s failed",
                            owner_type,
                            owner_id,
                        )

            # Filter by predicate (normalize)
            predicate_u = (predicate or "").upper()
            filtered = []
            for f in facts or []:
                pred_val = getattr(f, "predicate", None) if hasattr(f, "predicate") else (f.get("predicate") if isinstance(f, dict) else None)
                if pred_val and str(pred_val).upper() == predicate_u:
                    filtered.append(f)

            # If underlying store returned object wrappers, map to snippet
            snippets = [self._fact_snippet(f) for f in filtered]
            logger.debug("Environment.fetch_more_facts: returned %d", len(snippets))
            return snippets
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
    ) -> List[Dict[str, Any]]:
        """
        Vector search over episodic summaries.

        time_range (optional):
        - start: comparable timestamp (store-defined)
        - end: comparable timestamp (store-defined)
        - offset: int
        """
        if self._episodic_store is None:
            return []
        k = self._validate_k("Environment.search_episodic", k)
        offset = self._safe_offset(time_range.get("offset") if isinstance(time_range, dict) else None)

        try:
            user_subject = ensure_user_subject(user_id)

            try:
                episodes = await self._episodic_store.search(
                    query_embedding=list(query_embedding),
                    user_id=user_subject,
                    k=int(k),
                    offset=int(offset),
                )
            except TypeError:
                episodes = await self._episodic_store.search(
                    query_embedding=list(query_embedding),
                    user_id=user_subject,
                    k=int(k),
                )

            episodes = self._filter_time_range(episodes or [], time_range)
            snippets = [self._episode_snippet(ep) for ep in episodes]
            logger.debug("Environment.search_episodic: returned %d", len(snippets))
            return snippets
        except Exception:
            logger.exception("Environment.search_episodic failed")
            return []

    async def search_chunks(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        if self._chunk_store is None:
            return []
        k = self._validate_k("Environment.search_chunks", k)
        try:
            user_subject = ensure_user_subject(user_id)
            chunks: List[Any] = []
            for owner_type, owner_id in self._iter_owner_filters(
                user_subject=user_subject,
                agent_id=self._agent_id,
                project_id=self._project_id,
            ):
                try:
                    found = await self._chunk_store.search(
                        query_embedding=list(query_embedding),
                        owner_type=owner_type,
                        owner_id=owner_id,
                        k=int(k),
                    )
                    if found:
                        chunks.extend(found)
                except Exception:
                    logger.exception(
                        "Environment.search_chunks: owner=%s:%s failed",
                        owner_type,
                        owner_id,
                    )
            snippets = [
                {
                    "id": getattr(c, "id", None),
                    "doc_id": getattr(c, "doc_id", None),
                    "text": getattr(c, "text", None),
                    "page_range": getattr(c, "page_range", None),
                    "position": getattr(c, "position", None),
                    "meta": getattr(c, "meta", {}) or {},
                    "owner_type": getattr(c, "owner_type", None),
                    "owner_id": getattr(c, "owner_id", None),
                }
                for c in (chunks or [])
            ]
            logger.debug("Environment.search_chunks: returned %d", len(snippets))
            return snippets
        except Exception:
            logger.exception("Environment.search_chunks failed")
            return []
        

    async def fetch_episode_summaries(self, user_id: str, ids: List[str]) -> List[Dict[str, Any]]:
        if self._episodic_store is None or not ids:
            return []
        try:
            _ = ensure_user_subject(user_id)
            results = await self._episodic_store.fetch_summaries(ids) or []
            logger.debug("Environment.fetch_episode_summaries: returned %d", len(results))
            return results
        except Exception:
            logger.exception("Environment.fetch_episode_summaries failed")
            return []

    async def fetch_episode_transcripts(self, user_id: str, ids: List[str]) -> List[Dict[str, Any]]:
        if self._episodic_store is None or not ids:
            return []
        try:
            _ = ensure_user_subject(user_id)
            results = await self._episodic_store.fetch_transcripts(ids) or []
            logger.debug("Environment.fetch_episode_transcripts: returned %d", len(results))
            return results
        except Exception:
            logger.exception("Environment.fetch_episode_transcripts failed")
            return []

    async def episodic_cluster_summaries(
        self,
        user_id: str,
        k: int = 5,
        max_episodes: int = 50,
        time_range: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
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
            user_subject = ensure_user_subject(user_id)
            clusters = await episodic_core.list_cluster_summaries(
                user_id=user_subject,
                k=int(k),
                max_episodes=int(max_episodes),
                time_range=time_range,
            )
            clusters = clusters or []
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
    ) -> List[Dict[str, Any]]:
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
    # Procedural
    # ------------------------------------------------------------------

    async def search_procedural(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector search over procedural skills.

        NOTE:
        - Filters are store-dependent, but environment keeps them bounded.
        """
        if self._procedural_store is None:
            return []

        k = self._validate_k("Environment.search_procedural", k)
        try:
            user_subject = ensure_user_subject(user_id)
            # Procedural store should handle user scoping.
            skills = await self._procedural_store.search(
                query_embedding=list(query_embedding),
                user_id=user_subject,
                k=int(k),
            )
            snippets = [self._skill_snippet(s) for s in (skills or [])]
            logger.debug("Environment.search_procedural: returned %d", len(snippets))
            return snippets
        except Exception:
            logger.exception("Environment.search_procedural failed")
            return []

    async def fetch_skills_by_ids(self, user_id: str, ids: List[str]) -> List[Dict[str, Any]]:
        if self._procedural_store is None or not ids:
            return []
        try:
            _ = ensure_user_subject(user_id)
            skills = await self._procedural_store.fetch_skills_by_ids(ids)
            snippets = [self._skill_snippet(s) for s in (skills or [])]
            logger.debug("Environment.fetch_skills_by_ids: returned %d", len(snippets))
            return snippets
        except Exception:
            logger.exception("Environment.fetch_skills_by_ids failed")
            return []

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
    ) -> List[Dict[str, Any]]:
        """
        Safe bounded graph expansion.

        Requirements:
        - must be user-scoped
        - must be bounded (depth/k)
        - must not expose raw Cypher
        """
        if self._graph_core is None:
            return []

        k = self._validate_k("Environment.graph_neighbors", k)
        depth_i = self._safe_depth(depth)

        try:
            user_subject = ensure_user_subject(user_id)
            if not node_id:
                return []

            # Prefer a consistent async API on TemporalGraphCore
            if hasattr(self._graph_core, "neighbors"):
                results = self._graph_core.neighbors(
                    user_id=user_subject,
                    node_id=node_id,
                    predicate_scope=predicate_scope,
                    depth=depth_i,
                    k=k,
                )
                if asyncio.iscoroutine(results):
                    results = await results
                logger.debug("Environment.graph_neighbors: returned %d", len(results or []))
                return results or []

            logger.error("Environment.graph_neighbors: graph_core missing neighbors()")
            return []
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
    ) -> List[Dict[str, Any]]:
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

    async def resolve_conflicts(self, user_id: str, fact_ids: List[str]) -> List[Dict[str, Any]]:
        sanitized = self._limit_fact_ids(fact_ids)
        if not sanitized:
            return []
        if self._semantic_store is None:
            return []
        try:
            _ = ensure_user_subject(user_id)
            facts = await self.fetch_facts_by_ids(user_id, sanitized)
            return facts
        except Exception:
            logger.exception("Environment.resolve_conflicts failed")
            return []

    # ------------------------------------------------------------------
    # Snippet helpers (stable shapes)
    # ------------------------------------------------------------------

    def _fact_snippet(self, fact: Any) -> Dict[str, Any]:
        meta = getattr(fact, "meta", {}) or {}
        salience = meta.get("salience") if isinstance(meta, dict) else None
        return {
            "id": getattr(fact, "id", None),
            "subject": getattr(fact, "subject", None),
            "predicate": getattr(fact, "predicate", None),
            "object": getattr(fact, "object", None),
            "confidence": getattr(fact, "confidence", None),
            "salience": salience,
            "owner_type": getattr(fact, "owner_type", None),
            "owner_id": getattr(fact, "owner_id", None),
            "meta": meta if isinstance(meta, dict) else {},
        }

    def _episode_snippet(self, ep: Any) -> Dict[str, Any]:
        return {
            "id": getattr(ep, "id", None),
            "user_id": getattr(ep, "user_id", None),
            "timestamp": getattr(ep, "timestamp", None),
            "summary": getattr(ep, "summary", None),
        }

    def _skill_snippet(self, skill: Any) -> Dict[str, Any]:
        return {
            "id": getattr(skill, "id", None),
            "name": getattr(skill, "name", None),
            "description": getattr(skill, "description", None),
            "meta": getattr(skill, "meta", {}) or {},
        }

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


def _fact_topics(fact: Any) -> List[str]:
    meta = getattr(fact, "meta", {}) or {}
    if not isinstance(meta, dict):
        return []
    topics = meta.get("topics")
    if isinstance(topics, list):
        return [str(t) for t in topics if t]
    topic = meta.get("topic")
    if topic:
        return [str(topic)]
    return []
