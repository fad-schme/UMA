# uma/core/retrieval/rlm/environment.py

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Protocol, Union

from ...utils.identity import ensure_user_subject

logger = logging.getLogger(__name__)

NumericVector = List[Union[int, float]]


class MemoryEnvironment(Protocol):
    """
    Safe, read-only environment exposed to the RLM controller.

    IMPORTANT:
    - No raw DB access
    - No arbitrary queries
    - All calls must be bounded
    """

    async def search_semantic(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: ...

    async def fetch_facts_by_ids(self, user_id: str, ids: List[str]) -> List[Any]: ...

    async def search_episodic(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        time_range: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: ...

    async def fetch_episode_summaries(self, ids: List[str]) -> List[Dict[str, Any]]: ...

    async def fetch_episode_transcripts(self, ids: List[str]) -> List[Dict[str, Any]]: ...

    async def graph_neighbors(
        self,
        user_id: str,
        node_id: str,
        predicate_scope: Optional[List[str]] = None,
        depth: int = 1,
        k: int = 10,
    ) -> List[Dict[str, Any]]: ...

    async def episodic_cluster_summaries(
        self,
        user_id: str,
        k: int = 5,
        max_episodes: int = 50,
        time_range: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: ...

    async def get_working_memory(self, user_id: str, window: Optional[int] = None) -> List[Any]: ...
    async def get_query_embedding(self, query_text: str) -> NumericVector: ...

    # Legacy methods (kept for backward compatibility)
    async def retrieve_slice(
        self,
        user_id: str,
        memory_type: str,
        query: Union[str, NumericVector],
    ) -> List[Any]: ...

    async def retrieve_all(
        self,
        user_id: str,
        query: Union[str, NumericVector],
    ) -> Dict[str, List[Any]]: ...


class UMAMemoryEnvironment:
    """
    Production implementation of MemoryEnvironment.

    Wraps:
    - RetrievalService
    - WorkingMemoryCore

    This is the *only* surface the RLMController can see.
    """

    def __init__(self, memory: Any) -> None:
        self._retrieval = getattr(memory, "retrieval_service", None)
        self._wm = getattr(memory, "working_memory", None)
        self._semantic_store = getattr(memory, "semantic_store", None)
        self._episodic_store = getattr(memory, "episodic_store", None)
        self._graph_core = getattr(memory, "graph_core", None)
        self._embedder = getattr(memory, "embedder", None)
        self._allowed_topics = None
        retrieval_cfg = getattr(memory, "retrieval_cfg", None)
        ctx_cfg = getattr(retrieval_cfg, "context", None) if retrieval_cfg else None
        if ctx_cfg and getattr(ctx_cfg, "allowed_topics", None):
            self._allowed_topics = [t for t in ctx_cfg.allowed_topics if isinstance(t, str)]

        if self._retrieval is None:
            raise ValueError("UMAMemoryEnvironment requires retrieval_service")

        if self._wm is None:
            logger.warning("UMAMemoryEnvironment: working_memory missing")

        if self._semantic_store is None:
            logger.warning("UMAMemoryEnvironment: semantic_store missing")
        if self._episodic_store is None:
            logger.warning("UMAMemoryEnvironment: episodic_store missing")
        if self._embedder is None:
            logger.warning("UMAMemoryEnvironment: embedder missing")

    # ------------------------------------------------------------------
    # New granular API (snippet-first)
    # ------------------------------------------------------------------

    async def search_semantic(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._semantic_store is None:
            return []
        try:
            # Enforce user scoping: default subject to the requesting user to
            # prevent LLM-provided filters from performing cross-user retrieval.
            user_subject = ensure_user_subject(user_id)
            if isinstance(filters, dict) and "subject" in filters:
                provided = filters.get("subject")
                try:
                    provided_subject = ensure_user_subject(str(provided))
                except Exception as exc:
                    logger.warning(
                        "Environment.search_semantic: invalid subject filter %r: %s",
                        provided,
                        exc,
                    )
                    provided_subject = None

                # If the LLM supplied a subject, do not allow accessing other
                # users' data — enforce scoping to the requesting user.
                if provided_subject and provided_subject != user_subject:
                    logger.warning(
                        "Environment.search_semantic: subject filter %r ignored for user=%s",
                        provided_subject,
                        user_subject,
                    )
                    subject = user_subject
                else:
                    subject = provided_subject or user_subject
            else:
                subject = user_subject
            requested_topic = filters.get("topic") if isinstance(filters, dict) else None
            if isinstance(filters, dict):
                unsupported = [k for k in filters.keys() if k not in {"subject", "topic"}]
                if unsupported:
                    logger.warning(
                        "Environment.search_semantic: unsupported filters=%s",
                        unsupported,
                    )
            facts = await self._semantic_store.search(
                query_embedding=list(query_embedding),
                subject=subject,
                k=int(k),
            )
            if requested_topic:
                facts = [f for f in facts if (getattr(f, "meta", {}) or {}).get("topic") == requested_topic]
            if self._allowed_topics:
                facts = [f for f in facts if (getattr(f, "meta", {}) or {}).get("topic") in self._allowed_topics]
            return [self._fact_snippet(f) for f in facts]
        except Exception:
            logger.exception("Environment.search_semantic failed")
            raise

    async def fetch_facts_by_ids(self, user_id: str, ids: List[str]) -> List[Any]:
        if self._semantic_store is None:
            return []
        if not ids:
            return []
        try:
            return await self._semantic_store.fetch_facts_by_ids(ids)
        except Exception:
            logger.exception("Environment.fetch_facts_by_ids failed")
            raise

    async def search_episodic(
        self,
        user_id: str,
        query_embedding: NumericVector,
        k: int = 10,
        time_range: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._episodic_store is None:
            return []
        try:
            user_subject = ensure_user_subject(user_id)
            episodes = await self._episodic_store.search(
                query_embedding=list(query_embedding),
                user_id=user_subject,
                k=int(k),
            )
            filtered = self._filter_time_range(episodes, time_range)
            return [self._episode_snippet(ep) for ep in filtered]
        except Exception:
            logger.exception("Environment.search_episodic failed")
            raise

    async def fetch_episode_summaries(self, ids: List[str]) -> List[Dict[str, Any]]:
        if self._episodic_store is None:
            return []
        if not ids:
            return []
        try:
            # Fetch summaries only (small snippets), preserving input order downstream.
            return await self._episodic_store.fetch_summaries(ids)
        except Exception:
            logger.exception("Environment.fetch_episode_summaries failed")
            raise

    async def fetch_episode_transcripts(self, ids: List[str]) -> List[Dict[str, Any]]:
        if self._episodic_store is None:
            return []
        if not ids:
            return []
        try:
            # Fetch full transcripts only when explicitly requested.
            return await self._episodic_store.fetch_transcripts(ids)
        except Exception:
            logger.exception("Environment.fetch_episode_transcripts failed")
            raise

    async def graph_neighbors(
        self,
        user_id: str,
        node_id: str,
        predicate_scope: Optional[List[str]] = None,
        depth: int = 1,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        if self._graph_core is None:
            return []
        try:
            user_subject = ensure_user_subject(user_id)
            if hasattr(self._graph_core, "neighbors"):
                return self._graph_core.neighbors(
                    user_id=user_subject,
                    node_id=node_id,
                    predicate_scope=predicate_scope,
                    depth=depth,
                    k=k,
                )
            if hasattr(self._graph_core, "get_neighbors"):
                results = self._graph_core.get_neighbors(
                    entity_id=node_id,
                    depth=depth,
                )
                if k:
                    return results[: int(k)]
                return results
            logger.warning("Environment.graph_neighbors: graph_core has no neighbor query method.")
            return []
        except Exception:
            logger.exception("Environment.graph_neighbors failed")
            raise

    async def episodic_cluster_summaries(
        self,
        user_id: str,
        k: int = 5,
        max_episodes: int = 50,
        time_range: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._episodic_store is None:
            return []
        try:
            user_subject = ensure_user_subject(user_id)
            return await self._episodic_store.list_cluster_summaries(
                user_id=user_subject,
                k=int(k),
                time_range=time_range,
                max_episodes=max_episodes,
            )
        except Exception:
            logger.exception("Environment.episodic_cluster_summaries failed")
            raise

    async def get_working_memory(self, user_id: str, window: Optional[int] = None):
        try:
            user_subject = ensure_user_subject(user_id)
            return self._wm.get_context(user_subject, last_n=window) if self._wm else []
        except Exception:
            logger.exception("Environment.get_working_memory failed")
            return []

    async def get_query_embedding(self, query_text: str) -> NumericVector:
        if not isinstance(query_text, str) or not query_text.strip():
            return []
        if self._embedder is None:
            logger.error("Environment.get_query_embedding: embedder unavailable")
            return []
        try:
            vectors = await self._embedder.embed([query_text])
            if not vectors or not isinstance(vectors, list) or not vectors[0]:
                logger.error("Environment.get_query_embedding: empty embedding result")
                return []
            return [float(x) for x in vectors[0]]
        except Exception:
            logger.exception("Environment.get_query_embedding failed")
            return []

    # ------------------------------------------------------------------
    # Legacy API (kept for compatibility)
    # ------------------------------------------------------------------

    async def retrieve_slice(self, user_id: str, memory_type: str, query):
        try:
            res = await self._retrieval.retrieve(
                user_id=user_id,
                memory_type=memory_type,
                query_text_or_embedding=query,
            )
            return res if isinstance(res, list) else []
        except Exception:
            logger.exception("Environment.retrieve_slice failed")
            return []

    async def retrieve_all(self, user_id: str, query):
        try:
            res = await self._retrieval.retrieve(
                user_id=user_id,
                memory_type="all",
                query_text_or_embedding=query,
            )
            if not isinstance(res, dict):
                return {"episodes": [], "facts": [], "skills": [], "graph": []}
            return {
                "episodes": res.get("episodes", []) or [],
                "facts": res.get("facts", []) or [],
                "skills": res.get("skills", []) or [],
                "graph": res.get("graph", []) or [],
            }
        except Exception:
            logger.exception("Environment.retrieve_all failed")
            return {"episodes": [], "facts": [], "skills": [], "graph": []}

    # ------------------------------------------------------------------
    # Snippet helpers
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
            "meta": meta if isinstance(meta, dict) else {},
        }

    def _episode_snippet(self, ep: Any) -> Dict[str, Any]:
        return {
            "id": getattr(ep, "id", None),
            "user_id": getattr(ep, "user_id", None),
            "timestamp": getattr(ep, "timestamp", None),
            "summary": getattr(ep, "summary", None),
        }

    def _filter_time_range(self, episodes: List[Any], time_range: Optional[Dict[str, Any]]) -> List[Any]:
        if not time_range:
            return episodes
        start = time_range.get("start")
        end = time_range.get("end")
        if start is None and end is None:
            return episodes
        filtered = []
        for ep in episodes:
            ts = getattr(ep, "timestamp", None)
            if ts is None:
                continue
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            filtered.append(ep)
        return filtered
