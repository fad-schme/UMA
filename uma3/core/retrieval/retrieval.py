"""
uma3.core.retrieval
===================

MultiStoreRetriever — UMA-3’s unified raw retrieval engine (Parallel Version)

This retriever:
    • Accepts an embedding of the query text
    • Retrieves candidate items from:
        - Episodic memory    (async)
        - Semantic memory    (async)
        - Procedural memory  (async)
        - Temporal graph     (sync)
    • Uses asyncio.gather for concurrency
    • NEVER throws exceptions — all errors are logged
    • Returns RAW results (selector handles filtering)

Coding Agent Instructions
-------------------------
- Never add ranking logic here (belongs in MemorySelector).
- Always catch exceptions and return safe empty lists.
- Keep this backend-agnostic: stores define their search APIs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from uma3.core.uma3_memory import UMA3Memory

logger = logging.getLogger(__name__)


class MultiStoreRetriever:
    """
    Parallel multi-store retriever.

    Output structure (raw):
        {
            "episodes": [...],
            "facts": [...],
            "skills": [...],
            "graph": [...],
        }

    Coding Agent Instructions
    -------------------------
    - Do NOT import UMA3Config or read memory.config here.
      Limits (max_episodes, max_facts, ...) are passed in from RetrievalService.
    - Treat `memory` as a duck-typed object exposing:
        episodic_store, semantic_store, procedural_store, graph_core.
    """

    def __init__(
        self,
        max_episodes: int = 20,
        max_facts: int = 50,
        max_skills: int = 20,
        max_graph_items: int = 30,
    ) -> None:
        self.max_episodes = max(1, int(max_episodes))
        self.max_facts = max(1, int(max_facts))
        self.max_skills = max(1, int(max_skills))
        self.max_graph_items = max(1, int(max_graph_items))

        logger.info(
            "MultiStoreRetriever initialized (parallel version; "
            "episodes=%d, facts=%d, skills=%d, graph=%d)",
            self.max_episodes,
            self.max_facts,
            self.max_skills,
            self.max_graph_items,
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        memory: "UMA3Memory",
        query_embedding: List[float],
        user_id: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        """
        Perform RAW retrieval from all UMA-3 subsystems in parallel.

        Parameters
        ----------
        memory : UMA3Memory
            UMA3Memory instance (or compatible) providing episodic_store,
            semantic_store, procedural_store, graph_core.
        query_embedding : List[float]
            Numeric embedding of the retrieval query.
        user_id : Optional[str]
            User identifier to scope results.

        Returns
        -------
        Dict[str, List[Any]]
            Raw, unranked retrieval results per store:
            {
                "episodes": [...],
                "facts": [...],
                "skills": [...],
                "graph": [...],
            }
        """

        # Prepare async tasks
        tasks = {
            "episodes": asyncio.create_task(
                self._episodic(memory, query_embedding, user_id, self.max_episodes)
            ),
            "facts": asyncio.create_task(
                self._semantic(memory, query_embedding, user_id, self.max_facts)
            ),
            "skills": asyncio.create_task(
                self._procedural(memory, query_embedding, self.max_skills)
            ),
        }

        # Graph is synchronous, not awaited
        try:
            graph_res = self._graph(
                memory,
                user_id=user_id,
                limit=self.max_graph_items,
            )
        except Exception:
            logger.exception("MultiStoreRetriever: graph retrieval failed.")
            graph_res = []

        # Execute tasks concurrently
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Map task results back into dict
        final: Dict[str, List[Any]] = {
            "episodes": [],
            "facts": [],
            "skills": [],
            "graph": graph_res,
        }

        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.exception("Retriever task '%s' failed.", key)
                final[key] = []
            else:
                final[key] = result

        return final

    # ------------------------------------------------------------------
    # INDIVIDUAL RETRIEVAL TASKS
    # ------------------------------------------------------------------

    async def _episodic(
        self,
        memory: "UMA3Memory",
        emb: List[float],
        user_id: Optional[str],
        k: int,
    ) -> List[Any]:
        """
        Episodic vector search.

        Returns a list of episodic memory entries (episodes), or [] on error.
        """
        store = getattr(memory, "episodic_store", None)
        if store is None:
            return []

        try:
            # Use positional args to be more tolerant of different store signatures.
            episodes = await store.search(emb, user_id, k)
            if not isinstance(episodes, list):
                return []
            return episodes
        except Exception:
            logger.exception("Episodic retrieval failed.")
            return []

    async def _semantic(
        self,
        memory: "UMA3Memory",
        emb: List[float],
        user_id: Optional[str],
        k: int,
    ) -> List[Any]:
        """
        Semantic (fact) vector search.

        Returns a list of semantic fact objects, or [] on error.
        """
        store = getattr(memory, "semantic_store", None)
        if store is None:
            return []

        try:
            facts = await store.search(emb, user_id, k)
            if not isinstance(facts, list):
                return []
            return facts
        except Exception:
            logger.exception("Semantic retrieval failed.")
            return []

    async def _procedural(
        self,
        memory: "UMA3Memory",
        emb: List[float],
        k: int,
    ) -> List[Any]:
        """
        Procedural (skill) vector search.

        Returns a list of procedural skills, or [] on error.
        """
        store = getattr(memory, "procedural_store", None)
        if store is None:
            return []

        try:
            skills = await store.search(emb, k)
            if not isinstance(skills, list):
                return []
            return skills
        except Exception:
            logger.exception("Procedural retrieval failed.")
            return []

    # NOTE: Graph retrieval is synchronous for Neo4j/Memgraph drivers.
    # We call it outside asyncio.gather.
    def _graph(
        self,
        memory: "UMA3Memory",
        user_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        User-scoped, predicate-aware graph retrieval.

        Retrieves nodes connected to the user's episodes OR semantic facts via
        predicate-scoped relationships.

        Returns [] on any error or if graph is disabled.
        """
        graph_core = getattr(memory, "graph_core", None)
        if graph_core is None or user_id is None:
            return []

        try:
            # For real graph backends we assume .adapter.run_query(...)
            adapter = getattr(graph_core, "adapter", None)
            if adapter is None or not hasattr(adapter, "run_query"):
                logger.debug("MultiStoreRetriever._graph: no adapter.run_query; skipping.")
                return []

            return adapter.run_query(
                """
                MATCH (u:User {id: $user_id})
                  -[:HAS_EPISODE|MENTIONS|
                    LIKES|PREFERS|DISLIKES|
                    WORKS_ON|INTERESTED_IN*1..3]-> (n)
                RETURN DISTINCT n, labels(n) AS labels, properties(n) AS properties
                LIMIT $limit
                """,
                params={"user_id": user_id, "limit": limit},
            )
        except Exception:
            logger.exception("Graph retrieval failed.")
            return []