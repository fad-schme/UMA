"""
retrieval/service.py
====================

RetrievalService v3 — Developer-Facing Retrieval API for UMA-3

This service provides a high-level, developer-friendly interface to UMA-3
memory retrieval. It hides embedding details and lets developers express
their intent in terms of:

    - which user (user_id)
    - which memory type (episodic, semantic, procedural, graph, working_memory, all)
    - what query (plain text or precomputed embedding)

Design
------
- If the query is plain text (str), it is embedded via UMA-3's embedder.
- If the query is a numeric vector (List[float|int]), it is used directly.
- Retrieval is delegated to MultiStoreRetriever (episodic, semantic, procedural, graph).
- Ranking and top-k selection are delegated to MemorySelector.

Public API
----------
    results = await retrieval_service.retrieve(
        user_id="u123",
        memory_type="semantic",
        query_text_or_embedding="What does the user like?"
    )

Coding Agent Instructions
-------------------------
- Do not add store-specific logic here; that belongs in MultiStoreRetriever.
- Do not add ranking logic here; that belongs in MemorySelector.
- This class is allowed to access UMA3Memory (llm, embedder, stores, config).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from .retrieval import MultiStoreRetriever
from .selector import MemorySelector

logger = logging.getLogger(__name__)

NumericVector = List[Union[float, int]]


class RetrievalService:
    """
    UMA-3 Unified Retrieval Service (Developer-Facing API).

    Parameters
    ----------
    memory : UMA3Memory
        The UMA3Memory instance to operate on (for embedder, stores, config).
    retr_cfg : Any
        Configuration object with attributes:
            - max_episodes
            - max_facts
            - max_skills
            - max_graph_items
    """

    def __init__(self, memory: Any, retr_cfg: Any) -> None:
        self.memory = memory

        # Read configuration with safe defaults.
        max_episodes = getattr(retr_cfg, "max_episodes", 3)
        max_facts = getattr(retr_cfg, "max_facts", 10)
        max_skills = getattr(retr_cfg, "max_skills", 3)
        max_graph_items = getattr(retr_cfg, "max_graph_items", 5)

        # Core retriever (multi-store, parallel).
        # IMPORTANT: pass all limits in, do NOT let MultiStoreRetriever
        # reach into memory.config.
        self.retriever = MultiStoreRetriever(
            max_episodes=max_episodes,
            max_facts=max_facts,
            max_skills=max_skills,
            max_graph_items=max_graph_items,
        )

        # Selector: scoring + top-k truncation.
        self.selector = MemorySelector(
            max_episodes=max_episodes,
            max_facts=max_facts,
            max_skills=max_skills,
            max_graph_items=max_graph_items,
        )

        logger.info(
            "RetrievalService initialized "
            "(episodes=%d, facts=%d, skills=%d, graph=%d)",
            max_episodes,
            max_facts,
            max_skills,
            max_graph_items,
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        user_id: str,
        memory_type: str,
        query_text_or_embedding: Any,
    ) -> Any:
        """
        Retrieve memory for a given user, memory type, and query.

        Parameters
        ----------
        user_id : str
            User identifier.
        memory_type : str
            One of:
                - "episodic"
                - "semantic"
                - "procedural"
                - "graph"
                - "working_memory"
                - "all"
        query_text_or_embedding : Any
            Either:
                - str  : plain text query to be embedded
                - list : numeric vector to be used as embedding

        Returns
        -------
        Any
            - For memory_type in {"episodic","semantic","procedural","graph","working_memory"}:
                returns List[Any] specific to that memory type.
            - For memory_type == "all":
                returns Dict[str, List[Any]] with keys:
                    {"episodes", "facts", "skills", "graph"}.

        Raises
        ------
        ValueError
            If query_text_or_embedding is missing, or memory_type is unsupported.
        """

        if query_text_or_embedding is None:
            raise ValueError("RetrievalService.retrieve: query_text_or_embedding is required.")

        memory_type = (memory_type or "").strip().lower()
        if not memory_type:
            raise ValueError("RetrievalService.retrieve: memory_type must not be empty.")

        # Special case: working memory does not use vector search.
        if memory_type == "working_memory":
            return self._get_working_memory(user_id)

        # 1. Ensure we have a numeric embedding.
        try:
            embedding = await self._ensure_embedding(query_text_or_embedding)
        except Exception:
            logger.exception("RetrievalService: failed to obtain embedding.")
            return [] if memory_type != "all" else {}

        # 2. Raw multi-store retrieval.
        try:
            raw_results = await self.retriever.retrieve(
                memory=self.memory,
                query_embedding=embedding,
                user_id=user_id,
            )
        except Exception:
            logger.exception("RetrievalService: multi-store retrieval failed.")
            return [] if memory_type != "all" else {}

        # 3. Selection/ranking.
        try:
            selected = self.selector.select(raw_results)
        except Exception:
            logger.exception("RetrievalService: selection failed.")
            return [] if memory_type != "all" else {}

        # 4. Slice by requested memory_type (selector-filtered lists).
        if memory_type == "episodic":
            return selected.get("episodes", [])

        elif memory_type == "semantic":
            return selected.get("semantic", selected.get("facts", []))

        elif memory_type == "procedural":
            return selected.get("procedural", selected.get("skills", []))

        elif memory_type == "graph":
            return selected.get("graph", [])

        elif memory_type == "working_memory":
            return self._get_working_memory(user_id)

        elif memory_type == "all":
            return {
                "episodes": selected.get("episodes", []),
                "semantic": selected.get("semantic", selected.get("facts", [])),
                "procedural": selected.get("procedural", selected.get("skills", [])),
                "graph": selected.get("graph", []),
            }

        logger.error("RetrievalService.retrieve: unsupported memory_type=%r", memory_type)
        raise ValueError(f"Unsupported memory_type: {memory_type!r}")

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    async def _ensure_embedding(self, query: Any) -> NumericVector:
        """
        Ensure a numeric embedding from either text or precomputed vector.

        Parameters
        ----------
        query : Any
            str  -> will be embedded using UMA-3's embedder
            list -> assumed to be a precomputed embedding

        Returns
        -------
        List[float]
            Numeric embedding for use in vector search.

        Raises
        ------
        ValueError
            If the query cannot be interpreted as text or numeric vector.
        """
        # Case 1: precomputed embedding (list of numeric values).
        if isinstance(query, list) and query and all(
            isinstance(x, (float, int)) for x in query
        ):
            return [float(x) for x in query]

        # Case 2: plain text string to embed.
        if isinstance(query, str):
            try:
                vectors = await self.memory.embedder.embed([query])
                if not vectors or not isinstance(vectors, list):
                    raise ValueError("Embedder returned no vectors.")
                return [float(x) for x in vectors[0]]
            except Exception as exc:
                logger.exception("RetrievalService._ensure_embedding: embed failed.")
                raise ValueError("Failed to embed query text.") from exc

        raise ValueError(
            "RetrievalService._ensure_embedding: query must be either "
            "a string or a list of numeric values."
        )

    def _get_working_memory(self, user_id: str) -> List[Any]:
        """
        Return working memory messages for the given user_id.

        Working memory is NOT retrieval-based; it is a direct state lookup.
        """
        wm = []
        try:
            if hasattr(self.memory, "working_memory") and self.memory.working_memory:
                wm = self.memory.working_memory.get_context(user_id)
        except Exception:
            logger.exception("RetrievalService._get_working_memory failed.")
        return wm