"""
uma.core.retrieval.service
===========================

RetrievalService — Developer-facing retrieval API.

Responsibilities
----------------
- Validate input: user_id, memory_type, query required.
- Convert query -> embedding (text or numeric vector).
- Call MultiStoreRetriever for raw results.
- Call MemorySelector for ranking + truncation.
- Return only the requested slice (list) or the "all" dict.

Design principle
----------------
No store-specific behavior here (belongs to MultiStoreRetriever).
No ranking here (belongs to MemorySelector).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from .retrieval import MultiStoreRetriever
from .selector import MemorySelector
from ...adapters.observability.context import get_request_id, request_context
from ...adapters.observability.metrics import increment, timed
from ..utils.identity import ensure_user_subject

logger = logging.getLogger(__name__)

NumericVector = List[Union[float, int]]


class RetrievalService:
    """
    RetrievalService — Deterministic UMA Memory Retrieval API.

    This service provides **single-shot, deterministic retrieval** across all
    UMA memory stores:

        • Episodic memory
        • Semantic memory (facts)
        • Procedural memory (skills)
        • Temporal graph (optional)

    It is the **baseline retrieval mechanism** used by UMA and serves as:
        • The fallback path when RLM retrieval is disabled or fails
        • The underlying primitive used by the RLMController for recursive retrieval

    What this service DOES
    ----------------------
    • Accepts a user_id, memory_type, and query (text or embedding)
    • Performs vector-based retrieval across configured stores
    • Applies deterministic ranking and truncation
    • Returns structured memory slices (lists or dicts)

    What this service DOES NOT do
    -----------------------------
    • Does not perform recursive retrieval
    • Does not perform agent reasoning or planning
    • Does not construct prompts
    • Does not mutate memory
    • Does not call LLMs except for embedding generation

    Design notes
    ------------
    • RetrievalService is intentionally simple and predictable.
    • It performs **exactly one retrieval pass per call**.
    • More advanced retrieval strategies (e.g. recursive exploration)
      are implemented in `RLMController`, not here.

    Typical usage
    -------------
    RetrievalService is not usually called directly by developers.

    Instead, developers use:
        ctx = await memory.get_user_context(user_id, query)

    Internally:
        • `get_user_context()` delegates to RLMController if enabled
        • otherwise falls back to RetrievalService

    This separation ensures:
        • predictable baseline behavior
        • safe, bounded advanced retrieval
        • clean architectural layering
    """
    def __init__(self, memory: Any, retr_cfg: Any) -> None:
        self.memory = memory

        max_episodes = int(getattr(retr_cfg, "max_episodes"))
        max_facts = int(getattr(retr_cfg, "max_facts"))
        max_skills = int(getattr(retr_cfg, "max_skills"))
        max_graph_items = int(getattr(retr_cfg, "max_graph_items"))

        self.retriever = MultiStoreRetriever(
            max_episodes=max_episodes,
            max_facts=max_facts,
            max_skills=max_skills,
            max_graph_items=max_graph_items,
        )
        self.selector = MemorySelector(
            max_episodes=max_episodes,
            max_facts=max_facts,
            max_skills=max_skills,
            max_graph_items=max_graph_items,
        )

        logger.info(
            "RetrievalService initialized: episodes=%d facts=%d skills=%d graph=%d",
            max_episodes,
            max_facts,
            max_skills,
            max_graph_items,
        )

    async def retrieve(self, user_id: str, memory_type: str, query_text_or_embedding: Any) -> Any:
        """
        Retrieve memory.

        Returns
        -------
        - memory_type in {"episodic","semantic","procedural","graph","working_memory"} -> List[Any]
        - memory_type == "all" -> Dict[str, List[Any]]
        """
        with request_context(generate=(get_request_id() == "-")):
            memory_type = (memory_type or "").strip().lower()
            increment("retrieval.retrieve.count", tags={"memory_type": memory_type or "unknown"})
            try:
                with timed("retrieval.retrieve.latency_s", tags={"memory_type": memory_type or "unknown"}):
                    if not user_id or not isinstance(user_id, str):
                        raise ValueError("RetrievalService.retrieve: user_id must be a non-empty string.")

                    user_subject = ensure_user_subject(user_id)
                    if query_text_or_embedding is None:
                        raise ValueError("RetrievalService.retrieve: query_text_or_embedding is required.")

                    if not memory_type:
                        raise ValueError("RetrievalService.retrieve: memory_type must not be empty.")

                    if memory_type == "working_memory":
                        return self._get_working_memory(user_subject)

                    embedding = await self._ensure_embedding(query_text_or_embedding)

                    raw = await self.retriever.retrieve(
                        memory=self.memory,
                        query_embedding=[float(x) for x in embedding],
                        user_id=user_subject,
                    )

                    # selector expects keys: episodes/facts/skills/graph (+ optional WM)
                    selected = self.selector.select(raw)

                    # Route
                    if memory_type == "episodic":
                        return selected["episodes"]
                    if memory_type == "semantic":
                        return selected["facts"]
                    if memory_type == "procedural":
                        return selected["skills"]
                    if memory_type == "graph":
                        return selected["graph"]
                    if memory_type == "all":
                        return {
                            "episodes": selected["episodes"],
                            "facts": selected["facts"],
                            "skills": selected["skills"],
                            "graph": selected["graph"],
                        }

                    raise ValueError(f"RetrievalService.retrieve: unsupported memory_type={memory_type!r}")
            except Exception:
                increment("retrieval.retrieve.error", tags={"memory_type": memory_type or "unknown"})
                raise

    async def _ensure_embedding(self, query: Any) -> NumericVector:
        """Accept either a numeric vector or a text string (embed it)."""
        # numeric vector
        if isinstance(query, list) and query and all(isinstance(x, (int, float)) for x in query):
            return [float(x) for x in query]

        # text query
        if isinstance(query, str) and query.strip():
            try:
                # IMPORTANT: your embedders expect List[str] -> List[List[float]]
                vectors = await self.memory.embedder.embed([query])
                if not vectors or not isinstance(vectors, list) or not vectors[0]:
                    raise ValueError("Embedder returned empty embedding.")
                return [float(x) for x in vectors[0]]
            except Exception as exc:
                logger.exception("RetrievalService._ensure_embedding: embed failed.")
                raise ValueError("Failed to embed query text.") from exc

        raise ValueError("RetrievalService._ensure_embedding: query must be a non-empty str or numeric vector list.")

    def _get_working_memory(self, user_id: str) -> List[Any]:
        """Working memory is direct state lookup (not vector retrieval)."""
        try:
            wm = getattr(self.memory, "working_memory", None)
            if wm is None:
                return []
            return wm.get_context(user_id)
        except Exception:
            logger.exception("RetrievalService._get_working_memory failed.")
            return []
