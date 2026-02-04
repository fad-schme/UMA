from __future__ import annotations

"""
uma.core.retrieval.service
===========================

RetrievalService — Developer-facing retrieval API.

Responsibilities
----------------
- Validate input: user_id, memory_type, query required.
- Convert query -> embedding (text or numeric vector).
- Retrieve raw results from core subsystems.
- Call MemorySelector for ranking + truncation.
- Return only the requested slice (list) or the "all" dict.

Design principle
----------------
No store-specific behavior here (belongs to core subsystems).
No ranking here (belongs to MemorySelector).
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from .selector import MemorySelector
from .policy import RetrievalPolicy
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

    async def _retrieve_raw(
        self,
        *,
        user_subject: str,
        query_embedding: List[float],
        owner_type: str,
        owner_id: str,
    ) -> Dict[str, List[Any]]:
        """
        Perform raw retrieval from core subsystems.
        """
        tasks = {}

        episodic_core = getattr(self.memory, "episodic_core", None)
        if episodic_core is not None:
            tasks["episodes"] = asyncio.create_task(
                episodic_core.search(
                    user_id=user_subject,
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=self.selector.max_episodes,
                )
            )

        semantic_core = getattr(self.memory, "semantic_core", None)
        if semantic_core is not None:
            tasks["facts"] = asyncio.create_task(
                semantic_core.search(
                    subject=user_subject,
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=self.selector.max_facts,
                    offset=0,
                    filters=None,
                    query_text=None,
                    allowed_topics=None,
                )
            )

        chunk_core = getattr(self.memory, "chunk_core", None)
        if chunk_core is not None:
            tasks["chunks"] = asyncio.create_task(
                chunk_core.search(
                    user_id=user_subject,
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=self.selector.max_facts,
                    doc_id=None,
                )
            )

        procedural_core = getattr(self.memory, "procedural_core", None)
        if procedural_core is not None:
            tasks["skills"] = asyncio.create_task(
                procedural_core.search(
                    user_id=user_subject,
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=self.selector.max_skills,
                )
            )

        graph_core = getattr(self.memory, "graph_core", None)
        graph_res: List[Any] = []
        if graph_core is not None:
            try:
                graph_res = graph_core.neighbors(
                    user_id=user_subject,
                    node_id=user_subject,
                    depth=2,
                    k=self.selector.max_graph_items,
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
            except Exception:
                logger.exception("RetrievalService: graph retrieval failed.")
                graph_res = []

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        raw: Dict[str, List[Any]] = {
            "episodes": [],
            "facts": [],
            "chunks": [],
            "skills": [],
            "graph": graph_res,
        }

        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.exception("RetrievalService: task '%s' failed.", key)
                raw[key] = []
            else:
                raw[key] = result if isinstance(result, list) else []

        return raw

    async def retrieve(
        self,
        user_id: str,
        memory_type: str,
        query_text_or_embedding: Any,
        *,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Any:
        """
        Retrieve memory.

        Returns
        -------
        - memory_type in {"episodic","semantic","procedural","graph","chunks","working_memory"} -> List[Any]
        - memory_type == "all" -> Dict[str, List[Any]]
        """
        with request_context(generate=(get_request_id() == "-")):
            memory_type_norm = (memory_type or "").strip().lower()
            increment("retrieval.retrieve.count", tags={"memory_type": memory_type_norm or "unknown"})
            try:
                with timed("retrieval.retrieve.latency_s", tags={"memory_type": memory_type_norm or "unknown"}):
                    if not user_id or not isinstance(user_id, str):
                        raise ValueError("RetrievalService.retrieve: user_id must be a non-empty string.")

                    user_subject = ensure_user_subject(user_id)

                    if query_text_or_embedding is None:
                        raise ValueError("RetrievalService.retrieve: query_text_or_embedding is required.")

                    if not memory_type_norm:
                        raise ValueError("RetrievalService.retrieve: memory_type must not be empty.")

                    if memory_type_norm == "working_memory":
                        return self._get_working_memory(user_subject)

                    raw: Dict[str, List[Any]]
                    trace: List[Dict[str, Any]] = []
                    embedding: List[float]
                    policy = (
                        RetrievalPolicy(query_text_or_embedding)
                        if isinstance(query_text_or_embedding, str)
                        else None
                    )
                    if policy and policy.recall_score >= 0.75:
                        owner_type = "user"
                        owner_id = user_subject
                    else:
                        owner_type = "agent"
                        if not agent_id:
                            raise ValueError("RetrievalService.retrieve: agent_id is required for agent scope.")
                        owner_id = agent_id
                    try:
                        embedding = await self._ensure_embedding(query_text_or_embedding)
                        raw = await self._retrieve_raw(
                            user_subject=user_subject,
                            query_embedding=[float(x) for x in embedding],
                            owner_type=owner_type,
                            owner_id=owner_id,
                        )
                    except Exception:
                        logger.exception("RetrievalService: embedding failed.")
                        trace.append({"phase": "baseline", "event": "embedding_failed"})
                        strict = bool(getattr(self.memory, "retrieval_cfg", None) and self.memory.retrieval_cfg.strict)
                        if strict:
                            raise
                        # If embeddings fail (e.g., Ollama down), fall back to lexical chunks only.
                        raw = {"episodes": [], "facts": [], "chunks": [], "skills": [], "graph": []}

                    # Lexical fallback if chunk search returns nothing (or embed failed).
                    if not (raw.get("chunks") or []) and isinstance(query_text_or_embedding, str):
                        core = getattr(self.memory, "chunk_core", None)
                        if core is not None:
                            try:
                                chunks: List[Any] = []
                                found = await core.search_text(
                                    query_text_or_embedding,
                                    owner_type=owner_type,
                                    owner_id=owner_id,
                                    k=self.selector.max_facts,
                                )
                                if found:
                                    chunks.extend(found)
                                raw["chunks"] = chunks
                                trace.append(
                                    {
                                        "phase": "baseline",
                                        "event": "chunks_lexical_fallback",
                                        "reason": "chunks_empty",
                                    }
                                )
                                logger.info("RetrievalService: chunks lexical fallback (chunks_empty).")
                            except Exception:
                                logger.exception("RetrievalService: chunk lexical fallback failed.")
                                strict = bool(getattr(self.memory, "retrieval_cfg", None) and self.memory.retrieval_cfg.strict)
                                if strict:
                                    raise

                    # selector expects keys: episodes/facts/chunks/skills/graph (+ optional WM)
                    selected = self.selector.select(raw, policy=policy)

                    # Extra defensive truncation guard:
                    # Even if a store misbehaves or returns too many items,
                    # RetrievalService MUST NEVER violate configured budgets.
                    episodes = (selected.get("episodes") or [])[: self.selector.max_episodes]
                    facts = (selected.get("facts") or [])[: self.selector.max_facts]
                    chunks = (selected.get("chunks") or [])[: self.selector.max_facts]
                    # Strict keyword gate for traditional RAG: only keep chunks that match query terms.
                    if isinstance(query_text_or_embedding, str) and query_text_or_embedding.strip():
                        chunks = self._filter_chunks_by_query(chunks, query_text_or_embedding)
                    skills = (selected.get("skills") or [])[: self.selector.max_skills]
                    graph = (selected.get("graph") or [])[: self.selector.max_graph_items]

                    trace.append(
                        {
                            "phase": "baseline",
                            "event": "baseline_complete",
                            "counts": {
                                "episodes": len(episodes),
                                "facts": len(facts),
                                "chunks": len(chunks),
                                "skills": len(skills),
                                "graph": len(graph),
                            },
                        }
                    )

                    if memory_type_norm == "episodic":
                        return episodes
                    if memory_type_norm == "semantic":
                        return facts
                    if memory_type_norm == "chunks":
                        return chunks
                    if memory_type_norm == "procedural":
                        return skills
                    if memory_type_norm == "graph":
                        return graph
                    if memory_type_norm == "all":
                        return {
                            "episodes": episodes,
                            "facts": facts,
                            "chunks": chunks,
                            "skills": skills,
                            "graph": graph,
                            "trace": trace,
                        }

                    raise ValueError(f"RetrievalService.retrieve: unsupported memory_type={memory_type_norm!r}")
            except Exception:
                increment("retrieval.retrieve.error", tags={"memory_type": memory_type_norm or "unknown"})
                raise

    async def _ensure_embedding(self, query: Any) -> NumericVector:
        """Accept either a numeric vector or a text string (embed it)."""
        # numeric vector
        if isinstance(query, list) and query and all(isinstance(x, (int, float)) for x in query):
            return [float(x) for x in query]

        # text query
        if isinstance(query, str) and query.strip():
            try:
                # IMPORTANT: embedders expect List[str] -> List[List[float]]
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

    @staticmethod
    def _filter_chunks_by_query(chunks: List[Any], query_text: str) -> List[Any]:
        """
        Hard filter: only keep chunks containing query terms (case-insensitive).
        If no chunks match, return empty list.
        """
        try:
            from ..utils.user_query_helper import extract_query_terms, expand_query_terms
        except Exception:
            extract_query_terms = None
            expand_query_terms = None

        if not chunks or not query_text:
            return chunks
        if expand_query_terms:
            terms = expand_query_terms(query_text)
        elif extract_query_terms:
            terms = extract_query_terms(query_text)
        else:
            terms = []
        terms = [t for t in (terms or []) if t]
        if not terms:
            # Fallback to simple tokenization if helper returns nothing.
            import re
            terms = re.findall(r"[a-zA-Z0-9]+", query_text.lower())
            if not terms:
                return []
        # Drop stopwords and very short tokens to avoid matching everything.
        stop = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "to",
            "of", "in", "on", "at", "by", "with", "about", "what", "who", "why",
            "how", "when", "where", "is", "are", "was", "were", "be", "been",
            "do", "does", "did", "you", "your", "know", "tell", "me", "please",
        }
        filtered_terms = []
        for t in terms:
            tl = t.lower()
            if len(tl) < 3:
                continue
            if tl in stop:
                continue
            filtered_terms.append(tl)
        if not filtered_terms:
            return []

        def _text(ch: Any) -> str:
            txt = getattr(ch, "text", None)
            if txt is None and isinstance(ch, dict):
                txt = ch.get("text")
            return (txt or "").lower()

        kept = []
        for ch in chunks:
            hay = _text(ch)
            if any(t in hay for t in filtered_terms):
                kept.append(ch)
        return kept
