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
from ..utils.dedupe import dedupe_by_id
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
        ctx = await memory.get_structured_context(user_id, query)

    Internally:
        • `get_structured_context()` delegates to RLMController if enabled
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
        max_chunks = int(getattr(retr_cfg, "max_chunks", max_facts))
        max_skills = int(getattr(retr_cfg, "max_skills"))
        max_graph_items = int(getattr(retr_cfg, "max_graph_items"))

        self.selector = MemorySelector(
            max_episodes=max_episodes,
            max_facts=max_facts,
            max_chunks=max_chunks,
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
        query_text: Optional[str],
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
            # Subject filtering is optional:
            # - user scope: keep subject=user
            # - agent KB: subject=None for corpus-wide search within owner scope
            subject = user_subject if owner_type == "user" else None
            if not (query_text and str(query_text).strip()):
                logger.warning(
                    "RetrievalService._raw_retrieval: semantic_core.search running embedding-only "
                    "(query_text missing/blank); results may be noisier."
                )
            tasks["facts"] = asyncio.create_task(
                semantic_core.search(
                    subject=subject,
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=self.selector.max_facts,
                    offset=0,
                    filters=None,
                    query_text=query_text,
                    allowed_topics=None,
                )
            )

        chunk_core = getattr(self.memory, "chunk_core", None)
        if chunk_core is not None:
            # Chunk retrieval is centralized in ChunkCore.search_chunks.
            # Include neighbor expansion here so raw retrieval is complete/deterministic.
            try:
                neighbor_window = int(getattr(getattr(self.memory, "retrieval_cfg", None), "neighbor_window", 1))
            except Exception:
                neighbor_window = 1
            try:
                max_expanded_chunks = int(getattr(getattr(self.memory, "retrieval_cfg", None), "max_expanded_chunks", 24))
            except Exception:
                max_expanded_chunks = 24
            try:
                lexical_k = int(getattr(getattr(self.memory, "retrieval_cfg", None), "lexical_chunks_k", 15))
            except Exception:
                lexical_k = 15
            tasks["chunks"] = asyncio.create_task(
                chunk_core.search_chunks(
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=self.selector.max_chunks,
                    query_text=query_text,
                    lexical_k=lexical_k,
                    filter_terms=bool(query_text and query_text.strip()),
                    expand_neighbors=True,
                    neighbor_window=neighbor_window,
                    max_expanded_chunks=max_expanded_chunks,
                    shortlist_k=int(getattr(getattr(self.memory, "retrieval_cfg", None), "chunk_shortlist_k", 12)),
                    shortlist_max_per_doc=int(getattr(getattr(self.memory, "retrieval_cfg", None), "chunk_shortlist_max_per_doc", 3)),
                )
            )

        procedural_core = getattr(self.memory, "procedural_core", None)
        if procedural_core is not None:
            tasks["skills"] = asyncio.create_task(
                procedural_core.search(
                    user_id=None,
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=self.selector.max_skills,
                )
            )

        # Graph is a navigation layer; do not fetch it eagerly here.
        graph_res: List[Any] = []

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

        if any(isinstance(x, dict) for x in (raw.get("chunks") or [])):
            raise TypeError(
                "RetrievalService expected Chunk objects in raw['chunks']; got dict(s). "
                "Fix chunk store/core to return Chunk only."
            )

        return raw

    async def retrieve(
        self,
        user_id: str,
        memory_type: str,
        query_text_or_embedding: Any,
        *,
        agent_id: Optional[str] = None,
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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RetrievalService.retrieve: scope owner_type=%s owner_id=%s user=%s",
                            owner_type,
                            owner_id,
                            user_subject,
                        )
                    try:
                        embedding = await self._ensure_embedding(query_text_or_embedding)
                        raw = await self._retrieve_raw(
                            user_subject=user_subject,
                            query_embedding=[float(x) for x in embedding],
                            query_text=query_text_or_embedding if isinstance(query_text_or_embedding, str) else None,
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

                    # selector expects keys: episodes/facts/chunks/skills/graph (+ optional WM)
                    def _drop_invalid(items: List[Any], *, kind: str) -> List[Any]:
                        out: List[Any] = []
                        logged = False
                        for it in items or []:
                            # Episodes/facts/skills may still be dicts depending on store/backends,
                            # but chunks must be canonical Chunk objects end-to-end.
                            if kind == "chunk" and isinstance(it, dict):
                                raise TypeError(
                                    "RetrievalService expected Chunk objects for chunks; got dict. "
                                    "Fix chunk store/core to return Chunk only."
                                )
                            it_id = it.get("id") if isinstance(it, dict) else getattr(it, "id", None)
                            it_owner_type = it.get("owner_type") if isinstance(it, dict) else getattr(it, "owner_type", None)
                            it_owner_id = it.get("owner_id") if isinstance(it, dict) else getattr(it, "owner_id", None)
                            if not it_id or not it_owner_type or it_owner_id is None:
                                if not logged:
                                    logger.warning(
                                        "RetrievalService: dropping invalid %s item missing id/owner fields: %r",
                                        kind,
                                        it,
                                    )
                                    logged = True
                                continue
                            out.append(it)
                        return out

                    raw["episodes"] = _drop_invalid(raw.get("episodes") or [], kind="episode")
                    raw["facts"] = _drop_invalid(raw.get("facts") or [], kind="fact")
                    raw["chunks"] = _drop_invalid(raw.get("chunks") or [], kind="chunk")
                    raw["skills"] = _drop_invalid(raw.get("skills") or [], kind="skill")
                    raw["graph"] = raw.get("graph") or []

                    selected = self.selector.select(raw, policy=policy)

                    # Extra defensive truncation guard:
                    # Even if a store misbehaves or returns too many items,
                    # RetrievalService MUST NEVER violate configured budgets.
                    episodes = (selected.get("episodes") or [])[: self.selector.max_episodes]
                    facts = (selected.get("facts") or [])[: self.selector.max_facts]
                    chunks = (selected.get("chunks") or [])[: self.selector.max_chunks]

                    # Neighbor expansion happens inside ChunkCore.search_chunks(expand_neighbors=True).

                    # Evidence expansion: fetch cited chunks for selected facts (bounded).
                    if facts and isinstance(query_text_or_embedding, str) and query_text_or_embedding.strip():
                        core = getattr(self.memory, "chunk_core", None)
                        if core is not None:
                            try:
                                cited_ids: List[str] = []
                                for f in facts:
                                    src = f.get("source_ids") if isinstance(f, dict) else getattr(f, "source_ids", None)
                                    if isinstance(src, list):
                                        for sid in src:
                                            if sid:
                                                cited_ids.append(str(sid))
                                max_ev = int(getattr(getattr(self.memory, "retrieval_cfg", None), "max_evidence_chunks", 6))
                                max_ev = max(0, max_ev)
                                cited_ids = list(dict.fromkeys(cited_ids))[: max_ev]
                                if cited_ids:
                                    cited_chunks = await core._fetch_ranked_by_ids(
                                        ids=cited_ids,
                                        owner_type=owner_type,
                                        owner_id=owner_id,
                                        log_context="chunks_evidence_expand",
                                    )
                                    try:
                                        for ch in cited_chunks or []:
                                            if isinstance(ch, dict):
                                                raise TypeError(
                                                    "RetrievalService expected Chunk objects from chunk_core; got dict. "
                                                    "Fix the chunk store/core to return Chunk only."
                                                )
                                            meta = getattr(ch, "meta", None) or {}
                                            if not isinstance(meta, dict):
                                                meta = {}
                                            meta.setdefault("retrieval_route", "evidence")
                                            meta.setdefault("retrieval_stage", "evidence_expand")
                                            ch.meta = meta
                                    except Exception:
                                        pass
                                    # Role-aware merge with deterministic precedence:
                                    # 1) evidence chunks (fact-cited)
                                    # 2) query-hit chunks (vector/lexical)
                                    # 3) neighbor chunks
                                    query_hits = []
                                    neighbors = []
                                    for ch in chunks or []:
                                        if isinstance(ch, dict):
                                            raise TypeError("Expected Chunk objects in chunks; got dict.")
                                        meta = getattr(ch, "meta", None) or {}
                                        route = meta.get("retrieval_route") if isinstance(meta, dict) else None
                                        if route == "neighbor":
                                            neighbors.append(ch)
                                        else:
                                            query_hits.append(ch)

                                    merged = dedupe_by_id(list(cited_chunks or []) + list(query_hits) + list(neighbors))
                                    chunks = merged[: self.selector.max_chunks]
                                    trace.append(
                                        {
                                            "phase": "baseline",
                                            "event": "chunks_evidence_expand",
                                            "count": len(cited_chunks or []),
                                        }
                                    )
                            except Exception:
                                logger.exception("RetrievalService: evidence expansion failed.")
                                strict = bool(getattr(self.memory, "retrieval_cfg", None) and self.memory.retrieval_cfg.strict)
                                if strict:
                                    raise
                    # Strict keyword gate for traditional RAG: only keep chunks that match query terms.
                    # Evidence-expanded chunks may not contain the literal query tokens, so keep cited evidence.
                    if isinstance(query_text_or_embedding, str) and query_text_or_embedding.strip():
                        cited_set = set(cited_ids) if "cited_ids" in locals() else set()
                        filtered = self._filter_chunks_by_query(chunks, query_text_or_embedding)
                        # Re-add cited evidence that may not match lexically.
                        if cited_set:
                            filtered_ids = set()
                            for c in filtered or []:
                                if isinstance(c, dict):
                                    raise TypeError("Expected Chunk objects in filtered chunks; got dict.")
                                cid = getattr(c, "id", None)
                                if cid:
                                    filtered_ids.add(str(cid))
                            for c in chunks:
                                if isinstance(c, dict):
                                    raise TypeError("Expected Chunk objects in chunks; got dict.")
                                cid = getattr(c, "id", None)
                                if cid and str(cid) in cited_set and str(cid) not in filtered_ids:
                                    filtered.append(c)
                        chunks = filtered
                    skills = (selected.get("skills") or [])[: self.selector.max_skills]
                    graph = (selected.get("graph") or [])[: self.selector.max_graph_items]

                    # Graph last-mile navigation: only fetch after other layers are available.
                    graph_core = getattr(self.memory, "graph_core", None)
                    if graph_core is not None and self.selector.max_graph_items > 0:
                        try:
                            # Gate: only pull graph if we have some signal to expand around.
                            should_fetch = bool(facts or chunks or episodes)
                            if should_fetch:
                                graph = graph_core.neighbors(
                                    user_id=user_subject,
                                    node_id=user_subject,
                                    depth=2,
                                    k=self.selector.max_graph_items,
                                    owner_type=owner_type,
                                    owner_id=owner_id,
                                ) or []
                        except Exception:
                            logger.exception("RetrievalService: graph retrieval failed.")
                            graph = []

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
                expected_dim = getattr(self.memory.embedder, "dimension", None)
                if not isinstance(expected_dim, int) or expected_dim <= 0:
                    raise ValueError("Embedder.dimension must be a positive integer.")
                # IMPORTANT: embedders expect List[str] -> List[List[float]]
                vectors = await self.memory.embedder.embed([query])
                if not vectors or not isinstance(vectors, list) or not vectors[0]:
                    raise ValueError("Embedder returned empty embedding.")
                vec0 = vectors[0]
                if not isinstance(vec0, list) or len(vec0) != expected_dim:
                    raise ValueError(f"Embedder returned invalid dim (expected={expected_dim} got={len(vec0) if isinstance(vec0, list) else None}).")
                return [float(x) for x in vec0]
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
            from ..utils.user_query_helper import build_query_term_set, text_matches_query_terms
        except Exception:
            build_query_term_set = None
            text_matches_query_terms = None

        if not chunks or not query_text:
            return chunks
        if not text_matches_query_terms:
            return chunks

        if not build_query_term_set:
            return chunks
        term_set = build_query_term_set(query_text)
        if not term_set or (not term_set.terms and not term_set.phrases):
            return []

        def _text(ch: Any) -> str:
            if isinstance(ch, dict):
                raise TypeError(
                    "RetrievalService._filter_chunks_by_query expected Chunk objects; got dict. "
                    "Fix chunk store/core to return Chunk only."
                )
            return (getattr(ch, "text", "") or "").lower()

        kept = []
        for ch in chunks:
            hay = _text(ch)
            if text_matches_query_terms(hay, term_set, min_term_matches=2, max_terms_for_match=6):
                kept.append(ch)
        return kept
