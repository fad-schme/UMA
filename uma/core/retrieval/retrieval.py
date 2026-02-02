"""
uma.core.retrieval.retrieval
=============================

MultiStoreRetriever — UMA raw multi-store retrieval engine.

Responsibilities
----------------
- Accept an embedding vector (List[float]) and a user_id.
- Retrieve candidates from:
    - episodic_core    (async)
    - semantic_core    (async)
    - procedural_core  (async)
    - chunk_core       (async)
    - graph_core       (sync)
- Concurrency: uses asyncio.gather for async stores.
- NEVER raises: logs exceptions and returns safe empty lists.

Important
---------
- This module is backend-agnostic. It does NOT know store implementations.
- It does NOT do ranking, dedupe, or truncation logic beyond store-level k.
  Ranking/truncation belongs in MemorySelector.

Expected core signatures (duck-typing)
--------------------------------------
- episodic_core.search(owner_type, owner_id, embedding, k) -> List[Any]
- semantic_core.search(subject, embedding, k) -> List[Any]
- procedural_core.search(embedding, k) -> List[Any]
- chunk_core.search(embedding, k) -> List[Any]
- graph_core.query(cypher, params=dict) -> List[dict]
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..utils.identity import ensure_user_subject

if TYPE_CHECKING:
    from ..uma_memory import UMAMemory

logger = logging.getLogger(__name__)


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


class MultiStoreRetriever:
    """
    Raw multi-store retriever.

    Returns:
        {
            "episodes": [...],
            "facts": [...],
            "chunks": [...],
            "skills": [...],
            "graph": [...],
        }
    """

    def __init__(
        self,
        max_episodes: int,
        max_facts: int,
        max_skills: int,
        max_graph_items: int,
    ) -> None:
        self.max_episodes = max(1, int(max_episodes))
        self.max_facts = max(1, int(max_facts))
        self.max_skills = max(1, int(max_skills))
        self.max_graph_items = max(1, int(max_graph_items))

        logger.info(
            "MultiStoreRetriever initialized: episodes=%d facts=%d skills=%d graph=%d",
            self.max_episodes,
            self.max_facts,
            self.max_skills,
            self.max_graph_items,
        )

    async def retrieve(
        self,
        memory: "UMAMemory",
        query_embedding: List[float],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        """
        Perform raw retrieval from subsystems.

        This function never raises; errors are logged and replaced with [].
        """
        # Prepare async tasks (episodic/semantic/procedural)
        tasks = {
            "episodes": asyncio.create_task(
                self._episodic(memory, query_embedding, user_id, self.max_episodes)
            ),
            "facts": asyncio.create_task(
                self._semantic(
                    memory,
                    query_embedding,
                    user_id,
                    self.max_facts,
                    agent_id=agent_id,
                    project_id=project_id,
                )
            ),
            "chunks": asyncio.create_task(
                self._chunks(
                    memory,
                    query_embedding,
                    user_id,
                    self.max_facts,
                    agent_id=agent_id,
                    project_id=project_id,
                )
            ),
            "skills": asyncio.create_task(
                self._procedural(memory, query_embedding, self.max_skills)
            ),
        }

        # Graph retrieval is sync (driver-dependent); keep outside gather
        graph_res: List[Any]
        try:
            graph_res = self._graph(memory, user_id=user_id, limit=self.max_graph_items)
        except Exception:
            logger.exception("MultiStoreRetriever: graph retrieval failed.")
            graph_res = []

        # Run async tasks concurrently
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        final: Dict[str, List[Any]] = {
            "episodes": [],
            "facts": [],
            "chunks": [],
            "skills": [],
            "graph": graph_res,
        }

        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.exception("MultiStoreRetriever: task '%s' failed.", key)
                final[key] = []
            else:
                final[key] = result if isinstance(result, list) else []

        return final

    async def _episodic(
        self,
        memory: "UMAMemory",
        emb: List[float],
        user_id: Optional[str],
        k: int,
    ) -> List[Any]:
        core = getattr(memory, "episodic_core", None)
        if core is None:
            return []
        if user_id is None:
            return []
        try:
            subject = ensure_user_subject(user_id)
            res = await core.search(
                owner_type="user",
                owner_id=subject,
                query_embedding=emb,
                k=int(k),
            )
            return res if isinstance(res, list) else []
        except Exception:
            logger.exception("MultiStoreRetriever: episodic retrieval failed.")
            return []

    async def _semantic(
        self,
        memory: "UMAMemory",
        emb: List[float],
        user_id: Optional[str],
        k: int,
        *,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Semantic retrieval.

        IMPORTANT:
        SemanticSQLStore.search() has signature:
            search(query_embedding, subject=None, owner_type=None, owner_id=None, k=...)
        so we MUST call it using keyword arguments to avoid positional mismatch.
        """
        core = getattr(memory, "semantic_core", None)
        if core is None:
            return []
        if user_id is None:
            return []

        try:
            subject = ensure_user_subject(user_id)
            res: List[Any] = []
            for owner_type, owner_id in _iter_owner_filters(
                user_subject=subject,
                agent_id=agent_id,
                project_id=project_id,
            ):
                try:
                    found = await core.search(
                        subject=subject,
                        query_embedding=emb,
                        owner_type=owner_type,
                        owner_id=owner_id,
                        k=int(k),
                    )
                    if found:
                        res.extend(found)
                except Exception:
                    logger.exception(
                        "MultiStoreRetriever: semantic retrieval failed owner=%s:%s",
                        owner_type,
                        owner_id,
                    )
            return res if isinstance(res, list) else []
        except Exception:
            logger.exception("MultiStoreRetriever: semantic retrieval failed.")
            return []

    async def _procedural(
        self,
        memory: "UMAMemory",
        emb: List[float],
        k: int,
    ) -> List[Any]:
        core = getattr(memory, "procedural_core", None)
        if core is None:
            return []
        try:
            res = await core.search(query_embedding=emb, k=int(k))
            return res if isinstance(res, list) else []
        except Exception:
            logger.exception("MultiStoreRetriever: procedural retrieval failed.")
            return []

    async def _chunks(
        self,
        memory: "UMAMemory",
        emb: List[float],
        user_id: Optional[str],
        k: int,
        *,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[Any]:
        core = getattr(memory, "chunk_core", None)
        if core is None:
            return []
        if user_id is None:
            return []
        try:
            subject = ensure_user_subject(user_id)
            res: List[Any] = []
            for owner_type, owner_id in _iter_owner_filters(
                user_subject=subject,
                agent_id=agent_id,
                project_id=project_id,
            ):
                try:
                    found = await core.search(
                        query_embedding=emb,
                        owner_type=owner_type,
                        owner_id=owner_id,
                        k=int(k),
                    )
                    if found:
                        res.extend(found)
                except Exception:
                    logger.exception(
                        "MultiStoreRetriever: chunk retrieval failed owner=%s:%s",
                        owner_type,
                        owner_id,
                    )
            return res if isinstance(res, list) else []
        except Exception:
            logger.exception("MultiStoreRetriever: chunk retrieval failed.")
            return []
    def _graph(
        self,
        memory: "UMAMemory",
        user_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Graph retrieval (sync).

        Identity Convention (v1)
        ------------------------
        Graph retrieval ALWAYS uses the canonical user identity:

            User.id = "user:<id>"

        This ensures consistency with graph writes and semantic facts.
        """
        if user_id is None:
            return []

        graph_core = getattr(memory, "graph_core", None)
        if graph_core is None:
            return []

        try:
            subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception(
                "MultiStoreRetriever._graph: invalid user_id=%r", user_id
            )
            return []

        cypher = """
            OPTIONAL MATCH (u:User)-[r*1..2]->(n)
            WHERE u.id = $subject
            RETURN DISTINCT n, labels(n) AS labels, properties(n) AS properties
            LIMIT $limit
            """

        try:
            res = graph_core.query(
                cypher,
                params={
                    "subject": subject,
                    "limit": int(limit),
                },
            )
            return res if isinstance(res, list) else []
        except Exception:
            logger.exception("MultiStoreRetriever._graph: graph_core.query failed.")
            return []
