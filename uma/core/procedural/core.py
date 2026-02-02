"""
procedural/core.py
==================

ProceduralCore — unified interface for UMA procedural memory.

Responsibilities
----------------
- Provide a stable API for skill ingestion and retrieval
- Normalize ownership scoping
- Delegate persistence to ProceduralSQLStore
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from ...types_skill import Skill
from ..utils.identity import ensure_user_subject

logger = logging.getLogger(__name__)


class ProceduralCore:
    """
    High-level interface for UMA procedural memory.
    """

    def __init__(self, procedural_store: Any) -> None:
        self.store = procedural_store
        logger.info("ProceduralCore initialized.")

    # ------------------------------------------------------------------
    # PUBLIC API — ingest / CRUD
    # ------------------------------------------------------------------

    async def add_skill(self, skill: Skill, embedding: List[float]) -> Optional[Skill]:
        if self.store is None:
            return None
        try:
            return await self.store.add_skill(skill, embedding)
        except Exception:
            logger.exception("ProceduralCore.add_skill failed for id=%s", getattr(skill, "id", None))
            return None

    async def get_skill(self, skill_id: str) -> Optional[Skill]:
        if self.store is None or not skill_id:
            return None
        try:
            return await self.store.get_skill(skill_id)
        except Exception:
            logger.exception("ProceduralCore.get_skill failed for id=%s", skill_id)
            return None

    async def fetch_by_ids(self, ids: List[str]) -> List[Skill]:
        if self.store is None or not ids:
            return []
        try:
            if hasattr(self.store, "fetch_skills_by_ids"):
                return await self.store.fetch_skills_by_ids(ids)
            # Fallback: fetch one-by-one
            results: List[Skill] = []
            for sid in ids:
                skill = await self.store.get_skill(sid)
                if skill is not None:
                    results.append(skill)
            return results
        except Exception:
            logger.exception("ProceduralCore.fetch_by_ids failed")
            return []

    async def list_skills(self, limit: Optional[int] = None) -> List[Skill]:
        if self.store is None:
            return []
        try:
            return await self.store.list_skills(limit=limit)
        except Exception:
            logger.exception("ProceduralCore.list_skills failed")
            return []

    async def delete_skill(self, skill_id: str) -> bool:
        if self.store is None or not skill_id:
            return False
        try:
            await self.store.delete_skill(skill_id)
            return True
        except Exception:
            logger.exception("ProceduralCore.delete_skill failed for id=%s", skill_id)
            return False

    # ------------------------------------------------------------------
    # PUBLIC API — retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_owner_filters(
        *,
        user_subject: str,
        agent_id: Optional[str],
        project_id: Optional[str],
        owner_scope: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        scope = (owner_scope or "").lower()
        if scope:
            if scope == "user":
                return [("user", user_subject)]
            if scope == "agent" and agent_id:
                return [("agent", agent_id)]
            if scope == "project" and project_id:
                return [("project", f"{user_subject}:{project_id}")]
            return []

        filters: List[Tuple[str, str]] = [("user", user_subject)]
        if agent_id:
            filters.append(("agent", agent_id))
        if project_id:
            filters.append(("project", f"{user_subject}:{project_id}"))
        return filters

    async def search_tiered(
        self,
        user_id: str,
        query_embedding: List[float],
        *,
        k: int = 5,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        owner_scope: Optional[str] = None,
    ) -> List[Skill]:
        if self.store is None:
            return []
        try:
            user_subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("ProceduralCore.search_tiered: invalid subject=%r", user_id)
            return []

        skills: List[Skill] = []
        for owner_type, owner_id in self._iter_owner_filters(
            user_subject=user_subject,
            agent_id=agent_id,
            project_id=project_id,
            owner_scope=owner_scope,
        ):
            try:
                found = await self.store.search(
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(k),
                )
                if found:
                    skills.extend(found)
            except Exception:
                logger.exception(
                    "ProceduralCore.search_tiered failed owner=%s:%s",
                    owner_type,
                    owner_id,
                )
        return _dedupe_items(skills)

    async def search(
        self,
        query_embedding: List[float],
        *,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        k: int = 5,
    ) -> List[Skill]:
        if self.store is None:
            return []
        try:
            return await self.store.search(
                query_embedding=query_embedding,
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(k),
            )
        except Exception:
            logger.exception("ProceduralCore.search failed")
            return []

    def vector_index(self) -> Any:
        return getattr(self.store, "vector_index", None) if self.store is not None else None


def _dedupe_items(items: List[Any]) -> List[Any]:
    if not items:
        return []
    seen = set()
    out: List[Any] = []
    for it in items:
        key = None
        if isinstance(it, dict):
            key = it.get("id")
        else:
            key = getattr(it, "id", None)
        if key is None:
            key = id(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out
