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
from typing import Any, List, Optional

from ...types import Skill
from ..utils.dedupe import dedupe_by_id

logger = logging.getLogger(__name__)


class ProceduralCore:
    """
    High-level interface for UMA procedural memory.
    """

    def __init__(self, procedural_store: Any) -> None:
        self.store = procedural_store
        logger.debug("ProceduralCore initialized.")

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

    async def get_skill(
        self,
        skill_id: str,
        *,
        owner_type: str,
        owner_id: str,
    ) -> Optional[Skill]:
        if self.store is None or not skill_id:
            return None
        try:
            return await self.store.get_skill(
                skill_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("ProceduralCore.get_skill failed for id=%s", skill_id)
            return None

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        owner_type: str,
        owner_id: str,
    ) -> List[Skill]:
        if self.store is None or not ids:
            return []
        try:
            if hasattr(self.store, "fetch_skills_by_ids"):
                return await self.store.fetch_skills_by_ids(
                    ids,
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
            logger.error("ProceduralCore.fetch_by_ids requires fetch_skills_by_ids support")
            raise RuntimeError("ProceduralCore.fetch_by_ids requires fetch_skills_by_ids support")
        except Exception:
            logger.exception("ProceduralCore.fetch_by_ids failed")
            return []

    async def list_skills(
        self,
        *,
        owner_type: str,
        owner_id: str,
        limit: Optional[int] = None,
    ) -> List[Skill]:
        if self.store is None:
            return []
        try:
            return await self.store.list_skills(
                owner_type=owner_type,
                owner_id=owner_id,
                limit=limit,
            )
        except Exception:
            logger.exception("ProceduralCore.list_skills failed")
            return []

    async def delete_skill(
        self,
        skill_id: str,
        *,
        owner_type: str,
        owner_id: str,
    ) -> bool:
        if self.store is None or not skill_id:
            return False
        try:
            await self.store.delete_skill(
                skill_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
            return True
        except Exception:
            logger.exception("ProceduralCore.delete_skill failed for id=%s", skill_id)
            return False

    # ------------------------------------------------------------------
    # PUBLIC API — retrieval
    # ------------------------------------------------------------------

    async def search(
        self,
        user_id: str,
        query_embedding: List[float],
        *,
        owner_type: str,
        owner_id: str,
        k: int = 5,
    ) -> List[Skill]:
        if self.store is None:
            return []
        skills: List[Skill] = []
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
                "ProceduralCore.search failed owner=%s:%s",
                owner_type,
                owner_id,
            )
        return dedupe_by_id(skills)

    def vector_index(self) -> Any:
        return getattr(self.store, "vector_index", None) if self.store is not None else None
