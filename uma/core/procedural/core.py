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

from dataclasses import replace
import logging
from typing import Any, List, Optional

from ...stores.base_sql_store import DEFAULT_TENANT_ID
from ...types import OwnershipRef, Skill, TargetOwner
from ..utils.dedupe import dedupe_by_id
from ..utils.ownership import resolve_ownership_ref, resolve_target_owner

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

    def _target_owner_from_skill(self, skill: Skill) -> TargetOwner:
        return resolve_target_owner(
            tenant_id=getattr(skill, "tenant_id", None) or DEFAULT_TENANT_ID,
            owner_type=getattr(skill, "owner_type", None) or "user",
            owner_id=getattr(skill, "owner_id", None) or "",
            workspace_id=getattr(skill, "workspace_id", None),
            allowed_owner_types=("agent", "user", "workspace"),
        )

    def _read_owner_ref(
        self,
        *,
        owner: OwnershipRef | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> OwnershipRef:
        return resolve_ownership_ref(
            owner=owner,
            tenant_id=(owner.tenant_id if owner is not None else DEFAULT_TENANT_ID),
            owner_type=owner_type,
            owner_id=owner_id,
            allowed_owner_types=("agent", "user", "workspace"),
        )

    async def add_skill_for_owner(
        self,
        skill: Skill,
        embedding: List[float],
        *,
        target_owner: TargetOwner,
    ) -> Optional[Skill]:
        if self.store is None:
            return None
        try:
            normalized_owner = resolve_target_owner(
                target_owner=target_owner,
                allowed_owner_types=("agent", "user", "workspace"),
            )
            normalized_skill = replace(
                skill,
                tenant_id=normalized_owner.tenant_id,
                owner_type=normalized_owner.owner_type,
                owner_id=normalized_owner.owner_id,
                workspace_id=normalized_owner.workspace_id,
            )
            return await self.store.add_skill(normalized_skill, embedding)
        except Exception:
            logger.exception("ProceduralCore.add_skill failed for id=%s", getattr(skill, "id", None))
            return None

    async def add_skill(self, skill: Skill, embedding: List[float]) -> Optional[Skill]:
        if self.store is None:
            return None
        try:
            target_owner = self._target_owner_from_skill(skill)
            return await self.add_skill_for_owner(skill, embedding, target_owner=target_owner)
        except Exception:
            logger.exception("ProceduralCore.add_skill failed for id=%s", getattr(skill, "id", None))
            return None

    async def get_skill(
        self,
        skill_id: str,
        *,
        owner: OwnershipRef | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> Optional[Skill]:
        if self.store is None or not skill_id:
            return None
        try:
            read_owner = self._read_owner_ref(owner=owner, owner_type=owner_type, owner_id=owner_id)
        except ValueError:
            raise
        except Exception:
            logger.exception("ProceduralCore.get_skill failed for id=%s", skill_id)
            return None
        try:
            return await self.store.get_skill(
                skill_id,
                tenant_id=read_owner.tenant_id,
                owner_type=read_owner.owner_type,
                owner_id=read_owner.owner_id,
            )
        except Exception:
            logger.exception("ProceduralCore.get_skill failed for id=%s", skill_id)
            return None

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        owner: OwnershipRef | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> List[Skill]:
        if self.store is None or not ids:
            return []
        try:
            read_owner = self._read_owner_ref(owner=owner, owner_type=owner_type, owner_id=owner_id)
        except ValueError:
            raise
        except Exception:
            logger.exception("ProceduralCore.fetch_by_ids failed")
            return []
        try:
            if hasattr(self.store, "fetch_skills_by_ids"):
                return await self.store.fetch_skills_by_ids(
                    ids,
                    tenant_id=read_owner.tenant_id,
                    owner_type=read_owner.owner_type,
                    owner_id=read_owner.owner_id,
                )
            logger.error("ProceduralCore.fetch_by_ids requires fetch_skills_by_ids support")
            raise RuntimeError("ProceduralCore.fetch_by_ids requires fetch_skills_by_ids support")
        except Exception:
            logger.exception("ProceduralCore.fetch_by_ids failed")
            return []

    async def list_skills(
        self,
        *,
        owner: OwnershipRef | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
        limit: Optional[int] = None,
    ) -> List[Skill]:
        if self.store is None:
            return []
        try:
            read_owner = self._read_owner_ref(owner=owner, owner_type=owner_type, owner_id=owner_id)
        except ValueError:
            raise
        except Exception:
            logger.exception("ProceduralCore.list_skills failed")
            return []
        try:
            return await self.store.list_skills(
                tenant_id=read_owner.tenant_id,
                owner_type=read_owner.owner_type,
                owner_id=read_owner.owner_id,
                limit=limit,
            )
        except Exception:
            logger.exception("ProceduralCore.list_skills failed")
            return []

    async def delete_skill(
        self,
        skill_id: str,
        *,
        owner: OwnershipRef | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> bool:
        if self.store is None or not skill_id:
            return False
        try:
            read_owner = self._read_owner_ref(owner=owner, owner_type=owner_type, owner_id=owner_id)
        except ValueError:
            raise
        except Exception:
            logger.exception("ProceduralCore.delete_skill failed for id=%s", skill_id)
            return False
        try:
            await self.store.delete_skill(
                skill_id,
                tenant_id=read_owner.tenant_id,
                owner_type=read_owner.owner_type,
                owner_id=read_owner.owner_id,
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
        query_embedding: List[float],
        *,
        owner: OwnershipRef | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
        k: int = 5,
    ) -> List[Skill]:
        if self.store is None:
            return []
        skills: List[Skill] = []
        try:
            read_owner = self._read_owner_ref(owner=owner, owner_type=owner_type, owner_id=owner_id)
        except ValueError:
            raise
        except Exception:
            logger.exception(
                "ProceduralCore.search failed owner=%s:%s",
                owner_type if owner is None else owner.owner_type,
                owner_id if owner is None else owner.owner_id,
            )
            return []
        try:
            found = await self.store.search(
                query_embedding=query_embedding,
                tenant_id=read_owner.tenant_id,
                owner_type=read_owner.owner_type,
                owner_id=read_owner.owner_id,
                k=int(k),
            )
            if found:
                skills.extend(found)
        except Exception:
            logger.exception(
                "ProceduralCore.search failed owner=%s:%s",
                owner_type if owner is None else owner.owner_type,
                owner_id if owner is None else owner.owner_id,
            )
        return dedupe_by_id(skills)

    def vector_index(self) -> Any:
        return getattr(self.store, "vector_index", None) if self.store is not None else None
