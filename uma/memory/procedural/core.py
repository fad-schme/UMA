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
from datetime import datetime, timezone
import logging
from typing import Any, List, Optional

from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import AgentProfile, OwnershipRef, Skill
from uma.common.dedupe import dedupe_by_id
from uma.common.ownership import validate_explicit_owner


def _agent_profile_row_id(agent_id: str) -> str:
    """Deterministic row id for the agent-profile Skill row.

    One row per agent — the id is derived from agent_id so that upsert
    always finds the same row (idempotent overwrite semantics).
    """
    return f"skill_agent_profile:{agent_id}"


def _agent_profile_owner_id(agent_id: str) -> str:
    """Owner-scope id for the agent-profile Skill row.

    Matches the ``agent:<id>`` convention used by
    :meth:`PromotionPolicy.select_promotion_target` when it targets
    agent scope.
    """
    return f"agent:{agent_id}"

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

    async def add_skill_for_owner(
        self,
        skill: Skill,
        embedding: List[float],
        *,
        owner_type: str,
        owner_id: str,
        tenant_id: str | None = None,
        workspace_id: str | None = None,
    ) -> Optional[Skill]:
        """Add a skill for a specific owner, using the provided embedding."""
        if self.store is None:
            return None
        try:
            normalized_owner = validate_explicit_owner(
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                workspace_id=workspace_id,
            )
            if normalized_owner["owner_type"] not in {"agent", "user", "workspace"}:
                raise ValueError("owner_type must be one of: agent, user, workspace")
            normalized_skill = replace(
                skill,
                tenant_id=str(normalized_owner["tenant_id"]),
                owner_type=str(normalized_owner["owner_type"]),
                owner_id=str(normalized_owner["owner_id"]),
                workspace_id=normalized_owner["workspace_id"],
            )
            return await self.store.add_skill(normalized_skill, embedding)
        except Exception:
            logger.exception("ProceduralCore.add_skill failed for id=%s", getattr(skill, "id", None))
            return None

    async def add_skill(self, skill: Skill, embedding: List[float]) -> Optional[Skill]:
        """Add a skill to the procedural store, generating an embedding from its content."""
        if self.store is None:
            return None
        try:
            # Pass ownership through from the Skill object. The downstream
            # validation chain (validate_explicit_owner → SQL store
            # _validate_skill) will refuse missing/empty values; no need
            # for `or ""` fallbacks here.
            return await self.add_skill_for_owner(
                skill,
                embedding,
                tenant_id=getattr(skill, "tenant_id", None) or DEFAULT_TENANT_ID,
                owner_type=getattr(skill, "owner_type", None),
                owner_id=getattr(skill, "owner_id", None),
                workspace_id=getattr(skill, "workspace_id", None),
            )
        except Exception:
            logger.exception("ProceduralCore.add_skill failed for id=%s", getattr(skill, "id", None))
            return None

    async def get_skill(
        self,
        skill_id: str,
        *,
        tenant_id: str | None = None,
        owner: OwnershipRef | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> Optional[Skill]:
        """Fetch a single skill by ID within the ownership scope."""
        if self.store is None or not skill_id:
            return None
        try:
            normalized_owner = validate_explicit_owner(
                tenant_id=(owner.tenant_id if owner is not None else tenant_id or DEFAULT_TENANT_ID),
                owner_type=(owner.owner_type if owner is not None else owner_type or ""),
                owner_id=(owner.owner_id if owner is not None else owner_id or ""),
            )
            if normalized_owner["owner_type"] not in {"agent", "user", "workspace"}:
                raise ValueError("owner_type must be one of: agent, user, workspace")
        except ValueError:
            raise
        except Exception:
            logger.exception("ProceduralCore.get_skill failed for id=%s", skill_id)
            return None
        try:
            return await self.store.get_skill(
                skill_id,
                tenant_id=str(normalized_owner["tenant_id"]),
                owner_type=str(normalized_owner["owner_type"]),
                owner_id=str(normalized_owner["owner_id"]),
            )
        except Exception:
            logger.exception("ProceduralCore.get_skill failed for id=%s", skill_id)
            return None

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        tenant_id: str | None = None,
        owner: OwnershipRef | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> List[Skill]:
        """Bulk-fetch skills by ID list within the ownership scope."""
        if self.store is None or not ids:
            return []
        try:
            normalized_owner = validate_explicit_owner(
                tenant_id=(owner.tenant_id if owner is not None else tenant_id or DEFAULT_TENANT_ID),
                owner_type=(owner.owner_type if owner is not None else owner_type or ""),
                owner_id=(owner.owner_id if owner is not None else owner_id or ""),
            )
            if normalized_owner["owner_type"] not in {"agent", "user", "workspace"}:
                raise ValueError("owner_type must be one of: agent, user, workspace")
        except ValueError:
            raise
        except Exception:
            logger.exception("ProceduralCore.fetch_by_ids failed")
            return []
        try:
            if hasattr(self.store, "fetch_skills_by_ids"):
                return await self.store.fetch_skills_by_ids(
                    ids,
                    tenant_id=str(normalized_owner["tenant_id"]),
                    owner_type=str(normalized_owner["owner_type"]),
                    owner_id=str(normalized_owner["owner_id"]),
                )
            logger.error("ProceduralCore.fetch_by_ids requires fetch_skills_by_ids support")
            raise RuntimeError("ProceduralCore.fetch_by_ids requires fetch_skills_by_ids support")
        except Exception:
            logger.exception("ProceduralCore.fetch_by_ids failed")
            return []

    async def list_skills(
        self,
        *,
        tenant_id: str | None = None,
        owner: OwnershipRef | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
        limit: Optional[int] = None,
    ) -> List[Skill]:
        """List all skills for the ownership scope, optionally including quarantined records."""
        if self.store is None:
            return []
        try:
            normalized_owner = validate_explicit_owner(
                tenant_id=(owner.tenant_id if owner is not None else tenant_id or DEFAULT_TENANT_ID),
                owner_type=(owner.owner_type if owner is not None else owner_type or ""),
                owner_id=(owner.owner_id if owner is not None else owner_id or ""),
            )
            if normalized_owner["owner_type"] not in {"agent", "user", "workspace"}:
                raise ValueError("owner_type must be one of: agent, user, workspace")
        except ValueError:
            raise
        except Exception:
            logger.exception("ProceduralCore.list_skills failed")
            return []
        try:
            return await self.store.list_skills(
                tenant_id=str(normalized_owner["tenant_id"]),
                owner_type=str(normalized_owner["owner_type"]),
                owner_id=str(normalized_owner["owner_id"]),
                limit=limit,
            )
        except Exception:
            logger.exception("ProceduralCore.list_skills failed")
            return []

    async def delete_skill(
        self,
        skill_id: str,
        *,
        tenant_id: str | None = None,
        owner: OwnershipRef | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> bool:
        """Permanently delete a skill and its vector-index entry."""
        if self.store is None or not skill_id:
            return False
        try:
            normalized_owner = validate_explicit_owner(
                tenant_id=(owner.tenant_id if owner is not None else tenant_id or DEFAULT_TENANT_ID),
                owner_type=(owner.owner_type if owner is not None else owner_type or ""),
                owner_id=(owner.owner_id if owner is not None else owner_id or ""),
            )
            if normalized_owner["owner_type"] not in {"agent", "user", "workspace"}:
                raise ValueError("owner_type must be one of: agent, user, workspace")
        except ValueError:
            raise
        except Exception:
            logger.exception("ProceduralCore.delete_skill failed for id=%s", skill_id)
            return False
        try:
            await self.store.delete_skill(
                skill_id,
                tenant_id=str(normalized_owner["tenant_id"]),
                owner_type=str(normalized_owner["owner_type"]),
                owner_id=str(normalized_owner["owner_id"]),
            )
            return True
        except Exception:
            logger.exception("ProceduralCore.delete_skill failed for id=%s", skill_id)
            return False

    # ------------------------------------------------------------------
    # PUBLIC API — agent profile (memory-promotion feature)
    # ------------------------------------------------------------------

    async def upsert_agent_profile(
        self,
        *,
        agent_id: str,
        description: str,
        focus_areas: List[str],
        embedding: List[float],
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> AgentProfile:
        """Insert-or-overwrite the agent's profile row.

        The profile is persisted as a Skill row with ``kind='agent_profile'``.
        The row id is deterministic (``skill_agent_profile:<agent_id>``) so
        repeated calls overwrite the same row. Owner scope is
        ``('agent', 'agent:<agent_id>')``.

        Agent-profile rows never enter the vector index (they are only
        fetched by agent_id via :meth:`get_agent_profile`), so this write
        cannot leak into normal procedural search.
        """
        if self.store is None:
            raise RuntimeError("ProceduralCore.upsert_agent_profile: store is missing")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(focus_areas, list):
            raise ValueError("focus_areas must be a list of strings")
        for item in focus_areas:
            if not isinstance(item, str):
                raise ValueError("focus_areas entries must be strings")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("embedding must be a non-empty list of floats")

        now = datetime.now(timezone.utc)
        skill = Skill(
            id=_agent_profile_row_id(agent_id),
            name=f"agent profile: {agent_id}",
            description=description,
            kind="agent_profile",
            focus_areas=list(focus_areas),
            created_at=now,
            updated_at=now,
            owner_type="agent",
            owner_id=_agent_profile_owner_id(agent_id),
            tenant_id=tenant_id,
        )
        await self.store.add_skill(skill, embedding)
        return AgentProfile(
            agent_id=agent_id,
            description=description,
            focus_areas=list(focus_areas),
            profile_embedding=list(embedding),
            tenant_id=tenant_id,
        )

    async def get_agent_profile(
        self,
        *,
        agent_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> Optional[AgentProfile]:
        """Return the agent's profile if set, else None.

        Fetches directly by row id under the agent-owner scope. The
        embedding is decoded from the SQL ``profile_embedding`` BLOB by
        :meth:`ProceduralSQLStore._row_to_object` (the agent_profile row
        is never in the vector index).
        """
        if self.store is None:
            return None
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")

        try:
            skill = await self.store.get_skill(
                _agent_profile_row_id(agent_id),
                tenant_id=tenant_id,
                owner_type="agent",
                owner_id=_agent_profile_owner_id(agent_id),
            )
        except Exception:
            logger.exception(
                "ProceduralCore.get_agent_profile failed for agent_id=%s", agent_id
            )
            return None
        if skill is None:
            return None
        if getattr(skill, "kind", "procedural") != "agent_profile":
            # Defensive: an id collision with a normal skill would be a
            # bug elsewhere, but never return a non-profile as if it were
            # a profile.
            logger.warning(
                "ProceduralCore.get_agent_profile: row id=%s exists but kind=%r; ignoring",
                skill.id,
                skill.kind,
            )
            return None
        embedding = list(skill.embedding or [])
        if not embedding:
            logger.warning(
                "ProceduralCore.get_agent_profile: agent_id=%s has empty profile_embedding",
                agent_id,
            )
            return None
        return AgentProfile(
            agent_id=agent_id,
            description=skill.description,
            focus_areas=list(skill.focus_areas or []),
            profile_embedding=embedding,
            tenant_id=str(skill.tenant_id or tenant_id),
        )

    # ------------------------------------------------------------------
    # PUBLIC API — retrieval
    # ------------------------------------------------------------------

    async def search(
        self,
        query_embedding: List[float],
        *,
        tenant_id: str | None = None,
        owner: OwnershipRef | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
        k: int = 5,
    ) -> List[Skill]:
        """Vector-search skills for the given query embedding within the ownership scope."""
        if self.store is None:
            return []
        skills: List[Skill] = []
        try:
            normalized_owner = validate_explicit_owner(
                tenant_id=(owner.tenant_id if owner is not None else tenant_id or DEFAULT_TENANT_ID),
                owner_type=(owner.owner_type if owner is not None else owner_type or ""),
                owner_id=(owner.owner_id if owner is not None else owner_id or ""),
            )
            if normalized_owner["owner_type"] not in {"agent", "user", "workspace"}:
                raise ValueError("owner_type must be one of: agent, user, workspace")
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
                tenant_id=str(normalized_owner["tenant_id"]),
                owner_type=str(normalized_owner["owner_type"]),
                owner_id=str(normalized_owner["owner_id"]),
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
        """Return the underlying ``VectorIndex`` instance for this store."""
        return getattr(self.store, "vector_index", None) if self.store is not None else None