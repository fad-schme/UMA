"""
ProceduralSQLStore — Production-Grade SQL + VectorIndex Store
for skill-based procedural memory in UMA.

This implementation is fully aligned with:
    • UMAMemory initialization
    • RetrievalService v3 (memory_type="procedural")
    • RetrievalService (parallel store retrieval)
    • BaseVectorSQLStore (vector-ranking, SQL CRUD helpers)
    • ProceduralFeature (developer-facing API)
    • SkillIndexer (skill construction + embedding)

Enhancements in this version:
    • Added get_skill(skill_id)
    • Added delete_skill(skill_id)
    • Added strict validation of Skill objects
    • Improved logging + error handling
    • Ensures uniform store interface across UMA
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional, Any

from .base_vector_sql_store import BaseVectorSQLStore
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
from ..core.utils.store_metadata import ensure_store_metadata
from ..types_skill import Skill

logger = logging.getLogger(__name__)


class ProceduralSQLStore(BaseVectorSQLStore):
    """
    SQL + VectorIndex store for procedural / skill memory.

    Responsibilities:
    -----------------
    • Persist Skills in SQL
    • Maintain vector similarity index
    • Provide ranked semantic search (via BaseVectorSQLStore)
    • Expose CRUD: add_skill, get_skill, delete_skill
    """

    def __init__(self, db_adapter: DBAdapter, vector_index: VectorIndex) -> None:
        """
        Initialize the procedural store.

        Parameters
        ----------
        db_adapter : DBAdapter
            Abstraction that provides DB-API connections.
        vector_index : VectorIndex
            Pluggable vector backend for semantic retrieval.
        """
        super().__init__(db_adapter=db_adapter, vector_index=vector_index)
        self._init_db()
        logger.info("ProceduralSQLStore initialized with DB=%s", type(db_adapter).__name__)

    # ------------------------------------------------------------------ #
    # SQL Schema
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Create the skills table and indexes if missing."""
        conn = self._conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    trigger_phrases TEXT NOT NULL,
                    trigger_patterns TEXT NOT NULL,
                    plan TEXT NOT NULL,
                    tools TEXT NOT NULL,
                    example TEXT NOT NULL,
                    meta TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_owner ON skills(owner_type, owner_id);")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);")
            ensure_store_metadata(self, conn, store_name="procedural")
            conn.commit()
        except Exception:
            self._safe_rollback(conn, "init_db")
            logger.exception("ProceduralSQLStore: failed initializing schema.")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Required BaseVectorSQLStore properties
    # ------------------------------------------------------------------ #

    @property
    def _table_name(self) -> str:
        return "skills"

    @property
    def _id_column(self) -> str:
        return "id"

    # ------------------------------------------------------------------ #
    # Row → Skill conversion
    # ------------------------------------------------------------------ #

    def _row_to_object(self, row: Any) -> Skill:
        # Support both dict-like rows and sqlite3.Row objects (which lack .get)
        if hasattr(row, "get"):
            owner_type = row.get("owner_type", "user")
            owner_id = row.get("owner_id", "")
            trigger_phrases_val = row.get("trigger_phrases")
            trigger_patterns_val = row.get("trigger_patterns")
            plan_val = row.get("plan")
            tools_val = row.get("tools")
            meta_val = row.get("meta")
        else:
            keys = list(row.keys()) if hasattr(row, "keys") else []
            owner_type = row["owner_type"] if "owner_type" in keys else "user"
            owner_id = row["owner_id"] if "owner_id" in keys else ""
            trigger_phrases_val = row["trigger_phrases"] if "trigger_phrases" in keys else None
            trigger_patterns_val = row["trigger_patterns"] if "trigger_patterns" in keys else None
            plan_val = row["plan"] if "plan" in keys else None
            tools_val = row["tools"] if "tools" in keys else None
            meta_val = row["meta"] if "meta" in keys else None

        try:
            trigger_phrases = json.loads(trigger_phrases_val) if trigger_phrases_val else []
        except Exception:
            trigger_phrases = []
        try:
            trigger_patterns = json.loads(trigger_patterns_val) if trigger_patterns_val else []
        except Exception:
            trigger_patterns = []
        try:
            plan = json.loads(plan_val) if plan_val else {}
        except Exception:
            plan = {}
        try:
            tools = json.loads(tools_val) if tools_val else []
        except Exception:
            tools = []
        try:
            meta = json.loads(meta_val) if meta_val else {}
        except Exception:
            meta = {}

        return Skill(
            id=row["id"],
            name=row["name"],
            description="",
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            owner_type=owner_type,
            owner_id=owner_id,
            trigger_phrases=trigger_phrases,
            trigger_patterns=trigger_patterns,
            plan=plan,
            tools=tools,
            example=row["example"],
            meta=meta,
        )
    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_skill(self, skill: Skill) -> None:
        """Validate that Skill fields are well-formed. Raise ValueError if not."""
        if not isinstance(skill.id, str) or not skill.id.strip():
            raise ValueError("Skill.id must be a non-empty string.")

        if not isinstance(skill.name, str) or not skill.name.strip():
            raise ValueError("Skill.name must be a non-empty string.")

        if not isinstance(skill.trigger_phrases, list):
            raise ValueError("Skill.trigger_phrases must be a list.")

        if not isinstance(skill.trigger_patterns, list):
            raise ValueError("Skill.trigger_patterns must be a list.")

        if not isinstance(skill.plan, dict):
            raise ValueError("Skill.plan must be a dict.")

        if not isinstance(skill.tools, (list, dict)):
            raise ValueError("Skill.tools must be a list or dict.")

        if not isinstance(skill.meta, dict):
            raise ValueError("Skill.meta must be a dict.")

    # ------------------------------------------------------------------ #
    # CRUD operations
    # ------------------------------------------------------------------ #

    async def add_skill(self, skill: Skill, embedding: List[float]) -> Skill:
        """
        Insert or update a Skill record + vector embedding.

        Returns:
        --------
        Skill   (the stored/updated skill)
        """
        self._validate_skill(skill)

        conn = self._conn()
        now = datetime.utcnow().isoformat()

        try:
            payload = {
                "id": skill.id,
                "name": skill.name,
                "trigger_phrases": json.dumps(skill.trigger_phrases),
                "trigger_patterns": json.dumps(skill.trigger_patterns),
                "plan": json.dumps(skill.plan),
                "tools": json.dumps(skill.tools),
                "example": skill.example,
                "meta": json.dumps(skill.meta),
                "created_at": now,
                "updated_at": now,
                "owner_type": skill.owner_type or "user",
                "owner_id": skill.owner_id or "",
            }

            self._execute(
                conn,
                """
                INSERT INTO skills (
                    id, name, trigger_phrases, trigger_patterns, plan,
                    tools, example, meta, created_at, updated_at, owner_type, owner_id
                )
                VALUES (
                    :id, :name, :trigger_phrases, :trigger_patterns, :plan,
                    :tools, :example, :meta, :created_at, :updated_at, :owner_type, :owner_id
                )
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    trigger_phrases=excluded.trigger_phrases,
                    trigger_patterns=excluded.trigger_patterns,
                    plan=excluded.plan,
                    tools=excluded.tools,
                    example=excluded.example,
                    meta=excluded.meta,
                    updated_at=excluded.updated_at
                """,
                params=payload,
                log_context="add_skill",
            )
            try:
                self.vector_index.upsert(
                    ids=[skill.id],
                    vectors=[embedding],
                    metadata=[{
                        "name": skill.name,
                        "owner_type": skill.owner_type or "user",
                        "owner_id": skill.owner_id or "",
                        "scope_key": f"{skill.owner_type}:{skill.owner_id}",
                    }],
                )
            except Exception:
                logger.exception(
                    "ProceduralSQLStore: vector upsert failed for skill id=%s",
                    skill.id,
                )
                self._safe_rollback(conn, "add_skill")
                raise

            try:
                conn.commit()
            except Exception:
                self._safe_rollback(conn, "add_skill_commit")
                try:
                    self.vector_index.delete([skill.id])
                except Exception:
                    logger.exception(
                        "ProceduralSQLStore: vector delete failed after commit error id=%s",
                        skill.id,
                    )
                raise

            logger.info("ProceduralSQLStore: upserted skill id=%s", skill.id)
            return skill

        except Exception:
            logger.exception("ProceduralSQLStore.add_skill failed for id=%s", skill.id)
            raise

        finally:
            conn.close()

    async def get_skill(self, skill_id: str) -> Optional[Skill]:
        """
        Fetch a single skill by ID. Returns None if not found.
        """
        conn = self._conn()
        try:
            rows = self._query_all(
                conn,
                "SELECT * FROM skills WHERE id=?",
                params=[skill_id],
                log_context="get_skill",
            )
            if not rows:
                return None

            return self._row_to_object(rows[0])
        except Exception:
            logger.exception("ProceduralSQLStore.get_skill failed for id=%s", skill_id)
            raise
        finally:
            conn.close()

    async def list_skills(self, limit: Optional[int] = None) -> List[Skill]:
        """
        List skills ordered by updated_at DESC.
        """
        conn = self._conn()
        try:
            sql = "SELECT * FROM skills ORDER BY updated_at DESC"
            if limit:
                sql += f" LIMIT {int(limit)}"
            rows = self._query_all(conn, sql, log_context="list_skills")
            return [self._row_to_object(r) for r in rows]
        except Exception:
            logger.exception("ProceduralSQLStore.list_skills failed.")
            raise
        finally:
            conn.close()

    async def delete_skill(self, skill_id: str) -> None:
        """
        Remove a skill from SQL + vector store.
        """
        conn = self._conn()
        try:
            self._execute(
                conn,
                "DELETE FROM skills WHERE id=?",
                params=[skill_id],
                log_context="delete_skill",
            )
            conn.commit()

            try:
                self.vector_index.delete(ids=[skill_id])
            except Exception:
                logger.exception(
                    "ProceduralSQLStore.delete_skill: vector delete failed id=%s",
                    skill_id,
                )

            logger.info("ProceduralSQLStore: deleted skill id=%s", skill_id)

        except Exception:
            self._safe_rollback(conn, "delete_skill")
            logger.exception("ProceduralSQLStore.delete_skill failed id=%s", skill_id)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Semantic Search
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query_embedding: List[float],
        *,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        k: int = 5,
    ) -> List[Skill]:
        """
        Retrieve top-k procedural skills by vector similarity.

        Delegates ranking + row mapping to BaseVectorSQLStore.
        """
        try:
            return await self._semantic_search(
                query_embedding=query_embedding,
                k=k,
                filters=({"owner_type": owner_type, "owner_id": owner_id} if owner_type and owner_id else None),
                log_context="procedural_search",
            )
        except Exception:
            logger.exception("ProceduralSQLStore.search failed.")
            raise
