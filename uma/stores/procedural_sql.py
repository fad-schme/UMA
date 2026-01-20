"""
ProceduralSQLStore — Production-Grade SQL + VectorIndex Store
for skill-based procedural memory in UMA-3.

This implementation is fully aligned with:
    • UMAMemory initialization
    • RetrievalService v3 (memory_type="procedural")
    • MultiStoreRetriever v2 (parallel store retrieval)
    • BaseVectorSQLStore (vector-ranking, SQL CRUD helpers)
    • ProceduralFeature (developer-facing API)
    • SkillIndexer (skill construction + embedding)

Enhancements in this version:
    • Added get_skill(skill_id)
    • Added delete_skill(skill_id)
    • Added strict validation of Skill objects
    • Improved logging + error handling
    • Ensures uniform store interface across UMA-3
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional, Any

from .base_vector_sql_store import BaseVectorSQLStore
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
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
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);")
            conn.commit()
        except Exception:
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
        return Skill(
            id=row["id"],
            name=row["name"],
            trigger_phrases=json.loads(row["trigger_phrases"]),
            trigger_patterns=json.loads(row["trigger_patterns"]),
            plan=json.loads(row["plan"]),
            tools=json.loads(row["tools"]),
            example=row["example"],
            embedding=None,
            meta=json.loads(row["meta"]),
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
            }

            self._execute(
                conn,
                """
                INSERT INTO skills (
                    id, name, trigger_phrases, trigger_patterns, plan,
                    tools, example, meta, created_at, updated_at
                )
                VALUES (
                    :id, :name, :trigger_phrases, :trigger_patterns, :plan,
                    :tools, :example, :meta, :created_at, :updated_at
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
            conn.commit()

            # Vector upsert
            try:
                self.vector_index.upsert(
                    ids=[skill.id],
                    vectors=[embedding],
                    metadata=[{"name": skill.name}],
                )
            except Exception:
                logger.exception(
                    "ProceduralSQLStore: vector upsert failed for skill id=%s",
                    skill.id,
                )

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
            return None
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
            logger.exception("ProceduralSQLStore.delete_skill failed id=%s", skill_id)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Semantic Search
    # ------------------------------------------------------------------ #

    async def search(self, query_embedding: List[float], k: int = 5) -> List[Skill]:
        """
        Retrieve top-k procedural skills by vector similarity.

        Delegates ranking + row mapping to BaseVectorSQLStore.
        """
        return await self._semantic_search(
            query_embedding=query_embedding,
            k=k,
            filters=None,
            log_context="procedural_search",
        )