"""
TemporalGraphCore
=================

High-level interface for UMA temporal graph memory.

Responsibilities
----------------
- Hold the GraphAdapter backend
- Delegate all graph write logic to GraphUpdater
- Ensure graph failures NEVER crash UMA

Identity Convention (v1)
------------------------
All User nodes are keyed using the canonical form:

    "user:<id>"

This invariant is enforced by GraphUpdater via ensure_user_subject().
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Tuple

from ...adapters.graph.base import GraphAdapter
from ..utils.identity import ensure_user_subject
from .updater import GraphUpdater

logger = logging.getLogger(__name__)


class TemporalGraphCore:
    """
    Core temporal graph subsystem.

    This is a REQUIRED subsystem when graph.backend != "disabled".
    """

    def __init__(self, adapter: GraphAdapter):
        # Strict type validation for safety
        if not isinstance(adapter, GraphAdapter):
            raise TypeError(
                f"TemporalGraphCore requires a GraphAdapter instance, "
                f"got {type(adapter).__name__}"
            )

        self.adapter = adapter
        self.updater = GraphUpdater(self)

        logger.info(
            "TemporalGraphCore initialized with adapter=%s",
            adapter.__class__.__name__,
        )
    
    # ------------------------------------------------------------------
    # PUBLIC API
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

    def neighbors_tiered(
        self,
        user_id: str,
        node_id: str,
        *,
        predicate_scope: Optional[List[str]] = None,
        depth: int = 1,
        k: int = 10,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        owner_scope: Optional[str] = None,
    ) -> List[dict]:
        """
        Tiered neighbor expansion across user/agent/project scopes.
        """
        try:
            user_subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("TemporalGraphCore.neighbors_tiered: invalid subject=%r", user_id)
            return []

        results: List[dict] = []
        for owner_type, owner_id in self._iter_owner_filters(
            user_subject=user_subject,
            agent_id=agent_id,
            project_id=project_id,
            owner_scope=owner_scope,
        ):
            try:
                found = self.neighbors(
                    user_id=user_subject,
                    node_id=node_id,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    predicate_scope=predicate_scope,
                    depth=depth,
                    k=k,
                )
                if found:
                    results.extend(found)
            except Exception:
                logger.exception(
                    "TemporalGraphCore.neighbors_tiered failed owner=%s:%s",
                    owner_type,
                    owner_id,
                )
        return _dedupe_items(results)

    def add_episode(self, episode: Any) -> None:
        """Insert an Episode node and link it to its User."""
        try:
            self.updater.add_episode_node(episode)
        except Exception:
            logger.exception("TemporalGraphCore.add_episode failed.")

    def add_facts(self, facts: List[Any]) -> None:
        """Insert Fact nodes."""
        for fact in facts:
            try:
                self.updater.add_fact(fact)
            except Exception:
                logger.exception("TemporalGraphCore.add_facts failed.")

    def insert_fact_triplet(
        self,
        fact_id: str,
        subject: str,
        predicate: str,
        object: str,
        owner_type: str,
        owner_id: str,
        source_chunk_id: str,
        created_at: str,
        updated_at: str,
    ) -> bool:
        """
        Persist a fact triplet (subject, predicate, object) into Neo4j with full provenance.

        This closes the fact→graph provenance gap by ensuring graph materialization stores:
          - Relationship properties: fact_id, owner_type, owner_id, source_chunk_id, created_at, updated_at
          - A Fact node with the same provenance (so provenance exists even if edge props are lost)
          - Links between Fact and involved entities

        This method is safe, idempotent, and MUST NOT raise exceptions (graph failures never crash UMA).
        Returns:
            bool: True if the upsert succeeded, False otherwise.
        """
        try:
            rel_type = self._sanitize_predicate(predicate)

            cypher = f"""
            MERGE (subj:Entity {{id: $subject}})
            MERGE (obj:Entity {{id: $object}})

            MERGE (subj)-[r:{rel_type}]->(obj)
              SET
                r.fact_id = $fact_id,
                r.owner_type = $owner_type,
                r.owner_id = $owner_id,
                r.source_chunk_id = $source_chunk_id,
                r.created_at = $created_at,
                r.updated_at = $updated_at

            MERGE (f:Fact {{id: $fact_id}})
              SET
                f.subject = $subject,
                f.predicate = $predicate,
                f.object = $object,
                f.owner_type = $owner_type,
                f.owner_id = $owner_id,
                f.source_chunk_id = $source_chunk_id,
                f.created_at = $created_at,
                f.updated_at = $updated_at

            MERGE (f)-[:SUBJECT]->(subj)
            MERGE (f)-[:OBJECT]->(obj)
            MERGE (f)-[:ASSERTS {{predicate: $predicate}}]->(obj)
            """

            params = {
                "fact_id": fact_id,
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "owner_type": owner_type,
                "owner_id": owner_id,
                "source_chunk_id": source_chunk_id,
                "created_at": created_at,
                "updated_at": updated_at,
            }

            self.adapter.run_query(cypher, params=params)
            return True

        except Exception:
            logger.exception("TemporalGraphCore.insert_fact_triplet failed.")
            return False
        
    def link_episode_to_facts(self, episode: Any, facts: List[Any]) -> None:
        """Link Episode to its extracted Facts."""
        try:
            self.updater.link_episode_to_facts(episode, facts)
        except Exception:
            logger.exception("TemporalGraphCore.link_episode_to_facts failed.")

    def link_temporal(self, prev_ep: Any, next_ep: Any) -> None:
        """Link episodes in a temporal sequence."""
        try:
            self.updater.link_temporal(prev_ep, next_ep)
        except Exception:
            logger.exception("TemporalGraphCore.link_temporal failed.")

    def neighbors(
        self,
        user_id: str,
        node_id: str,
        owner_type: str,
        owner_id: str,
        predicate_scope: Optional[List[str]] = None,
        depth: int = 1,
        k: int = 10,
    ) -> List[dict]:
        """
        Fetch graph neighbors for a node, STRICTLY scoped by ownership.
        This is a read-only helper intended for RLM navigation.
        DAT invariant (critical)
        ------------------------
        This method enforces that traversal ONLY follows relationships
        that match (owner_type, owner_id). Cross-scope leakage is impossible.

        Parameters
        ----------
        user_id : str
            Requesting user (for sanity/logging only).
        node_id : str
            Starting node id.
        owner_type : str
            One of {'user','agent','project'}.
        owner_id : str
            Canonical owner identifier (e.g. 'user:u1').
        predicate_scope : Optional[List[str]]
            Optional list of predicate names to restrict traversal.
        depth : int
            Maximum hop count (bounded).
        k : int
            Maximum number of results.
        """
        
        if not user_id or not node_id:
            logger.warning("TemporalGraphCore.neighbors: missing user_id or node_id")
            return []
        
        if not owner_type or not owner_id:
            logger.error("TemporalGraphCore.neighbors: owner_type and owner_id are required")
            return []
        
        depth_i = max(1, min(5, int(depth)))
        limit = max(1, int(k))

        preds = None
        if predicate_scope:
            preds = [self._sanitize_predicate(p) for p in predicate_scope if p]
            preds = [p for p in preds if p]

        cypher = f"""
        MATCH (n {{id: $node_id}})-[rs*1..{depth_i}]-(m)
        WHERE ALL(r IN rs WHERE r.owner_type = $owner_type AND r.owner_id = $owner_id)
        AND ($preds IS NULL OR ALL(r IN rs WHERE type(r) IN $preds))
        RETURN DISTINCT m AS node, labels(m) AS labels, properties(m) AS properties
        LIMIT $limit
        """
        params = {          
            "node_id": node_id,
            "owner_type": owner_type,
            "owner_id": owner_id,
            "preds": preds,
            "limit": limit,
        }

        try:
            return self.adapter.run_query(cypher, params=params)
        except Exception:
            logger.exception("TemporalGraphCore.neighbors failed.")
            return []

    def query(
        self,
        cypher: str,
        params: Optional[dict] = None,
    ) -> List[dict]:
        """
        Run a raw Cypher query through the adapter.

        This is intended for controlled, internal call sites only.
        """
        try:
            return self.adapter.run_query(cypher, params=params or {})
        except Exception:
            logger.exception("TemporalGraphCore.query failed.")
            return []

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close graph backend safely."""
        try:
            self.adapter.close()
        except Exception:
            logger.exception("TemporalGraphCore.close failed.")

    def _sanitize_predicate(self, predicate: str) -> str:
        """Ensure predicate is a valid Cypher relationship type."""
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", predicate or "").strip("_")
        if not cleaned or not cleaned[0].isalpha():
            return "RELATES_TO"
        return cleaned.upper()

    def run_query(self, cypher: str, params: dict):
        return self.adapter.run_query(cypher, params)


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
