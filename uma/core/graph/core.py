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
from typing import Any, List, Optional

from ...adapters.graph.base import GraphAdapter
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
        self.updater = GraphUpdater(adapter)

        logger.info(
            "TemporalGraphCore initialized with adapter=%s",
            adapter.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

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
                self.updater.add_fact_node(fact)
            except Exception:
                logger.exception("TemporalGraphCore.add_facts failed.")

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
        predicate_scope: Optional[List[str]] = None,
        depth: int = 1,
        k: int = 10,
    ) -> List[dict]:
        """
        Fetch graph neighbors for a node, scoped to a user.

        This is a read-only helper intended for RLM navigation.
        """
        if not user_id or not node_id:
            logger.warning("TemporalGraphCore.neighbors: missing user_id or node_id.")
            return []

        depth = max(1, int(depth))
        limit = max(1, int(k))

        preds = None
        if predicate_scope:
            preds = [self._sanitize_predicate(p) for p in predicate_scope if p]
            preds = [p for p in preds if p]

        cypher = """
        MATCH (u:User {id: $user_id})
        MATCH (u)-[*0..3]->(n {id: $node_id})
        MATCH path = (n)-[r*1..$depth]-(m)
        WHERE $preds IS NULL OR all(rel IN r WHERE type(rel) IN $preds)
        RETURN DISTINCT m, labels(m) AS labels, properties(m) AS properties
        LIMIT $limit
        """
        params = {
            "user_id": user_id,
            "node_id": node_id,
            "depth": depth,
            "limit": limit,
            "preds": preds,
        }

        try:
            return self.adapter.run_query(cypher, params=params)
        except Exception:
            logger.exception("TemporalGraphCore.neighbors failed.")
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
