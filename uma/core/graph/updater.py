"""
Temporal Graph Updater (Core)
=============================

This module defines how UMA projects episodic and semantic memory
into the temporal knowledge graph.

Design Principles
-----------------
- Backend-agnostic: talks only to GraphAdapter
- Safe: never raises exceptions to callers
- Deterministic: no retrieval or reasoning logic here
- Identity-consistent: ALL User nodes use "user:<id>"

This class is effectively the *graph mapper* for UMA v1.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List

from ...adapters.graph.base import GraphAdapter
from ..utils.identity import ensure_user_subject

logger = logging.getLogger(__name__)


class GraphUpdater:
    """
    High-level temporal graph update logic.

    Projects:
        - Episodes
        - Facts
        - Entities
        - Temporal relationships

    into a graph structure using Cypher (via GraphAdapter).
    """

    def __init__(self, graph: GraphAdapter):
        """
        Initialize the GraphUpdater.

        Parameters
        ----------
        graph : GraphAdapter
            Concrete backend adapter (Neo4jAdapter, MemgraphAdapter, etc.).
        """
        self.graph = graph
        logger.info("GraphUpdater initialized.")

    # ------------------------------------------------------------------
    # EPISODES
    # ------------------------------------------------------------------

    def add_episode_node(self, episode: Any) -> None:
        """
        Insert an Episode node and link it to its User.

        Graph structure:

            (u:User {id: "user:<id>"})
                -[:HAS_EPISODE]-> (e:Episode {id: episode.id})
        """
        if not hasattr(episode, "id") or not hasattr(episode, "timestamp"):
            logger.error(
                "GraphUpdater.add_episode_node: invalid episode object=%r",
                episode,
            )
            return

        try:
            subject = ensure_user_subject(getattr(episode, "user_id", ""))

            self.graph.run_query(
                """
                MERGE (u:User {id: $subject})
                MERGE (e:Episode {id: $id})
                SET e.summary = $summary,
                    e.timestamp = $timestamp
                MERGE (u)-[:HAS_EPISODE]->(e)
                """,
                {
                    "id": episode.id,
                    "subject": subject,
                    "summary": getattr(episode, "summary", None),
                    "timestamp": (
                        episode.timestamp.isoformat()
                        if hasattr(episode, "timestamp")
                        else None
                    ),
                },
            )
        except Exception:
            logger.exception("GraphUpdater.add_episode_node failed.")

    # ------------------------------------------------------------------
    # FACTS (SEMANTIC)
    # ------------------------------------------------------------------

    def add_fact_node(self, fact: Any) -> None:
        """
        Insert a Fact node AND a predicate-scoped semantic edge.

        Structure added:

            (u:User {id: fact.subject})
                -[:PREDICATE {confidence, updated_at}]-> (o:Entity {id: fact.object})

        Also preserves a Fact node for auditing/debugging:

            (f:Fact {id, subject, predicate, object, updated_at})
        """
        if (
            not hasattr(fact, "id")
            or not hasattr(fact, "subject")
            or not hasattr(fact, "predicate")
        ):
            logger.error(
                "GraphUpdater.add_fact_node: invalid fact object=%r",
                fact,
            )
            return

        try:
            subject = ensure_user_subject(getattr(fact, "subject", ""))
            raw_pred = str(getattr(fact, "predicate", "")).upper()
            predicate = self._sanitize_predicate(raw_pred)
            obj = str(getattr(fact, "object", ""))
            confidence = float(getattr(fact, "confidence", 1.0))
            updated_at = (
                fact.updated_at.isoformat()
                if hasattr(fact, "updated_at")
                else None
            )

            # User anchor
            self.graph.run_query(
                "MERGE (u:User {id: $subject})",
                {"subject": subject},
            )

            # Object entity
            self.graph.run_query(
                "MERGE (o:Entity {id: $object})",
                {"object": obj},
            )

            # Predicate-scoped relationship
            self.graph.run_query(
                f"""
                MATCH (u:User {{id: $subject}})
                MATCH (o:Entity {{id: $object}})
                MERGE (u)-[r:{predicate}]->(o)
                SET r.confidence = $confidence,
                    r.updated_at = $updated_at
                """,
                {
                    "subject": subject,
                    "object": obj,
                    "confidence": confidence,
                    "updated_at": updated_at,
                },
            )

            # Fact node (audit trail)
            self.graph.run_query(
                """
                MERGE (f:Fact {id: $id})
                SET f.subject = $subject,
                    f.predicate = $predicate,
                    f.object = $object,
                    f.updated_at = $updated_at
                """,
                {
                    "id": fact.id,
                    "subject": subject,
                    "predicate": raw_pred,
                    "object": obj,
                    "updated_at": updated_at,
                },
            )

        except Exception:
            logger.exception("GraphUpdater.add_fact_node failed.")

    # ------------------------------------------------------------------
    # LINKS: EPISODE ↔ FACTS
    # ------------------------------------------------------------------

    def link_episode_to_facts(self, episode: Any, facts: List[Any]) -> None:
        """
        Connect an Episode to its extracted Facts.

        Adds:
            (Episode)-[:MENTIONS]->(Fact)

        Also adds:
            (Episode)-[:PREDICATE]->(Entity)

        where PREDICATE is fact.predicate.upper().
        """
        if not isinstance(facts, list):
            logger.error(
                "GraphUpdater.link_episode_to_facts: facts must be a list, got=%r",
                type(facts),
            )
            return

        if not hasattr(episode, "id"):
            logger.error(
                "GraphUpdater.link_episode_to_facts: episode missing id=%r",
                episode,
            )
            return

        for fact in facts:
            try:
                # Episode → Fact
                self.graph.run_query(
                    """
                    MATCH (e:Episode {id: $ep_id})
                    MATCH (f:Fact {id: $f_id})
                    MERGE (e)-[:MENTIONS]->(f)
                    """,
                    {
                        "ep_id": episode.id,
                        "f_id": getattr(fact, "id", None),
                    },
                )

                # Episode → Entity (predicate edge)
                if hasattr(fact, "object") and hasattr(fact, "predicate"):
                    raw_pred = str(getattr(fact, "predicate", "")).upper()
                    predicate = self._sanitize_predicate(raw_pred)
                    obj = str(getattr(fact, "object", ""))

                    self.graph.run_query(
                        f"""
                        MATCH (e:Episode {{id: $ep_id}})
                        MATCH (o:Entity {{id: $object}})
                        MERGE (e)-[:{predicate}]->(o)
                        """,
                        {
                            "ep_id": episode.id,
                            "object": obj,
                        },
                    )
            except Exception:
                logger.exception("GraphUpdater.link_episode_to_facts failed.")

    # ------------------------------------------------------------------
    # TEMPORAL LINKS
    # ------------------------------------------------------------------

    def link_temporal(self, ep_prev: Any, ep_next: Any) -> None:
        """
        Add PRECEDES/FOLLOWS relationships between episodes.

        Graph pattern:

            (a:Episode)-[:PRECEDES]->(b:Episode)
            (b:Episode)-[:FOLLOWS]->(a:Episode)
        """
        if not hasattr(ep_prev, "id") or not hasattr(ep_next, "id"):
            logger.error(
                "GraphUpdater.link_temporal: invalid episode objects prev=%r next=%r",
                ep_prev,
                ep_next,
            )
            return

        try:
            self.graph.run_query(
                """
                MATCH (a:Episode {id: $a})
                MATCH (b:Episode {id: $b})
                MERGE (a)-[:PRECEDES]->(b)
                MERGE (b)-[:FOLLOWS]->(a)
                """,
                {
                    "a": ep_prev.id,
                    "b": ep_next.id,
                },
            )
        except Exception:
            logger.exception("GraphUpdater.link_temporal failed.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sanitize_predicate(self, predicate: str) -> str:
        """
        Ensure predicate is a valid Cypher relationship type:
        - uppercase alphanumerics + underscore
        - must start with a letter
        """
        cleaned = re.sub(r"[^A-Z0-9_]", "_", predicate or "").strip("_")
        if not cleaned or not cleaned[0].isalpha():
            cleaned = f"REL_{cleaned}" if cleaned else "RELATES_TO"
        if cleaned != (predicate or ""):
            logger.debug("Sanitized predicate %r -> %r", predicate, cleaned)
        return cleaned
