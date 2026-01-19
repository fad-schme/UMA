"""
Temporal Graph Updater (Core)
=============================

This module defines logic for how UMA-3 updates the temporal graph:
- Episode nodes
- Fact nodes
- Links between them
- Temporal relations (PRECEDES/FOLLOWS)

Coding Agent Instructions
-------------------------
- This should contain graph *logic*, not DB backend code.
- DB communication happens through GraphAdapter.
"""

from __future__ import annotations
import logging
from typing import Any

from ...adapters.graph.base import GraphAdapter

logger = logging.getLogger(__name__)


class GraphUpdater:
    """
    High-level temporal graph update logic.

    This class encapsulates how episodic and semantic memory objects are
    projected into the temporal knowledge graph.

    It should remain:
        - Backend-agnostic (only talks to GraphAdapter)
        - Safe (never raise exceptions to callers)
        - Focused on graph *logic* (no driver-specific code)
    """

    def __init__(self, graph: GraphAdapter):
        """
        Initialize the GraphUpdater with a concrete GraphAdapter.

        Parameters
        ----------
        graph : GraphAdapter
            Backend graph adapter (e.g., Neo4jAdapter, MemgraphAdapter).
        """
        self.graph = graph
        logger.info("GraphUpdater initialized.")

    # ------------------------------------------------------------------
    # EPISODES
    # ------------------------------------------------------------------

    def add_episode_node(self, episode: Any):
        """
        Insert an Episode node and link it to its User.

        Graph structure:

            (u:User {id: episode.user_id})
                -[:HAS_EPISODE]-> (e:Episode {id: episode.id})

        Episode properties:
            - summary
            - timestamp
        """

        if not hasattr(episode, "id") or not hasattr(episode, "timestamp"):
            logger.error(
                "GraphUpdater.add_episode_node: invalid episode object=%r", episode
            )
            return

        try:
            self.graph.run_query(
                """
                MERGE (u:User {id: $user_id})
                MERGE (e:Episode {id: $id})
                SET e.summary = $summary,
                    e.timestamp = $timestamp
                MERGE (u)-[:HAS_EPISODE]->(e)
                """,
                {
                    "id": episode.id,
                    "user_id": episode.user_id,
                    "summary": episode.summary,
                    "timestamp": episode.timestamp.isoformat(),
                },
            )
        except Exception:
            logger.exception("GraphUpdater.add_episode_node failed.")

    # ------------------------------------------------------------------
    # FACTS (SEMANTIC)
    # ------------------------------------------------------------------

    def add_fact_node(self, fact: Any):
        """
        Insert a Fact node AND a predicate-scoped semantic edge.

        Structure added:

            (u:User {id:fact.subject})
                -[:PREDICATE {confidence, ...}]-> (o:Entity {id:fact.object})

        And preserve a Fact node for auditing:

            (f:Fact {id, subject, predicate, object, updated_at})

        The goal is to keep both:
            - a clean, semantic knowledge graph (User/Entity + edge)
            - a detailed fact node for history / debugging / export.
        """

        # Validate object
        if (
            not hasattr(fact, "id")
            or not hasattr(fact, "subject")
            or not hasattr(fact, "predicate")
        ):
            logger.error(
                "GraphUpdater.add_fact_node: invalid fact object=%r", fact
            )
            return

        subject = fact.subject
        predicate = fact.predicate.upper()
        obj = str(fact.object)
        conf = getattr(fact, "confidence", 1.0)
        updated = fact.updated_at.isoformat() if hasattr(fact, "updated_at") else None

        try:
            # 1) User anchor
            self.graph.run_query(
                """
                MERGE (u:User {id: $subject})
                """,
                {"subject": subject},
            )

            # 2) Object as entity
            self.graph.run_query(
                """
                MERGE (o:Entity {id: $object})
                """,
                {"object": obj},
            )

            # 3) Predicate-scoped relationship
            self.graph.run_query(
                f"""
                MATCH (u:User {{id: $subject}}), (o:Entity {{id: $object}})
                MERGE (u)-[r:{predicate}]->(o)
                SET r.confidence = $confidence,
                    r.updated_at = $updated_at
                """,
                {
                    "subject": subject,
                    "object": obj,
                    "confidence": conf,
                    "updated_at": updated,
                },
            )

            # 4) Preserve Fact node (audit/reasoning)
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
                    "predicate": fact.predicate,
                    "object": obj,
                    "updated_at": updated,
                },
            )

        except Exception:
            logger.exception("GraphUpdater.add_fact_node failed.")

    # ------------------------------------------------------------------
    # LINKS: EPISODE ↔ FACTS
    # ------------------------------------------------------------------

    def link_episode_to_facts(self, episode: Any, facts: list[Any]):
        """
        Connect an episode to its extracted semantic facts.

        Adds:
            (Episode)-[:MENTIONS]->(Fact)

        And, when possible, also adds a semantic edge to the fact's object:

            (Episode)-[:PREDICATE]->(Entity)

        where PREDICATE is the upper-cased fact.predicate.
        """

        if not isinstance(facts, list):
            logger.error(
                "GraphUpdater.link_episode_to_facts: facts must be a list, got=%r",
                type(facts),
            )
            return

        for fact in facts:
            try:
                # Episode → Fact node connection
                self.graph.run_query(
                    """
                    MATCH (e:Episode {id: $ep_id}), (f:Fact {id: $f_id})
                    MERGE (e)-[:MENTIONS]->(f)
                    """,
                    {"ep_id": episode.id, "f_id": fact.id},
                )

                # ALSO: Episode → semantic object (predicate-scoped) link
                if hasattr(fact, "object") and hasattr(fact, "predicate"):
                    pred = fact.predicate.upper()
                    self.graph.run_query(
                        f"""
                        MATCH (e:Episode {{id: $ep_id}}), (o:Entity {{id: $object}})
                        MERGE (e)-[:{pred}]->(o)
                        """,
                        {
                            "ep_id": episode.id,
                            "object": str(fact.object),
                        },
                    )
            except Exception:
                logger.exception("GraphUpdater.link_episode_to_facts failed.")

    # ------------------------------------------------------------------
    # TEMPORAL LINKS
    # ------------------------------------------------------------------

    def link_temporal(self, ep_prev: Any, ep_next: Any):
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
                MATCH (a:Episode {id: $a}), (b:Episode {id: $b})
                MERGE (a)-[:PRECEDES]->(b)
                MERGE (b)-[:FOLLOWS]->(a)
                """,
                {"a": ep_prev.id, "b": ep_next.id},
            )
        except Exception:
            logger.exception("GraphUpdater.link_temporal failed.")