"""
TemporalGraphCore
=================

A unified graph reasoning subsystem for UMA-3.

Responsibilities:
-----------------
• Manage entities as graph nodes
• Manage relationships as graph edges
• Link episodic memories to graph entities
• Insert semantic facts into graph (subject → predicate → object)
• Query subgraphs and multi-hop paths
• Work with ANY backend implementing GraphAdapter (Neo4j, Memgraph, etc.)

This class is intentionally thin and backend-agnostic:
- ALL Cypher / Bolt I/O happens in GraphAdapter
- This class handles naming, schema, and conceptual graph operations
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...adapters.graph.base import GraphAdapter

logger = logging.getLogger(__name__)


class TemporalGraphCore:
    """
    Unified, production-grade graph engine for UMA-3.

    Provides:
    - add_entity
    - add_relationship
    - relate_episode
    - insert_fact_triplet
    - get_neighbors
    - get_paths
    """

    def __init__(self, adapter: GraphAdapter) -> None:
        """
        Parameters
        ----------
        adapter : GraphAdapter
            Concrete graph backend (Neo4jAdapter or MemgraphAdapter).
        """
        if not isinstance(adapter, GraphAdapter):
            raise TypeError("TemporalGraphCore requires a GraphAdapter instance")

        self.adapter = adapter
        logger.info(
            "TemporalGraphCore initialized with backend=%s",
            adapter.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Entity Handling
    # ------------------------------------------------------------------

    def add_entity(
        self,
        entity_id: str,
        labels: Optional[List[str]] = None,
        **props: Any,
    ) -> None:
        """
        Create (or merge) an entity node.

        Parameters
        ----------
        entity_id : str
            Unique identifier for the node (e.g., 'user:1', 'topic:db').
        labels : Optional[List[str]]
            Node labels (e.g., ['User'], ['Concept'], ['Place']).
        props : dict
            Additional properties stored in the node.
        """
        labels = labels or ["Entity"]
        labels_str = ":".join(labels)

        cypher = f"""
        MERGE (n:{labels_str} {{id: $id}})
        SET n += $props
        RETURN n
        """

        params = {"id": entity_id, "props": props}
        self.adapter.run_query(cypher, params)

    # ------------------------------------------------------------------
    # Relationship Handling
    # ------------------------------------------------------------------

    def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        **props: Any,
    ) -> None:
        """
        Create or update a directed relationship.

        Parameters
        ----------
        source_id : str
            Source node 'from'.
        relation : str
            Relationship type (e.g., LIKES, MENTIONED, PREFERS).
        target_id : str
            Target node 'to'.
        props : dict
            Custom properties for the edge (e.g., timestamp, confidence).
        """
        rel = relation.upper()

        cypher = f"""
        MATCH (a {{id: $source_id}}),
              (b {{id: $target_id}})
        MERGE (a)-[r:{rel}]->(b)
        SET r += $props
        RETURN r
        """

        params = {"source_id": source_id, "target_id": target_id, "props": props}
        self.adapter.run_query(cypher, params)

    # ------------------------------------------------------------------
    # Episodic Linking
    # ------------------------------------------------------------------

    def relate_episode(
        self,
        episode_id: str,
        user_id: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Connect an episode to a user in the graph.

        Relations added:
            (User)-[:HAS_EPISODE]->(Episode)
        """
        ts = timestamp or datetime.utcnow()

        # Ensure both nodes exist
        self.add_entity(episode_id, labels=["Episode"], timestamp=str(ts))
        self.add_entity(user_id, labels=["User"])

        cypher = """
        MATCH (u {id: $user_id}), (e {id: $episode_id})
        MERGE (u)-[r:HAS_EPISODE]->(e)
        SET r.timestamp = $ts
        RETURN r
        """
        params = {"user_id": user_id, "episode_id": episode_id, "ts": str(ts)}

        self.adapter.run_query(cypher, params)

    # ------------------------------------------------------------------
    # Fact Ingestion (predicate-scoped)
    # ------------------------------------------------------------------

    def insert_fact_triplet(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        **meta: Any,
    ) -> None:
        """
        Insert a semantic fact as a predicate-scoped edge:

            (subject)-[:PREDICATE {confidence, ...}]->(object)

        Additionally, preserve a Fact node for auditing:

            (f:Fact {id, subject, predicate, object, updated_at})

        Parameters
        ----------
        subject : str
            Entity identifier for subject (e.g., 'user:1').
        predicate : str
            Relation type, will be upper-cased for edge label.
        obj : str
            Entity identifier for object.
        confidence : float
            Confidence score for fact storage.
        meta : dict
            Additional metadata stored on the edge.
        """

        # Ensure subject + object nodes exist
        self.add_entity(
            subject,
            labels=["User"] if subject.startswith("user:") else ["Entity"],
        )
        self.add_entity(obj, labels=["Entity"])

        rel = predicate.upper()
        props = {"confidence": confidence, **meta}

        # Create predicate-scoped relationship
        self.adapter.run_query(
            f"""
            MATCH (s {{id: $subject}}), (o {{id: $object}})
            MERGE (s)-[r:{rel}]->(o)
            SET r += $props
            RETURN r
            """,
            {"subject": subject, "object": obj, "props": props},
        )

        # Also preserve a Fact node for auditing
        self.adapter.run_query(
            """
            MERGE (f:Fact {id: $id})
            SET f.subject = $subject,
                f.predicate = $predicate,
                f.object = $object,
                f.updated_at = datetime()
            """,
            {
                "id": f"{subject}:{predicate}:{obj}",
                "subject": subject,
                "predicate": predicate,
                "object": obj,
            },
        )

    # ------------------------------------------------------------------
    # Queries and Reasoning
    # ------------------------------------------------------------------

    def get_neighbors(self, entity_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Return neighbors of an entity up to N hops.

        Useful for:
            - preference extraction
            - interest graphs
            - temporal reasoning
        """
        cypher = """
        MATCH (n {id: $id})-[*1..$depth]-(m)
        RETURN DISTINCT m.id AS neighbor
        """
        params = {"id": entity_id, "depth": depth}

        return self.adapter.run_query(cypher, params)

    def get_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Multi-hop reasoning: find paths between two nodes.

        Parameters
        ----------
        source_id : str
            Start entity id.
        target_id : str
            End entity id.
        max_depth : int
            Maximum hops to explore.
        """
        cypher = """
        MATCH p = (a {id: $source})-[*1..$max_depth]->(b {id: $target})
        RETURN p
        """
        params = {"source": source_id, "target": target_id, "max_depth": max_depth}

        return self.adapter.run_query(cypher, params)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close the underlying graph adapter.

        Must never raise; logs any errors.
        """
        try:
            self.adapter.close()
        except Exception:
            logger.exception("TemporalGraphCore.close() failed.")