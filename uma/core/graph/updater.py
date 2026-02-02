"""
Temporal Graph Updater (Core)
=============================

Projects episodic and semantic memory into the temporal knowledge graph.

DAT invariants enforced
----------------------
- Graph is DERIVED (never authoritative)
- Fact → graph provenance is mandatory
- Ownership is explicit on all nodes/edges used for traversal
- Graph failures never crash UMA (best-effort)

Strict mode (UMA-RLM v1)
------------------------
- No legacy Cypher fallback for facts
- Ownership is stamped on ALL traversable relationship types
"""

from __future__ import annotations

import logging
from typing import Any, List

from ..utils.identity import ensure_user_subject

logger = logging.getLogger(__name__)


class GraphUpdater:
    """
    High-level temporal graph mapper.

    Responsibilities
    ----------------
    - Insert Episode nodes and link them to User (HAS_EPISODE)
    - Insert Fact triplets with full provenance (via TemporalGraphCore.insert_fact_triplet)
    - Link Episodes ↔ Facts (MENTIONS) WITH ownership
    - Link Episode → Entity (predicate edges) WITH ownership
    - Add temporal PRECEDES/FOLLOWS edges WITH ownership

    Notes
    -----
    - This class is write-only.
    - Reads are handled by TemporalGraphCore.neighbors/get_paths with strict ownership.
    """

    def __init__(self, graph_core: Any):
        """
        Parameters
        ----------
        graph_core :
            Instance of TemporalGraphCore.

        Notes
        -----
        GraphUpdater depends on graph_core.insert_fact_triplet().
        If that API is missing, graph ingestion is disabled.
        """
        self.graph_core = graph_core
        logger.info("GraphUpdater initialized (strict DAT-safe mode).")

    # ------------------------------------------------------------------
    # EPISODES
    # ------------------------------------------------------------------

    def add_episode_node(self, episode: Any) -> None:
        """
        Insert an Episode node and link it to its owner User.

        Graph shape:
            (User)-[:HAS_EPISODE {owner_type, owner_id, timestamp}]->(Episode)
        """
        try:
            if not hasattr(episode, "id") or not hasattr(episode, "timestamp"):
                raise ValueError("Invalid episode object")

            owner_type = str(getattr(episode, "owner_type", "user") or "user")
            owner_id_raw = str(getattr(episode, "owner_id", "") or "")
            owner_id = ensure_user_subject(owner_id_raw) if owner_type == "user" else owner_id_raw

            self.graph_core.adapter.run_query(
                """
                MERGE (u:User {id: $user_id})
                MERGE (e:Episode {id: $episode_id})
                SET e.summary = $summary,
                    e.timestamp = $timestamp,
                    e.owner_type = $owner_type,
                    e.owner_id = $owner_id
                MERGE (u)-[r:HAS_EPISODE]->(e)
                SET r.timestamp = $timestamp,
                    r.owner_type = $owner_type,
                    r.owner_id = $owner_id
                """,
                {
                    "user_id": owner_id,
                    "episode_id": str(episode.id),
                    "summary": getattr(episode, "summary", None),
                    "timestamp": episode.timestamp.isoformat(),
                    "owner_type": owner_type,
                    "owner_id": owner_id,
                },
            )

        except Exception:
            logger.exception("GraphUpdater.add_episode_node failed (ignored).")

    # ------------------------------------------------------------------
    # FACTS (SEMANTIC) — STRICT PROVENANCE
    # ------------------------------------------------------------------

    def add_fact(self, fact: Any) -> None:
        """
        Insert a semantic Fact into the graph with full provenance.

        REQUIRED
        --------
        graph_core.insert_fact_triplet MUST exist.
        """
        try:
            insert = getattr(self.graph_core, "insert_fact_triplet", None)
            if not callable(insert):
                logger.error("GraphUpdater.add_fact_node skipped: insert_fact_triplet missing.")
                return

            subj = ensure_user_subject(str(getattr(fact, "subject", "") or ""))
            pred = str(getattr(fact, "predicate", "") or "")
            obj = str(getattr(fact, "object", "") or "")

            owner_type = str(getattr(fact, "owner_type", "user") or "user")
            owner_id_raw = str(getattr(fact, "owner_id", "") or "")
            owner_id = ensure_user_subject(owner_id_raw) if owner_type == "user" else owner_id_raw

            meta = getattr(fact, "meta", {}) or {}
            source_chunk_id = meta.get("source_chunk_id")

            insert(
                subject=subj,
                predicate=pred,
                object=obj,
                confidence=float(getattr(fact, "confidence", 1.0) or 1.0),
                fact_id=str(getattr(fact, "id", "")),
                owner_type=owner_type,
                owner_id=owner_id,
                source_chunk_id=source_chunk_id,
                created_at=getattr(fact, "created_at", None),
                updated_at=getattr(fact, "updated_at", None),
            )

        except Exception:
            logger.exception("GraphUpdater.add_fact_node failed (ignored).")

    # ------------------------------------------------------------------
    # LINKS: EPISODE ↔ FACTS (STAMP OWNERSHIP)
    # ------------------------------------------------------------------

    def link_episode_to_facts(self, episode: Any, facts: List[Any]) -> None:
        """
        Link an Episode to the Facts it mentions.

        Graph edges created:
            (Episode)-[:MENTIONS {owner_type, owner_id}]->(Fact)
            (Episode)-[:<PREDICATE> {owner_type, owner_id}]->(Entity)

        This is REQUIRED for owner-scoped graph traversal.
        """
        if not facts or not hasattr(episode, "id"):
            return

        try:
            owner_type = str(getattr(episode, "owner_type", "user") or "user")
            owner_id_raw = str(getattr(episode, "owner_id", "") or "")
            owner_id = ensure_user_subject(owner_id_raw) if owner_type == "user" else owner_id_raw
        except Exception:
            logger.exception("GraphUpdater.link_episode_to_facts: invalid episode ownership")
            return

        for fact in facts:
            try:
                # Episode → Fact (MENTIONS)
                self.graph_core.run_query(
                    """
                    MATCH (e:Episode {id: $ep_id})
                    MATCH (f:Fact {id: $fact_id})
                    MERGE (e)-[r:MENTIONS]->(f)
                    SET r.owner_type = $owner_type,
                        r.owner_id = $owner_id
                    """,
                    {
                        "ep_id": str(episode.id),
                        "fact_id": str(getattr(fact, "id", "")),
                        "owner_type": owner_type,
                        "owner_id": owner_id,
                    },
                )

                # Episode → Entity (predicate edge)
                raw_pred = str(getattr(fact, "predicate", "")).upper()
                predicate = self.graph_core._sanitize_predicate(raw_pred)
                obj = str(getattr(fact, "object", "") or "")

                self.graph_core.run_query(
                    f"""
                    MATCH (e:Episode {{id: $ep_id}})
                    MERGE (o:Entity {{id: $object}})
                    MERGE (e)-[r:{predicate}]->(o)
                    SET r.owner_type = $owner_type,
                        r.owner_id = $owner_id
                    """,
                    {
                        "ep_id": str(episode.id),
                        "object": obj,
                        "owner_type": owner_type,
                        "owner_id": owner_id,
                    },
                )

            except Exception:
                logger.exception(
                    "GraphUpdater.link_episode_to_facts failed "
                    "(episode_id=%s, fact_id=%s)",
                    getattr(episode, "id", None),
                    getattr(fact, "id", None),
                )

    # ------------------------------------------------------------------
    # TEMPORAL LINKS (STAMP OWNERSHIP)
    # ------------------------------------------------------------------

    def link_temporal(self, ep_prev: Any, ep_next: Any) -> None:
        """
        Add PRECEDES / FOLLOWS relationships between episodes with ownership.
        """
        try:
            owner_type = str(getattr(ep_prev, "owner_type", "user") or "user")
            owner_id_raw = str(getattr(ep_prev, "owner_id", "") or "")
            owner_id = ensure_user_subject(owner_id_raw) if owner_type == "user" else owner_id_raw

            self.graph_core.adapter.run_query(
                """
                MATCH (a:Episode {id: $a})
                MATCH (b:Episode {id: $b})
                MERGE (a)-[p:PRECEDES]->(b)
                SET p.owner_type = $owner_type,
                    p.owner_id = $owner_id
                MERGE (b)-[f:FOLLOWS]->(a)
                SET f.owner_type = $owner_type,
                    f.owner_id = $owner_id
                """,
                {
                    "a": str(ep_prev.id),
                    "b": str(ep_next.id),
                    "owner_type": owner_type,
                    "owner_id": owner_id,
                },
            )

        except Exception:
            logger.exception("GraphUpdater.link_temporal failed (ignored).")

