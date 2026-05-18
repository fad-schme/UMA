"""
Temporal Graph Updater (Core)
=============================

Maps episodic and semantic memory into the temporal knowledge graph.

DAT invariants enforced
----------------------
- Graph is DERIVED (never authoritative)
- Fact → graph provenance is mandatory
- Ownership is explicit on all nodes/edges used for traversal
- Graph failures never crash UMA (best-effort)

Strict mode (UMA v1)
------------------------
- No Cypher fallback for facts
- Ownership is stamped on ALL traversable relationship types
"""

from __future__ import annotations

import json
import logging
from typing import Any, List

from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types.types_scope import validate_owner_type, validate_tenant_id
from uma.common.identity import normalize_user_id

logger = logging.getLogger(__name__)


def _graph_scope_from_object(object: Any) -> tuple[str, str, str, str | None]:
    tenant_id = str(getattr(object, "tenant_id", None) or DEFAULT_TENANT_ID)
    owner_type = str(getattr(object, "owner_type", "") or "").strip()
    owner_id_raw = str(getattr(object, "owner_id", "") or "").strip()
    scope_model_version = getattr(object, "scope_model_version", None)

    tenant_id = validate_tenant_id(tenant_id)
    owner_type = validate_owner_type(owner_type)
    if owner_type == "system":
        raise ValueError("system owner_type is not supported for graph writes")
    if owner_type not in {"agent", "user", "workspace"}:
        raise ValueError(f"unsupported owner_type for graph writes: {owner_type!r}")
    owner_id = normalize_user_id(owner_id_raw) if owner_type == "user" else owner_id_raw
    if not owner_id:
        raise ValueError("owner_id is required for graph writes")
    return tenant_id, owner_type, owner_id, (str(scope_model_version) if scope_model_version else None)


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
        logger.debug("GraphUpdater initialized (strict DAT-safe mode).")

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

            tenant_id, owner_type, owner_id, scope_model_version = _graph_scope_from_object(episode)

            self.graph_core.adapter.run_query(
                """
                MERGE (u:User {id: $user_id})
                MERGE (e:Episode {id: $episode_id})
                SET e.summary = $summary,
                    e.timestamp = $timestamp,
                    e.tenant_id = $tenant_id,
                    e.owner_type = $owner_type,
                    e.owner_id = $owner_id,
                    e.scope_model_version = $scope_model_version
                MERGE (u)-[r:HAS_EPISODE]->(e)
                SET r.timestamp = $timestamp,
                    r.tenant_id = $tenant_id,
                    r.owner_type = $owner_type,
                    r.owner_id = $owner_id,
                    r.scope_model_version = $scope_model_version
                """,
                {
                    "user_id": owner_id,
                    "episode_id": str(episode.id),
                    "summary": getattr(episode, "summary", None),
                    "timestamp": episode.timestamp.isoformat(),
                    "tenant_id": tenant_id,
                    "owner_type": owner_type,
                    "owner_id": owner_id,
                    "scope_model_version": scope_model_version,
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
                logger.error("GraphUpdater.add_fact skipped: insert_fact_triplet missing.")
                return

            # Ownership is mandatory for graph navigation safety.
            tenant_id, owner_type, owner_id, scope_model_version = _graph_scope_from_object(fact)

            # IMPORTANT: do NOT normalize KB fact.subject into user:<id>.
            subj = str(getattr(fact, "subject", "") or "").strip()
            pred = str(getattr(fact, "predicate", "") or "").strip()
            obj = str(getattr(fact, "object", "") or "").strip()

            if not subj or not obj:
                logger.warning(
                    "GraphUpdater.add_fact skipped: missing subject/object (fact_id=%s subj=%r obj_len=%d)",
                    getattr(fact, "id", None),
                    subj,
                    len(obj) if isinstance(obj, str) else 0,
                )
                return
            if not pred:
                logger.warning(
                    "GraphUpdater.add_fact skipped: missing predicate (fact_id=%s subj=%r)",
                    getattr(fact, "id", None),
                    subj,
                )
                return

            # Provenance: use Fact.source_ids[0] when present.
            meta = getattr(fact, "meta", {}) or {}
            source_chunk_id = ""
            try:
                source_ids = getattr(fact, "source_ids", None)
                if isinstance(source_ids, list) and source_ids:
                    source_chunk_id = str(source_ids[0] or "")
            except Exception:
                source_chunk_id = ""

            # Timestamps must be strings (TemporalGraphCore.insert_fact_triplet expects str).
            def _to_iso(x: Any) -> str:
                if x is None:
                    return ""
                try:
                    if hasattr(x, "isoformat"):
                        return str(x.isoformat())
                    return str(x)
                except Exception:
                    return ""

            created_at_s = _to_iso(getattr(fact, "created_at", None))
            updated_at_s = _to_iso(getattr(fact, "updated_at", None))

            # Optional meta_json for diagnostics/provenance. Keep it bounded and safe.
            meta_json = None
            if isinstance(meta, dict) and meta:
                try:
                    meta_json = json.dumps(meta, ensure_ascii=False)[:4000]
                except Exception:
                    meta_json = None

            domain = None
            try:
                if isinstance(meta, dict):
                    d = meta.get("domain")
                    if isinstance(d, str) and d.strip():
                        domain = d.strip().lower()
            except Exception:
                domain = None

            insert(
                fact_id=str(getattr(fact, "id", "")),
                subject=subj,
                predicate=pred,
                object=obj,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                source_chunk_id=source_chunk_id,
                created_at=created_at_s,
                updated_at=updated_at_s,
                scope_model_version=scope_model_version,
                meta_json=meta_json,
                domain=domain,
            )

        except Exception:
            logger.exception("GraphUpdater.add_fact failed (ignored).")

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
            tenant_id, owner_type, owner_id, scope_model_version = _graph_scope_from_object(episode)
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
                    SET r.tenant_id = $tenant_id,
                        r.owner_type = $owner_type,
                        r.owner_id = $owner_id,
                        r.scope_model_version = $scope_model_version
                    """,
                    {
                        "ep_id": str(episode.id),
                        "fact_id": str(getattr(fact, "id", "")),
                        "tenant_id": tenant_id,
                        "owner_type": owner_type,
                        "owner_id": owner_id,
                        "scope_model_version": scope_model_version,
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
                    SET r.tenant_id = $tenant_id,
                        r.owner_type = $owner_type,
                        r.owner_id = $owner_id,
                        r.scope_model_version = $scope_model_version
                    """,
                    {
                        "ep_id": str(episode.id),
                        "object": obj,
                        "tenant_id": tenant_id,
                        "owner_type": owner_type,
                        "owner_id": owner_id,
                        "scope_model_version": scope_model_version,
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
            tenant_id, owner_type, owner_id, scope_model_version = _graph_scope_from_object(ep_prev)

            self.graph_core.adapter.run_query(
                """
                MATCH (a:Episode {id: $a})
                MATCH (b:Episode {id: $b})
                MERGE (a)-[p:PRECEDES]->(b)
                SET p.tenant_id = $tenant_id,
                    p.owner_type = $owner_type,
                    p.owner_id = $owner_id,
                    p.scope_model_version = $scope_model_version
                MERGE (b)-[f:FOLLOWS]->(a)
                SET f.tenant_id = $tenant_id,
                    f.owner_type = $owner_type,
                    f.owner_id = $owner_id,
                    f.scope_model_version = $scope_model_version
                """,
                {
                    "a": str(ep_prev.id),
                    "b": str(ep_next.id),
                    "tenant_id": tenant_id,
                    "owner_type": owner_type,
                    "owner_id": owner_id,
                    "scope_model_version": scope_model_version,
                },
            )

        except Exception:
            logger.exception("GraphUpdater.link_temporal failed (ignored).")
