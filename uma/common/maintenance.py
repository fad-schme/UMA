"""
Maintenance helpers for data consistency and recovery.

These utilities rebuild derived indexes from SQL-backed authoritative data.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Optional

from uma.common.types.types_scope import DEFAULT_TENANT_ID
from uma.common.types import Fact
from uma.common.types import OwnershipRef, Skill
from uma.common.identity import normalize_user_id
from uma.common.results import (
    DerivedRebuildReport,
    GraphRebuildReport,
    LaneRebuildStatus,
    VectorRebuildReport,
)
from uma.common.text import build_fact_embedding_text

logger = logging.getLogger(__name__)


def _ensure_async_lock(memory: Any, attr_name: str) -> asyncio.Lock:
    lock = getattr(memory, attr_name, None)
    if lock is not None:
        return lock

    lifecycle_lock = getattr(memory, "_lifecycle_lock", None)
    if lifecycle_lock is not None:
        with lifecycle_lock:
            existing = getattr(memory, attr_name, None)
            if existing is not None:
                return existing
            created = asyncio.Lock()
            setattr(memory, attr_name, created)
            return created

    created = asyncio.Lock()
    setattr(memory, attr_name, created)
    return created


def _ownership_ref(tenant_id: str, owner_type: str, owner_id: str) -> OwnershipRef:
    return OwnershipRef(
        tenant_id=tenant_id,
        owner_type=owner_type,
        owner_id=owner_id,
    )


def _scope_metadata_from_object(object: Any, *, include_session_id: bool) -> dict[str, Any]:
    owner_type = str(getattr(object, "owner_type", "") or "").strip()
    owner_id = str(getattr(object, "owner_id", "") or "").strip()
    if owner_type == "user":
        owner_id = normalize_user_id(owner_id)
    metadata: dict[str, Any] = {
        "tenant_id": str(getattr(object, "tenant_id", None) or "default"),
        "owner_type": owner_type,
        "owner_id": owner_id,
        "workspace_id": getattr(object, "workspace_id", None),
        "scope_model_version": getattr(object, "scope_model_version", None),
        "scope_key": f"{owner_type}:{owner_id}",
    }
    if include_session_id:
        metadata["session_id"] = getattr(object, "session_id", None)
    return metadata


def _split_isolation_from_metas(
    metas: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    """C1: split a list of pre-contract metadata dicts into the four
    parallel lists the new VectorIndex contract requires.

    Pulls `tenant_id` / `owner_type` / `owner_id` out of each dict into
    their respective parallel lists. The remaining keys become the
    per-row extra_metadata dict. Used by `rebuild_vector_indexes` which
    aggregates metadata dicts from per-row helpers.
    """
    tenant_ids: list[str] = []
    owner_types: list[str] = []
    owner_ids: list[str] = []
    extras: list[dict[str, Any]] = []
    for m in metas:
        m = dict(m or {})
        tenant_ids.append(str(m.pop("tenant_id", "") or ""))
        owner_types.append(str(m.pop("owner_type", "") or ""))
        owner_ids.append(str(m.pop("owner_id", "") or ""))
        extras.append(m)
    return tenant_ids, owner_types, owner_ids, extras


def _fact_vector_metadata(fact: Fact) -> dict[str, Any]:
    meta = _scope_metadata_from_object(fact, include_session_id=True)
    meta.update(
        {
            "subject": fact.subject,
            "predicate": fact.predicate,
        }
    )
    topic = (fact.meta or {}).get("topic") if isinstance(fact.meta, dict) else None
    if topic:
        meta["topic"] = topic
    return meta


def _skill_vector_metadata(skill: Skill) -> dict[str, Any]:
    meta = _scope_metadata_from_object(skill, include_session_id=False)
    meta["name"] = skill.name
    return meta


def _episode_vector_metadata(episode: Any) -> dict[str, Any]:
    meta = _scope_metadata_from_object(episode, include_session_id=True)
    meta["user_id"] = getattr(episode, "user_id", None)
    return meta

async def _embed_in_batches(embedder: Any, texts: list[str], batch_size: int) -> list[list[float]]:
    if not texts:
        return []
    expected_dim = getattr(embedder, "dimension", None)
    if not isinstance(expected_dim, int) or expected_dim <= 0:
        raise ValueError("_embed_in_batches: embedder.dimension must be a positive integer")
    if batch_size <= 0:
        vectors = await embedder.embed(texts)
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("_embed_in_batches: embedder returned invalid shape")
        for v in vectors:
            if not isinstance(v, list) or len(v) != expected_dim:
                raise ValueError("_embed_in_batches: invalid embedding dim")
        return vectors
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_vectors = await embedder.embed(batch)
        if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch):
            raise ValueError("_embed_in_batches: embedder returned invalid batch shape")
        for v in batch_vectors:
            if not isinstance(v, list) or len(v) != expected_dim:
                raise ValueError("_embed_in_batches: invalid embedding dim")
        vectors.extend(batch_vectors)
    return vectors


async def rebuild_vector_indexes(
    memory: Any,
    *,
    tenant_id: Optional[str] = None,
    owner_type: Optional[str] = None,
    owner_id: Optional[str] = None,
    include_episodic: bool = True,
    include_semantic: bool = True,
    include_procedural: bool = True,
    batch_size: int = 32,
) -> VectorRebuildReport:
    """
    Rebuild vector indexes from SQL stores.

    Notes
    -----
    - Episodic embeddings are reused if present; otherwise summaries are embedded.
    - Semantic and procedural embeddings are recomputed from text representations.
    - If tenant_id, owner_type or owner_id is not provided, semantic and episodic rebuilds are skipped.
    """
    lock = _ensure_async_lock(memory, "_vector_rebuild_lock")
    async with lock:
        return await _rebuild_vector_indexes_unlocked(
            memory,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
            include_procedural=include_procedural,
            batch_size=batch_size,
        )


async def _rebuild_vector_indexes_unlocked(
    memory: Any,
    *,
    tenant_id: Optional[str] = None,
    owner_type: Optional[str] = None,
    owner_id: Optional[str] = None,
    include_episodic: bool = True,
    include_semantic: bool = True,
    include_procedural: bool = True,
    batch_size: int = 32,
) -> VectorRebuildReport:
    report: dict[str, Any] = {
        "episodic": {"status": "skipped", "count": 0},
        "semantic": {"status": "skipped", "count": 0},
        "procedural": {"status": "skipped", "count": 0},
    }

    embedder = getattr(memory, "embedder", None)
    embedding_cfg = getattr(memory, "embedding_cfg", None)
    if embedding_cfg is None:
        raise ValueError("rebuild_vector_indexes: memory.embedding_cfg is required")
    if not getattr(embedding_cfg, "model", None):
        raise ValueError("rebuild_vector_indexes: embedding_cfg.model is required")
    dim = int(getattr(embedding_cfg, "dimension", 0) or 0)
    if dim <= 0:
        raise ValueError("rebuild_vector_indexes: embedding_cfg.dimension must be a positive integer")

    llm_cfg = getattr(memory, "llm_cfg", None)
    if llm_cfg is None:
        raise ValueError("rebuild_vector_indexes: memory.llm_cfg is required")
    if not getattr(llm_cfg, "provider", None):
        raise ValueError("rebuild_vector_indexes: llm_cfg.provider is required")
    if not getattr(llm_cfg, "model", None):
        raise ValueError("rebuild_vector_indexes: llm_cfg.model is required")

    agent_llm_cfg = getattr(memory, "agent_llm_cfg", None)
    if agent_llm_cfg is None:
        raise ValueError("rebuild_vector_indexes: memory.agent_llm_cfg is required")
    if not getattr(agent_llm_cfg, "provider", None):
        raise ValueError("rebuild_vector_indexes: agent_llm_cfg.provider is required")
    if not getattr(agent_llm_cfg, "model", None):
        raise ValueError("rebuild_vector_indexes: agent_llm_cfg.model is required")
    if embedder is None:
        return VectorRebuildReport(
            status="error",
            error="embedder not initialized",
            report={k: LaneRebuildStatus(**v) for k, v in report.items()},
        )

    tenant_id = tenant_id or getattr(memory, "tenant_id", None) or DEFAULT_TENANT_ID

    if include_episodic:
        if not owner_type or not owner_id:
            report["episodic"]["status"] = "skipped"
        else:
            try:
                episodes = await memory.episodic_core.list_episodes(tenant_id, owner_type, owner_id)
                ids: list[str] = []
                vectors: list[list[float]] = []
                metas: list[dict[str, Any]] = []
                texts: list[str] = []
                text_ids: list[str] = []
                text_metas: list[dict[str, Any]] = []

                for ep in episodes:
                    if ep.embedding and len(ep.embedding) == dim:
                        ids.append(ep.id)
                        vectors.append(ep.embedding)
                        metas.append(_episode_vector_metadata(ep))
                    else:
                        text_ids.append(ep.id)
                        texts.append(ep.summary or "")
                        text_metas.append(_episode_vector_metadata(ep))

                if texts:
                    text_vectors = await _embed_in_batches(embedder, texts, batch_size)
                    ids.extend(text_ids)
                    vectors.extend(text_vectors)
                    metas.extend(text_metas)

                if ids:
                    idx = memory.episodic_core.vector_index() if memory.episodic_core else None
                    if idx is None:
                        raise RuntimeError("episodic vector index missing")
                    tids, otypes, oids, extras = _split_isolation_from_metas(metas)
                    idx.upsert(
                        ids=ids,
                        vectors=vectors,
                        tenant_ids=tids,
                        owner_types=otypes,
                        owner_ids=oids,
                        extra_metadata=extras,
                    )
                report["episodic"] = {"status": "ok", "count": len(ids)}
            except Exception:
                logger.exception("rebuild_vector_indexes: episodic rebuild failed.")
                report["episodic"] = {"status": "error", "count": 0}

    if include_semantic:
        if not owner_type or not owner_id:
            report["semantic"]["status"] = "skipped"
        else:
            try:
                scoped_owner_id = owner_id
                if owner_type == "user":
                    scoped_owner_id = normalize_user_id(owner_id)
                facts: list[Fact] = await memory.semantic_core.list_facts_for_owner(
                    tenant_id=tenant_id,
                    owner_type=owner_type,
                    owner_id=scoped_owner_id,
                    limit=None,
                )
                if facts:
                    texts = [build_fact_embedding_text(f) for f in facts]
                    vectors = await _embed_in_batches(embedder, texts, batch_size)
                    ids = [f.id for f in facts]
                    metas = [_fact_vector_metadata(f) for f in facts]
                    idx = memory.semantic_core.vector_index() if memory.semantic_core else None
                    if idx is None:
                        raise RuntimeError("semantic vector index missing")
                    tids, otypes, oids, extras = _split_isolation_from_metas(metas)
                    idx.upsert(
                        ids=ids,
                        vectors=vectors,
                        tenant_ids=tids,
                        owner_types=otypes,
                        owner_ids=oids,
                        extra_metadata=extras,
                    )
                report["semantic"] = {"status": "ok", "count": len(facts)}
            except Exception:
                logger.exception("rebuild_vector_indexes: semantic rebuild failed.")
                report["semantic"] = {"status": "error", "count": 0}

    if include_procedural:
        try:
            if not owner_type or not owner_id:
                report["procedural"]["status"] = "skipped"
            else:
                skills = await memory.procedural_core.list_skills(
                    owner=_ownership_ref(tenant_id, owner_type, owner_id),
                )
                if skills:
                    texts = [_skill_embedding_text(s) for s in skills]
                    vectors = await _embed_in_batches(embedder, texts, batch_size)
                    ids = [s.id for s in skills]
                    metas = [_skill_vector_metadata(s) for s in skills]
                    idx = memory.procedural_core.vector_index() if memory.procedural_core else None
                    if idx is None:
                        raise RuntimeError("procedural vector index missing")
                    tids, otypes, oids, extras = _split_isolation_from_metas(metas)
                    idx.upsert(
                        ids=ids,
                        vectors=vectors,
                        tenant_ids=tids,
                        owner_types=otypes,
                        owner_ids=oids,
                        extra_metadata=extras,
                    )
                report["procedural"] = {"status": "ok", "count": len(skills)}
        except Exception:
            logger.exception("rebuild_vector_indexes: procedural rebuild failed.")
            report["procedural"] = {"status": "error", "count": 0}

    overall = "ok"
    if any(section["status"] == "error" for section in report.values()):
        overall = "error"
    elif any(section["status"] == "skipped" for section in report.values()):
        overall = "degraded"

    return VectorRebuildReport(
        status=overall,
        report={k: LaneRebuildStatus(**v) for k, v in report.items()},
    )


async def rebuild_derived_indexes(
    memory: Any,
    *,
    tenant_id: Optional[str] = None,
    owner_type: Optional[str] = None,
    owner_id: Optional[str] = None,
    include_episodic: bool = True,
    include_semantic: bool = True,
    include_procedural: bool = True,
    include_graph: bool = True,
    batch_size: int = 32,
) -> DerivedRebuildReport:
    vector_result = await rebuild_vector_indexes(
        memory,
        tenant_id=tenant_id,
        owner_type=owner_type,
        owner_id=owner_id,
        include_episodic=include_episodic,
        include_semantic=include_semantic,
        include_procedural=include_procedural,
        batch_size=batch_size,
    )
    graph_report = await _rebuild_graph_from_authoritative_stores(
        memory,
        tenant_id=tenant_id,
        owner_type=owner_type,
        owner_id=owner_id,
        include_graph=include_graph,
    )
    overall = "ok"
    if vector_result.status == "error" or graph_report.status == "error":
        overall = "error"
    elif vector_result.status != "ok" or graph_report.status != "ok":
        overall = "degraded"
    return DerivedRebuildReport(
        status=overall,
        vector=vector_result,
        graph=graph_report,
    )


def _clear_scoped_graph_materialization(
    graph: Any,
    *,
    tenant_id: str,
    owner_type: str,
    owner_id: str,
) -> None:
    if not hasattr(graph, "run_query"):
        raise RuntimeError("graph core missing run_query")

    params = {
        "tenant_id": tenant_id,
        "owner_type": owner_type,
        "owner_id": owner_id,
    }

    graph.run_query(
        """
        MATCH ()-[r]-()
        WHERE r.tenant_id = $tenant_id
          AND r.owner_type = $owner_type
          AND r.owner_id = $owner_id
        DELETE r
        """,
        params,
    )
    graph.run_query(
        """
        MATCH (f:Fact)
        WHERE f.tenant_id = $tenant_id
          AND f.owner_type = $owner_type
          AND f.owner_id = $owner_id
        DETACH DELETE f
        """,
        params,
    )
    graph.run_query(
        """
        MATCH (e:Episode)
        WHERE e.tenant_id = $tenant_id
          AND e.owner_type = $owner_type
          AND e.owner_id = $owner_id
        DETACH DELETE e
        """,
        params,
    )


async def _rebuild_graph_from_authoritative_stores(
    memory: Any,
    *,
    tenant_id: Optional[str],
    owner_type: Optional[str],
    owner_id: Optional[str],
    include_graph: bool,
) -> GraphRebuildReport:
    lock = _ensure_async_lock(memory, "_graph_rebuild_lock")
    async with lock:
        return await _rebuild_graph_from_authoritative_stores_unlocked(
            memory,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            include_graph=include_graph,
        )


async def _rebuild_graph_from_authoritative_stores_unlocked(
    memory: Any,
    *,
    tenant_id: Optional[str],
    owner_type: Optional[str],
    owner_id: Optional[str],
    include_graph: bool,
) -> GraphRebuildReport:
    report: dict[str, Any] = {
        "status": "skipped",
        "episodes": 0,
        "facts": 0,
        "episode_fact_links": 0,
        "temporal_links": 0,
    }
    if not include_graph:
        return GraphRebuildReport(**report)
    tenant_id = tenant_id or getattr(memory, "tenant_id", None) or DEFAULT_TENANT_ID
    if not owner_type or not owner_id:
        return GraphRebuildReport(**report)

    graph = getattr(memory, "graph_core", None)
    episodic_core = getattr(memory, "episodic_core", None)
    semantic_core = getattr(memory, "semantic_core", None)
    if graph is None or episodic_core is None or semantic_core is None:
        return GraphRebuildReport(**report)

    try:
        scoped_owner_id = normalize_user_id(owner_id) if owner_type == "user" else owner_id
        _clear_scoped_graph_materialization(
            graph,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=scoped_owner_id,
        )
        episodes = await episodic_core.list_episodes(tenant_id, owner_type, scoped_owner_id) if include_graph else []
        facts: list[Fact] = await semantic_core.list_facts_for_owner(
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=scoped_owner_id,
            limit=None,
        )

        for episode in episodes or []:
            graph.add_episode(episode)
        if facts:
            graph.add_facts(list(facts))

        facts_by_turn_id: dict[str, list[Fact]] = defaultdict(list)
        for fact in facts or []:
            meta = getattr(fact, "meta", None) or {}
            turn_id = str(meta.get("turn_id") or "").strip() if isinstance(meta, dict) else ""
            if turn_id:
                facts_by_turn_id[turn_id].append(fact)

        episode_fact_links = 0
        for episode in episodes or []:
            meta = getattr(episode, "meta", None) or {}
            turn_id = str(meta.get("turn_id") or "").strip() if isinstance(meta, dict) else ""
            if not turn_id:
                continue
            linked_facts = facts_by_turn_id.get(turn_id) or []
            if not linked_facts:
                continue
            graph.link_episode_to_facts(episode, linked_facts)
            episode_fact_links += len(linked_facts)

        temporal_links = 0
        scoped_sequences: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for episode in episodes or []:
            session_id = str(getattr(episode, "session_id", "") or "").strip()
            origin_agent_id = str(getattr(episode, "origin_agent_id", "") or "").strip()
            if not session_id or not origin_agent_id:
                continue
            scoped_sequences[(session_id, origin_agent_id)].append(episode)
        for scoped_episodes in scoped_sequences.values():
            ordered = sorted(scoped_episodes, key=lambda ep: (getattr(ep, "timestamp", None), getattr(ep, "id", "")))
            for previous, current in zip(ordered, ordered[1:]):
                graph.link_temporal(previous, current)
                temporal_links += 1

        report.update(
            {
                "status": "ok",
                "episodes": len(episodes or []),
                "facts": len(facts or []),
                "episode_fact_links": episode_fact_links,
                "temporal_links": temporal_links,
            }
        )
        return GraphRebuildReport(**report)
    except Exception:
        logger.exception("rebuild_derived_indexes: graph rebuild failed.")
        report["status"] = "error"
        return GraphRebuildReport(**report)


def _skill_embedding_text(skill: Skill) -> str:
    return (
        f"name: {skill.name}\n"
        f"phrases: {skill.trigger_phrases}\n"
        f"patterns: {skill.trigger_patterns}\n"
        f"example: {skill.example}\n"
        f"plan: {skill.plan}\n"
        f"tools: {skill.tools}\n"
    )
