"""
Maintenance helpers for data consistency and recovery.

These utilities focus on reindexing vector stores from SQL-backed data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...types import Fact
from ...types import Skill
from .identity import normalize_user_id
from .user_query_helper import build_fact_embedding_text

logger = logging.getLogger(__name__)

async def _embed_in_batches(embedder: Any, texts: List[str], batch_size: int) -> List[List[float]]:
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
    vectors: List[List[float]] = []
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
    owner_type: Optional[str] = None,
    owner_id: Optional[str] = None,
    include_episodic: bool = True,
    include_semantic: bool = True,
    include_procedural: bool = True,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Rebuild vector indexes from SQL stores.

    Notes
    -----
    - Episodic embeddings are reused if present; otherwise summaries are embedded.
    - Semantic and procedural embeddings are recomputed from text representations.
    - If owner_type or owner_id is not provided, semantic and episodic rebuilds are skipped.
    """
    report: Dict[str, Any] = {
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
        return {
            "status": "error",
            "error": "embedder not initialized",
            "report": report,
        }

    if include_episodic:
        if not owner_type or not owner_id:
            report["episodic"]["status"] = "skipped"
        else:
            try:
                episodes = await memory.episodic_core.list_episodes(owner_type, owner_id)
                ids: List[str] = []
                vectors: List[List[float]] = []
                metas: List[Dict[str, Any]] = []
                texts: List[str] = []
                text_ids: List[str] = []
                text_metas: List[Dict[str, Any]] = []

                for ep in episodes:
                    if ep.embedding and len(ep.embedding) == dim:
                        ids.append(ep.id)
                        vectors.append(ep.embedding)
                        metas.append({"owner_type": ep.owner_type, "owner_id": ep.owner_id, "user_id": ep.user_id})
                    else:
                        text_ids.append(ep.id)
                        texts.append(ep.summary or "")
                        text_metas.append({"owner_type": ep.owner_type, "owner_id": ep.owner_id, "user_id": ep.user_id})

                if texts:
                    text_vectors = await _embed_in_batches(embedder, texts, batch_size)
                    ids.extend(text_ids)
                    vectors.extend(text_vectors)
                    metas.extend(text_metas)

                if ids:
                    idx = memory.episodic_core.vector_index() if memory.episodic_core else None
                    if idx is None:
                        raise RuntimeError("episodic vector index missing")
                    idx.upsert(ids=ids, vectors=vectors, metadata=metas)
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
                facts: List[Fact] = await memory.semantic_core.list_facts_for_owner(
                    owner_type=owner_type,
                    owner_id=scoped_owner_id,
                    limit=None,
                )
                if facts:
                    texts = [build_fact_embedding_text(f) for f in facts]
                    vectors = await _embed_in_batches(embedder, texts, batch_size)
                    ids = [f.id for f in facts]
                    metas = [
                        {
                            "subject": f.subject,
                            "predicate": f.predicate,
                            "owner_type": f.owner_type,
                            "owner_id": f.owner_id,
                            "scope_key": f"{f.owner_type}:{f.owner_id}",
                        }
                        for f in facts
                    ]
                    idx = memory.semantic_core.vector_index() if memory.semantic_core else None
                    if idx is None:
                        raise RuntimeError("semantic vector index missing")
                    idx.upsert(ids=ids, vectors=vectors, metadata=metas)
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
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
                if skills:
                    texts = [_skill_embedding_text(s) for s in skills]
                    vectors = await _embed_in_batches(embedder, texts, batch_size)
                    ids = [s.id for s in skills]
                    metas = [{"name": s.name, "owner_type": s.owner_type, "owner_id": s.owner_id} for s in skills]
                    idx = memory.procedural_core.vector_index() if memory.procedural_core else None
                    if idx is None:
                        raise RuntimeError("procedural vector index missing")
                    idx.upsert(ids=ids, vectors=vectors, metadata=metas)
                report["procedural"] = {"status": "ok", "count": len(skills)}
        except Exception:
            logger.exception("rebuild_vector_indexes: procedural rebuild failed.")
            report["procedural"] = {"status": "error", "count": 0}

    overall = "ok"
    if any(section["status"] == "error" for section in report.values()):
        overall = "error"
    elif any(section["status"] == "skipped" for section in report.values()):
        overall = "degraded"

    return {"status": overall, "report": report}


def _skill_embedding_text(skill: Skill) -> str:
    return (
        f"name: {skill.name}\n"
        f"phrases: {skill.trigger_phrases}\n"
        f"patterns: {skill.trigger_patterns}\n"
        f"example: {skill.example}\n"
        f"plan: {skill.plan}\n"
        f"tools: {skill.tools}\n"
    )
