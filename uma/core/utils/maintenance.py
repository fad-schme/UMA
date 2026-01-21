"""
Maintenance helpers for data consistency and recovery.

These utilities focus on reindexing vector stores from SQL-backed data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...types_fact import Fact
from ...types_skill import Skill
from .identity import ensure_user_subject

logger = logging.getLogger(__name__)

async def _embed_in_batches(embedder: Any, texts: List[str], batch_size: int) -> List[List[float]]:
    if not texts:
        return []
    if batch_size <= 0:
        return await embedder.embed(texts)
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_vectors = await embedder.embed(batch)
        vectors.extend(batch_vectors)
    return vectors


async def rebuild_vector_indexes(
    memory: Any,
    *,
    user_id: Optional[str] = None,
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
    - If user_id is not provided, semantic and episodic rebuilds are skipped.
    """
    report: Dict[str, Any] = {
        "episodic": {"status": "skipped", "count": 0},
        "semantic": {"status": "skipped", "count": 0},
        "procedural": {"status": "skipped", "count": 0},
    }

    embedder = getattr(memory, "embedder", None)
    dim = int(getattr(getattr(memory, "embedding_cfg", None), "dimension", 0) or 0)
    if embedder is None:
        return {
            "status": "error",
            "error": "embedder not initialized",
            "report": report,
        }

    if include_episodic:
        if not user_id:
            report["episodic"]["status"] = "skipped"
        else:
            try:
                episodes = await memory.episodic_store.list_episodes(user_id)
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
                        metas.append({"user_id": ep.user_id})
                    else:
                        text_ids.append(ep.id)
                        texts.append(ep.summary or "")
                        text_metas.append({"user_id": ep.user_id})

                if texts:
                    text_vectors = await _embed_in_batches(embedder, texts, batch_size)
                    ids.extend(text_ids)
                    vectors.extend(text_vectors)
                    metas.extend(text_metas)

                if ids:
                    memory.episodic_store.vector_index.upsert(ids=ids, vectors=vectors, metadata=metas)
                report["episodic"] = {"status": "ok", "count": len(ids)}
            except Exception:
                logger.exception("rebuild_vector_indexes: episodic rebuild failed.")
                report["episodic"] = {"status": "error", "count": 0}

    if include_semantic:
        if not user_id:
            report["semantic"]["status"] = "skipped"
        else:
            try:
                subject = ensure_user_subject(user_id)
                facts: List[Fact] = await memory.semantic_store.list_facts_for_subject(subject)
                if facts:
                    texts = [f"{f.subject} {f.predicate} {f.object}" for f in facts]
                    vectors = await _embed_in_batches(embedder, texts, batch_size)
                    ids = [f.id for f in facts]
                    metas = [{"subject": f.subject, "predicate": f.predicate} for f in facts]
                    memory.semantic_store.vector_index.upsert(ids=ids, vectors=vectors, metadata=metas)
                report["semantic"] = {"status": "ok", "count": len(facts)}
            except Exception:
                logger.exception("rebuild_vector_indexes: semantic rebuild failed.")
                report["semantic"] = {"status": "error", "count": 0}

    if include_procedural:
        try:
            skills: List[Skill] = await memory.procedural_store.list_skills()
            if skills:
                texts = [_skill_embedding_text(s) for s in skills]
                vectors = await _embed_in_batches(embedder, texts, batch_size)
                ids = [s.id for s in skills]
                metas = [{"name": s.name} for s in skills]
                memory.procedural_store.vector_index.upsert(ids=ids, vectors=vectors, metadata=metas)
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
