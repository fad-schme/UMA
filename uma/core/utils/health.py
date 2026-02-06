"""
Health checks and dependency readiness probes for UMA-RLM.

This module provides lightweight checks for core dependencies such as
SQL backends, vector indices, and graph adapters. It avoids network-heavy
operations by default and reports a structured readiness summary.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from ...adapters.db.base import DBAdapter
from ...adapters.vector.base import VectorIndex
from ...adapters.graph.base import GraphAdapter
from ...adapters.llm.base import LLMInterface, EmbeddingInterface

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    name: str
    status: str
    detail: str
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _timed(detail_fn):
    start = time.monotonic()
    status, detail = detail_fn()
    latency = (time.monotonic() - start) * 1000.0
    return status, detail, latency


def _check_db(name: str, adapter: Optional[DBAdapter]) -> HealthCheck:
    if adapter is None:
        return HealthCheck(name=name, status="skipped", detail="adapter missing")

    def _probe():
        conn = adapter.get_connection()
        try:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            finally:
                try:
                    cursor.close()
                except Exception:
                    logger.debug("Health check cursor close failed for %s", name)
            return "ok", "connection ok"
        except Exception as exc:
            return "error", f"connection failed: {exc}"
        finally:
            try:
                conn.close()
            except Exception:
                logger.debug("Health check connection close failed for %s", name)

    status, detail, latency = _timed(_probe)
    return HealthCheck(name=name, status=status, detail=detail, latency_ms=latency)


def _check_vector(name: str, index: Optional[VectorIndex], dim: int) -> HealthCheck:
    if index is None:
        return HealthCheck(name=name, status="skipped", detail="index missing")

    def _probe():
        if hasattr(index, "dim"):
            index_dim = getattr(index, "dim", None)
        else:
            index_dim = getattr(index, "dimension", None)

        if index_dim and int(index_dim) != int(dim):
            return "error", f"dimension mismatch: index={index_dim} config={dim}"

        verify = getattr(index, "verify_connectivity", None)
        if callable(verify):
            return ("ok", "connectivity ok") if verify() else ("error", "connectivity failed")

        try:
            index.query([0.0] * dim, k=1)
            return "ok", "query ok"
        except Exception as exc:
            return "error", f"query failed: {exc}"

    status, detail, latency = _timed(_probe)
    return HealthCheck(name=name, status=status, detail=detail, latency_ms=latency)


def _check_graph(name: str, adapter: Optional[GraphAdapter]) -> HealthCheck:
    if adapter is None:
        return HealthCheck(name=name, status="skipped", detail="graph disabled")

    def _probe():
        verify = getattr(adapter, "verify_connectivity", None)
        if callable(verify):
            return ("ok", "connectivity ok") if verify() else ("error", "connectivity failed")
        try:
            adapter.run_query("RETURN 1 AS ok")
            return "ok", "query ok"
        except Exception as exc:
            return "error", f"query failed: {exc}"

    status, detail, latency = _timed(_probe)
    return HealthCheck(name=name, status=status, detail=detail, latency_ms=latency)


def _check_llm(llm: Optional[LLMInterface]) -> HealthCheck:
    if llm is None:
        return HealthCheck(name="llm", status="error", detail="not initialized")
    return HealthCheck(name="llm", status="ok", detail=llm.__class__.__name__)


def _check_embedder(embedder: Optional[EmbeddingInterface], dim: int) -> HealthCheck:
    if embedder is None:
        return HealthCheck(name="embedding", status="error", detail="not initialized")
    if embedder.dimension != dim:
        return HealthCheck(
            name="embedding",
            status="error",
            detail=f"dimension mismatch: embedder={embedder.dimension} config={dim}",
        )
    return HealthCheck(name="embedding", status="ok", detail=embedder.__class__.__name__)


def run_health_checks(memory: Any) -> Dict[str, Any]:
    """
    Run basic readiness checks for UMA-RLM dependencies.

    Returns a dict with overall status and per-check details.
    """
    checks: Dict[str, HealthCheck] = {}

    epi_store = getattr(getattr(memory, "episodic_core", None), "store", None)
    sem_core = getattr(memory, "semantic_core", None)
    sem_store = getattr(getattr(sem_core, "ingestor", None), "semantic_store", None)
    proc_store = getattr(getattr(memory, "procedural_core", None), "store", None)

    checks["db:episodic"] = _check_db(
        "db:episodic", getattr(epi_store, "_db_adapter", None)
    )
    checks["db:semantic"] = _check_db(
        "db:semantic", getattr(sem_store, "_db_adapter", None)
    )
    checks["db:procedural"] = _check_db(
        "db:procedural", getattr(proc_store, "_db_adapter", None)
    )

    embedding_cfg = getattr(memory, "embedding_cfg", None)
    if embedding_cfg is None:
        raise ValueError("health_check: memory.embedding_cfg is required")
    if not getattr(embedding_cfg, "model", None):
        raise ValueError("health_check: embedding_cfg.model is required")
    dim = int(getattr(embedding_cfg, "dimension", 0) or 0)
    if dim <= 0:
        raise ValueError("health_check: embedding_cfg.dimension must be a positive integer")

    llm_cfg = getattr(memory, "llm_cfg", None)
    if llm_cfg is None:
        raise ValueError("health_check: memory.llm_cfg is required")
    if not getattr(llm_cfg, "provider", None):
        raise ValueError("health_check: llm_cfg.provider is required")
    if not getattr(llm_cfg, "model", None):
        raise ValueError("health_check: llm_cfg.model is required")

    agent_llm_cfg = getattr(memory, "agent_llm_cfg", None)
    if agent_llm_cfg is None:
        raise ValueError("health_check: memory.agent_llm_cfg is required")
    if not getattr(agent_llm_cfg, "provider", None):
        raise ValueError("health_check: agent_llm_cfg.provider is required")
    if not getattr(agent_llm_cfg, "model", None):
        raise ValueError("health_check: agent_llm_cfg.model is required")
    checks["vector:episodic"] = _check_vector(
        "vector:episodic", getattr(epi_store, "vector_index", None), dim
    )
    checks["vector:semantic"] = _check_vector(
        "vector:semantic", getattr(sem_store, "vector_index", None), dim
    )
    checks["vector:procedural"] = _check_vector(
        "vector:procedural", getattr(proc_store, "vector_index", None), dim
    )

    graph_core = getattr(memory, "graph_core", None)
    graph_adapter = getattr(graph_core, "adapter", None) if graph_core else None
    checks["graph"] = _check_graph("graph", graph_adapter)

    checks["llm"] = _check_llm(getattr(memory, "llm", None))
    checks["embedding"] = _check_embedder(getattr(memory, "embedder", None), dim)

    statuses = {check.status for check in checks.values()}
    if "error" in statuses:
        overall = "error"
    elif "skipped" in statuses:
        overall = "degraded"
    else:
        overall = "ok"

    return {
        "status": overall,
        "checks": {name: check.to_dict() for name, check in checks.items()},
    }
