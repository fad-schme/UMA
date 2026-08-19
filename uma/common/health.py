"""
Health checks and dependency readiness probes for UMA.

This module provides lightweight checks for core dependencies such as
SQL backends, vector indices, and graph adapters. It avoids network-heavy
operations by default and reports a structured readiness summary.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any, Awaitable, Callable, Optional

from uma.adapters.db.base import DBAdapter
from uma.adapters.vector.base import VectorIndex
from uma.adapters.graph.base import GraphAdapter
from uma.adapters.llm.base import LLMInterface, EmbeddingInterface
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from uma.common.results import HealthStatus
logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """One dependency probe.

    `status` is one of:

    - ``ok``       — probe succeeded
    - ``error``    — probe failed, or a required dependency is missing
    - ``disabled`` — optional subsystem intentionally switched off in config;
                     neutral, never degrades the overall status
    - ``degraded`` — reachable but partially functional
    """

    name: str
    status: str
    detail: str
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timed(detail_fn):
    start = time.monotonic()
    status, detail = detail_fn()
    latency = (time.monotonic() - start) * 1000.0
    return status, detail, latency


def _check_db(name: str, adapter: Optional[DBAdapter]) -> HealthCheck:
    if adapter is None:
        # Lite always provisions a SQL adapter per lane. A missing one is a
        # broken runtime, not an unconfigured optional subsystem.
        return HealthCheck(name=name, status="error", detail="adapter missing")

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
            logger.debug("_check_sql: connection check failed: %s", exc, exc_info=True)
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
        # Same reasoning as _check_db: the vector index is always provisioned
        # in lite, so its absence is a failure rather than a skipped option.
        return HealthCheck(name=name, status="error", detail="index missing")

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
            # C1: vector index queries require explicit isolation. This
            # probe uses sentinel values that will not match any real
            # row — we only care that the query call goes through
            # without raising, not what it returns.
            index.query(
                [0.0] * dim,
                tenant_id="__health__",
                owner_type="system",
                owner_id="__health__",
                k=1,
            )
            return "ok", "query ok"
        except Exception as exc:
            logger.debug("_check_vector: query check failed: %s", exc, exc_info=True)
            return "error", f"query failed: {exc}"

    status, detail, latency = _timed(_probe)
    return HealthCheck(name=name, status=status, detail=detail, latency_ms=latency)


def _check_graph(
    name: str,
    adapter: Optional[GraphAdapter],
    *,
    configured: bool,
) -> HealthCheck:
    if adapter is None:
        if not configured:
            # Graph is opt-in and user-supplied: a graph DB cannot be embedded
            # the way SQLite and LanceDB are. `graph_backend: disabled` is the
            # intended lite posture, so it must not degrade overall health.
            return HealthCheck(name=name, status="disabled", detail="graph disabled")
        # A backend was configured but no adapter reached the runtime.
        return HealthCheck(
            name=name,
            status="error",
            detail="graph backend configured but adapter is unavailable",
        )

    def _probe():
        verify = getattr(adapter, "verify_connectivity", None)
        if callable(verify):
            return ("ok", "connectivity ok") if verify() else ("error", "connectivity failed")
        try:
            adapter.run_query("RETURN 1 AS ok")
            return "ok", "query ok"
        except Exception as exc:
            logger.debug("_check_graph: query check failed: %s", exc, exc_info=True)
            return "error", f"query failed: {exc}"

    status, detail, latency = _timed(_probe)
    return HealthCheck(name=name, status=status, detail=detail, latency_ms=latency)


_DEFAULT_PROBE_TIMEOUT = 30.0


def _probe_timeout(cfg: Any) -> float:
    """Probe deadline for a provider: its own configured request timeout."""
    raw = (getattr(cfg, "config", None) or {}).get("timeout")
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_PROBE_TIMEOUT
    return timeout if timeout > 0 else _DEFAULT_PROBE_TIMEOUT


def _run_probe(make_coro: Callable[[], Awaitable[Any]], timeout: float) -> Any:
    """Run one async provider probe from sync health-check code.

    `health_check()` is sync but the provider interfaces are async-only, so
    the probe needs its own loop. When the caller already has a running loop
    (health checked from async application code) we cannot nest `asyncio.run`,
    so the probe goes to a worker thread with a loop of its own.
    """

    async def _runner() -> Any:
        return await asyncio.wait_for(make_coro(), timeout=timeout)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_runner())

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _runner()).result()


def _provider_label(cfg: Any, fallback: str) -> str:
    provider = getattr(cfg, "provider", None) or "unknown"
    model = getattr(cfg, "model", None) or "unknown"
    host = (getattr(cfg, "config", None) or {}).get("host")
    label = f"{provider}:{model}"
    return f"{label} @ {host}" if host else f"{label} ({fallback})"


def _check_llm(llm: Optional[LLMInterface], cfg: Any) -> HealthCheck:
    """Confirm the UMA LLM is reachable.

    UMA is RLM-first: both retrieval and ingestion require an LLM, so an
    unreachable provider is a hard failure, not a warning. The probe is the
    smallest possible generation (1 token).
    """
    if llm is None:
        return HealthCheck(name="llm", status="error", detail="not initialized")

    label = _provider_label(cfg, llm.__class__.__name__)
    timeout = _probe_timeout(cfg)

    def _probe() -> tuple[str, str]:
        try:
            _run_probe(
                lambda: llm.generate(
                    [{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0.0,
                ),
                timeout,
            )
            return "ok", f"{label} reachable"
        except asyncio.TimeoutError:
            logger.error(
                "health: llm probe timed out provider=%s timeout=%.1fs",
                label,
                timeout,
            )
            return "error", f"{label} did not respond within {timeout:.0f}s"
        except Exception as exc:
            logger.error("health: llm probe failed provider=%s error=%s", label, exc)
            return "error", f"{label} unreachable: {exc}"

    status, detail, latency = _timed(_probe)
    return HealthCheck(name="llm", status=status, detail=detail, latency_ms=latency)


def _check_embedder(
    embedder: Optional[EmbeddingInterface],
    dim: int,
    cfg: Any,
) -> HealthCheck:
    """Confirm the embedding provider is reachable and returns the right dim."""
    if embedder is None:
        return HealthCheck(name="embedding", status="error", detail="not initialized")
    if embedder.dimension != dim:
        return HealthCheck(
            name="embedding",
            status="error",
            detail=f"dimension mismatch: embedder={embedder.dimension} config={dim}",
        )

    label = _provider_label(cfg, embedder.__class__.__name__)
    timeout = _probe_timeout(cfg)

    def _probe() -> tuple[str, str]:
        try:
            vectors = _run_probe(lambda: embedder.embed(["health"]), timeout)
        except asyncio.TimeoutError:
            logger.error(
                "health: embedding probe timed out provider=%s timeout=%.1fs",
                label,
                timeout,
            )
            return "error", f"{label} did not respond within {timeout:.0f}s"
        except Exception as exc:
            logger.error(
                "health: embedding probe failed provider=%s error=%s", label, exc
            )
            return "error", f"{label} unreachable: {exc}"

        if not vectors or not isinstance(vectors, list) or not vectors[0]:
            return "error", f"{label} returned an empty embedding"
        returned = len(vectors[0])
        if returned != dim:
            return (
                "error",
                f"{label} returned dim={returned}, config expects {dim}",
            )
        return "ok", f"{label} reachable"

    status, detail, latency = _timed(_probe)
    return HealthCheck(name="embedding", status=status, detail=detail, latency_ms=latency)


def run_health_checks(memory: Any) -> "HealthStatus":
    """
    Run basic readiness checks for UMA dependencies.

    Returns a `HealthStatus` with an overall status literal and a per-check
    map keyed by check name. Overall status is ``error`` if any check failed,
    ``degraded`` if any check is partially functional, otherwise ``ok`` —
    disabled optional subsystems are neutral.
    """
    from uma.common.results import HealthStatus

    checks: dict[str, HealthCheck] = {}

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
    # `_init_graph_core` sets graph_core to None only when the backend is
    # "disabled" and raises on every initialization failure, so the config
    # value is an exact statement of operator intent here.
    graph_backend = getattr(getattr(memory, "raw_config", None), "storage", None)
    graph_backend = getattr(graph_backend, "graph_backend", "disabled")
    checks["graph"] = _check_graph(
        "graph",
        graph_adapter,
        configured=str(graph_backend) != "disabled",
    )

    checks["llm"] = _check_llm(getattr(memory, "llm", None), llm_cfg)
    checks["embedding"] = _check_embedder(
        getattr(memory, "embedder", None), dim, embedding_cfg
    )

    # Only a failing check degrades overall health. An intentionally disabled
    # optional subsystem reports "disabled" and is neutral — a correct lite
    # install with graph off must report "ok" and exit 0, because operators
    # wire `uma health` into container healthchecks and CI gates.
    statuses = {check.status for check in checks.values()}
    if "error" in statuses:
        overall = "error"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "ok"

    return HealthStatus(status=overall, checks=checks)
