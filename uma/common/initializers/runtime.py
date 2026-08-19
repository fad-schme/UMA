"""
initializers/runtime.py
=======================

Orchestration helpers for UMAMemory initialization.

Goals:
- UMAConfig.load_yaml() calls init_runtime_env(cfg) (lightweight only).
- UMAMemory.from_yaml() makes UMA retrieval-ready synchronously (predictable cost).
- Ingestion-heavy subsystems and optional memory features are warmed up in the background.
- Retrieval remains functional even if warmup fails.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Callable, Iterable

from .providers import ensure_embedder, ensure_llm
from .stores import initialize_stores

if TYPE_CHECKING:
    from ..uma_memory import UMAMemory

logger = logging.getLogger(__name__)

_ADAPTER_ROOTS_ENV = "UMA_ADAPTER_ROOTS"


def _run_singleflight_initializer(
    memory: "UMAMemory",
    *,
    key: str,
    ready_attr: str,
    initializer: Callable[[], None],
) -> None:
    """
    Run one initializer at a time per readiness key.

    The coordination lock only protects state transitions. The actual
    initialization work runs outside the lock so DB/provider/model setup is
    not globally serialized.
    """
    with memory._init_condition:
        if getattr(memory, ready_attr, False):
            return
        if key in memory._init_inflight:
            while key in memory._init_inflight and not getattr(memory, ready_attr, False):
                memory._init_condition.wait()
            if getattr(memory, ready_attr, False):
                return
            raise RuntimeError(f"UMA initialization failed for {key}.")
        memory._init_inflight.add(key)

    try:
        initializer()
    finally:
        with memory._init_condition:
            memory._init_inflight.discard(key)
            memory._init_condition.notify_all()


# ---------------------------------------------------------------------
# Runtime environment (called during config load)
# ---------------------------------------------------------------------
def _iter_adapter_roots(cfg: object) -> Iterable[str]:
    """
    Yield adapter roots that should be made importable for plugin specs.

    Order matters:
      1. Explicit external adapter roots from the environment.
    """
    seen: set[str] = set()

    env_roots = os.environ.get(_ADAPTER_ROOTS_ENV, "")
    for raw_path in env_roots.split(os.pathsep):
        path = raw_path.strip()
        if not path:
            continue
        normalized = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        if normalized in seen:
            continue
        seen.add(normalized)
        yield normalized


def init_runtime_env(cfg: object) -> None:
    """
    Initialize UMA runtime environment as early as config load time.

    Intentionally lightweight:
      - register explicit external adapter roots on sys.path
      - DO NOT initialize providers, DBs, vector indexes, or other heavy services
    """
    try:
        valid_roots: list[str] = []
        for path in _iter_adapter_roots(cfg):
            if not os.path.isdir(path):
                continue
            valid_roots.append(path)

        for path in reversed(valid_roots):
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
            logger.info("Registered adapter root on sys.path: %s", path)

        source_dir = getattr(cfg, "_source_dir", None)
        if valid_roots and source_dir:
            os.environ.setdefault("UMA_CONFIG_DIR", str(source_dir))
    except Exception:
        logger.exception("Failed to initialize UMA runtime environment (non-fatal).")


# ---------------------------------------------------------------------
# Stores / cores / features / pipeline (orchestration-level ensures)
# ---------------------------------------------------------------------
def ensure_stores(memory: "UMAMemory") -> None:
    if not memory._stores:
        memory._stores = initialize_stores(memory)


def ensure_cores(memory: "UMAMemory") -> None:
    # Heavy: initializes WM + episodic/semantic/procedural/chunk cores (ingestion path).
    if (
        memory.working_memory is None
        or memory.episodic_core is None
        or memory.semantic_core is None
        or memory.procedural_core is None
        or memory.chunk_core is None
    ):
        memory._init_core_subsystems()


def ensure_graph(memory: "UMAMemory") -> None:
    if memory.graph_core is None and memory.cfg.storage.graph_backend != "disabled":
        memory._init_graph_core()


def ensure_features(memory: "UMAMemory") -> None:
    if not getattr(memory, "_features_initialized", False):
        memory._init_optional_features()
        memory._features_initialized = True


def ensure_pipeline(memory: "UMAMemory") -> None:
    if getattr(memory, "pipeline", None) is None:
        from uma.ingest.pipeline import MemoryPipeline

        memory.pipeline = MemoryPipeline(
            memory_client=memory,
            hooks=memory.hooks,
        )


def ensure_rlm(memory: "UMAMemory") -> None:
    """
    Wire UMA controller (the only supported retrieval mode).
    """
    rlm_cfg = memory.retrieval_cfg.rlm
    if rlm_cfg is None or not rlm_cfg.enabled:
        raise RuntimeError("RLM retrieval is required (retrieval.rlm.enabled must be true).")
    if getattr(memory, "_rlm_controller", None) is not None:
        return
    if memory.llm is None:
        raise RuntimeError("RLM retrieval requires an LLM (memory.llm is None).")

    try:
        from uma.retrieve.rlm.environment import UMAMemoryEnvironment
        from uma.retrieve.rlm.controller import RLMController

        memory.memory_env = UMAMemoryEnvironment(memory)
        memory._rlm_controller = RLMController(
            llm=memory.llm,
            env=memory.memory_env,
        )
        logger.info("RLMController enabled and wired.")
    except Exception:
        logger.exception("Failed to initialize RLMController.")
        raise


def init_base_ready(memory: "UMAMemory") -> None:
    """
    Initialize shared retrieval/ingestion base components once.

    This coordinates the common cold-start path across retrieval and ingestion so
    stores/providers/cores cannot duplicate when both paths overlap.
    """

    def _initialize() -> None:
        ensure_stores(memory)
        ensure_llm(memory)
        ensure_embedder(memory)
        ensure_cores(memory)
        memory._base_ready = True

    _run_singleflight_initializer(
        memory,
        key="base",
        ready_attr="_base_ready",
        initializer=_initialize,
    )


# ---------------------------------------------------------------------
# Startup: predictable retrieval-ready init
# ---------------------------------------------------------------------
def init_retrieval_ready(memory: "UMAMemory") -> None:
    """
    Make UMA retrieval-ready synchronously (predictable startup cost).

    Guarantees:
      - stores wired (SQL + vector)
      - LLM initialized
      - embedder initialized
      - retrieval cores initialized (episodic/semantic/procedural/chunk + WM)
      - graph initialized if enabled (best-effort)
      - RLM wired (required)

    MUST NOT initialize ingestion pipeline/features.
    """
    def _initialize() -> None:
        init_base_ready(memory)

        # Graph is optional; never fail retrieval startup because of graph.
        try:
            ensure_graph(memory)
        except Exception:
            logger.exception("Graph init failed during retrieval startup (non-fatal).")
            memory.graph_core = None

        ensure_rlm(memory)
        memory._retrieval_ready = True

    _run_singleflight_initializer(
        memory,
        key="retrieval",
        ready_attr="_retrieval_ready",
        initializer=_initialize,
    )


# ---------------------------------------------------------------------
# Heavy init (ingestion-ready) used by warmup and by ingestion calls
# ---------------------------------------------------------------------
def init_ingestion_ready(memory: "UMAMemory") -> None:
    """
    Initialize ingestion-heavy subsystems:
      - stores (if missing)
      - LLM (required)
      - embedder
      - cores
      - optional features
      - pipeline

    Safe to call multiple times.
    """
    def _initialize() -> None:
        init_base_ready(memory)

        # features optional; should not abort readiness
        try:
            ensure_features(memory)
        except Exception:
            logger.exception("Feature init failed (non-fatal).")

        ensure_pipeline(memory)
        memory._ingestion_ready = True

    _run_singleflight_initializer(
        memory,
        key="ingestion",
        ready_attr="_ingestion_ready",
        initializer=_initialize,
    )


# ---------------------------------------------------------------------
# Background warmup scheduling (no developer involvement)
# ---------------------------------------------------------------------
def schedule_ingestion_warmup(memory: "UMAMemory") -> None:
    """
    Best-effort background warmup for ingestion-heavy subsystems.

    - Does not block retrieval-ready startup.
    - If warmup fails, ingestion APIs will still self-heal by calling init_ingestion_ready.
    """
    with memory._lifecycle_lock:
        if getattr(memory, "_warmup_scheduled", False):
            return
        memory._warmup_scheduled = True

    async def _warmup_async() -> None:
        try:
            if getattr(memory, "_ingestion_ready", False):
                return
            init_ingestion_ready(memory)
            logger.info("UMA ingestion warmup completed.")
        except Exception:
            with memory._lifecycle_lock:
                if not getattr(memory, "_ingestion_ready", False):
                    memory._warmup_scheduled = False
            logger.exception("UMA ingestion warmup failed (non-fatal).")

    # Prefer scheduling on a running loop if present.
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_warmup_async())
        return
    except RuntimeError:  # nosec B110 — expected: no running event loop; fall through to thread path
        pass
    except Exception:
        logger.exception("Failed scheduling warmup on loop; falling back to thread.")

    # Thread fallback.
    def _warmup_thread() -> None:
        try:
            if getattr(memory, "_ingestion_ready", False):
                return
            init_ingestion_ready(memory)
            logger.info("UMA ingestion warmup completed (thread).")
        except Exception:
            with memory._lifecycle_lock:
                if not getattr(memory, "_ingestion_ready", False):
                    memory._warmup_scheduled = False
            logger.exception("UMA ingestion warmup failed (thread, non-fatal).")

    t = threading.Thread(target=_warmup_thread, name="uma-ingestion-warmup", daemon=True)
    t.start()
