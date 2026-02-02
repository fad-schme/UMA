"""
initializers/runtime.py
=======================

Lazy initialization helpers for UMAMemory.

These functions keep UMAMemory lean by consolidating service wiring
and capability initialization in one place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .providers import initialize_embedder, initialize_llm
from .stores import initialize_stores
from ..retrieval.service import RetrievalService

if TYPE_CHECKING:
    from ..uma_memory import UMAMemory

logger = logging.getLogger(__name__)


def ensure_llm(memory: "UMAMemory") -> None:
    if memory.llm is None:
        initialize_llm(memory)


def ensure_embedder(memory: "UMAMemory") -> None:
    if memory.embedder is None:
        initialize_embedder(memory)


def ensure_stores(memory: "UMAMemory") -> None:
    if not memory._stores:
        memory._stores = initialize_stores(memory)


def ensure_cores(memory: "UMAMemory") -> None:
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
        from ..utils.pipeline import MemoryPipeline

        memory.pipeline = MemoryPipeline(
            memory_client=memory,
            hooks=memory.hooks,
            promotion_policy=memory.promotion_policy,
        )


def ensure_retrieval(memory: "UMAMemory") -> None:
    if memory.retrieval_service is None:
        memory.retrieval_service = RetrievalService(
            memory=memory,
            retr_cfg=memory.retrieval_cfg,
        )


def ensure_rlm(memory: "UMAMemory") -> None:
    rlm_cfg = memory.retrieval_cfg.rlm
    if rlm_cfg is None or not rlm_cfg.enabled:
        return
    if memory.rlm_controller is not None:
        return
    try:
        from ..retrieval.rlm.environment import UMAMemoryEnvironment
        from ..retrieval.rlm.controller import RLMController

        memory.memory_env = UMAMemoryEnvironment(memory)
        memory.rlm_controller = RLMController(
            llm=memory.llm,
            env=memory.memory_env,
        )
        logger.info("RLMController enabled and wired.")
    except Exception:
        logger.exception(
            "Failed to initialize RLMController; falling back to classic retrieval."
        )
        memory.rlm_controller = None
