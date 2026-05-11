"""
Provider initialization helpers split out of `uma_memory` for readability.
"""

from __future__ import annotations

import logging
from typing import Any

from uma.common.config_types import LLMConfig, EmbeddingConfig
from uma.adapters.llm.provider_registry import get_embedder_factory, get_llm_factory

logger = logging.getLogger(__name__)


def _unsupported_llm_provider_error(provider: str) -> ValueError:
    return ValueError(
        f"Unsupported provider '{provider}'. Supported providers: ollama, openai, anthropic."
    )


def _unsupported_embedding_provider_error(provider: str) -> ValueError:
    return ValueError(
        f"Unsupported embedding provider '{provider}'. Supported providers: ollama, openai."
    )


def initialize_llm(memory: Any) -> None:
    """
    Initialize UMA LLM and optional agent LLM from typed configs.
    """
    llm_cfg: LLMConfig = memory.llm_cfg

    llm_factory = get_llm_factory(llm_cfg.provider)
    if llm_factory:
        memory.llm = llm_factory(llm_cfg)
        logger.info(
            "Loaded %s LLM (model=%s)",
            llm_cfg.provider,
            getattr(memory.llm, "model", "unknown"),
        )
    else:
        raise _unsupported_llm_provider_error(llm_cfg.provider)

    # Agent LLM (optional; defaults to UMA LLM)
    agent_cfg = getattr(memory, "agent_llm_cfg", None)
    if agent_cfg and agent_cfg != llm_cfg:
        agent_factory = get_llm_factory(agent_cfg.provider)
        if agent_factory:
            memory.agent_llm = agent_factory(agent_cfg)
        else:
            raise _unsupported_llm_provider_error(agent_cfg.provider)
    if memory.agent_llm is None:
        memory.agent_llm = memory.llm

    logger.info("LLM initialization successful.")


def initialize_embedder(memory: Any) -> None:
    """
    Initialize the embedding model based on typed config.
    """
    embedding_cfg: EmbeddingConfig = memory.embedding_cfg
    if not isinstance(embedding_cfg.dimension, int) or embedding_cfg.dimension <= 0:
        raise ValueError("embedding.dimension must be a positive integer")
    embed_factory = get_embedder_factory(embedding_cfg.provider)
    if embed_factory:
        memory.embedder = embed_factory(embedding_cfg)
        logger.info(
            "Loaded %s embedder (model=%s, dimension=%s)",
            embedding_cfg.provider,
            getattr(memory.embedder, "model", embedding_cfg.model),
            getattr(memory.embedder, "dimension", embedding_cfg.dimension),
        )
    else:
        raise _unsupported_embedding_provider_error(embedding_cfg.provider)

    # Enforce the global invariant: every embedder must expose a valid dimension and match config.
    embedder_dim = getattr(memory.embedder, "dimension", None)
    if not isinstance(embedder_dim, int) or embedder_dim <= 0:
        raise ValueError(f"Embedder returned invalid dimension={embedder_dim!r}")
    if embedder_dim != embedding_cfg.dimension:
        raise ValueError(
            f"Embedder dimension mismatch: embedder={embedder_dim} config={embedding_cfg.dimension}"
        )

    # Best-effort preflight if the embedder supports it.
    try:
        preflight = getattr(memory.embedder, "preflight", None)
        if callable(preflight):
            preflight()
    except Exception:
        logger.exception("Embedder preflight failed at init; continuing.")

    logger.info("Embedder initialization successful.")


def ensure_llm(memory: Any) -> None:
    """
    Ensure UMA LLM is initialized.

    UMA is RLM-first: retrieval requires an LLM.
    Ingestion also requires an LLM.
    """
    if getattr(memory, "llm", None) is not None:
        return
    try:
        initialize_llm(memory)
        if getattr(memory, "llm", None) is None:
            raise RuntimeError("LLM initialization completed but memory.llm is still None")
    except Exception:
        logger.exception("LLM initialization failed.")
        raise


def ensure_embedder(memory: Any) -> None:
    """Ensure embedder exists (required for retrieval query embedding)."""
    if getattr(memory, "embedder", None) is None:
        initialize_embedder(memory)
