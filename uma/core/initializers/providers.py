"""
Provider initialization helpers split out of `uma_memory` for readability.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from ..utils.config_types import LLMConfig, EmbeddingConfig, parse_plugin_spec
from ...adapters.llm.base import EmbeddingInterface, LLMInterface
from ...adapters.llm.callable_adapter import CallableEmbedderAdapter, CallableLLMAdapter
from ...adapters.llm.provider_registry import get_embedder_factory, get_llm_factory

logger = logging.getLogger(__name__)


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
        llm_cls = parse_plugin_spec(llm_cfg.provider)
        llm_kwargs = {**llm_cfg.config}
        if llm_cfg.model and "model" not in llm_kwargs:
            llm_kwargs["model"] = llm_cfg.model
        if isinstance(llm_cls, LLMInterface):
            memory.llm = llm_cls
        elif inspect.isclass(llm_cls):
            memory.llm = llm_cls(**llm_kwargs)
        elif callable(llm_cls):
            memory.llm = CallableLLMAdapter(
                callable_fn=llm_cls,
                name=llm_cfg.provider,
                preflight=bool(llm_kwargs.pop("preflight", True)),
                default_kwargs=llm_kwargs,
            )
        else:
            raise TypeError(f"Unsupported LLM provider type: {type(llm_cls)}")
        logger.info("Loaded custom LLM adapter (%s)", llm_cfg.provider)

    # Agent LLM (optional; defaults to UMA LLM)
    agent_cfg = getattr(memory, "agent_llm_cfg", None)
    if agent_cfg and agent_cfg != llm_cfg:
        agent_factory = get_llm_factory(agent_cfg.provider)
        if agent_factory:
            memory.agent_llm = agent_factory(agent_cfg)
        else:
            agent_cls = parse_plugin_spec(agent_cfg.provider)
            agent_kwargs = {**agent_cfg.config}
            if agent_cfg.model and "model" not in agent_kwargs:
                agent_kwargs["model"] = agent_cfg.model
            if isinstance(agent_cls, LLMInterface):
                memory.agent_llm = agent_cls
            elif inspect.isclass(agent_cls):
                memory.agent_llm = agent_cls(**agent_kwargs)
            elif callable(agent_cls):
                memory.agent_llm = CallableLLMAdapter(
                    callable_fn=agent_cls,
                    name=agent_cfg.provider,
                    preflight=bool(agent_kwargs.pop("preflight", True)),
                    default_kwargs=agent_kwargs,
                )
            else:
                raise TypeError(f"Unsupported agent LLM provider type: {type(agent_cls)}")
    if memory.agent_llm is None:
        memory.agent_llm = memory.llm

    logger.info("LLM initialization successful.")


def initialize_embedder(memory: Any) -> None:
    """
    Initialize the embedding model based on typed config.
    """
    embedding_cfg: EmbeddingConfig = memory.embedding_cfg
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
        embed_cls = parse_plugin_spec(embedding_cfg.provider)
        embed_kwargs = {**embedding_cfg.config}
        if embedding_cfg.model and "model" not in embed_kwargs:
            embed_kwargs["model"] = embedding_cfg.model
        if "dimension" not in embed_kwargs:
            embed_kwargs["dimension"] = embedding_cfg.dimension
        if isinstance(embed_cls, EmbeddingInterface):
            memory.embedder = embed_cls
        elif inspect.isclass(embed_cls):
            memory.embedder = embed_cls(**embed_kwargs)
        elif callable(embed_cls):
            memory.embedder = CallableEmbedderAdapter(
                callable_fn=embed_cls,
                dimension=embedding_cfg.dimension,
                name=embedding_cfg.provider,
                preflight=bool(embed_kwargs.pop("preflight", True)),
                default_kwargs=embed_kwargs,
            )
        else:
            raise TypeError(f"Unsupported embedder provider type: {type(embed_cls)}")
        logger.info("Loaded custom embedder adapter (%s)", embedding_cfg.provider)

    # Best-effort preflight if the embedder supports it.
    try:
        preflight = getattr(memory.embedder, "preflight", None)
        if callable(preflight):
            preflight()
    except Exception:
        logger.exception("Embedder preflight failed at init; continuing.")

    logger.info("Embedder initialization successful.")
