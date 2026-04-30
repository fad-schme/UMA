"""
Provider registries for UMA's built-in LLMs and embedders.

This module centralizes factory lookup so UMA core never needs to branch
on specific provider names. Builtin adapters register themselves here via
their `from_config` helpers defined in their modules.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from uma.common.config_types import EmbeddingConfig, LLMConfig
from .base import EmbeddingInterface, LLMInterface

LLMFactory = Callable[[LLMConfig], LLMInterface]
EmbeddingFactory = Callable[[EmbeddingConfig], EmbeddingInterface]

_llm_factories: Dict[str, LLMFactory] = {}
_embedder_factories: Dict[str, EmbeddingFactory] = {}


def register_llm_provider(name: str, factory: LLMFactory) -> None:
    _llm_factories[name.lower()] = factory


def register_embedder_provider(name: str, factory: EmbeddingFactory) -> None:
    _embedder_factories[name.lower()] = factory


def get_llm_factory(name: str) -> Optional[LLMFactory]:
    return _llm_factories.get(name.lower())


def get_embedder_factory(name: str) -> Optional[EmbeddingFactory]:
    return _embedder_factories.get(name.lower())


# Register builtin providers automatically.
from .ollama_llm import OllamaLLM  # noqa: E402
from .ollama_embedding import OllamaEmbedder  # noqa: E402

register_llm_provider("ollama", OllamaLLM.from_config)
register_embedder_provider("ollama", OllamaEmbedder.from_config)
