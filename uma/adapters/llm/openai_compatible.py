from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

from uma.common.config_types import EmbeddingConfig, LLMConfig

from .base import EmbeddingInterface, LLMInterface
from .retry_utils import retryable, should_retry_openai

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI  # type: ignore
except Exception as exc:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]
    logger.error("Failed to import openai: %s", exc)


def _normalize_base_url(*, provider_name: str, host: Optional[str], base_url: Optional[str]) -> str:
    if base_url and isinstance(base_url, str) and base_url.strip():
        return base_url.strip().rstrip("/")

    if provider_name == "ollama":
        raw_host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip()
        raw_host = raw_host.rstrip("/")
        return raw_host if raw_host.endswith("/v1") else f"{raw_host}/v1"

    return "https://api.openai.com/v1"


def _resolve_api_key(
    *,
    provider_name: str,
    api_key: Optional[str],
    api_key_env: Optional[str],
) -> str:
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()

    env_name = ""
    if isinstance(api_key_env, str) and api_key_env.strip():
        env_name = api_key_env.strip()
    elif provider_name == "openai":
        env_name = "OPENAI_API_KEY"

    if env_name:
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return env_value.strip()

    if provider_name == "ollama":
        return "ollama"

    raise RuntimeError(
        "OpenAI-compatible provider requires an API key. Set config.api_key or config.api_key_env."
    )


def _coerce_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


class OpenAICompatibleLLM(LLMInterface):
    """Shared OpenAI-compatible chat adapter used for public `ollama` and `openai` providers."""

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError(
                "OpenAI-compatible LLM adapter requires the 'openai' package to be installed."
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAICompatibleLLM: model must be a non-empty string.")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("OpenAICompatibleLLM: base_url must be a non-empty string.")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("OpenAICompatibleLLM: api_key must be a non-empty string.")
        if not isinstance(timeout, (int, float)) or float(timeout) <= 0:
            raise ValueError("OpenAICompatibleLLM: timeout must be > 0.")

        self.provider_name = provider_name
        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.timeout = float(timeout)
        self._client = AsyncOpenAI(
            api_key=api_key.strip(),
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @retryable(should_retry=should_retry_openai)
    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return _coerce_text_content(content).strip()

    @classmethod
    def from_config(
        cls,
        cfg: LLMConfig,
        *,
        provider_name: str,
    ) -> "OpenAICompatibleLLM":
        llm_kwargs = {**cfg.config}
        host = llm_kwargs.pop("host", None)
        base_url = llm_kwargs.pop("base_url", None)
        api_key = llm_kwargs.pop("api_key", None)
        api_key_env = llm_kwargs.pop("api_key_env", None)
        timeout = llm_kwargs.pop("timeout", 30.0)
        llm_kwargs.pop("model", None)
        llm_kwargs.pop("ollama_model", None)
        if llm_kwargs:
            logger.debug(
                "OpenAICompatibleLLM ignored extra config keys for provider=%s: %s",
                provider_name,
                sorted(llm_kwargs.keys()),
            )

        model = cfg.ollama_model or cfg.model or ("llama3" if provider_name == "ollama" else "gpt-4o-mini")
        return cls(
            provider_name=provider_name,
            model=model,
            base_url=_normalize_base_url(
                provider_name=provider_name,
                host=host,
                base_url=base_url,
            ),
            api_key=_resolve_api_key(
                provider_name=provider_name,
                api_key=api_key,
                api_key_env=api_key_env,
            ),
            timeout=float(timeout),
        )


class OpenAICompatibleEmbedder(EmbeddingInterface):
    """Shared OpenAI-compatible embedding adapter used for public `ollama` and `openai` providers."""

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        dimension: int,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        max_input_chars: int = 0,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError(
                "OpenAI-compatible embedder requires the 'openai' package to be installed."
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAICompatibleEmbedder: model must be a non-empty string.")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("OpenAICompatibleEmbedder: dimension must be a positive integer.")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("OpenAICompatibleEmbedder: base_url must be a non-empty string.")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("OpenAICompatibleEmbedder: api_key must be a non-empty string.")
        if not isinstance(timeout, (int, float)) or float(timeout) <= 0:
            raise ValueError("OpenAICompatibleEmbedder: timeout must be > 0.")

        self.provider_name = provider_name
        self.model = model.strip()
        self._dimension = dimension
        self.base_url = base_url.strip().rstrip("/")
        self.timeout = float(timeout)
        self.max_input_chars = max(0, int(max_input_chars or 0))
        self._client = AsyncOpenAI(
            api_key=api_key.strip(),
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @retryable(should_retry=should_retry_openai)
    async def embed(self, texts: Iterable[str]) -> list[list[float]]:
        if isinstance(texts, str):
            raise TypeError(
                "OpenAICompatibleEmbedder.embed expects Iterable[str], not a single string."
            )
        normalized = [str(text).strip() for text in list(texts) if str(text).strip()]
        if not normalized:
            return []

        response = await self._client.embeddings.create(
            model=self.model,
            input=normalized,
        )
        vectors: list[list[float]] = []
        for item in getattr(response, "data", []) or []:
            vectors.append([float(value) for value in getattr(item, "embedding", [])])
        return vectors

    @classmethod
    def from_config(
        cls,
        cfg: EmbeddingConfig,
        *,
        provider_name: str,
    ) -> "OpenAICompatibleEmbedder":
        embed_kwargs = {**cfg.config}
        host = embed_kwargs.pop("host", None)
        base_url = embed_kwargs.pop("base_url", None)
        api_key = embed_kwargs.pop("api_key", None)
        api_key_env = embed_kwargs.pop("api_key_env", None)
        timeout = embed_kwargs.pop("timeout", 30.0)
        max_input_chars = int(embed_kwargs.pop("max_input_chars", 0) or 0)
        embed_kwargs.pop("model", None)
        embed_kwargs.pop("dimension", None)
        embed_kwargs.pop("batch_size", None)
        if embed_kwargs:
            logger.debug(
                "OpenAICompatibleEmbedder ignored extra config keys for provider=%s: %s",
                provider_name,
                sorted(embed_kwargs.keys()),
            )

        default_model = "nomic-embed-text" if provider_name == "ollama" else "text-embedding-3-small"
        default_dimension = 768 if provider_name == "ollama" else 1536
        return cls(
            provider_name=provider_name,
            model=cfg.model or default_model,
            dimension=cfg.dimension or default_dimension,
            base_url=_normalize_base_url(
                provider_name=provider_name,
                host=host,
                base_url=base_url,
            ),
            api_key=_resolve_api_key(
                provider_name=provider_name,
                api_key=api_key,
                api_key_env=api_key_env,
            ),
            timeout=float(timeout),
            max_input_chars=max_input_chars,
        )
