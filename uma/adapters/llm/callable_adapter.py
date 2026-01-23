"""
Callable adapters for plugging in custom LLM/embedding callables.

These adapters wrap any callable into the UMA LLMInterface/EmbeddingInterface
while performing lightweight preflight validation to reduce runtime errors.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

from .base import EmbeddingInterface, LLMInterface

logger = logging.getLogger(__name__)


def _supports_param(sig: inspect.Signature, name: str) -> bool:
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if p.name == name:
            return True
    return False


class CallableLLMAdapter(LLMInterface):
    """
    Wrap an arbitrary callable into the LLMInterface.

    The callable is expected to accept `messages` and return a string (or
    an awaitable that resolves to a string).
    """

    def __init__(
        self,
        callable_fn: Callable[..., Any],
        name: Optional[str] = None,
        preflight: bool = True,
        default_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not callable(callable_fn):
            raise TypeError("callable_fn must be callable")
        self._callable = callable_fn
        self._name = name or getattr(callable_fn, "__name__", "callable_llm")
        self._default_kwargs = default_kwargs or {}
        sig = inspect.signature(callable_fn)
        self._accepts_messages = _supports_param(sig, "messages")
        self._accepts_max_tokens = _supports_param(sig, "max_tokens")
        self._accepts_temperature = _supports_param(sig, "temperature")
        self._accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if preflight and not self._accepts_messages:
            raise TypeError(
                f"{self._name} must accept a 'messages' argument or **kwargs"
            )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        try:
            call_kwargs = {}
            if self._accepts_messages or self._accepts_kwargs:
                call_kwargs["messages"] = messages
            if self._accepts_max_tokens or self._accepts_kwargs:
                call_kwargs["max_tokens"] = max_tokens
            if self._accepts_temperature or self._accepts_kwargs:
                call_kwargs["temperature"] = temperature
            call_kwargs.update(self._default_kwargs)
            call_kwargs.update(kwargs)
            result = self._callable(**call_kwargs)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, str):
                raise TypeError(
                    f"{self._name} returned non-string result: {type(result)}"
                )
            return result
        except Exception as exc:
            logger.exception("CallableLLMAdapter(%s) failed.", self._name)
            raise RuntimeError(f"Callable LLM failed: {self._name}") from exc


class CallableEmbedderAdapter(EmbeddingInterface):
    """
    Wrap an arbitrary callable into the EmbeddingInterface.

    The callable is expected to accept `texts` and return a List[List[float]]
    (or an awaitable that resolves to that).
    """

    def __init__(
        self,
        callable_fn: Callable[..., Any],
        dimension: int,
        name: Optional[str] = None,
        preflight: bool = True,
        default_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not callable(callable_fn):
            raise TypeError("callable_fn must be callable")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self._callable = callable_fn
        self._dimension = dimension
        self._name = name or getattr(callable_fn, "__name__", "callable_embedder")
        self._default_kwargs = default_kwargs or {}
        sig = inspect.signature(callable_fn)
        self._accepts_texts = _supports_param(sig, "texts")
        self._accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if preflight and not self._accepts_texts:
            raise TypeError(
                f"{self._name} must accept a 'texts' argument or **kwargs"
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Iterable[str]) -> List[List[float]]:
        try:
            if isinstance(texts, str):
                raise TypeError(
                    "CallableEmbedderAdapter.embed expects Iterable[str], not a single string."
                )

            texts = list(texts)
            if not texts:
                return []

            normalized = []
            for t in texts:
                if not isinstance(t, str):
                    raise TypeError(
                        f"CallableEmbedderAdapter.embed expects strings, got {type(t)}"
                    )
                stripped = t.strip()
                if not stripped:
                    logger.warning(
                        "CallableEmbedderAdapter.embed skipping empty/whitespace text."
                    )
                    continue
                normalized.append(stripped)

            if not normalized:
                return []
    
            call_kwargs = {}
            if self._accepts_texts or self._accepts_kwargs:
                call_kwargs["texts"] = list(normalized)
            call_kwargs.update(self._default_kwargs)
            result = self._callable(**call_kwargs)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, list):
                raise TypeError(
                    f"{self._name} returned non-list result: {type(result)}"
                )
            for vec in result:
                if not isinstance(vec, list):
                    raise TypeError(
                        f"{self._name} returned invalid vector type: {type(vec)}"
                    )
                if len(vec) != self._dimension:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: expected {self._dimension}, got {len(vec)}"
                    )
            return result
        except Exception as exc:
            logger.exception("CallableEmbedderAdapter(%s) failed.", self._name)
            raise RuntimeError(f"Callable embedder failed: {self._name}") from exc
