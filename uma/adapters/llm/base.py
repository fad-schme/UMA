"""
LLM and embedding interfaces for UMA.

This module defines the abstraction layer over any LLM / embeddings
provider (OpenAI, Azure, Anthropic, etc).

Coding agent instructions
-------------------------
- Implement provider-specific classes implementing these interfaces.
- Handle retries, timeouts, and logging in those concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class LLMInterface(ABC):
    """
    Abstract chat-style language model interface.

    Concrete implementations might call:
    - OpenAI ChatCompletions
    - Azure OpenAI
    - Local models, etc.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response for a chat-like prompt.

        Parameters
        ----------
        messages:
            List of {"role": "system"|"user"|"assistant", "content": str}.
        max_tokens:
            Maximum tokens in the response.
        temperature:
            Sampling temperature.

        Returns
        -------
        str
            Generated text.
        """
        raise NotImplementedError


class EmbeddingInterface(ABC):
    """
    Abstract text embedding interface.

    Concrete implementations might call:
    - OpenAI Embeddings
    - Local sentence transformers, etc.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""

    @abstractmethod
    async def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """
        Compute vector embeddings for a list of texts.

        Parameters
        ----------
        texts:
            Iterable of strings.

        Returns
        -------
        List[List[float]]
            One vector per input text.
        """
        raise NotImplementedError
