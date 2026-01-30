"""
Template LLM and embedder adapters.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from uma.adapters.llm.base import LLMInterface, EmbeddingInterface


class ExampleLLM(LLMInterface):
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError


class ExampleEmbedder(EmbeddingInterface):
    def __init__(self, dimension: int, **kwargs: Any) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Iterable[str]) -> List[List[float]]:
        raise NotImplementedError
