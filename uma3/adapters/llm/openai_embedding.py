"""
OpenAI Embedding adapter (production-grade).

Implements:
- EmbeddingInterface
- Automatic batching
- Retries on transient errors

Coding agent instructions:
--------------------------
- Use model "text-embedding-3-small" or "text-embedding-3-large".
- Configure batch_size according to your rate limits.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Iterable, List

from openai import AsyncOpenAI
from ...adapters.llm.base import EmbeddingInterface
from .retry_utils import retryable

logger = logging.getLogger(__name__)


class OpenAIEmbedder(EmbeddingInterface):

    def __init__(
        self,
        model: str,
        dimension: int,
        batch_size: int = 32,
        timeout: float = 20.0,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found.")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self._dimension = dimension
        logger.info("OpenAIEmbedder initialized with model=%s", model)

    @property
    def dimension(self) -> int:
        return self._dimension

    @retryable()
    async def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """
        Embed texts in batches.

        Returns
        -------
        List[List[float]]
        """
        texts = list(texts)
        vectors: List[List[float]] = []

        async def _embed_batch(batch):
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            return [item.embedding for item in response.data]

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                chunk_vecs = await asyncio.wait_for(
                    _embed_batch(batch), timeout=self.timeout
                )
                vectors.extend(chunk_vecs)
            except Exception:
                logger.exception("Embedding batch failed.")
                raise

        return vectors