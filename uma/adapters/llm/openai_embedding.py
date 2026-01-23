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
        Embed texts in batches using the OpenAI embeddings API.

        Contract
        --------
        - `texts` MUST be an iterable of strings.
        - Passing a single string is a programming error.

        Returns
        -------
        List[List[float]]
            One embedding vector per input text (order preserved).
        """
        # Guard against accidental str usage
        if isinstance(texts, str):
            raise TypeError(
                "OpenAIEmbedder.embed expects Iterable[str], not a single string."
            )

        texts_list = list(texts)
        if not texts_list:
            logger.debug("OpenAIEmbedder.embed called with empty iterable.")
            return []

        # Normalize inputs
        normalized: List[str] = []
        for t in texts_list:
            if not isinstance(t, str):
                raise TypeError(
                    f"OpenAIEmbedder.embed expects strings, got {type(t)}"
                )
            stripped = t.strip()
            if not stripped:
                logger.warning("OpenAIEmbedder.embed skipping empty/whitespace text.")
                continue
            normalized.append(stripped)

        if not normalized:
            logger.warning(
                "OpenAIEmbedder.embed: no valid texts after normalization."
            )
            return []

        vectors: List[List[float]] = []

        async def _embed_batch(batch: List[str]) -> List[List[float]]:
            """
            Embed a single batch of texts.

            Separated as a nested helper for clarity and reuse.
            """
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            return [item.embedding for item in response.data]

        # IMPORTANT: batch over *normalized*, not original texts
        for i in range(0, len(normalized), self.batch_size):
            batch = normalized[i : i + self.batch_size]
            try:
                chunk_vecs = await asyncio.wait_for(
                    _embed_batch(batch),
                    timeout=self.timeout,
                )
                for vec in chunk_vecs:
                    if len(vec) != self._dimension:
                        raise RuntimeError(
                            f"Embedding dimension mismatch: expected {self._dimension}, got {len(vec)}"
                        )
                vectors.extend(chunk_vecs)
            except Exception:
                logger.exception("OpenAIEmbedder: embedding batch failed.")
                raise

        return vectors