"""
OllamaEmbedder — UMA EmbeddingInterface implementation.

This version uses the official `ollama` Python package instead of HTTP calls.

IMPORTANT:
----------
- Ollama must be installed locally: `pip install ollama`.
- The Ollama daemon must be running on the host.
- The model specified MUST support the `.embed()` API.
- This adapter supports only two modes:
    - "native"   → call `ollama.embed()` for real embeddings
    - "disabled" → raise error on use (safe default)

Rationale:
----------
The Python API is more robust, faster, and officially recommended by Ollama.
This avoids custom HTTP endpoints entirely.

Coding agent instructions:
--------------------------
- Do NOT fabricate embeddings.
- Ensure the model exists locally via `ollama pull <model>`.
- If the model does not support embeddings, a RuntimeError will be raised.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List, Optional

from ...adapters.llm.base import EmbeddingInterface
from .retry_utils import retryable

logger = logging.getLogger(__name__)

# Try importing ollama; if missing, we raise at runtime.
try:
    import ollama
except Exception:
    ollama = None


class OllamaEmbedder(EmbeddingInterface):
    """
    UMA embedding adapter for Ollama.

    Parameters
    ----------
    model : str
        The Ollama model name to use for embedding.
    mode : str
        "native" or "disabled".
    base_url, endpoint, timeout : ignored
        Present for API compatibility only.
    """

    def __init__(
        self,
        model: str,
        dimension: int,
        timeout: float = 30.0,
        mode: str = "disabled",
    ) -> None:
        if mode not in {"native", "disabled"}:
            raise ValueError(
                f"Unsupported OllamaEmbedder mode '{mode}'. Allowed: 'native', 'disabled'."
            )

        self.model = model
        self.mode = mode
        self._dimension = dimension
        self.timeout = timeout

        logger.info("OllamaEmbedder initialized (mode=%s, model=%s)", mode, model)
        
    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """
        Embed text using Ollama's Python API.

        Parameters
        ----------
        texts : Iterable[str] or str
            Input texts.

        Returns
        -------
        List[List[float]]
        """

        # Accept single string
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        if not texts_list:
            logger.debug("OllamaEmbedder.embed called with empty list.")
            return []

        if self.mode == "disabled":
            logger.error("Embedding called while mode='disabled'.")
            raise RuntimeError(
                "Ollama embedding is disabled. Enable mode='native' to use it."
            )

        if self.mode == "native":
            return await self._embed_native(texts_list)

        logger.critical("Unexpected OllamaEmbedder mode: %s", self.mode)
        raise RuntimeError(f"Invalid mode: {self.mode}")

    @retryable()
    async def _embed_native(self, texts: List[str]) -> List[List[float]]:
        """
        Perform embedding using the official Ollama Python API.
        """

        if ollama is None:
            logger.error("Ollama Python package not installed.")
            raise RuntimeError(
                "The 'ollama' Python package is not installed. Install with: pip install ollama"
            )

        try:
            # ollama.embed returns a dict with an "embeddings" field containing a list of vectors.
            # For multiple inputs, the recommended call is one-at-a-time OR if supported:
            #   ollama.embed(model=..., input=[...])
            # The Python client supports list input.
            response = await asyncio.to_thread(ollama.embed, model=self.model, input=texts)
        except Exception as exc:
            logger.exception("Ollama embedding call failed: %s", exc)
            raise RuntimeError("Ollama embedding request failed.") from exc

        # Expected response format:
        #   {"model": ..., "embeddings": [[...], [...]] }
        try:
            vectors = response.get("embeddings") or response.get("data")
        except Exception:
            logger.exception("Invalid embedding response structure: %r", response)
            raise RuntimeError("Malformed embedding response from Ollama.")

        if not isinstance(vectors, list):
            logger.error("Embedding response missing 'embeddings' list: %r", response)
            raise RuntimeError("Ollama embedding response malformed.")

        if len(vectors) != len(texts):
            logger.error(
                "Ollama returned mismatched embedding count: expected %d, got %d",
                len(texts), len(vectors)
            )
            raise RuntimeError("Embedding count mismatch from Ollama.")

        for vec in vectors:
            if len(vec) != self._dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {self._dimension}, got {len(vec)}"
                )

        logger.debug("Received %d embeddings from Ollama.", len(vectors))
        return vectors
