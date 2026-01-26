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

Embedding Contract (UMA3-RLM v1)
--------------------------------
    async def embed(texts: Iterable[str]) -> List[List[float]]

- `texts` MUST be an iterable of strings
- Passing a single string is a programming error
- Output ordering MUST match input ordering

Coding agent instructions:
--------------------------
- Do NOT fabricate embeddings.
- Ensure the model exists locally via `ollama pull <model>`.
- If the model does not support embeddings, a RuntimeError will be raised.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List

from ...core.utils.config_types import EmbeddingConfig
from .base import EmbeddingInterface
from .retry_utils import retryable

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency if unused
try:
    import ollama  # type: ignore
except Exception:
    ollama = None


class OllamaEmbedder(EmbeddingInterface):
    """
    UMA embedding adapter for Ollama.

    Parameters
    ----------
    model : str
        Ollama model name (e.g. "nomic-embed-text")
    dimension : int
        Expected embedding dimensionality
    timeout : float
        Max seconds per embedding call
    mode : str
        "native" or "disabled"
    """

    def __init__(
        self,
        model: str,
        dimension: int,
        timeout: float = 30.0,
        mode: str = "disabled",
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OllamaEmbedder: model must be a non-empty string.")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("OllamaEmbedder: dimension must be a positive integer.")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("OllamaEmbedder: timeout must be > 0.")
        if mode not in {"native", "disabled"}:
            raise ValueError(
                f"OllamaEmbedder: invalid mode '{mode}'. Allowed: 'native', 'disabled'."
            )

        self.model = model
        self._dimension = dimension
        self.timeout = float(timeout)
        self.mode = mode

        logger.info(
            "OllamaEmbedder initialized (model=%s, dimension=%d, mode=%s)",
            self.model,
            self._dimension,
            self.mode,
        )

    # ------------------------------------------------------------------
    # EmbeddingInterface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        return self._dimension

    async def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """
        Embed a batch of texts using Ollama.

        Contract
        --------
        - `texts` MUST be Iterable[str]
        - Passing a single string is an error
        """
        # Prevent accidental character-level embedding
        if isinstance(texts, str):
            raise TypeError(
                "OllamaEmbedder.embed expects Iterable[str], not a single string."
            )

        texts_list = list(texts)
        if not texts_list:
            logger.debug("OllamaEmbedder.embed called with empty iterable.")
            return []

        # Normalize input
        normalized: List[str] = []
        for t in texts_list:
            if not isinstance(t, str):
                raise TypeError(
                    f"OllamaEmbedder.embed expects strings, got {type(t)}"
                )
            stripped = t.strip()
            if not stripped:
                logger.warning("OllamaEmbedder.embed skipping empty/whitespace text.")
                continue
            normalized.append(stripped)

        if not normalized:
            logger.warning("OllamaEmbedder.embed: no valid texts after normalization.")
            return []

        if self.mode == "disabled":
            logger.error("OllamaEmbedder.embed called while mode='disabled'.")
            raise RuntimeError(
                "Ollama embedding is disabled. Enable mode='native' to use it."
            )

        # Enforce timeout around native embedding
        try:
            return await asyncio.wait_for(
                self._embed_native(normalized),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            logger.exception("OllamaEmbedder.embed timed out.")
            raise RuntimeError("Ollama embedding timed out.")

    # ------------------------------------------------------------------
    # Native embedding
    # ------------------------------------------------------------------

    @retryable()
    async def _embed_native(self, texts: List[str]) -> List[List[float]]:
        """
        Perform embedding using Ollama's Python API.

        This method embeds the entire batch at once.
        (No batching is used in v1; acceptable for local Ollama models.)
        """
        if ollama is None:
            logger.error("Ollama Python package not installed.")
            raise RuntimeError(
                "OllamaEmbedder requires the 'ollama' package. Install with: pip install ollama"
            )

        try:
            response = await asyncio.to_thread(
                ollama.embed,
                model=self.model,
                input=texts,
            )
        except Exception as exc:
            logger.exception("Ollama embedding call failed.")
            raise RuntimeError("Ollama embedding request failed.") from exc

        # Ollama official response key
        vectors = response.get("embeddings")
        if vectors is None:
            logger.error("Unexpected Ollama embedding response: %r", response)
            raise RuntimeError("Ollama embedding response missing 'embeddings'.")

        if not isinstance(vectors, list):
            raise RuntimeError("Ollama embedding response malformed.")

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(vectors)} embeddings for {len(texts)} inputs."
            )

        # Validate vectors
        cleaned: List[List[float]] = []
        for i, vec in enumerate(vectors):
            if not isinstance(vec, list):
                raise RuntimeError(f"Invalid embedding vector at index {i}.")
            flt_vec = [float(x) for x in vec]
            if len(flt_vec) != self._dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {self._dimension}, got {len(flt_vec)}"
                )
            cleaned.append(flt_vec)

        logger.debug("OllamaEmbedder: received %d embeddings.", len(cleaned))
        return cleaned

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: EmbeddingConfig) -> "OllamaEmbedder":
        """
        Construct an Ollama embedder from a typed config.
        """
        model = cfg.model
        if not model:
            raise ValueError("Ollama embedding config must define 'model'.")
        embed_kwargs = {**cfg.config}
        embed_kwargs.pop("model", None)
        embed_kwargs.pop("dimension", None)
        if "mode" not in embed_kwargs:
            embed_kwargs["mode"] = "native"
        return cls(
            model=model,
            dimension=cfg.dimension,
            **embed_kwargs,
        )
