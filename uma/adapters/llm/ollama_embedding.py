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
from typing import Iterable, List, Optional

import os

from ...core.utils.config_types import EmbeddingConfig
from .base import EmbeddingInterface
from .retry_utils import retryable

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency if unused
try:
    from ollama import Client  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Client = None


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
        host: Optional[str] = None,
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
        self.host = host
        self._preflight_checked = False

        if Client is None:
            raise RuntimeError(
                "OllamaEmbedder requires the 'ollama' package. Install with: pip install ollama"
            )
        client_kwargs = {}
        if host is not None:
            client_kwargs["host"] = host
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client = Client(**client_kwargs)

        logger.info(
            "OllamaEmbedder initialized (model=%s, dimension=%d, mode=%s, host=%s)",
            self.model,
            self._dimension,
            self.mode,
            self.host or os.getenv("OLLAMA_HOST"),
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

        logger.debug(
            "OllamaEmbedder.embed: model=%s host=%s batch=%d",
            self.model,
            self.host or os.getenv("OLLAMA_HOST"),
            len(normalized),
        )

        # One-time connectivity check to fail fast if Ollama is down.
        if not self._preflight_checked:
            try:
                await asyncio.to_thread(self._client.embed, model=self.model, input=["ping"])
                self._preflight_checked = True
            except Exception as exc:
                logger.error(
                    "Ollama embedder preflight failed (host=%s, model=%s): %s",
                    self.host or os.getenv("OLLAMA_HOST"),
                    self.model,
                    exc,
                )
                raise RuntimeError(
                    "Ollama embedder preflight failed. Check that Ollama is running, "
                    "the model supports embeddings, and OLLAMA_HOST/host are correct."
                ) from exc

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
        try:
            response = await asyncio.to_thread(
                self._client.embed,
                model=self.model,
                input=texts,
            )
        except Exception as exc:
            logger.exception(
                "Ollama embedding call failed (host=%s, model=%s).",
                self.host or os.getenv("OLLAMA_HOST"),
                self.model,
            )
            raise RuntimeError(
                "Ollama embedding request failed. Verify Ollama is reachable and the model supports embeddings."
            ) from exc

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
    # Startup preflight (sync)
    # ------------------------------------------------------------------

    def preflight(self) -> None:
        """
        Best-effort connectivity check used at initialization time.
        Logs a warning on failure but does not raise.
        """
        if self._preflight_checked:
            return
        try:
            self._client.embed(model=self.model, input=["ping"])
            self._preflight_checked = True
            logger.info(
                "OllamaEmbedder preflight OK (host=%s, model=%s)",
                self.host or os.getenv("OLLAMA_HOST"),
                self.model,
            )
        except Exception as exc:
            logger.warning(
                "OllamaEmbedder preflight failed at startup (host=%s, model=%s): %s",
                self.host or os.getenv("OLLAMA_HOST"),
                self.model,
                exc,
            )

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
        host = embed_kwargs.pop("host", None)
        embed_kwargs.pop("dimension", None)
        if not host and not os.getenv("OLLAMA_HOST"):
            raise ValueError(
                "Ollama embedding config must include 'host' or set OLLAMA_HOST."
            )
        if "mode" not in embed_kwargs:
            embed_kwargs["mode"] = "native"
        return cls(
            model=model,
            dimension=cfg.dimension,
            host=host,
            **embed_kwargs,
        )
