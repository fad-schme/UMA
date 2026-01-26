"""
Production-ready Ollama LLM adapter for UMA.

This module implements:
- LLMInterface for chat-style generation
- Async HTTP client using aiohttp
- Robust retries (via retryable decorator)
- Timeout control
- Centralized logging

This allows UMA to run fully locally using models like:
- llama3
- mistral
- mixtral
- phi3
- any custom ollama model

Coding agent instructions:
--------------------------
- Ensure 'ollama' is installed locally: https://ollama.com
- Use this adapter interchangeably with OpenAILLM.
- You can add streaming or function-calling later.

Supported Ollama endpoints:
- POST /api/chat (messages)
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Dict, Any

try:
    import aiohttp
except Exception:  # pragma: no cover - optional dependency
    aiohttp = None

from ...adapters.llm.base import LLMInterface
from ...core.utils.config_types import LLMConfig
from .retry_utils import retryable

logger = logging.getLogger(__name__)


class OllamaLLM(LLMInterface):
    """
    UMA adapter that calls a locally running Ollama server.

    Default endpoint:
        http://localhost:11434/api/chat

    Behavior:
    - Build a chat-style prompt
    - Send to Ollama
    - Retrieve assistant response from JSON
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://192.168.178.101:11434",
        chat_endpoint: str = "/api/chat",
        timeout: float = 30.0,
    ):
        if aiohttp is None:
            raise RuntimeError(
                "OllamaLLM requires 'aiohttp' to be installed. Install with: pip install aiohttp"
            )
        self.model = model
        self.base_url = host.rstrip("/")
        self.chat_url = f"{self.base_url}{chat_endpoint}"
        self.timeout = timeout

        logger.info("OllamaLLM initialized (model=%s)", model)

    @retryable()
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Generate a response using Ollama's /api/chat endpoint.

        Parameters
        ----------
        messages:
            Chat-style conversation history.
        max_tokens:
            Ignored by Ollama (for now), but kept for API symmetry.
        temperature:
            Temperature setting passed to ollama.

        Returns
        -------
        str
            Assistant reply text.
        """

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if "format" in kwargs and kwargs["format"]:
            payload["format"] = kwargs["format"]

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(self.chat_url, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error("Ollama LLM error: HTTP %s: %s", resp.status, text)
                        raise RuntimeError(f"Ollama error: {resp.status}")

                    data = await resp.json()

            except asyncio.TimeoutError:
                logger.exception("OllamaLLM.generate timeout after %.2f sec", self.timeout)
                raise
            except Exception:
                logger.exception("OllamaLLM.generate network exception")
                raise

        # Ollama returns: {"message": {"role": "assistant", "content": "..."}}
        try:
            return data["message"]["content"]
        except Exception:
            logger.error("Malformed Ollama response: %s", data)
            raise RuntimeError("Malformed Ollama response.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: LLMConfig) -> "OllamaLLM":
        """
        Build an OllamaLLM instance from UMA's typed LLMConfig.
        """
        model = cfg.ollama_model or cfg.model or "llama3"
        llm_kwargs = {**cfg.config}
        llm_kwargs.pop("model", None)
        return cls(model=model, **llm_kwargs)
