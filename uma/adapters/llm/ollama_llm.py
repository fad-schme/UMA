"""
Production-ready Ollama LLM adapter for UMA-3.

This module implements:
- LLMInterface for chat-style generation
- Async HTTP client using aiohttp
- Robust retries (via retryable decorator)
- Timeout control
- Centralized logging

This allows UMA-3 to run fully locally using models like:
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
from .retry_utils import retryable

logger = logging.getLogger(__name__)


class OllamaLLM(LLMInterface):
    """
    UMA-3 adapter that calls a locally running Ollama server.

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
        host: str = "http://localhost:11434",
        chat_endpoint: str = "/api/chat",
        timeout: float = 30.0,
    ):
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
            "options": {
                "temperature": temperature,
            },
        }

        if aiohttp is None:
            raise RuntimeError(
                "OllamaLLM requires 'aiohttp' to be installed. Install with: pip install aiohttp"
            )

        async with aiohttp.ClientSession() as session:
            try:
                async with asyncio.wait_for(
                    session.post(self.chat_url, json=payload), timeout=self.timeout
                ) as resp:

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
            return ""