"""
OpenAI ChatCompletion adapter (production-grade).

Implements:
- LLMInterface
- Async OpenAI API calls
- Retries (via retryable decorator)
- Timeout handling
- Logging

Coding agent instructions:
--------------------------
- Set OPENAI_API_KEY in environment variables.
- For Azure OpenAI, fork this class and adjust endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Dict, Any

from openai import AsyncOpenAI
from ...adapters.llm.base import LLMInterface
from .retry_utils import retryable

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """
    UMA adapter for OpenAI ChatCompletion models.

    Recommended models:
    - "gpt-4.1-mini"
    - "gpt-4.1"
    - "gpt-4.1-turbo"
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        timeout: float = 20.0,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment.")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.timeout = timeout

        logger.info("OpenAILLM initialized with model=%s", model)

    @retryable()
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Call OpenAI ChatCompletion API.

        Error handling:
        - Retries on rate-limit / transient failures.
        - Timeout enforced manually.

        Returns
        -------
        str
            Assistant message content.
        """

        async def _call():
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as exc:
                logger.exception("OpenAI ChatCompletion failed: %s", exc)
                raise

        return await asyncio.wait_for(_call(), timeout=self.timeout)