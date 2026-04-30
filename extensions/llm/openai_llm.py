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
from typing import List, Dict

from openai import AsyncOpenAI
from uma.adapters.llm.base import LLMInterface
from uma.common.config_types import LLMConfig
from uma.adapters.llm.retry_utils import retryable, should_retry_openai

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

    @retryable(should_retry=should_retry_openai)
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: LLMConfig) -> "OpenAILLM":
        """
        Build an OpenAILLM instance from UMA's typed LLMConfig.
        """
        model = cfg.model or "gpt-4.1-mini"
        llm_kwargs = {**cfg.config}
        llm_kwargs.pop("model", None)
        return cls(model=model, **llm_kwargs)
