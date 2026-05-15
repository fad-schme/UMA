from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from uma.common.config_types import LLMConfig

from .base import LLMInterface
from .retry_utils import retryable, should_retry_anthropic

logger = logging.getLogger(__name__)

try:
    from anthropic import AsyncAnthropic  # type: ignore
except Exception as exc:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore[assignment]
    #logger.error("Failed to import anthropic: %s", exc)


def _resolve_api_key(
    *,
    api_key: Optional[str],
    api_key_env: Optional[str],
) -> str:
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()

    env_name = "ANTHROPIC_API_KEY"
    if isinstance(api_key_env, str) and api_key_env.strip():
        env_name = api_key_env.strip()

    env_value = os.getenv(env_name)
    if env_value and env_value.strip():
        return env_value.strip()

    raise RuntimeError(
        "Anthropic provider requires an API key. Set config.api_key or config.api_key_env."
    )


def _coerce_role(role: Any) -> str:
    if role == "assistant":
        return "assistant"
    return "user"


def _coerce_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


class AnthropicLLM(LLMInterface):
    """Native Anthropic chat adapter for the public `anthropic` provider."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout: float = 30.0,
    ) -> None:
        if AsyncAnthropic is None:
            raise RuntimeError(
                "Anthropic LLM adapter requires the 'anthropic' package to be installed."
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("AnthropicLLM: model must be a non-empty string.")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("AnthropicLLM: api_key must be a non-empty string.")
        if not isinstance(timeout, (int, float)) or float(timeout) <= 0:
            raise ValueError("AnthropicLLM: timeout must be > 0.")

        self.provider_name = "anthropic"
        self.model = model.strip()
        self.timeout = float(timeout)
        self._client = AsyncAnthropic(
            api_key=api_key.strip(),
            timeout=self.timeout,
        )

    @retryable(should_retry=should_retry_anthropic)
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        system_parts: List[str] = []
        anthropic_messages: List[Dict[str, str]] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            content = str(message.get("content") or "")
            if not content.strip():
                continue
            role = str(message.get("role") or "user")
            if role == "system":
                system_parts.append(content)
                continue
            anthropic_messages.append(
                {
                    "role": _coerce_role(role),
                    "content": content,
                }
            )

        if not anthropic_messages:
            anthropic_messages = [{"role": "user", "content": ""}]

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        system_text = "\n\n".join(system_parts).strip()
        if system_text:
            request_kwargs["system"] = system_text
        request_kwargs.update(kwargs)

        response = await self._client.messages.create(**request_kwargs)
        return _coerce_text_content(getattr(response, "content", None)).strip()

    @classmethod
    def from_config(cls, cfg: LLMConfig) -> "AnthropicLLM":
        llm_kwargs = {**cfg.config}
        api_key = llm_kwargs.pop("api_key", None)
        api_key_env = llm_kwargs.pop("api_key_env", None)
        timeout = llm_kwargs.pop("timeout", 30.0)
        llm_kwargs.pop("model", None)
        if llm_kwargs:
            logger.debug(
                "AnthropicLLM ignored extra config keys: %s",
                sorted(llm_kwargs.keys()),
            )

        return cls(
            model=cfg.model or "claude-3-5-haiku-latest",
            api_key=_resolve_api_key(api_key=api_key, api_key_env=api_key_env),
            timeout=float(timeout),
        )
