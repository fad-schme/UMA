from __future__ import annotations

from types import SimpleNamespace

import pytest

from uma.adapters.llm import anthropic as anthropic_module
from uma.adapters.llm import openai_compatible as shared_module
from uma.adapters.llm.provider_registry import get_embedder_factory, get_llm_factory
from uma.common.initializers.providers import initialize_embedder, initialize_llm
from uma.common.config import UMAConfig
from uma.common.config_types import EmbeddingConfig, LLMConfig


class _FakeAsyncOpenAI:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))
        self.embeddings = SimpleNamespace(create=self._embedding_create)

    async def _chat_create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    async def _embedding_create(self, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])


class _FakeAsyncAnthropic:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.messages = SimpleNamespace(create=self._messages_create)
        self.last_create_kwargs = None

    async def _messages_create(self, **kwargs):
        self.last_create_kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(text="anthropic-ok")])


def test_registry_resolves_ollama_and_openai_llm_and_embedding(monkeypatch) -> None:
    monkeypatch.setattr(shared_module, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(anthropic_module, "AsyncAnthropic", _FakeAsyncAnthropic)

    llm_cfg = LLMConfig(provider="ollama", model="llama3", config={"host": "http://localhost:11434"})
    embed_cfg = EmbeddingConfig(
        provider="ollama",
        model="nomic-embed-text",
        dimension=768,
        config={"host": "http://localhost:11434"},
    )

    assert get_llm_factory("ollama") is not None
    assert get_llm_factory("openai") is not None
    assert get_llm_factory("anthropic") is not None
    assert get_embedder_factory("ollama") is not None
    assert get_embedder_factory("openai") is not None
    assert get_embedder_factory("anthropic") is None

    assert get_llm_factory("ollama")(llm_cfg).model == "llama3"
    assert get_embedder_factory("ollama")(embed_cfg).model == "nomic-embed-text"


def test_ollama_provider_normalizes_host_to_v1(monkeypatch) -> None:
    monkeypatch.setattr(shared_module, "AsyncOpenAI", _FakeAsyncOpenAI)

    cfg = LLMConfig(
        provider="ollama",
        model="llama3",
        config={"host": "http://localhost:11434"},
    )
    llm = get_llm_factory("ollama")(cfg)

    assert llm.base_url == "http://localhost:11434/v1"
    assert llm._client.init_kwargs["base_url"] == "http://localhost:11434/v1"
    assert llm._client.init_kwargs["api_key"] == "ollama"


def test_openai_llm_uses_openai_base_url(monkeypatch) -> None:
    monkeypatch.setattr(shared_module, "AsyncOpenAI", _FakeAsyncOpenAI)

    cfg = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        config={"api_key": "test-key"},
    )
    llm = get_llm_factory("openai")(cfg)

    assert llm.base_url == "https://api.openai.com/v1"
    assert llm._client.init_kwargs["base_url"] == "https://api.openai.com/v1"
    assert llm._client.init_kwargs["api_key"] == "test-key"


def test_openai_api_key_env_is_accepted_without_network(monkeypatch) -> None:
    monkeypatch.setattr(shared_module, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setenv("UMA_OPENAI_TEST_KEY", "env-key")

    cfg = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        dimension=1536,
        config={"api_key_env": "UMA_OPENAI_TEST_KEY"},
    )
    embedder = get_embedder_factory("openai")(cfg)

    assert embedder.base_url == "https://api.openai.com/v1"
    assert embedder._client.init_kwargs["api_key"] == "env-key"
    assert embedder.dimension == 1536


def test_anthropic_llm_supports_api_key(monkeypatch) -> None:
    monkeypatch.setattr(anthropic_module, "AsyncAnthropic", _FakeAsyncAnthropic)

    cfg = LLMConfig(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        config={"api_key": "anthropic-test-key"},
    )
    llm = get_llm_factory("anthropic")(cfg)

    assert llm.model == "claude-3-5-haiku-latest"
    assert llm._client.init_kwargs["api_key"] == "anthropic-test-key"


def test_anthropic_llm_supports_api_key_env(monkeypatch) -> None:
    monkeypatch.setattr(anthropic_module, "AsyncAnthropic", _FakeAsyncAnthropic)
    monkeypatch.setenv("UMA_ANTHROPIC_TEST_KEY", "env-anthropic-key")

    cfg = LLMConfig(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        config={"api_key_env": "UMA_ANTHROPIC_TEST_KEY"},
    )
    llm = get_llm_factory("anthropic")(cfg)

    assert llm._client.init_kwargs["api_key"] == "env-anthropic-key"


def test_anthropic_explicit_api_key_wins_over_env(monkeypatch) -> None:
    monkeypatch.setattr(anthropic_module, "AsyncAnthropic", _FakeAsyncAnthropic)
    monkeypatch.setenv("UMA_ANTHROPIC_TEST_KEY", "env-anthropic-key")

    cfg = LLMConfig(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        config={
            "api_key": "explicit-anthropic-key",
            "api_key_env": "UMA_ANTHROPIC_TEST_KEY",
        },
    )
    llm = get_llm_factory("anthropic")(cfg)

    assert llm._client.init_kwargs["api_key"] == "explicit-anthropic-key"


@pytest.mark.asyncio
async def test_anthropic_adapter_formats_messages_from_uma_interface(monkeypatch) -> None:
    monkeypatch.setattr(anthropic_module, "AsyncAnthropic", _FakeAsyncAnthropic)

    cfg = LLMConfig(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        config={"api_key": "anthropic-test-key"},
    )
    llm = get_llm_factory("anthropic")(cfg)

    out = await llm.generate(
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "tool-result"},
        ],
        max_tokens=99,
        temperature=0.2,
    )

    assert out == "anthropic-ok"
    assert llm._client.last_create_kwargs["system"] == "You are concise."
    assert llm._client.last_create_kwargs["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "tool-result"},
    ]
    assert llm._client.last_create_kwargs["max_tokens"] == 99
    assert llm._client.last_create_kwargs["temperature"] == 0.2


def test_public_configs_remain_ollama_based() -> None:
    for path in ("config/uma.yaml", "config/uma_lite.yaml", "config/uma_cont.yaml"):
        cfg = UMAConfig.load_yaml(path)
        assert cfg.embedding.provider == "ollama"
        assert cfg.llms.agent.provider == "ollama"
        assert cfg.llms.uma.provider == "ollama"


def test_initializer_rejects_unsupported_public_provider() -> None:
    memory = SimpleNamespace(
        llm_cfg=LLMConfig(provider="tests.helpers.providers:fake_llm", model="fake", config={}),
        agent_llm_cfg=None,
        embedding_cfg=EmbeddingConfig(
            provider="tests.helpers.providers:fake_embed",
            model="fake",
            dimension=64,
            config={},
        ),
        llm=None,
        agent_llm=None,
        embedder=None,
    )

    with pytest.raises(ValueError, match="Unsupported provider 'tests\\.helpers\\.providers:fake_llm'"):
        initialize_llm(memory)

    with pytest.raises(
        ValueError,
        match="Unsupported embedding provider 'tests\\.helpers\\.providers:fake_embed'",
    ):
        initialize_embedder(memory)


def test_initializer_rejects_anthropic_embedding_provider() -> None:
    memory = SimpleNamespace(
        embedding_cfg=EmbeddingConfig(
            provider="anthropic",
            model="not-supported",
            dimension=1536,
            config={"api_key": "anthropic-test-key"},
        ),
        embedder=None,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported embedding provider 'anthropic'. Supported providers: ollama, openai.",
    ):
        initialize_embedder(memory)
