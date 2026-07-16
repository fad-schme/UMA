"""Configuration, providers, features, and infrastructure: config types, secrets, LLM/embedding providers.

Covers UMAConfig parsing, SecretsProvider wiring, LLM/embedding provider
resolution (Ollama/OpenAI/Anthropic), feature loading, health checks,
and SQLite adapter rollback safety.
"""
from __future__ import annotations
from pathlib import Path
from tests.helpers.runtime import build_test_config
from tests.helpers.runtime import init_uma_for_tests
from types import SimpleNamespace
from uma.adapters.llm.provider_registry import get_embedder_factory, get_llm_factory
from uma.adapters.db.sqlite_adapter import SQLiteAdapter
from uma.adapters.llm import anthropic as anthropic_module
from uma.adapters.llm import openai_compatible as shared_module
from uma.adapters.llm.provider_registry import get_embedder_factory, get_llm_factory
from uma.api.memory import UMAMemory
from uma.common.config import UMAConfig
from uma.common.config_types import EmbeddingConfig, LLMConfig, RuntimeConfig
from uma.common.config_types import RetrievalConfig
from uma.common.initializers.providers import initialize_embedder, initialize_llm
from uma.stores.base_sql_store import BaseSQLStore
import pytest
import yaml

# ── test_config_types ──────────────────────────────────────────



def test_retrieval_config_from_dict_with_rlm():
    cfg = RetrievalConfig.from_dict(
        {
            "max_episodes": 2,
            "max_facts": 3,
            "max_skills": 4,
            "max_graph_items": 5,
            "rlm": {
                "enabled": True,
                "max_steps": 7,
            },
        }
    )

    assert cfg.max_episodes == 2
    assert cfg.max_facts == 3
    assert cfg.max_skills == 4
    assert cfg.max_graph_items == 5
    assert cfg.rlm is not None
    assert cfg.rlm.enabled is True
    assert cfg.rlm.max_steps == 7
    # Defaults preserved
    assert cfg.rlm.max_actions_per_step == 2
    assert cfg.rlm.max_items_per_type == 30


# ── test_secrets_config ──────────────────────────────────────────


from uma.adapters.secrets.secrets import EnvVarProvider, Secret, SecretNotFound, SecretsProvider, SecretsProviderError


def _write_config(tmp_path: Path, cfg: dict) -> Path:
    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_public_adapter_surface_exports_secrets_types() -> None:
    assert SecretsProvider is not None
    assert EnvVarProvider is not None
    assert Secret is not None
    assert SecretsProviderError is not None
    assert SecretNotFound is not None


def test_from_yaml_without_secrets_block_preserves_lite_default(tmp_path: Path) -> None:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg_path = _write_config(tmp_path, cfg)

    memory = UMAMemory.from_yaml(str(cfg_path))
    try:
        assert memory._secrets_provider is None
    finally:
        memory.shutdown()


def test_from_yaml_with_secrets_block_initializes_provider(tmp_path: Path) -> None:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg["secrets"] = {
        "provider": "uma.adapters.secrets.EnvVarProvider",
        "options": {"prefix": "UMA"},
    }
    cfg_path = _write_config(tmp_path, cfg)

    memory = UMAMemory.from_yaml(str(cfg_path))
    try:
        assert isinstance(memory._secrets_provider, EnvVarProvider)
        assert memory._secrets_cfg is not None
        assert memory._secrets_cfg.provider == "uma.adapters.secrets.EnvVarProvider"
    finally:
        memory.shutdown()


def test_from_yaml_rejects_unresolvable_secrets_provider(tmp_path: Path) -> None:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg["secrets"] = {
        "provider": "uma.adapters.secrets.DoesNotExist",
        "options": {},
    }
    cfg_path = _write_config(tmp_path, cfg)

    with pytest.raises(RuntimeError, match=r"secrets\.provider"):
        UMAMemory.from_yaml(str(cfg_path))


def test_from_yaml_rejects_invalid_secrets_options(tmp_path: Path) -> None:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg["secrets"] = {
        "provider": "uma.adapters.secrets.EnvVarProvider",
        "options": {"prefix": ""},
    }
    cfg_path = _write_config(tmp_path, cfg)

    with pytest.raises(RuntimeError, match=r"secrets\.options"):
        UMAMemory.from_yaml(str(cfg_path))


# ── test_openai_compatible_providers ──────────────────────────────────────────






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


def test_public_configs_declare_registered_providers() -> None:
    for path in ("config/uma.yaml",):
        cfg = UMAConfig.load_yaml(path)
        assert get_embedder_factory(cfg.embedding.provider) is not None
        assert get_llm_factory(cfg.llms.uma.provider) is not None


def test_features_section_is_optional_and_uses_runtime_defaults(tmp_path) -> None:
    config_data = build_test_config(db_root=tmp_path / "db")
    config_data.pop("features")
    cfg = UMAConfig.load_yaml(str(_write_config(tmp_path, config_data)))

    runtime = RuntimeConfig.from_uma_config(cfg)

    assert runtime.features.procedural_enabled is True
    assert runtime.features.consolidation_enabled is True
    assert {item["name"] for item in runtime.features.load} == {
        "procedural",
        "consolidation",
    }


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


# ── test_feature_loading ──────────────────────────────────────────






@pytest.mark.asyncio
async def test_feature_loader_attaches_procedural_feature(tmp_path):
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)

    cfg = build_test_config(db_root=db_root)
    cfg["features"] = {
        "load": [
            {
                "name": "procedural",
                "enabled": True,
                "provider": "uma.memory.procedural.feature:ProceduralFeature",
                "config": {"max_k": 3},
            }
        ],
        "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False},
    }

    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    memory = UMAMemory.from_yaml(str(cfg_path))
    memory._ensure_ingestion_ready()

    assert "procedural" in memory.features
    assert callable(getattr(memory, "procedural_health"))


@pytest.mark.asyncio
async def test_feature_loader_skips_failed_attachment_when_policy_is_log_and_skip(tmp_path):
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)

    cfg = build_test_config(db_root=db_root)
    cfg["features"] = {
        "load": [
            {
                "name": "procedural",
                "enabled": True,
                "provider": "uma.memory.procedural.feature:DoesNotExist",
                "config": {"max_k": 3},
            }
        ],
        "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False},
    }

    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    memory = UMAMemory.from_yaml(str(cfg_path))
    memory._ensure_ingestion_ready()

    assert "procedural" not in memory.features



# ── test_health_check ──────────────────────────────────────────





_EXPECTED_CHECK_KEYS = {
    "db:episodic",
    "db:semantic",
    "db:procedural",
    "vector:episodic",
    "vector:semantic",
    "vector:procedural",
    "graph",
    "llm",
    "embedding",
}


@pytest.mark.asyncio
async def test_health_check_returns_ok_or_degraded_on_initialized_instance(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)
    result = memory.health_check()

    assert result["status"] in ("ok", "degraded")
    assert "checks" in result
    assert _EXPECTED_CHECK_KEYS.issubset(result["checks"].keys())

    for check in result["checks"].values():
        assert "name" in check
        assert "status" in check
        assert check["status"] in ("ok", "error", "skipped")


@pytest.mark.asyncio
async def test_health_check_returns_error_when_not_initialized(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)
    # Simulate the defensive guard: initialized=False is only reachable if
    # someone constructs UMAMemory directly (not via from_yaml) without warmup.
    memory.initialized = False

    result = memory.health_check()

    assert result["status"] == "error"
    assert "checks" in result
    assert "memory" in result["checks"]
    assert result["checks"]["memory"]["status"] == "error"


# ── test_db_adapters ──────────────────────────────────────────



def test_sqlite_adapter_row_access_and_rollback(tmp_path):
    db_path = tmp_path / "test.db"
    adapter = SQLiteAdapter(str(db_path))

    conn = adapter.get_connection()
    try:
        cur = conn.cursor()
        cur.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        cur.execute("INSERT INTO items (name) VALUES (?)", ("alpha",))
        conn.commit()

        cur.execute("SELECT id, name FROM items")
        row = cur.fetchone()
        assert row["name"] == "alpha"

        conn.rollback()
    finally:
        conn.close()


# ── test_db_rollbacks ──────────────────────────────────────────



class _FailingConn:
    def rollback(self):
        raise RuntimeError("rollback failed")


class _OkConn:
    def __init__(self):
        self.called = False

    def rollback(self):
        self.called = True


def test_safe_rollback_swallows_errors(tmp_path):
    store = BaseSQLStore(SQLiteAdapter(str(tmp_path / "t.db")))
    store._safe_rollback(_FailingConn(), "test")


def test_safe_rollback_calls_connection(tmp_path):
    store = BaseSQLStore(SQLiteAdapter(str(tmp_path / "t.db")))
    conn = _OkConn()
    store._safe_rollback(conn, "test")
    assert conn.called is True
