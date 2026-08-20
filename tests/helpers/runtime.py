from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import yaml

from tests.helpers.providers import fake_embed, fake_llm
from uma.api.memory import UMAMemory
from uma.adapters.llm import openai_compatible as shared_llm_module


_TEST_EMBED_DIM = 64


class _FakeAsyncOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))
        self.embeddings = SimpleNamespace(create=self._embedding_create)

    async def _chat_create(self, **kwargs: Any) -> Any:
        content = await fake_llm(
            messages=list(kwargs.get("messages") or []),
            max_tokens=int(kwargs.get("max_tokens", 256)),
            temperature=float(kwargs.get("temperature", 0.0)),
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    async def _embedding_create(self, **kwargs: Any) -> Any:
        vectors = await fake_embed(
            texts=list(kwargs.get("input") or []),
            dimension=_TEST_EMBED_DIM,
        )
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=vector) for vector in vectors]
        )


def build_test_config(
    *,
    db_root: Path,
    embedding_dim: int = 64,
    graph_backend: str = "disabled",
    graph_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build a minimal UMA config suitable for deterministic, offline CI tests.

    Notes:
    - Uses real UMA bootstrapping (UMAMemory.from_yaml).
    - Uses callable-based LLM + embedder providers (no network).
    - Uses sqlite + InMemoryVectorIndex (fast, dependency-free).
    """
    db_root = Path(db_root)
    graph_config = graph_config or {}
    global _TEST_EMBED_DIM
    _TEST_EMBED_DIM = int(embedding_dim)
    shared_llm_module.AsyncOpenAI = _FakeAsyncOpenAI  # type: ignore[assignment]

    cfg: dict[str, Any] = {
        "storage": {
            "db_root": str(db_root) + "/",
            "sql_backend": "sqlite",
            "vector_backend": "inmemory",
            "graph_backend": graph_backend,
        },
        "working_memory": {
            "max_tokens": 512,
            "warning_ratio": 0.7,
            "hard_limit_ratio": 0.95,
            "chunk_size": 10,
            "keep_recent_messages": 2,
            "keep_recent_token_fraction": 0.1,
        },
        "embedding": {
            "provider": "ollama",
            "model": "nomic-embed-text",
            "dimension": int(embedding_dim),
            "config": {"host": "http://localhost:11434"},
        },
        "llms": {
            "agent": {"provider": "ollama", "model": "llama3", "config": {"host": "http://localhost:11434"}},
            "uma": {"provider": "ollama", "model": "llama3", "config": {"host": "http://localhost:11434"}},
        },
        "retrieval": {
            "max_episodes": 5,
            "max_facts": 10,
            "max_skills": 5,
            "max_graph_items": 5,
            "max_evidence_chunks": 6,
            "strict": True,
            "hybrid": {"enabled": True, "top_k_dense": 0, "top_k_sparse": 15, "fusion_strategy": "rrf"},
            "context": {
                "max_working_messages": 6,
                "max_episodic": 2,
                "max_semantic": 4,
                "max_chunks": 4,
                "max_procedural": 3,
                "max_graph": 3,
                "snippet_max_chars": 600,
                "snippet_refiner_top_k": 6,
                "include_working_memory": True,
                "include_episodic": True,
                "include_graph": True,
                "include_procedural": True,
            },
            "rlm": {
                "test_mode": True,
                "max_steps": 2,
                "max_actions_per_step": 1,
                "max_items_per_type": 30,
                "timeout_s": 5.0,
                "max_env_calls": 6,
            },
        },
        "semantic": {"salience_threshold": 0.1},
        "consolidation": {
            "enabled": False,
            "cluster_similarity": 0.75,
            "max_episodes_per_cycle": 50,
            "prune_min_fact_salience": 0.2,
        },
        "features": {"load": [], "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False}},
    }

    if graph_backend != "disabled":
        cfg["storage"]["graph_config"] = graph_config

    return cfg


TEST_AGENT_ID = "agent-default"


async def init_uma_for_tests(
    tmp_path: Path,
    *,
    embedding_dim: int = 64,
    graph_backend: str = "disabled",
    graph_config: Optional[dict[str, Any]] = None,
) -> UMAMemory:
    """
    Initialize a fully bootstrapped UMAMemory instance for tests.

    Uses the same public entrypoint (`UMAMemory.from_yaml`)
    and then forces ingestion-ready initialization to avoid background warmup
    races in CI.
    """
    tmp_path = Path(tmp_path)
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)

    cfg = build_test_config(
        db_root=db_root,
        embedding_dim=embedding_dim,
        graph_backend=graph_backend,
        graph_config=graph_config,
    )
    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Identity is per call, not per instance. `agent_id` is kept on the helper
    # only so tests have one canonical value to pass into the public API.
    memory = UMAMemory.from_yaml(str(cfg_path))
    memory._ensure_ingestion_ready()
    return memory
