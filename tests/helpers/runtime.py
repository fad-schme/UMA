from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from uma.core.uma_memory import UMAMemory


def build_test_config(
    *,
    db_root: Path,
    embedding_dim: int = 64,
    graph_backend: str = "disabled",
    graph_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a minimal UMA config suitable for deterministic, offline CI tests.

    Notes:
    - Uses real UMA bootstrapping (UMAMemory.from_yaml).
    - Uses callable-based LLM + embedder providers (no network).
    - Uses sqlite + InMemoryVectorIndex (fast, dependency-free).
    """
    db_root = Path(db_root)
    graph_config = graph_config or {}

    cfg: Dict[str, Any] = {
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
            "provider": "tests.helpers.providers:fake_embed",
            "model": "fake-embed",
            "dimension": int(embedding_dim),
        },
        "llms": {
            "agent": {"provider": "tests.helpers.providers:fake_llm", "model": "fake-llm", "config": {}},
            "uma": {"provider": "tests.helpers.providers:fake_llm", "model": "fake-llm", "config": {}},
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
                "snippet_refiner_enabled": True,
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
        "semantic": {"salience_threshold": 0.1, "doc_min_fact_words": 5, "doc_summary_enabled": False},
        "consolidation": {
            "enabled": False,
            "cluster_similarity": 0.75,
            "max_episodes_per_cycle": 50,
            "prune_min_fact_salience": 0.2,
        },
        "features": {"load": [], "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False}},
        "pipeline": {"defer_post_turn": False, "post_turn_queue_max": 50},
    }

    if graph_backend != "disabled":
        cfg["storage"]["graph_config"] = graph_config

    return cfg


async def init_uma_for_tests(
    tmp_path: Path,
    *,
    agent_id: str = "agent-default",
    embedding_dim: int = 64,
    graph_backend: str = "disabled",
    graph_config: Optional[Dict[str, Any]] = None,
) -> UMAMemory:
    """
    Initialize a fully bootstrapped UMAMemory instance for tests.

    Uses the same entrypoint as production/examples (`UMAMemory.from_yaml`)
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

    memory = UMAMemory.from_yaml(str(cfg_path))
    memory.agent_id = agent_id
    memory._ensure_ingestion_ready()
    return memory
