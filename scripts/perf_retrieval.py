"""
Simple retrieval performance harness for UMA-RLM.

Measures end-to-end latency from prompt to retrieval using in-memory backends.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

from uma.core.uma_memory import UMAMemory
from uma.core.utils.identity import ensure_user_subject
from uma.types_episode import Episode
from uma.types_fact import Fact
from uma.types_skill import Skill
from uma.adapters.observability.metrics import snapshot


async def fake_llm(messages=None, **kwargs):
    return "ok"


async def fake_embed(texts=None, **kwargs):
    texts = texts or []
    return [[0.1, 0.1, 0.1] for _ in texts]


def build_config(db_root: str, rlm_enabled: bool) -> dict:
    return {
        "storage": {
            "db_root": db_root,
            "sql_backend": "sqlite",
            "vector_backend": "inmemory",
            "graph_backend": "disabled",
        },
        "working_memory": {
            "max_tokens": 512,
            "warning_ratio": 0.7,
            "hard_limit_ratio": 0.95,
            "chunk_size": 10,
        },
        "embedding": {
            "provider": "scripts.perf_retrieval:fake_embed",
            "dimension": 3,
        },
        "llm": {
            "provider": "scripts.perf_retrieval:fake_llm",
        },
        "retrieval": {
            "max_episodes": 5,
            "max_facts": 5,
            "max_skills": 5,
            "max_graph_items": 5,
            "rlm": {
                "enabled": rlm_enabled,
                "max_steps": 2,
                "max_actions_per_step": 2,
                "max_items_per_type": 10,
                "llm_max_tokens": 120,
                "timeout_s": 5.0,
                "max_env_calls": 6,
                "max_return_chars": 400,
            },
        },
        "consolidation": {
            "enabled": True,
            "cluster_similarity": 0.75,
            "max_episodes_per_cycle": 50,
            "prune_min_fact_salience": 0.2,
        },
        "features": {
            "load": [],
        },
    }


async def seed_memory(memory: UMAMemory, user_id: str) -> None:
    embedding = [0.1, 0.1, 0.1]
    episode = Episode(
        id="ep-1",
        user_id=user_id,
        timestamp=datetime.utcnow(),
        summary="user likes coffee",
        raw="user likes coffee and cold brew",
        tags=["pref"],
        embedding=embedding,
    )
    await memory.episodic_store.add_episode(episode, embedding)

    fact = Fact(
        id="fact-1",
        subject=ensure_user_subject(user_id),
        predicate="likes",
        object="coffee",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=[episode.id],
        confidence=0.9,
    )
    await memory.semantic_store.upsert_fact(fact, embedding)

    skill = Skill(
        id="skill-1",
        name="Make coffee",
        trigger_phrases=["coffee"],
        trigger_patterns=[],
        plan={"steps": ["brew", "serve"]},
        tools=["kettle"],
        example="Make coffee",
        meta={"tag": "demo"},
    )
    await memory.procedural_store.add_skill(skill, embedding)


async def run_perf(iterations: int, concurrency: int, rlm_enabled: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_root = str(Path(tmp) / "db") + "/"
        cfg = build_config(db_root, rlm_enabled)
        cfg_path = Path(tmp) / "perf.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        memory = UMAMemory.from_yaml(str(cfg_path))
        memory.initialize()

        user_id = "user-123"
        await seed_memory(memory, user_id)

        async def _call(i: int) -> None:
            await memory.get_user_context(user_id=user_id, query_text=f"query {i}")

        sem = asyncio.Semaphore(concurrency)

        async def _guarded(i: int) -> None:
            async with sem:
                await _call(i)

        await asyncio.gather(*[_guarded(i) for i in range(iterations)])

        print("Metrics snapshot:")
        print(snapshot())


def main() -> None:
    parser = argparse.ArgumentParser(description="UMA-RLM retrieval performance test")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--rlm", action="store_true", help="Enable RLM retrieval")
    args = parser.parse_args()

    asyncio.run(run_perf(args.iterations, args.concurrency, args.rlm))


if __name__ == "__main__":
    main()
