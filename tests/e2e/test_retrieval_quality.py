"""Opt-in retrieval quality measurement against a local Ollama model.

Companion to ``test_fact_extraction_quality.py``. That test measures how much
the extractor gets out of a passage; this one measures whether
``retrieve_context`` surfaces the right source pages for a question.

Scoring is at page granularity: gold names corpus filenames, and a retrieved
chunk votes for the page it came from. Chunk ids are generated at ingest and
cannot be hand-authored, so page-level gold is the only form that stays
sealed and stable across re-ingests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.e2e.ollama_env import resolve_ollama_host
from uma.api.memory import UMAMemory

from tests.helpers.runtime import TEST_AGENT_ID

AGENT_ID = TEST_AGENT_ID


pytestmark = [
    pytest.mark.asyncio,
    # Ingesting the corpus and running every gold query against a local model
    # far exceeds the 60s default in pyproject.toml.
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
        os.getenv("RUN_E2E") != "1",
        reason="set RUN_E2E=1 to run tests that require local Ollama",
    ),
]

_FIXTURES = Path(__file__).parent / "fixtures"
_CORPUS_DIR = _FIXTURES / "retrieval_corpus"
_GOLD_PATH = _FIXTURES / "retrieval_gold.json"

_OWNER_ID = "user:eval"
# Top-3 distinct pages: tighter than the retrieval caps and close to what a
# caller actually feeds an LLM. Recall saturates at every cutoff on a corpus
# this small (see the baseline note in README.md) — r-precision carries the
# signal, recall is a coverage regression guard.
_CUTOFF = 3


def _build_config(db_root: Path) -> dict[str, Any]:
    """Real-provider config. Deliberately not `tests.helpers.runtime`, which
    installs fake LLM/embedding providers as a global side effect."""
    host = resolve_ollama_host()
    chat_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    embed_dim = int(os.getenv("OLLAMA_EMBED_DIM", "768"))
    llm = {"provider": "ollama", "model": chat_model, "config": {"host": host}}

    return {
        "storage": {
            "db_root": str(db_root) + "/",
            "sql_backend": "sqlite",
            "vector_backend": "inmemory",
            "graph_backend": "disabled",
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
            "model": embed_model,
            "dimension": embed_dim,
            "config": {"host": host},
        },
        "llms": {"agent": llm, "uma": llm},
        "retrieval": {
            "max_episodes": 5,
            "max_facts": 10,
            "max_skills": 5,
            "max_graph_items": 5,
            # Wide enough that the top-5 page cutoff is applied by the scorer,
            # not by the retrieval caps.
            "max_evidence_chunks": 20,
            "strict": True,
            "hybrid": {"enabled": True, "top_k_dense": 15, "top_k_sparse": 15, "fusion_strategy": "rrf"},
            "context": {
                "max_working_messages": 6,
                "max_episodic": 2,
                "max_semantic": 4,
                "max_chunks": 12,
                "max_procedural": 3,
                "max_graph": 3,
                "snippet_max_chars": 600,
                "snippet_refiner_top_k": 6,
                "include_working_memory": False,
                "include_episodic": True,
                "include_graph": False,
                "include_procedural": True,
            },
            "rlm": {
                "test_mode": True,
                "max_steps": 2,
                "max_actions_per_step": 1,
                "max_items_per_type": 30,
                "timeout_s": 60.0,
                "max_env_calls": 6,
            },
        },
        "semantic": {"salience_threshold": 0.1},
        "consolidation": {"enabled": False, "cluster_similarity": 0.75, "max_episodes_per_cycle": 50, "prune_min_fact_salience": 0.2},
        "features": {"load": [], "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False}},
    }


def _ranked_pages(chunks: list[Any], limit: int) -> list[str]:
    """Distinct source pages in retrieval rank order, best first."""
    pages: list[str] = []
    for chunk in chunks:
        name = Path(str(getattr(chunk, "source_path", "") or "")).name
        if name and name not in pages:
            pages.append(name)
        if len(pages) >= limit:
            break
    return pages


def _recall_at_k(ranked: list[str], gold: set[str]) -> float:
    return len(set(ranked[:_CUTOFF]) & gold) / len(gold)


def _r_precision(ranked: list[str], gold: set[str]) -> float:
    """Precision over the first |gold| results.

    Fixed-cutoff precision is not reported: with 8 corpus pages and gold sets
    of 1-2 pages, P@k is capped by gold-set size rather than by ranking
    quality, so it cannot move when ranking changes. R-precision is
    rank-sensitive and normalised for gold-set size.
    """
    cut = len(gold)
    return len(set(ranked[:cut]) & gold) / cut


async def test_local_ollama_retrieval_precision_recall(tmp_path) -> None:
    """Measure retrieve_context page-level quality without touching the hermetic suite."""
    corpus_pages = sorted(_CORPUS_DIR.glob("*.md"))
    assert corpus_pages, f"no corpus pages under {_CORPUS_DIR}"

    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / "uma_eval.yaml"
    cfg_path.write_text(yaml.safe_dump(_build_config(db_root)), encoding="utf-8")

    memory = UMAMemory.from_yaml(str(cfg_path)).set_context(agent_id="retrieval-eval")
    memory._ensure_ingestion_ready()
    try:
        for page in corpus_pages:
            report = await memory.ingest_document(
                str(page),
                owner_type="user",
                owner_id=_OWNER_ID,
                agent_id=AGENT_ID,
            )
            assert report.chunks_created > 0, f"{page.name} produced no chunks: {report.warnings}"

        # Gold is read only after ingest has completed, so it cannot reach the
        # ingest path even by accident.
        gold_doc = json.loads(_GOLD_PATH.read_text(encoding="utf-8"))
        known_pages = {p.name for p in corpus_pages}

        recalls: list[float] = []
        precisions: list[float] = []
        query_results: list[dict[str, Any]] = []
        for query in gold_doc["queries"]:
            gold = set(query["relevant_pages"])
            unknown = gold - known_pages
            assert not unknown, f"gold query {query['id']} names missing pages: {sorted(unknown)}"

            bundle = await memory.retrieve_context(
                query_text=query["text"],
                user_id=_OWNER_ID,
                agent_id=AGENT_ID,
            )
            ranked = _ranked_pages(list(bundle.chunks), _CUTOFF)
            recall = _recall_at_k(ranked, gold)
            precision = _r_precision(ranked, gold)
            recalls.append(recall)
            precisions.append(precision)
            query_results.append(
                {
                    "id": query["id"],
                    "gold": sorted(gold),
                    "retrieved": ranked,
                    f"recall_at_{_CUTOFF}": round(recall, 4),
                    "r_precision": round(precision, 4),
                }
            )

        macro_recall = sum(recalls) / len(recalls)
        macro_precision = sum(precisions) / len(precisions)
        metrics = {
            "chat_model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            "embed_model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            "pages": len(corpus_pages),
            "queries": len(recalls),
            f"recall_at_{_CUTOFF}": round(macro_recall, 4),
            "r_precision": round(macro_precision, 4),
            "query_results": query_results,
        }
        print("RETRIEVAL_QUALITY=" + json.dumps(metrics, sort_keys=True))

        min_recall = float(os.getenv("E2E_MIN_RETRIEVAL_RECALL", "0.82"))
        min_precision = float(os.getenv("E2E_MIN_RETRIEVAL_R_PRECISION", "0.65"))
        assert macro_recall >= min_recall, metrics
        assert macro_precision >= min_precision, metrics
    finally:
        try:
            memory.shutdown()
        except Exception:
            pass
