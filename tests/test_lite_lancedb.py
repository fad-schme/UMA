from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from uma.adapters.llm import openai_compatible as shared_module
from uma.adapters.vector.lancedb import LanceDBIndex
from uma.api.memory import UMAMemory


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
        input_items = list(kwargs.get("input") or [])
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.0] * 64) for _ in input_items]
        )


def test_lancedb_index_upsert_query_and_filters(tmp_path) -> None:
    index = LanceDBIndex(dim=3, path=str(tmp_path / "vectors"), table_name="test_vectors")
    index.upsert(
        ids=["doc-a", "doc-b"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        metadata=[
            {"owner_type": "user", "owner_id": "user:u1", "kb_lane": "raw"},
            {"owner_type": "workspace", "owner_id": "ws:1", "kb_lane": "raw"},
        ],
    )

    results = index.query([1.0, 0.0, 0.0], k=2)
    assert results
    assert results[0][0] == "doc-a"
    assert isinstance(results[0][1], float)

    filtered = index.query([1.0, 0.0, 0.0], k=2, filters={"owner_type": "user"})
    assert filtered == [results[0]]

    table = index._open_table()
    assert table is not None
    rows = table.search([1.0, 0.0, 0.0]).limit(2).to_list()
    stored = LanceDBIndex._parse_metadata(rows[0]["metadata_json"])
    assert stored["owner_type"] == "user"
    assert stored["owner_id"] == "user:u1"
    assert stored["kb_lane"] == "raw"

    index.delete(["doc-a"])
    remaining = index.query([1.0, 0.0, 0.0], k=2)
    assert all(item_id != "doc-a" for item_id, _ in remaining)


@pytest.mark.asyncio
async def test_lite_config_initializes_sqlite_and_lancedb_without_graph_services(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shared_module, "AsyncOpenAI", _FakeAsyncOpenAI)

    config_data = yaml.safe_load(Path("config/uma_lite.yaml").read_text(encoding="utf-8"))
    config_data["storage"]["db_root"] = str(tmp_path / "db")
    config_data["storage"]["vector_config"]["path"] = str(tmp_path / "vectors")
    config_data["embedding"] = {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "dimension": 64,
        "config": {"host": "http://localhost:11434"},
    }
    config_data["llms"] = {
        "agent": {
            "provider": "ollama",
            "model": "llama3",
            "config": {"host": "http://localhost:11434"},
        },
        "uma": {
            "provider": "ollama",
            "model": "llama3",
            "config": {"host": "http://localhost:11434"},
        },
    }
    config_data["consolidation"]["enabled"] = False
    config_data["features"]["load"] = []

    config_path = tmp_path / "uma_lite_test.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    memory = UMAMemory.from_yaml(str(config_path))
    try:
        assert memory.raw_config.profile == "lite"
        assert isinstance(memory._stores["episodic"].vector_index, LanceDBIndex)
        assert isinstance(memory._stores["semantic"].vector_index, LanceDBIndex)
        assert isinstance(memory._stores["procedural"].vector_index, LanceDBIndex)
        assert isinstance(memory._stores["chunk"].vector_index, LanceDBIndex)
        assert memory.graph_core is None

        # Prove the real runtime path can write to the embedded vector backend.
        vectors_root = tmp_path / "vectors"
        assert vectors_root.exists()
    finally:
        memory.shutdown()
