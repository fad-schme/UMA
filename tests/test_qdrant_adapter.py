from __future__ import annotations

import importlib.metadata as md
import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import uma.adapters.vector.qdrant as qdrant_module
from uma.adapters.vector.base import VectorIndex
from uma.adapters.vector.qdrant import QdrantIndex
from uma.common.config import UMAConfig


def _has_qdrant_client() -> bool:
    try:
        md.version("qdrant-client")
    except md.PackageNotFoundError:
        return False
    return True


@dataclass
class _FakeMatchValue:
    value: object


@dataclass
class _FakeFieldCondition:
    key: str
    match: _FakeMatchValue


@dataclass
class _FakeFilter:
    must: list[_FakeFieldCondition]


@dataclass
class _FakeVectorParams:
    size: int
    distance: object


@dataclass
class _FakePointStruct:
    id: str
    vector: list[float]
    payload: dict


@dataclass
class _FakePointIdsList:
    points: list[str]


class _FakeDistance:
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"


class _FakeQModels:
    Distance = _FakeDistance
    MatchValue = _FakeMatchValue
    FieldCondition = _FakeFieldCondition
    Filter = _FakeFilter
    VectorParams = _FakeVectorParams
    PointStruct = _FakePointStruct
    PointIdsList = _FakePointIdsList


class _FakeClient:
    def __init__(self, *, url: str, api_key: str | None = None) -> None:
        self.url = url
        self.api_key = api_key
        self.created = []
        self.upserts = []
        self.deletes = []
        self.queries = []

    def collection_exists(self, name: str) -> bool:
        return False

    def create_collection(self, **kwargs) -> None:
        self.created.append(kwargs)

    def upsert(self, collection_name: str, *, points, wait: bool) -> None:
        self.upserts.append((collection_name, points, wait))

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="ignored",
                    payload={"uma_id": "doc-a", "owner": "u1"},
                    score=0.75,
                )
            ]
        )

    def delete(self, *, collection_name: str, points_selector, wait: bool) -> None:
        self.deletes.append((collection_name, points_selector, wait))


def test_qdrant_module_import_does_not_require_network() -> None:
    assert QdrantIndex is not None


def test_qdrant_adapter_imports_and_matches_vector_contract() -> None:
    assert issubclass(QdrantIndex, VectorIndex)
    assert callable(QdrantIndex.upsert)
    assert callable(QdrantIndex.query)
    assert callable(QdrantIndex.delete)

    signature = inspect.signature(QdrantIndex.__init__)
    assert "collection" in signature.parameters
    assert "url" in signature.parameters


def test_qdrant_adapter_fails_clearly_without_client_dependency() -> None:
    if _has_qdrant_client():
        pytest.skip("qdrant-client is installed; this guard only covers the no-client path.")

    with pytest.raises(RuntimeError, match="qdrant-client is not installed"):
        QdrantIndex(dim=3, url="http://qdrant:6333", collection="uma_vectors")


def test_point_id_is_deterministic_for_non_uuid_strings() -> None:
    first = QdrantIndex._point_id("doc-a")
    second = QdrantIndex._point_id("doc-a")
    third = QdrantIndex._point_id("doc-b")

    assert first == second
    assert first != third


def test_build_filter_creates_simple_exact_match_and_filter(monkeypatch) -> None:
    monkeypatch.setattr(qdrant_module, "qmodels", _FakeQModels)

    query_filter = QdrantIndex._build_filter({"owner": "u1", "lane": "semantic"})

    assert query_filter is not None
    assert [item.key for item in query_filter.must] == ["owner", "lane"]
    assert [item.match.value for item in query_filter.must] == ["u1", "semantic"]


def test_qdrant_index_uses_simple_dense_client_calls(monkeypatch) -> None:
    monkeypatch.setattr(qdrant_module, "qmodels", _FakeQModels)
    monkeypatch.setattr(qdrant_module, "QdrantClient", _FakeClient)

    index = QdrantIndex(dim=3, url="http://qdrant:6333", collection="uma_vectors")
    assert isinstance(index._client, _FakeClient)
    assert index._client.created

    index.upsert(
        ids=["doc-a"],
        vectors=[[1, 0, 0]],
        metadata=[{"owner": "u1", "lane": "semantic"}],
    )
    collection_name, points, wait = index._client.upserts[0]
    assert collection_name == "uma_vectors"
    assert wait is True
    assert points[0].payload["uma_id"] == "doc-a"
    assert points[0].vector == [1.0, 0.0, 0.0]

    results = index.query([1, 0, 0], k=2, filters={"owner": "u1"})
    assert results == [("doc-a", 0.75)]
    assert index._client.queries[0]["limit"] == 2
    assert index._client.queries[0]["query_filter"] is not None

    index.delete(["doc-a"])
    delete_collection, selector, delete_wait = index._client.deletes[0]
    assert delete_collection == "uma_vectors"
    assert delete_wait is True
    assert selector.points == [QdrantIndex._point_id("doc-a")]


def test_cont_config_points_to_packaged_qdrant_backend() -> None:
    cfg = UMAConfig.load_yaml("config/uma_cont.yaml")
    assert cfg.storage.vector_backend == "uma.adapters.vector.qdrant:QdrantIndex"
    assert cfg.storage.vector_config["url"] == "http://qdrant:6333"
