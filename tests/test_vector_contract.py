"""Vector contract, LanceDB, and low-level infra: score plumbing, payload shape, isolation, lexical search.

Covers vector adapter score plumbing end-to-end, minimal payload contract,
delete, LanceDB isolation, lexical termset determinism, chunk ID determinism,
neighbor expansion, PDF text normalization, and deduplication helpers.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uma.adapters.db.sqlite_adapter import SQLiteAdapter
from uma.adapters.vector.base import VectorIndex
from uma.adapters.vector.inmemory import InMemoryVectorIndex
from uma.adapters.vector.lancedb import LanceDBIndex
from uma.adapters.vector import qdrant as qdrant_module
from uma.api.memory import UMAMemory
from uma.common.initializers import providers as provider_initializers
from uma.common.dedupe import dedupe_by_id
from uma.common.types import Chunk
from uma.ingest.chunker import chunk_sections
from uma.ingest.normalizer import _clean_page_text, _drop_repeated_lines_across_pages
from uma.ingest.types import NormalizedSection
from uma.memory.chunk.core import ChunkCore
from uma.retrieve.rlm.entity_seed import extract_candidate_entities
from uma.retrieve.user_query_helper import build_query_term_set
from uma.stores.chunk_sql import ChunkSQLStore
import pytest
import yaml

# ── test_vector_scores_plumbed ──────────────────────────────────────────






def test_vector_index_query_returns_id_and_score() -> None:
    idx = InMemoryVectorIndex(dim=3)
    idx.upsert(
        ids=["a", "b"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        tenant_ids=["default", "default"],
        owner_types=["user", "user"],
        owner_ids=["user:u1", "user:u1"],
        extra_metadata=[{}, {}],
    )
    res = idx.query(
        vector=[1.0, 0.0, 0.0],
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        k=2,
    )
    assert res, "expected non-empty vector results"
    assert all(isinstance(t, tuple) and len(t) == 2 for t in res)
    assert all(isinstance(t[0], str) and isinstance(t[1], float) for t in res)


@pytest.mark.asyncio
async def test_chunk_core_preserves_vector_score(tmp_path) -> None:
    db = SQLiteAdapter(str(tmp_path / "uma_test_vector_scores.sqlite"))
    vec = InMemoryVectorIndex(dim=3)
    store = ChunkSQLStore(db_adapter=db, vector_index=vec)
    core = ChunkCore(store)

    now = datetime.now(timezone.utc)
    c1 = Chunk(
        id="chunk_1",
        doc_id="doc1",
        text="hello.",
        page_range=(1, 1),
        position=1,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:u1",
        meta={},
    )
    c2 = Chunk(
        id="chunk_2",
        doc_id="doc1",
        text="world.",
        page_range=(1, 1),
        position=2,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:u1",
        meta={},
    )

    await store.upsert_chunk(c1, embedding=[1.0, 0.0, 0.0])
    await store.upsert_chunk(c2, embedding=[0.0, 1.0, 0.0])

    out = await core.search_chunks(query_embedding=[1.0, 0.0, 0.0], owner_type="user", owner_id="user:u1", k=2)
    assert [c.id for c in out][:2] == ["chunk_1", "chunk_2"]
    assert "vector_score" in (out[0].meta or {})
    assert float(out[0].meta["vector_score"]) >= float(out[1].meta.get("vector_score", -1.0))


# ── test_vector_payload_minimal ──────────────────────────────────────────






class _SpyVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self.last_ids = None
        self.last_vectors = None
        self.last_tenant_ids = None
        self.last_owner_types = None
        self.last_owner_ids = None
        self.last_extra_metadata = None

    def upsert(self, ids, vectors, *, tenant_ids, owner_types, owner_ids, extra_metadata=None) -> None:
        self.last_ids = list(ids or [])
        self.last_vectors = list(vectors or [])
        self.last_tenant_ids = list(tenant_ids or [])
        self.last_owner_types = list(owner_types or [])
        self.last_owner_ids = list(owner_ids or [])
        self.last_extra_metadata = list(extra_metadata or [])

    def query(self, vector, *, tenant_id, owner_type, owner_id, k=10, extra_filters=None):
        return []

    def delete(self, ids) -> None:
        return None


@pytest.mark.asyncio
async def test_chunk_vector_payload_is_minimal_and_excludes_text(tmp_path) -> None:
    db = SQLiteAdapter(str(tmp_path / "uma_test_payload.sqlite"))
    spy = _SpyVectorIndex()
    store = ChunkSQLStore(db_adapter=db, vector_index=spy)

    now = datetime.now(timezone.utc)
    chunk = Chunk(
        id="chunk_1",
        doc_id="doc_1",
        text="This is the canonical chunk text stored in SQL.",
        page_range=(3, 4),
        position=7,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:u1",
        meta={},
    )

    await store.upsert_chunk(chunk, embedding=[0.0, 0.0, 0.0])

    assert spy.last_ids == ["chunk_1"]
    assert spy.last_tenant_ids == ["default"]
    assert spy.last_owner_types == ["user"]
    assert spy.last_owner_ids == ["user:u1"]
    assert spy.last_extra_metadata and isinstance(spy.last_extra_metadata[0], dict)
    meta = spy.last_extra_metadata[0]

    # Minimal, filterable fields only.
    assert meta.get("doc_id") == "doc_1"
    assert meta.get("kb_lane") == "raw"
    assert meta.get("position") == 7
    assert meta.get("page_start") == 3
    assert meta.get("page_end") == 4

    # Never duplicate full chunk text in vector payload.
    assert "text" not in meta
    assert "__text" not in meta


# ── test_vector_index_delete ──────────────────────────────────────────



def test_inmemory_delete_removes_vectors():
    idx = InMemoryVectorIndex(dim=3)
    idx.upsert(
        ids=["a", "b"],
        vectors=[[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]],
        tenant_ids=["default", "default"],
        owner_types=["user", "user"],
        owner_ids=["user:u1", "user:u1"],
        extra_metadata=[{}, {}],
    )

    assert "a" in idx._vectors
    idx.delete(["a"])
    assert "a" not in idx._vectors
    assert "b" in idx._vectors


# ── test_lite_lancedb ──────────────────────────────────────────






def test_lancedb_index_upsert_query_and_filters(tmp_path) -> None:
    index = LanceDBIndex(dim=3, path=str(tmp_path / "vectors"), table_name="test_vectors")
    index.upsert(
        ids=["doc-a", "doc-b"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        tenant_ids=["default", "default"],
        owner_types=["user", "workspace"],
        owner_ids=["user:u1", "ws:1"],
        extra_metadata=[
            {"kb_lane": "raw"},
            {"kb_lane": "raw"},
        ],
    )

    results = index.query([1.0, 0.0, 0.0], tenant_id="default", owner_type="user", owner_id="user:u1", k=2)
    assert results
    assert results[0][0] == "doc-a"
    assert isinstance(results[0][1], float)

    filtered = index.query(
        [1.0, 0.0, 0.0],
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
        k=2,
        extra_filters={"kb_lane": "raw"},
    )
    assert filtered == [results[0]]

    table = index._open_table()
    assert table is not None
    rows = table.search([1.0, 0.0, 0.0]).limit(2).to_list()
    stored = LanceDBIndex._parse_metadata(rows[0]["metadata_json"])
    assert rows[0]["tenant_id"] == "default"
    assert rows[0]["owner_type"] == "user"
    assert rows[0]["owner_id"] == "user:u1"
    assert stored["kb_lane"] == "raw"

    index.delete(["doc-a"])
    remaining = index.query([1.0, 0.0, 0.0], tenant_id="default", owner_type="user", owner_id="user:u1", k=2)
    assert all(item_id != "doc-a" for item_id, _ in remaining)


class _FakeQdrantClient:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.created = []
        self.upserts = []
        self.queries = []
        self.deletes = []

    def collection_exists(self, collection: str) -> bool:
        return False

    def create_collection(self, **kwargs) -> None:
        self.created.append(kwargs)

    def upsert(self, collection: str, **kwargs) -> None:
        self.upserts.append((collection, kwargs))

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-id",
                    payload={"uma_id": "fact-a"},
                    score=0.91,
                )
            ]
        )

    def delete(self, **kwargs) -> None:
        self.deletes.append(kwargs)


class _FakeQdrantModels:
    class Distance:
        COSINE = "cosine"
        DOT = "dot"
        EUCLID = "euclid"

    @staticmethod
    def VectorParams(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def PointStruct(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def MatchValue(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def FieldCondition(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def Filter(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def PointIdsList(**kwargs):
        return SimpleNamespace(**kwargs)


def _qdrant_index(monkeypatch) -> qdrant_module.QdrantIndex:
    monkeypatch.setattr(qdrant_module, "QdrantClient", _FakeQdrantClient)
    monkeypatch.setattr(qdrant_module, "qmodels", _FakeQdrantModels)
    return qdrant_module.QdrantIndex(
        3,
        url="http://qdrant.test",
        table_name="vectors_semantic",
    )


def test_qdrant_adapter_enforces_scope_in_native_payload_and_filter(
    monkeypatch,
) -> None:
    index = _qdrant_index(monkeypatch)

    index.upsert(
        ["fact-a"],
        [[1.0, 0.0, 0.0]],
        tenant_ids=["tenant-a"],
        owner_types=["user"],
        owner_ids=["user-a"],
        extra_metadata=[{"kb_lane": "semantic"}],
    )
    point = index._client.upserts[0][1]["points"][0]
    assert point.payload == {
        "uma_id": "fact-a",
        "tenant_id": "tenant-a",
        "owner_type": "user",
        "owner_id": "user-a",
        "kb_lane": "semantic",
    }

    assert index.query(
        [1.0, 0.0, 0.0],
        tenant_id="tenant-a",
        owner_type="user",
        owner_id="user-a",
        extra_filters={"kb_lane": "semantic"},
    ) == [("fact-a", 0.91)]
    conditions = index._client.queries[0]["query_filter"].must
    assert {
        condition.key: condition.match.value
        for condition in conditions
    } == {
        "tenant_id": "tenant-a",
        "owner_type": "user",
        "owner_id": "user-a",
        "kb_lane": "semantic",
    }


def test_qdrant_adapter_validates_complete_batch_before_write(
    monkeypatch,
) -> None:
    index = _qdrant_index(monkeypatch)

    with pytest.raises(ValueError, match="reserved isolation key"):
        index.upsert(
            ["fact-a"],
            [[1.0, 0.0, 0.0]],
            tenant_ids=["tenant-a"],
            owner_types=["user"],
            owner_ids=["user-a"],
            extra_metadata=[{"tenant_id": "other"}],
        )

    assert index._client.upserts == []


@pytest.mark.asyncio
async def test_lite_config_initializes_sqlite_and_lancedb_without_graph_services(tmp_path, monkeypatch) -> None:
    config_data = yaml.safe_load(Path("config/uma.yaml").read_text(encoding="utf-8"))
    config_data["storage"]["db_root"] = str(tmp_path / "db")
    config_data["storage"]["vector_config"]["path"] = str(tmp_path / "vectors")
    config_data["embedding"]["dimension"] = 64
    config_data.setdefault("features", {})["load"] = []

    expected_llm_provider = config_data["llms"]["uma"]["provider"]
    expected_embedding_provider = config_data["embedding"]["provider"]
    resolved_providers = {"llm": [], "embedding": []}

    def get_test_llm_factory(provider):
        resolved_providers["llm"].append(provider)
        return lambda cfg: SimpleNamespace(provider_name=cfg.provider, model=cfg.model)

    def get_test_embedder_factory(provider):
        resolved_providers["embedding"].append(provider)
        return lambda cfg: SimpleNamespace(
            provider_name=cfg.provider,
            model=cfg.model,
            dimension=cfg.dimension,
        )

    monkeypatch.setattr(provider_initializers, "get_llm_factory", get_test_llm_factory)
    monkeypatch.setattr(provider_initializers, "get_embedder_factory", get_test_embedder_factory)

    config_path = tmp_path / "uma_test.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    memory = UMAMemory.from_yaml(str(config_path))
    try:
        assert memory.raw_config.profile == "lite"
        assert isinstance(memory._stores["episodic"].vector_index, LanceDBIndex)
        assert isinstance(memory._stores["semantic"].vector_index, LanceDBIndex)
        assert isinstance(memory._stores["procedural"].vector_index, LanceDBIndex)
        assert isinstance(memory._stores["chunk"].vector_index, LanceDBIndex)
        assert memory.graph_core is None
        assert memory.llm.provider_name == expected_llm_provider
        assert memory.embedder.provider_name == expected_embedding_provider
        assert expected_llm_provider in resolved_providers["llm"]
        assert expected_embedding_provider in resolved_providers["embedding"]

        # Prove the real runtime path can write to the embedded vector backend.
        vectors_root = tmp_path / "vectors"
        assert vectors_root.exists()
    finally:
        memory.shutdown()


# ── test_lexical_termset_determinism ──────────────────────────────────────────




def test_build_query_term_set_is_deterministic() -> None:
    q = 'How do I "reset MFA" for AWS IAM users, and why does it fail? Explain the details.'
    a = build_query_term_set(q, max_terms=10, max_phrases=4)
    b = build_query_term_set(q, max_terms=10, max_phrases=4)
    assert a == b


def test_build_query_term_set_filters_noise() -> None:
    q = "What is 123 456? how to explain a guide to the the the."
    ts = build_query_term_set(q, max_terms=10, max_phrases=4)
    assert "123" not in ts.terms
    assert "456" not in ts.terms
    assert "how" not in ts.terms
    assert "explain" not in ts.terms
    assert "guide" not in ts.terms



# ── test_chunk_ids_deterministic ──────────────────────────────────────────



def test_chunk_ids_are_deterministic():
    sections = [
        NormalizedSection(section_id="s1", doc_id="doc1", text="hello world " * 200, page_range=(1, 1)),
        NormalizedSection(section_id="s2", doc_id="doc1", text="another section " * 200, page_range=(2, 2)),
    ]

    chunks_a = chunk_sections(sections, chunk_size_tokens=50, overlap_tokens=10)
    chunks_b = chunk_sections(sections, chunk_size_tokens=50, overlap_tokens=10)

    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert [c.position for c in chunks_a] == [c.position for c in chunks_b]


def test_chunk_ids_do_not_depend_on_section_iteration_order():
    sections_a = [
        NormalizedSection(section_id="s1", doc_id="doc1", text="hello world " * 200, page_range=(1, 1)),
        NormalizedSection(section_id="s2", doc_id="doc1", text="another section " * 200, page_range=(2, 2)),
    ]
    sections_b = list(reversed(sections_a))

    chunks_a = chunk_sections(sections_a, chunk_size_tokens=50, overlap_tokens=10)
    chunks_b = chunk_sections(sections_b, chunk_size_tokens=50, overlap_tokens=10)

    # IDs should be stable even if upstream section order changes.
    assert sorted([c.chunk_id for c in chunks_a]) == sorted([c.chunk_id for c in chunks_b])


# ── test_chunk_neighbor_expansion ──────────────────────────────────────────






def _mk(doc_id: str, pos: int, *, owner_type: str, owner_id: str) -> Chunk:
    now = datetime.now(timezone.utc)
    return Chunk(
        id=f"chunk_{doc_id}_{pos}",
        doc_id=doc_id,
        text=f"text {doc_id} {pos}.",
        page_range=(1, 1),
        position=pos,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type=owner_type,
        owner_id=owner_id,
        meta={},
    )


@pytest.mark.asyncio
async def test_expand_neighbors_single_anchor_window_1(uma_memory) -> None:
    memory = uma_memory
    owner_type = "user"
    owner_id = "user:u1"

    chunks = [_mk("d1", p, owner_type=owner_type, owner_id=owner_id) for p in range(1, 11)]
    embs = await memory.embedder.embed([c.text for c in chunks])
    for c, e in zip(chunks, embs):
        await memory.chunk_core.upsert_chunk(c, e)

    anchors = [_mk("d1", 5, owner_type=owner_type, owner_id=owner_id)]
    expanded = await memory.chunk_core.expand_neighbors(
        owner_type=owner_type,
        owner_id=owner_id,
        anchors=anchors,
        window=1,
        max_total=24,
    )
    assert [c.position for c in expanded] == [5, 4, 6]


@pytest.mark.asyncio
async def test_expand_neighbors_overlapping_anchors_dedupes(uma_memory) -> None:
    memory = uma_memory
    owner_type = "user"
    owner_id = "user:u1"

    chunks = [_mk("d1", p, owner_type=owner_type, owner_id=owner_id) for p in range(1, 11)]
    embs = await memory.embedder.embed([c.text for c in chunks])
    for c, e in zip(chunks, embs):
        await memory.chunk_core.upsert_chunk(c, e)

    anchors = [
        _mk("d1", 5, owner_type=owner_type, owner_id=owner_id),
        _mk("d1", 6, owner_type=owner_type, owner_id=owner_id),
    ]
    expanded = await memory.chunk_core.expand_neighbors(
        owner_type=owner_type,
        owner_id=owner_id,
        anchors=anchors,
        window=1,
        max_total=24,
    )
    assert [c.position for c in expanded] == [5, 4, 6, 7]


@pytest.mark.asyncio
async def test_expand_neighbors_enforces_max_total(uma_memory) -> None:
    memory = uma_memory
    owner_type = "user"
    owner_id = "user:u1"

    chunks = [_mk("d1", p, owner_type=owner_type, owner_id=owner_id) for p in range(1, 100)]
    embs = await memory.embedder.embed([c.text for c in chunks[:32]])
    # Keep this fast: only upsert a prefix large enough to cover anchors + window.
    for c, e in zip(chunks[:32], embs):
        await memory.chunk_core.upsert_chunk(c, e)

    anchors = [
        _mk("d1", 10, owner_type=owner_type, owner_id=owner_id),
        _mk("d1", 20, owner_type=owner_type, owner_id=owner_id),
        _mk("d1", 30, owner_type=owner_type, owner_id=owner_id),
    ]
    expanded = await memory.chunk_core.expand_neighbors(
        owner_type=owner_type,
        owner_id=owner_id,
        anchors=anchors,
        window=3,
        max_total=5,
    )
    assert len(expanded) == 5



# ── test_pdf_text_normalizer_reflow ──────────────────────────────────────────




def test_clean_page_text_dehyphenates_and_reflows_soft_wraps() -> None:
    raw = (
        "This doc describes inter-\n"
        "nal VLANs and internal ACLs\n"
        "tightly controlled (only restore processes can read them).\n"
        "\n"
        "Second paragraph starts here.\n"
    )
    cleaned = _clean_page_text(raw)
    assert "internal VLANs" in cleaned
    # Soft wrap should reflow line break into a space.
    assert "ACLs tightly controlled" in cleaned
    # Blank line boundary preserved (paragraph split still possible).
    assert "\n\n" in cleaned


def test_drop_repeated_lines_across_pages_removes_headers() -> None:
    pages = [
        "CONFIDENTIAL\nBody A line 1\nBody A line 2\n1\n",
        "CONFIDENTIAL\nBody B line 1\nBody B line 2\n2\n",
        "CONFIDENTIAL\nBody C line 1\nBody C line 2\n3\n",
    ]
    out = _drop_repeated_lines_across_pages(pages, min_repeats=3)
    assert len(out) == 3
    assert all("CONFIDENTIAL" not in p for p in out)



# ── test_dedupe_by_id_helper ──────────────────────────────────────────



class Obj:
    def __init__(self, id):
        self.id = id


def test_dedupe_by_id_handles_dicts_and_objects():
    items = [{"id": "a"}, Obj("a"), {"id": "b"}, Obj("b"), {"id": "a"}]
    out = dedupe_by_id(items)
    assert [getattr(x, "id", x.get("id")) for x in out] == ["a", "b"]



# ── test_entity_seed ──────────────────────────────────────────




def test_extract_candidate_entities_includes_acronyms_and_is_bounded() -> None:
    q = "How do IAM and VPC integrate with KMS for TLS?"
    out = extract_candidate_entities(q, facts=[], chunks=[], limit=3)
    assert out == ["IAM", "VPC", "KMS"]


def test_extract_candidate_entities_dedupes_case_insensitive() -> None:
    q = "IAM iam VPC vpc"
    out = extract_candidate_entities(q, facts=[], chunks=[], limit=10)
    assert out[:2] == ["IAM", "VPC"]
