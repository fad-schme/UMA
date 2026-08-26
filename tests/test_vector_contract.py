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
from uma.api.memory import UMAMemory
from uma.common.initializers import providers as provider_initializers
from uma.common.dedupe import dedupe_by_id
from uma.common.types import Chunk
from uma.ingest.chunker import chunk_sections
from uma.ingest.normalizer import _clean_page_text, _drop_repeated_lines_across_pages
from uma.ingest.types import NormalizedSection
from uma.memory.chunk.core import ChunkCore
from uma.common.text import build_query_term_set
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


@pytest.mark.asyncio
async def test_chunk_core_mmr_selection_prefers_diversity_when_enabled(tmp_path) -> None:
    """Regression for retrieval-ranking-gap ticket 07 (end-to-end through
    ChunkCore.search_chunks, not just the pure mmr_select function): with
    mmr_enabled, a distinct-but-lower-scoring candidate should win a slot
    over a near-duplicate of an already-selected higher-scoring one. Plain
    top-2-by-score would pick a+b (both point toward the query, redundant);
    MMR should pick a+c (diverse)."""
    db = SQLiteAdapter(str(tmp_path / "uma_test_mmr.sqlite"))
    vec = InMemoryVectorIndex(dim=2)
    store = ChunkSQLStore(db_adapter=db, vector_index=vec)
    memory = SimpleNamespace(retrieval_cfg=SimpleNamespace(mmr_enabled=True, mmr_lambda=0.5))
    core = ChunkCore(store, memory=memory)

    now = datetime.now(timezone.utc)

    def make(cid: str, text: str, position: int) -> Chunk:
        return Chunk(
            id=cid, doc_id="doc1", text=text, page_range=(1, 1), position=position,
            source_path="/tmp/x", source_hash="h", created_at=now, updated_at=now,
            owner_type="user", owner_id="user:u1", meta={},
        )

    await store.upsert_chunk(make("a", "alpha text", 1), embedding=[1.0, 0.0])
    await store.upsert_chunk(make("b", "alpha text near-duplicate", 2), embedding=[0.99, 0.01])
    await store.upsert_chunk(make("c", "distinct fact", 3), embedding=[0.0, 1.0])

    out = await core.search_chunks(query_embedding=[1.0, 0.0], owner_type="user", owner_id="user:u1", k=2)
    assert {c.id for c in out} == {"a", "c"}


@pytest.mark.asyncio
async def test_chunk_core_select_chunks_falls_back_to_top_k_when_vectors_incomplete() -> None:
    """A backend that can't provide vectors for every candidate must not
    produce a partial/silent MMR pass -- falls back to plain top-k order."""

    class _PartialVectorIndex:
        def get_vectors(self, ids, *, tenant_id, owner_type, owner_id):
            return {}  # simulates a backend with no get_vectors support

    class _Store:
        vector_index = _PartialVectorIndex()

    memory = SimpleNamespace(retrieval_cfg=SimpleNamespace(mmr_enabled=True, mmr_lambda=0.5))
    core = ChunkCore(_Store(), memory=memory)

    now = datetime.now(timezone.utc)

    def make(cid: str, score: float) -> Chunk:
        return Chunk(
            id=cid, doc_id="doc1", text=cid, page_range=(1, 1), position=0,
            source_path="/tmp/x", source_hash="h", created_at=now, updated_at=now,
            owner_type="user", owner_id="user:u1", meta={"vector_score": score},
        )

    candidates = [make("a", 0.9), make("b", 0.8), make("c", 0.5)]
    out = await core._select_chunks(
        candidates, 2, query_text=None, tenant_id="default", owner_type="user", owner_id="user:u1"
    )
    assert [c.id for c in out] == ["a", "b"]


@pytest.mark.asyncio
async def test_chunk_core_select_chunks_uses_id_membership_not_count_for_vector_availability() -> None:
    """Review finding: the original fallback check compared len(vectors) to
    len(candidates), which is wrong when candidates contains a duplicate id
    -- vectors (a dict) can never have more entries than unique ids, so a
    duplicate id alone would trip a spurious fallback even though every id
    actually resolved. Must check id membership, not count."""

    class _AllVectorsIndex:
        def get_vectors(self, ids, *, tenant_id, owner_type, owner_id):
            return {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}

    class _Store:
        vector_index = _AllVectorsIndex()

    memory = SimpleNamespace(retrieval_cfg=SimpleNamespace(mmr_enabled=True, mmr_lambda=0.5))
    core = ChunkCore(_Store(), memory=memory)

    now = datetime.now(timezone.utc)

    def make(cid: str, score: float) -> Chunk:
        return Chunk(
            id=cid, doc_id="doc1", text=cid, page_range=(1, 1), position=0,
            source_path="/tmp/x", source_hash="h", created_at=now, updated_at=now,
            owner_type="user", owner_id="user:u1", meta={"vector_score": score},
        )

    # "a" appears twice: len(candidates)=4 > len(vectors)=3 unique ids, but
    # every candidate's id is actually present in vectors -- MMR should run,
    # not fall back. The buggy count-based check would return
    # candidates[:2] verbatim (both "a" duplicates); the fixed id-based
    # check lets MMR consider diversity instead.
    candidates = [make("a", 0.9), make("a", 0.9), make("b", 0.8), make("c", 0.5)]
    out = await core._select_chunks(
        candidates, 2, query_text=None, tenant_id="default", owner_type="user", owner_id="user:u1"
    )
    assert out != candidates[:2]


@pytest.mark.asyncio
async def test_chunk_core_select_chunks_disabled_by_default() -> None:
    """mmr_enabled defaults to False -- plain top-k, no vector_index calls."""

    class _ExplodingVectorIndex:
        def get_vectors(self, ids, *, tenant_id, owner_type, owner_id):
            raise AssertionError("get_vectors must not be called when mmr_enabled is False")

    class _Store:
        vector_index = _ExplodingVectorIndex()

    memory = SimpleNamespace(retrieval_cfg=SimpleNamespace(mmr_enabled=False, mmr_lambda=0.5))
    core = ChunkCore(_Store(), memory=memory)

    now = datetime.now(timezone.utc)

    def make(cid: str, score: float) -> Chunk:
        return Chunk(
            id=cid, doc_id="doc1", text=cid, page_range=(1, 1), position=0,
            source_path="/tmp/x", source_hash="h", created_at=now, updated_at=now,
            owner_type="user", owner_id="user:u1", meta={"vector_score": score},
        )

    candidates = [make("a", 0.9), make("b", 0.8), make("c", 0.5)]
    out = await core._select_chunks(
        candidates, 2, query_text=None, tenant_id="default", owner_type="user", owner_id="user:u1"
    )
    assert [c.id for c in out] == ["a", "b"]


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


def test_inmemory_get_vectors_returns_scoped_vectors():
    """Regression for retrieval-ranking-gap ticket 07 (MMR chunk selection):
    get_vectors must return only ids within the given isolation scope, and
    must not invent an entry for an unknown id."""
    idx = InMemoryVectorIndex(dim=3)
    idx.upsert(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        tenant_ids=["default", "default", "default"],
        owner_types=["user", "user", "user"],
        owner_ids=["user:u1", "user:u1", "user:u2"],
        extra_metadata=[{}, {}, {}],
    )
    out = idx.get_vectors(["a", "b", "c", "missing"], tenant_id="default", owner_type="user", owner_id="user:u1")
    assert out == {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]}


def test_inmemory_get_vectors_rejects_blank_isolation_args():
    """Matches query()'s own validation -- a blank scope arg should raise,
    not silently return an empty/partial result (review finding: get_vectors
    initially skipped this check while query() had it)."""
    idx = InMemoryVectorIndex(dim=3)
    with pytest.raises(ValueError):
        idx.get_vectors(["a"], tenant_id="", owner_type="user", owner_id="user:u1")


def test_vector_index_base_get_vectors_defaults_to_empty():
    """The ABC default (no override) must degrade to 'unavailable', never raise."""
    spy = _SpyVectorIndex()
    assert spy.get_vectors(["a", "b"], tenant_id="default", owner_type="user", owner_id="user:u1") == {}


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


def test_lancedb_get_vectors_returns_scoped_vectors(tmp_path) -> None:
    """Regression for retrieval-ranking-gap ticket 07: the vector column is
    already fetched by the underlying table read (see LanceDBIndex.query's
    docstring) -- get_vectors reads it back out, scoped, without re-embedding."""
    index = LanceDBIndex(dim=3, path=str(tmp_path / "vectors"), table_name="test_get_vectors")
    index.upsert(
        ids=["doc-a", "doc-b"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        tenant_ids=["default", "default"],
        owner_types=["user", "workspace"],
        owner_ids=["user:u1", "ws:1"],
        extra_metadata=[{"kb_lane": "raw"}, {"kb_lane": "raw"}],
    )

    out = index.get_vectors(["doc-a", "doc-b", "missing"], tenant_id="default", owner_type="user", owner_id="user:u1")
    assert list(out.keys()) == ["doc-a"]
    assert out["doc-a"] == pytest.approx([1.0, 0.0, 0.0])


def test_lancedb_get_vectors_rejects_blank_isolation_args(tmp_path) -> None:
    """Matches query()'s own validation (review finding: get_vectors
    initially skipped this check while query() had it)."""
    index = LanceDBIndex(dim=3, path=str(tmp_path / "vectors"), table_name="test_get_vectors_validation")
    with pytest.raises(ValueError):
        index.get_vectors(["a"], tenant_id="", owner_type="user", owner_id="user:u1")


def test_faiss_get_vectors_reconstructs_scoped_vectors() -> None:
    """Regression for retrieval-ranking-gap ticket 07: IndexFlatIP stores
    vectors verbatim, so reconstruct() is a cheap in-memory lookup, not a
    re-embed. Skipped when the optional faiss-cpu/faiss-gpu extra isn't
    installed."""
    pytest.importorskip("faiss")
    from uma.adapters.vector.faiss_adapter import FaissIndex

    index = FaissIndex(dim=3)
    index.upsert(
        ids=["doc-a", "doc-b"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        tenant_ids=["default", "default"],
        owner_types=["user", "workspace"],
        owner_ids=["user:u1", "ws:1"],
        extra_metadata=[{}, {}],
    )

    out = index.get_vectors(["doc-a", "doc-b", "missing"], tenant_id="default", owner_type="user", owner_id="user:u1")
    assert list(out.keys()) == ["doc-a"]
    assert out["doc-a"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-5)


def test_faiss_get_vectors_rejects_blank_isolation_args() -> None:
    """Matches query()'s own validation (review finding: get_vectors
    initially skipped this check while query() had it)."""
    pytest.importorskip("faiss")
    from uma.adapters.vector.faiss_adapter import FaissIndex

    index = FaissIndex(dim=3)
    with pytest.raises(ValueError):
        index.get_vectors(["a"], tenant_id="", owner_type="user", owner_id="user:u1")


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


# ── test_entity_extraction ──────────────────────────────────────────


def test_query_term_set_entities_includes_acronyms_and_is_bounded() -> None:
    q = "How do IAM and VPC integrate with KMS for TLS?"
    out = build_query_term_set(q).entities
    assert out[:3] == ["iam", "vpc", "kms"]


def test_query_term_set_entities_dedupes_case_insensitive() -> None:
    q = "IAM iam VPC vpc"
    out = build_query_term_set(q).entities
    assert out[:2] == ["iam", "vpc"]
