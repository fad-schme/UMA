from __future__ import annotations

from datetime import datetime, timezone

from uma.core.retrieval.ranking import Ranker, fuse_candidates, rerank_candidates
from uma.types import Chunk, Fact


def test_fuse_candidates_sparse_precedes_and_dedupes_with_rrf() -> None:
    dense = [{"id": "a"}, {"id": "b"}]
    sparse = [{"id": "b"}, {"id": "c"}]
    out = fuse_candidates(dense=dense, sparse=sparse, strategy="rrf", rrf_k=60)
    # Overlap (b) must win. Remaining are ordered by RRF rank bonus deterministically.
    assert [x["id"] for x in out] == ["b", "a", "c"]


def test_rank_facts_reranks_by_query_overlap_without_dropping() -> None:
    now = datetime.now(timezone.utc)
    f_sushi = Fact(
        id="f1",
        subject="user",
        predicate="LIKES",
        object="sushi",
        created_at=now,
        updated_at=now,
        meta={"vector_score": 0.1},
        owner_type="user",
        owner_id="user:u1",
    )
    f_coffee = Fact(
        id="f2",
        subject="user",
        predicate="LIKES",
        object="coffee",
        created_at=now,
        updated_at=now,
        meta={"vector_score": 10.0},
        owner_type="user",
        owner_id="user:u1",
    )

    r = Ranker()
    ranked = r.rank_facts([f_coffee, f_sushi], query_text="sushi")
    assert [f.id for f in ranked][:2] == ["f1", "f2"]
    assert len(ranked) == 2
    assert "rerank_score" in (ranked[0].meta or {})
    assert "final_score" in (ranked[0].meta or {})


def test_rank_chunks_enforces_route_precedence() -> None:
    now = datetime.now(timezone.utc)
    evidence = Chunk(
        id="chunk_ev",
        doc_id="d1",
        text="evidence",
        page_range=(1, 1),
        position=1,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:u1",
        meta={"retrieval_route": "evidence", "retrieval_method": "vector", "vector_score": 0.0},
    )
    query_hit = Chunk(
        id="chunk_q",
        doc_id="d1",
        text="query hit",
        page_range=(1, 1),
        position=2,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:u1",
        meta={"retrieval_route": "query", "retrieval_method": "vector", "vector_score": 999.0},
    )
    r = Ranker()
    ranked = r.rank_chunks([query_hit, evidence], query_text="evidence")
    assert [c.id for c in ranked][:2] == ["chunk_ev", "chunk_q"]


def test_rerank_candidates_preserves_membership_and_is_deterministic() -> None:
    items = [
        {"id": "a", "owner_type": "agent", "owner_id": "agent:1", "text": "alpha", "meta": {"vector_score": 0.0}},
        {"id": "b", "owner_type": "agent", "owner_id": "agent:1", "text": "contains sushi", "meta": {"vector_score": 0.0}},
        {"id": "c", "owner_type": "user", "owner_id": "user:u1", "text": "sushi too", "meta": {"vector_score": 0.0}},
        {"id": "d", "owner_type": "user", "owner_id": "user:u1", "text": "delta", "meta": {"vector_score": 0.0}},
    ]
    out1 = rerank_candidates("sushi", items)
    out2 = rerank_candidates("sushi", items)
    assert [x["id"] for x in out1] == [x["id"] for x in out2]
    assert set(x["id"] for x in out1) == {"a", "b", "c", "d"}
    # Owner groups keep their relative order (agent group first, then user group),
    # but within each group, overlaps should bubble up.
    assert [x["id"] for x in out1][:2] == ["b", "a"]
    assert [x["id"] for x in out1][2:] == ["c", "d"]
