from __future__ import annotations

from datetime import datetime, timezone

from uma.core.retrieval.policy import RetrievalPolicy
from uma.core.retrieval.selector import MemorySelector
from uma.types import Chunk


def test_debug_scores_attaches_score_card_to_chunks() -> None:
    sel = MemorySelector(
        max_episodes=1,
        max_facts=1,
        max_chunks=3,
        max_skills=1,
        max_graph_items=1,
        debug_scores=True,
    )
    policy = RetrievalPolicy("sushi")

    now = datetime.now(timezone.utc)
    ch = Chunk(
        id="chunk_1",
        doc_id="doc_1",
        text="This mentions sushi and should get a strong rerank score.",
        page_range=(1, 1),
        position=1,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:u1",
        meta={"vector_score": 0.83, "lexical_score": 2.0, "retrieval_route": "query", "retrieval_method": "vector"},
    )

    out = sel.select({"chunks": [ch]}, policy=policy)["chunks"]
    assert out and out[0].id == "chunk_1"
    meta = out[0].meta or {}
    assert "score_card" in meta
    card = meta["score_card"]
    assert isinstance(card, dict)
    assert card.get("id") == "chunk_1"
    assert "vector_score" in card
    assert "lexical_score" in card
    assert "rerank_score" in card
    assert "final_score" in card
    assert card.get("route") == "query"
    assert "text" not in card

