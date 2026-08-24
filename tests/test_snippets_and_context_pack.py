"""Snippets and semantic surface: refiner, ranking, and semantic read paths.

Covers snippet refiner presentation-only contract, ranking score cards,
semantic paging, subject-optional search, and multiple-object upsert.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from uma.common.identity import normalize_user_id
from uma.common.types import Chunk, Fact
from uma.ingest.normalizer import _clean_page_text, _drop_repeated_lines_across_pages
from uma.retrieve.policy import RetrievalPolicy
from uma.retrieve.ranking import Ranker
from uma.retrieve.rlm.snippet_refiner import SnippetRefiner
import asyncio
import pytest

from tests.helpers.runtime import TEST_AGENT_ID

AGENT_ID = TEST_AGENT_ID

# ── test_snippet_refiner_presentation_only ──────────────────────────────────────────










def test_snippet_refiner_presentation_only_does_not_filter_by_relevance() -> None:
    class _Cfg:
        max_chunks = 3
        snippet_max_chars = 60

    now = datetime.now(timezone.utc)
    chunks = [
        Chunk(
            id="chunk_1",
            doc_id="doc1",
            text="This chunk is about apples. It should not be dropped.",
            page_range=(1, 1),
            position=1,
            source_path="/tmp/x",
            source_hash="h",
            created_at=now,
            updated_at=now,
            owner_type="user",
            owner_id="user:u1",
            meta={},
        ),
        Chunk(
            id="chunk_2",
            doc_id="doc2",
            text="This chunk is about bananas. It should not be dropped either.",
            page_range=(1, 1),
            position=1,
            source_path="/tmp/y",
            source_hash="h2",
            created_at=now,
            updated_at=now,
            owner_type="user",
            owner_id="user:u1",
            meta={},
        ),
    ]

    refiner = SnippetRefiner(llm=None, cfg=_Cfg())

    async def run():
        return await refiner.refine(query_text="zzz-not-present", facts=[], chunks=chunks)

    out = asyncio.run(run())
    assert len(out) == 2
    assert all(isinstance(s, dict) for s in out)
    assert all(s.get("text") for s in out)
    assert all(len(s["text"]) <= _Cfg.snippet_max_chars for s in out)
    assert out[0]["source"]["doc_id"] == "doc1"
    assert out[1]["source"]["doc_id"] == "doc2"



# ── test_ranking_score_cards ──────────────────────────────────────────





def test_debug_scores_attaches_score_card_to_chunks() -> None:
    ranker = Ranker(debug_scores=True)
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

    out = ranker.rank_chunks([ch], query_text=policy.query_text)
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


# ── test_semantic_fetch_more_facts_paging ──────────────────────────────────────────






@pytest.mark.asyncio
async def test_fetch_more_facts_pages_deterministically_by_offset(uma_memory):
    memory = uma_memory

    owner_type = "user"
    owner_id = "user:u1"

    now = datetime.now(timezone.utc)
    emb = (await memory.embedder.embed(["shared"]))[0]

    # Ensure deterministic ordering: SemanticSQLStore orders by updated_at DESC, then id ASC.
    facts = [
        Fact(
            id="fact_1",
            subject=owner_id,
            predicate="P",
            object="a",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            meta={},
            salience=0.1,
            owner_type=owner_type,
            owner_id=owner_id,
        ),
        Fact(
            id="fact_2",
            subject=owner_id,
            predicate="P",
            object="b",
            created_at=now,
            updated_at=now - timedelta(seconds=1),
            source_ids=[],
            confidence=0.9,
            meta={},
            salience=0.1,
            owner_type=owner_type,
            owner_id=owner_id,
        ),
        Fact(
            id="fact_3",
            subject=owner_id,
            predicate="Q",
            object="c",
            created_at=now,
            updated_at=now - timedelta(seconds=2),
            source_ids=[],
            confidence=0.9,
            meta={},
            salience=0.1,
            owner_type=owner_type,
            owner_id=owner_id,
        ),
        Fact(
            id="fact_4",
            subject=owner_id,
            predicate="P",
            object="d",
            created_at=now,
            updated_at=now - timedelta(seconds=3),
            source_ids=[],
            confidence=0.9,
            meta={},
            salience=0.1,
            owner_type=owner_type,
            owner_id=owner_id,
        ),
    ]

    for f in facts:
        await memory.semantic_core.upsert_fact(f, emb)

    page1 = await memory.semantic_core.fetch_more_facts("P", owner_type=owner_type, owner_id=owner_id, k=2, offset=0)
    page2 = await memory.semantic_core.fetch_more_facts("P", owner_type=owner_type, owner_id=owner_id, k=2, offset=2)

    assert [f.id for f in page1] == ["fact_1", "fact_2"]
    assert [f.id for f in page2] == ["fact_4"]


# ── test_semantic_search_subject_optional ──────────────────────────────────────────






@pytest.mark.asyncio
async def test_semantic_search_subject_optional(uma_memory):
    """
    Semantic retrieval is ownership-only; subject is not a gating filter.
    """
    memory = uma_memory
    owner_type = "agent"
    owner_id = AGENT_ID

    now = datetime.now(timezone.utc)
    emb = (await memory.embedder.embed(["shared"]))[0]

    facts = [
        Fact(
            id="fact_zt",
            subject="entity:zero_trust",
            predicate="PRINCIPLE",
            object="least privilege",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            salience=0.9,
            owner_type=owner_type,
            owner_id=owner_id,
            meta={},
        ),
        Fact(
            id="fact_cloud",
            subject="entity:cloud_security",
            predicate="PRINCIPLE",
            object="segmentation",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            salience=0.9,
            owner_type=owner_type,
            owner_id=owner_id,
            meta={},
        ),
        Fact(
            id="fact_userish",
            subject="user:local",
            predicate="REMEMBERED",
            object="note",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            salience=0.9,
            owner_type=owner_type,
            owner_id=owner_id,
            meta={},
        ),
    ]

    for f in facts:
        await memory.semantic_core.upsert_fact(f, emb)

    all_facts = await memory.semantic_core.search(
        query_embedding=emb,
        owner_type=owner_type,
        owner_id=owner_id,
        k=10,
        filters=None,
        query_text=None,
    )
    assert len(all_facts) == 3

    # Subject filters are ignored (ownership-only retrieval).
    filtered = await memory.semantic_core.search(
        query_embedding=emb,
        owner_type=owner_type,
        owner_id=owner_id,
        k=10,
        filters={"subject": "user:local"},
        query_text=None,
    )
    assert len(filtered) == 3



# ── test_semantic_upsert_multiple_objects ──────────────────────────────────────────





@pytest.mark.asyncio
async def test_semantic_upsert_allows_multiple_objects_same_predicate(uma_memory):
    """
    Regression test:
    - We must NOT drop distinct objects for the same (owner, subject, predicate).
      e.g., user LIKES sushi AND user LIKES pizza should both persist and be retrievable.
    """
    memory = uma_memory

    user_id = "user:123"
    owner_id = normalize_user_id(user_id)
    now = datetime.utcnow()

    sushi = Fact(
        id="fact_sushi",
        subject=owner_id,
        predicate="LIKES",
        object="sushi",
        created_at=now,
        updated_at=now,
        source_ids=[],
        confidence=0.8,
        owner_type="user",
        owner_id=owner_id,
        meta={},
        salience=0.9,
    )
    pizza = Fact(
        id="fact_pizza",
        subject=owner_id,
        predicate="LIKES",
        object="pizza",
        created_at=now,
        updated_at=now,
        source_ids=[],
        confidence=0.8,
        owner_type="user",
        owner_id=owner_id,
        meta={},
        salience=0.9,
    )

    sushi_emb, pizza_emb = await memory.embedder.embed(["sushi", "pizza"])
    await memory.semantic_core.upsert_fact(sushi, sushi_emb)
    await memory.semantic_core.upsert_fact(pizza, pizza_emb)

    facts = await memory.semantic_core.list_facts_for_owner(owner_type="user", owner_id=owner_id, limit=None)
    likes = [f for f in facts if f.subject == owner_id and f.predicate == "LIKES"]
    objects = {str(getattr(f, "object", "")) for f in likes}
    assert {"sushi", "pizza"}.issubset(objects)

    # Vector retrieval should return the correct fact when queried near its embedding.
    found_sushi = await memory.semantic_core.search(
        query_embedding=sushi_emb,
        owner_type="user",
        owner_id=owner_id,
        k=10,
        offset=0,
        filters=None,
        query_text=None,
    )
    assert any(getattr(f, "id", None) == "fact_sushi" for f in found_sushi)

    found_pizza = await memory.semantic_core.search(
        query_embedding=pizza_emb,
        owner_type="user",
        owner_id=owner_id,
        k=10,
        offset=0,
        filters=None,
        query_text=None,
    )
    assert any(getattr(f, "id", None) == "fact_pizza" for f in found_pizza)


# ── test_semantic_ingest_user_facts_persisted ──────────────────────────────────────────




@pytest.mark.asyncio
async def test_semantic_core_ingest_persists_multiple_user_facts(uma_memory):
    memory = uma_memory

    user_id = "user:123"
    owner_id = normalize_user_id(user_id)

    persisted = await memory.semantic_core.ingest(
        owner_id,
        "user likes sushi and pizza",
        extra_meta={"turn_id": "t1"},
    )
    assert persisted

    facts = await memory.semantic_core.list_facts_for_owner(owner_type="user", owner_id=owner_id, limit=None)
    likes = [f for f in facts if getattr(f, "owner_id", None) == owner_id and getattr(f, "predicate", "") == "LIKES"]
    assert likes and all(getattr(f, "subject", None) == "user" for f in likes)
    objects = {str(getattr(f, "object", "")) for f in likes}
    assert {"sushi", "pizza"}.issubset(objects)


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

