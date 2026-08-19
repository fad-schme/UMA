"""Retrieval pipeline: planner intent routing, stop policy, ranker, score cards, recall scope.

Covers the canonical retrieval pipeline from intent classification through
fusion, rerank, trust adjustment, and selection; plus lane recall scope
contracts for chunk, semantic, and procedural lanes.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from uma.common.storage_metadata import EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE, RAW_LANE, SEMANTIC_LANE, WIKI_LANE
from uma.common.types import Chunk, Fact, OwnershipRef, Skill
from uma.memory.chunk.core import ChunkSearchOptions
from uma.retrieve.planner import build_retrieval_plan
from uma.retrieve.policy import RetrievalPolicy, should_stop
from uma.retrieve.ranking import Ranker, fuse_candidates, rerank_candidates
from uma.retrieve.rlm.snippet_refiner import SnippetRefiner
import ast
import pytest
import re

from tests.helpers.runtime import TEST_AGENT_ID

AGENT_ID = TEST_AGENT_ID

# ── test_retrieval_planner ──────────────────────────────────────────




def _excluded_reason(plan, lane: str) -> str:
    for item in plan.excluded_lanes:
        if item["lane"] == lane:
            return item["reason"]
    raise AssertionError(f"lane {lane!r} not found in excluded_lanes")


def test_context_plan_defaults_to_evidence_lanes_for_topical_queries() -> None:
    plan = build_retrieval_plan(
        product="context",
        query_text="How should a zero-trust rollout be structured?",
        available_lanes=[RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
    )

    assert plan.participating_lanes == (RAW_LANE, SEMANTIC_LANE)
    assert plan.active_domains == ("kb_doc",)
    assert _excluded_reason(plan, PROFILE_LANE) == "profile_not_selected_for_topical_query"
    assert _excluded_reason(plan, WIKI_LANE) == "wiki_not_enabled_by_default_for_context"


def test_context_plan_uses_profile_lane_without_confusing_it_with_user_kb() -> None:
    plan = build_retrieval_plan(
        product="context",
        query_text="What do I like?",
        available_lanes=[RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
    )

    assert plan.participating_lanes == (PROFILE_LANE, PROCEDURAL_LANE, SEMANTIC_LANE, EPISODIC_LANE)
    assert plan.active_domains == ("user_profile", "procedural", "kb_doc")


def test_memory_plan_prefers_wiki_but_surfaces_runtime_unavailability_explicitly() -> None:
    plan = build_retrieval_plan(
        product="memory",
        query_text="What decisions did I make earlier about the rollout?",
        available_lanes=[RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
        memory_intent="continuity",
    )

    assert plan.requested_lanes[:4] == (WIKI_LANE, RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE)
    assert plan.participating_lanes == (RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE)
    assert plan.requires_compiled_memory is True
    assert plan.evidence_expansion is True
    assert _excluded_reason(plan, WIKI_LANE) == "lane_not_available_in_runtime"


def test_memory_plan_uses_wiki_when_runtime_supports_compiled_memory_lane() -> None:
    plan = build_retrieval_plan(
        product="memory",
        query_text="What decisions did I make earlier about the rollout?",
        available_lanes=[WIKI_LANE, RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
        memory_intent="continuity",
    )

    assert plan.participating_lanes[:4] == (WIKI_LANE, RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE)
    assert plan.requires_compiled_memory is True
    assert plan.evidence_expansion is True


def test_memory_plan_uses_profile_lane_only_for_profile_style_requests() -> None:
    plan = build_retrieval_plan(
        product="memory",
        query_text="What do I like?",
        available_lanes=[WIKI_LANE, RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
        memory_intent="profile",
    )

    assert plan.participating_lanes == (
        WIKI_LANE,
        RAW_LANE,
        SEMANTIC_LANE,
        PROFILE_LANE,
    )
    assert plan.active_domains == ("kb_doc", "user_profile")


def test_planner_trace_stays_backend_agnostic() -> None:
    plan = build_retrieval_plan(
        product="memory",
        query_text="What decisions did I make earlier about the rollout?",
        available_lanes=[WIKI_LANE, RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
        memory_intent="continuity",
    )

    trace = plan.to_trace()

    assert "participating_lanes" in trace
    assert "active_domains" in trace
    assert "fusion_strategy" not in trace
    assert "top_k_dense" not in trace
    assert "top_k_sparse" not in trace
    assert "backend" not in trace


# ── test_retrieval_policy_should_stop ──────────────────────────────────────────




def test_should_stop_on_max_calls():
    stop, reason = should_stop(
        recall_score=0.0,
        coverage={"confidence": 0.0},
        calls_made=10,
        max_calls=10,
        tokens_used=0,
        token_budget=10000,
        user_results_count=0,
    )
    assert stop is True
    assert reason == "max_calls_reached"


def test_should_stop_on_token_budget():
    stop, reason = should_stop(
        recall_score=0.0,
        coverage={"confidence": 1.0},
        calls_made=0,
        max_calls=10,
        tokens_used=6000,
        token_budget=5000,
        user_results_count=0,
    )
    assert stop is True
    assert reason == "token_budget_exhausted"


def test_recall_expected_but_no_user_results_continues():
    stop, reason = should_stop(
        recall_score=1.0,
        coverage={"confidence": 0.1},
        calls_made=1,
        max_calls=10,
        tokens_used=100,
        token_budget=5000,
        user_results_count=0,
    )
    assert stop is False
    assert reason == "recall_expected_but_no_user_results"


def test_recall_with_user_results_and_high_confidence_stops():
    stop, reason = should_stop(
        recall_score=1.0,
        coverage={"confidence": 0.95},
        calls_made=1,
        max_calls=10,
        tokens_used=100,
        token_budget=5000,
        user_results_count=2,
    )
    assert stop is True
    assert reason == "coverage_confident_and_recall_satisfied"


def test_non_recall_high_confidence_stops():
    stop, reason = should_stop(
        recall_score=0.0,
        coverage={"confidence": 0.9, "facts": 0, "episodes": 0},
        calls_made=1,
        max_calls=10,
        tokens_used=100,
        token_budget=5000,
        user_results_count=0,
    )
    assert stop is True
    assert reason == "coverage_confident"

# ── test_ranker ──────────────────────────────────────────





def test_fuse_candidates_sparse_precedes_and_dedupes_with_rrf() -> None:
    dense = [{"id": "a"}, {"id": "b"}]
    sparse = [{"id": "b"}, {"id": "c"}]
    out = fuse_candidates(dense=dense, sparse=sparse, strategy="rrf", rrf_k=60)
    # Overlap (b) must win. Remaining are ordered by RRF rank bonus deterministically.
    assert [x["id"] for x in out] == ["b", "a", "c"]


def test_score_card_emission_follows_request_debug_flag() -> None:
    """`score_card` is opt-in per request and must not leak by default.

    The Ranker is shared across concurrent requests, so the debug flag is
    passed per call rather than held on the instance. Both the per-request
    flag and the global `retrieval.debug_scores` config default enable it.
    """
    now = datetime.now(timezone.utc)

    def _fact(fact_id: str) -> Fact:
        return Fact(
            id=fact_id,
            subject="user",
            predicate="LIKES",
            object="sushi",
            created_at=now,
            updated_at=now,
            meta={"vector_score": 0.5},
            owner_type="user",
            owner_id="user:u1",
        )

    # Default: no emission — score cards stay out of the normal product.
    ranked = Ranker().rank_facts([_fact("f1")], query_text="sushi")
    assert "score_card" not in (ranked[0].meta or {})

    # Per-request opt-in.
    ranked = Ranker().rank_facts([_fact("f1")], query_text="sushi", debug=True)
    card = (ranked[0].meta or {}).get("score_card")
    assert card is not None
    assert card["id"] == "f1"
    assert set(card) >= {
        "vector_score",
        "lexical_score",
        "rerank_score",
        "route",
        "final_score",
        "trust_score",
        "final_score_with_trust",
    }

    # Global config default still emits without a per-request flag.
    ranked = Ranker(debug_scores=True).rank_facts([_fact("f1")], query_text="sushi")
    assert "score_card" in (ranked[0].meta or {})


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


def test_rank_facts_prefers_specific_answer_bearing_fact_over_generic_placeholder() -> None:
    now = datetime.now(timezone.utc)
    generic = Fact(
        id="f_generic",
        subject="user",
        predicate="current projects or research topics",
        object="research",
        created_at=now,
        updated_at=now,
        meta={"vector_score": 1.0},
        owner_type="user",
        owner_id="user:u1",
        confidence=0.9,
        salience=0.9,
    )
    specific = Fact(
        id="f_specific",
        subject="user",
        predicate="current research topic",
        object="adoption agencies",
        created_at=now,
        updated_at=now,
        meta={"vector_score": 1.0},
        owner_type="user",
        owner_id="user:u1",
        confidence=0.9,
        salience=0.9,
    )

    ranked = Ranker().rank_facts([generic, specific], query_text="What did the user research?")
    assert [fact.id for fact in ranked][:2] == ["f_specific", "f_generic"]


def test_rank_facts_prefers_specific_education_interest_over_generic_context() -> None:
    now = datetime.now(timezone.utc)
    generic = Fact(
        id="f_journey",
        subject="user",
        predicate="community affiliation",
        object="journey with Caroline",
        created_at=now,
        updated_at=now,
        meta={"vector_score": 1.0},
        owner_type="user",
        owner_id="user:u1",
        confidence=0.9,
        salience=0.9,
    )
    specific = Fact(
        id="f_education",
        subject="user",
        predicate="career or education plans",
        object="continue education and counseling or mental health work",
        created_at=now,
        updated_at=now,
        meta={"vector_score": 1.0},
        owner_type="user",
        owner_id="user:u1",
        confidence=0.9,
        salience=0.9,
    )

    ranked = Ranker().rank_facts(
        [generic, specific],
        query_text="What fields would the user likely pursue in education?",
    )
    assert [fact.id for fact in ranked][:2] == ["f_education", "f_journey"]


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


# ── test_retrieval_service_recall_scope ──────────────────────────────────────────





@pytest.mark.asyncio
async def test_rlm_lane_recall_scopes_user_only(uma_memory, tmp_path):
    memory = uma_memory
    assert AGENT_ID, "test runtime must set agent_id"
    agent_doc = tmp_path / "agent_doc.txt"
    agent_doc.write_text(
        (
            "Agent KB document. It contains hello world but does not mention the recall cue. "
            "This sentence is padding to ensure strict chunk validation passes for ingestion in CI. "
            "Additional padding words to reach the minimum chunk length requirement.\n"
        ),
        encoding="utf-8",
    )
    user_doc = tmp_path / "user_doc.txt"
    user_doc.write_text(
        (
            "User document. Remember last time we talked about hello world and a private user note. "
            "This sentence is padding to ensure strict chunk validation passes for ingestion in CI. "
            "Additional padding words to reach the minimum chunk length requirement.\n"
        ),
        encoding="utf-8",
    )

    await memory.ingest_document(str(agent_doc), owner_type="agent", owner_id=AGENT_ID)
    await memory.ingest_document(str(user_doc), owner_type="user", owner_id="user:u1")

    ctx = await memory.retrieve_context(
        request_id="req-recall-user-only",
        user_id="user:u1",
        session_id="session:user:u1",
        query_text="remember last time hello world",
        agent_id=AGENT_ID,
    )
    facts = ctx.facts
    chunks = ctx.chunks
    assert all(getattr(f, "owner_type", None) == "user" for f in facts)
    assert all(getattr(f, "owner_id", None) == "user:u1" for f in facts)
    assert all(getattr(c, "owner_type", None) == "user" for c in chunks)
    assert all(getattr(c, "owner_id", None) == "user:u1" for c in chunks)


@pytest.mark.asyncio
async def test_rlm_lane_kb_scopes_agent_and_user(uma_memory, tmp_path):
    memory = uma_memory
    assert AGENT_ID, "test runtime must set agent_id"
    agent_doc = tmp_path / "agent_doc.txt"
    agent_doc.write_text(
        (
            "Agent KB document. It contains hello world and an agent-only guideline. "
            "This sentence is padding to ensure strict chunk validation passes for ingestion in CI. "
            "Additional padding words to reach the minimum chunk length requirement.\n"
        ),
        encoding="utf-8",
    )
    user_doc = tmp_path / "user_doc.txt"
    user_doc.write_text(
        (
            "User document. It also contains hello world but is user-owned. "
            "This sentence is padding to ensure strict chunk validation passes for ingestion in CI. "
            "Additional padding words to reach the minimum chunk length requirement.\n"
        ),
        encoding="utf-8",
    )

    await memory.ingest_document(str(agent_doc), owner_type="agent", owner_id=AGENT_ID)
    await memory.ingest_document(str(user_doc), owner_type="user", owner_id="user:u1")

    ctx = await memory.retrieve_context(
        request_id="req-recall-kb",
        user_id="user:u1",
        session_id="session:user:u1",
        query_text="hello world",
        agent_id=AGENT_ID,
    )
    chunks = list(ctx.chunks)
    assert any(getattr(c, "owner_type", None) == "agent" for c in chunks)
    assert any(getattr(c, "owner_type", None) == "user" for c in chunks)


# ── test_ownership_only_retrieval ──────────────────────────────────────────




def _assert_no_subject_keyword_in_search_calls(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != "search":
            continue
        if any(isinstance(kw, ast.keyword) and kw.arg == "subject" for kw in node.keywords or []):
            raise AssertionError(f"{path} contains a .search(subject=...) call; retrieval must be ownership-only")


def test_rlm_deterministic_decision_does_not_inject_subject_filter():
    from uma.retrieve.rlm.decisions import deterministic_decision

    class _Pack:
        facts = []
        chunks = []
        steps = []
        owner_type = "user"
        user_id = "user:123"

    class _Coverage:
        needs_semantic = True
        needs_clusters = False

    decision = deterministic_decision(_Pack(), _Coverage(), cfg={})
    assert decision is not None
    assert decision.actions
    assert decision.actions[0].action == "search_semantic"
    assert decision.actions[0].filters is None


@pytest.mark.parametrize(
    "relpath",
    [
        "uma/retrieve/rlm/controller.py",
        "uma/retrieve/rlm/environment.py",
    ],
)
def test_no_subject_keyword_in_search_calls(relpath: str):
    _assert_no_subject_keyword_in_search_calls(Path(relpath))


@pytest.mark.parametrize(
    ("relpath", "forbidden"),
    [
        ("uma/retrieve/rlm/environment.py", "_agent_id"),
        ("uma/retrieve/rlm/environment.py", "owner_id or self._agent_id"),
        ("uma/retrieve/rlm/controller.py", "env._agent_id"),
    ],
)
def test_no_ambient_agent_scope_fallback_in_retrieval_internals(relpath: str, forbidden: str):
    src = Path(relpath).read_text(encoding="utf-8")
    if forbidden == "_agent_id":
        assert re.search(r"(?<![A-Za-z0-9])_agent_id(?![A-Za-z0-9])", src) is None
        return
    assert forbidden not in src


# ── test_chunk_and_procedural_search_no_subject ──────────────────────────────────────────






@pytest.mark.asyncio
async def test_chunk_search_does_not_require_subject(uma_memory, tmp_path):
    memory = uma_memory
    owner_type = "agent"
    owner_id = AGENT_ID

    doc = tmp_path / "doc.txt"
    doc.write_text(
        "This is a test document used for UMA retrieval. "
        "It contains the phrase hello world in a longer passage so lexical search can match it reliably. "
        "The rest of this sentence is padding to ensure the stored chunk is long enough for LIKE-based lexical search.\n"
    )
    await memory.ingest_document(str(doc), owner_type=owner_type, owner_id=owner_id)

    q = "hello world"
    query_embedding = (await memory.embedder.embed([q]))[0]

    res = await memory.chunk_core.search_chunks(
        query_embedding=query_embedding,
        owner_type=owner_type,
        owner_id=owner_id,
        k=5,
    )
    assert res, "Expected dense chunk retrieval to return at least one result"

    res2 = await memory.chunk_core.search_chunks(
        query_embedding=query_embedding,
        owner_type=owner_type,
        owner_id=owner_id,
        k=5,
        options=ChunkSearchOptions(query_text=q, filter_terms=False),
    )
    assert res2, "Expected hybrid chunk retrieval to return at least one result"

    if hasattr(memory.chunk_core.store, "lexical_search"):
        assert any(
            (getattr(ch, "meta", None) or {}).get("retrieval_method") == "lexical"
            for ch in res2
        ), "Expected lexical capability to tag at least one chunk as lexical"
    else:
        assert all(
            (getattr(ch, "meta", None) or {}).get("retrieval_method") == "vector"
            for ch in res2
        ), "Expected vector-only path when lexical capability is absent"


@pytest.mark.asyncio
async def test_procedural_search_does_not_require_subject(uma_memory):
    memory = uma_memory
    owner_type = "agent"
    owner_id = AGENT_ID

    skill = Skill(
        id="skill_s1",
        name="Test skill",
        description="How to do the hello world procedure safely and deterministically.",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        owner_type=owner_type,
        owner_id=owner_id,
    )
    emb = (await memory.embedder.embed([skill.description]))[0]
    persisted = await memory.procedural_core.add_skill(skill, emb)
    assert persisted is not None

    query_embedding = (await memory.embedder.embed(["hello world procedure"]))[0]
    res = await memory.procedural_core.search(
        query_embedding=query_embedding,
        owner=OwnershipRef(tenant_id="default", owner_type=owner_type, owner_id=owner_id),
        k=5,
    )
    assert res and res[0].id == "skill_s1"


# ── test_chunk_retrieval_returns_objects ──────────────────────────────────────────






@pytest.mark.asyncio
async def test_chunk_retrieval_returns_chunk_objects(uma_memory, tmp_path) -> None:
    memory = uma_memory
    owner_type = "agent"
    owner_id = AGENT_ID

    doc = tmp_path / "doc.txt"
    doc.write_text(
        "This document contains hello world and enough text to be chunked and retrieved.\n"
        "Second sentence for stability.\n",
        encoding="utf-8",
    )
    await memory.ingest_document(str(doc), owner_type=owner_type, owner_id=owner_id)

    q = "hello world"
    query_embedding = (await memory.embedder.embed([q]))[0]
    res = await memory.chunk_core.search_chunks(
        query_embedding=query_embedding,
        owner_type=owner_type,
        owner_id=owner_id,
        k=5,
        options=ChunkSearchOptions(query_text=q, filter_terms=False),
    )
    assert res
    assert all(isinstance(c, Chunk) for c in res)


@pytest.mark.asyncio
async def test_snippet_refiner_accepts_object_facts_and_chunks(uma_memory) -> None:
    memory = uma_memory

    class _Cfg:
        snippet_refiner_top_k = 3
        max_chunks = 2
        snippet_max_chars = 120

    now = datetime.now(timezone.utc)
    fact = Fact(
        id="fact_1",
        subject="user",
        predicate="STATES",
        object="Something happened.",
        created_at=now,
        updated_at=now,
        source_ids=[],
        confidence=0.9,
        salience=0.5,
        owner_type="user",
        owner_id="user:u1",
        meta={},
    )
    chunks = [
        Chunk(
            id="chunk_1",
            doc_id="doc_1",
            text="Something happened. More context here.",
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
    ]

    refiner = SnippetRefiner(llm=memory.llm, cfg=_Cfg())
    out = await refiner.refine(query_text="something", facts=[fact], chunks=chunks)
    assert isinstance(out, list)
    assert out and isinstance(out[0], dict)

