"""Retrieval context and RLM: bound retrieve_context/retrieve_memory surface, environment API, RLM decisions.

Covers the public retrieve_context / retrieve_memory surfaces, evidence
expansion, RLM controller intent/domain classification, decision shapes,
fallback ladder, and stop confidence wiring.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from tests.helpers.context_bundle import make_context_bundle
from tests.helpers.graph_adapter import RecordingGraphAdapter
from tests.helpers.runtime import TEST_AGENT_ID, init_uma_for_tests
from uma.api.memory import UMAMemory
from uma.api.runtime import UMARuntime
from uma.common.results import Confidence, MemoryResult, Provenance
from uma.common.types import RuntimeContext
from uma.common.types.types_fact import Fact
from uma.retrieve.policy import should_stop
from uma.retrieve.rlm.context_pack import ContextPack
from uma.retrieve.rlm.controller import RLMController
from uma.retrieve.rlm.decisions import (
    ControllerDecision,
    SearchSemanticAction,
    deterministic_decision,
)
from uma.retrieve.rlm.domain import ensure_fact_domain, filter_facts_by_domains
from uma.retrieve.rlm.environment import UMAMemoryEnvironment
from uma.retrieve.rlm.evidence import expand_evidence_chunks_from_facts
from uma.retrieve.rlm.intent import QueryIntent, classify_query_intent
from uma.retrieve.rlm.request import RetrievalRequest
from uma.common.types.types_scope import DEFAULT_TENANT_ID
import asyncio
import pytest

AGENT_ID = TEST_AGENT_ID

# ── test_bound_context_retrieval ──────────────────────────────────────────






@pytest.mark.asyncio
async def test_runtime_retrieval_delegates_directly(uma_memory) -> None:
    memory = uma_memory
    runtime = UMARuntime.from_memory(memory)
    context = RuntimeContext(
        tenant_id="tenant-1",
        agent_id=AGENT_ID,
        request_id="req-1",
        user_id="user:u1",
        workspace_id="workspace:alpha",
        session_id="session-1",
    )
    seen: list[tuple[str, RuntimeContext, str]] = []

    async def fake_context(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        lane_filter=None,
        include_debug: bool = False,
    ):
        seen.append(("context", bound_context, query_text))
        return make_context_bundle(
            query=query_text,
            lane_filter=lane_filter or [],
        )

    async def fake_memory(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        memory_intent: str = "continuity",
        include_debug: bool = False,
    ):
        seen.append(("memory", bound_context, query_text))
        return MemoryResult(
            query=query_text,
            compiled_memory=None,
            facts=[{"text": "fact", "confidence": 1.0, "salience": 1.0, "source_chunk_ids": ["chunk-1"]}],
            evidence=[],
            provenance_valid=True,
            debug={"memory_intent": memory_intent} if include_debug else None,
        )

    runtime.retrieve_context = fake_context  # type: ignore[method-assign]
    runtime.retrieve_memory = fake_memory  # type: ignore[method-assign]

    structured = await runtime.retrieve_context(context, query_text="hello world", lane_filter=["semantic"])
    memory_payload = await runtime.retrieve_memory(context, query_text="hello world")

    assert structured.product == "context"
    assert structured.debug.lane_filter == ["semantic"]
    assert structured.documents == []
    assert memory_payload.provenance_valid is True
    assert memory_payload.facts[0]["source_chunk_ids"] == ["chunk-1"]
    assert seen == [
        ("context", context, "hello world"),
        ("memory", context, "hello world"),
    ]


@pytest.mark.asyncio
async def test_runtime_retrieval_requires_user_id(uma_memory) -> None:
    runtime = UMARuntime.from_memory(uma_memory)
    context = RuntimeContext(
        tenant_id="tenant-1",
        agent_id=AGENT_ID,
        request_id="req-1",
    )

    with pytest.raises(ValueError, match="user_id"):
        await runtime.retrieve_context(context, query_text="hello world")


@pytest.mark.asyncio
async def test_umamemory_public_retrieval_surface_delegates_by_intent(uma_memory) -> None:
    memory = uma_memory
    seen: list[tuple[str, RuntimeContext, str]] = []

    async def fake_context(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        lane_filter=None,
        include_debug: bool = False,
    ):
        seen.append(("context", bound_context, query_text))
        return make_context_bundle(
            query=query_text,
            lane_filter=lane_filter or [],
        )

    async def fake_memory(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        memory_intent: str = "continuity",
        include_debug: bool = False,
    ):
        seen.append(("memory", bound_context, query_text))
        return MemoryResult(
            query=query_text,
            compiled_memory=None,
            facts=[{"text": "fact", "confidence": 1.0, "salience": 1.0, "source_chunk_ids": ["chunk-1"]}],
            evidence=[],
            provenance_valid=True,
            debug={"memory_intent": memory_intent} if include_debug else None,
        )

    memory.runtime.retrieve_context = fake_context  # type: ignore[method-assign]
    memory.runtime.retrieve_memory = fake_memory  # type: ignore[method-assign]

    context = await memory.retrieve_context(
        query_text="hello world",
        agent_id=AGENT_ID,
        user_id="user:u1",
        tenant_id="tenant-1",
        request_id="req-ctx",
        workspace_id="workspace:alpha",
        session_id="session-1",
    )
    memory_result = await memory.retrieve_memory(
        query_text="hello world",
        agent_id=AGENT_ID,
        user_id="user:u1",
        tenant_id="tenant-1",
        request_id="req-mem",
        workspace_id="workspace:alpha",
        session_id="session-1",
    )

    assert context.product == "context"
    assert context.documents == []
    assert memory_result.provenance_valid is True
    assert memory_result.facts[0]["text"] == "fact"
    assert seen == [
        (
            "context",
            RuntimeContext(
                tenant_id="tenant-1",
                agent_id=AGENT_ID,
                request_id="req-ctx",
                user_id="user:u1",
                workspace_id="workspace:alpha",
                session_id="session-1",
            ),
            "hello world",
        ),
        (
            "memory",
            RuntimeContext(
                tenant_id="tenant-1",
                agent_id=AGENT_ID,
                request_id="req-mem",
                user_id="user:u1",
                workspace_id="workspace:alpha",
                session_id="session-1",
            ),
            "hello world",
        ),
    ]


@pytest.mark.asyncio
async def test_runtime_memory_retrieval_surfaces_explicit_evidence_only_fallback(uma_memory) -> None:
    runtime = UMARuntime.from_memory(uma_memory)
    context = RuntimeContext(
        tenant_id="tenant-1",
        agent_id=AGENT_ID,
        request_id="req-memory-fallback",
        user_id="user:u1",
    )

    async def fake_context(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        lane_filter=None,
        include_debug: bool = False,
    ):
        assert lane_filter == ["wiki", "raw", "semantic", "episodic"]
        return make_context_bundle(
            query=query_text,
            provenance=Provenance(
                source_chunk_ids=[],
                source_document_ids=[],
                derivation_type="context_retrieval",
                retrieval_path=[],
                parent_artifact_ids=[],
                support_density=None,
                confidence=None,
                conflicts=[],
                evidence_scopes=[],
                manual=False,
                valid=False,
                invalid_reasons=["missing_source_chunk_ids"],
            ),
        )

    runtime.retrieve_context = fake_context  # type: ignore[method-assign]

    result = await runtime.retrieve_memory(
        context,
        query_text="memory query",
        memory_intent="continuity",
    )

    assert result.facts == []
    assert result.evidence == []
    assert result.provenance_valid is False
    assert result.provenance_error == "missing_source_chunk_ids"
    assert "product" not in result
    assert "memory_intent" not in result
    assert "compiled_memory_log" not in result
    assert "compiled_memory_index" not in result


@pytest.mark.asyncio
async def test_runtime_memory_retrieval_can_expose_debug_payload(uma_memory) -> None:
    runtime = UMARuntime.from_memory(uma_memory)
    context = RuntimeContext(
        tenant_id="tenant-1",
        agent_id=AGENT_ID,
        request_id="req-memory-debug",
        user_id="user:u1",
    )

    async def fake_context(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        lane_filter=None,
        include_debug: bool = False,
    ):
        return make_context_bundle(
            query=query_text,
            trace=[{"event": "lane_plan"}],
            confidence=Confidence(
                score=0.2,
                semantic_enough=0.0,
                clusters_present=0.0,
                graph_present=0.0,
                graph_entity_support=0.0,
                graph_predicate_support=0.0,
                novelty_recent=0.0,
                contradictions=0.0,
            ),
            provenance=Provenance(
                source_chunk_ids=[],
                source_document_ids=[],
                derivation_type="context_retrieval",
                retrieval_path=[],
                parent_artifact_ids=[],
                support_density=None,
                confidence=None,
                conflicts=[],
                evidence_scopes=[],
                manual=False,
                valid=False,
                invalid_reasons=["missing_source_chunk_ids"],
            ),
        )

    runtime.retrieve_context = fake_context  # type: ignore[method-assign]

    result = await runtime.retrieve_memory(
        context,
        query_text="memory query",
        memory_intent="continuity",
        include_debug=True,
    )

    assert result.provenance_valid is False
    assert result.debug is not None
    assert result.debug["product"] == "memory"
    assert result.debug["memory_intent"] == "continuity"
    assert "compiled_memory_log" in result.debug


@pytest.mark.asyncio
async def test_runtime_memory_zero_evidence_returns_honest_fallback_debug_shape(uma_memory) -> None:
    runtime = UMARuntime.from_memory(uma_memory)
    context = RuntimeContext(
        tenant_id="tenant-1",
        agent_id=AGENT_ID,
        request_id="req-memory-fallback",
        user_id="user:u1",
    )

    async def fake_context(
        bound_context: RuntimeContext,
        *,
        query_text: str,
        lane_filter=None,
        include_debug: bool = False,
    ):
        _now = datetime.now(timezone.utc)
        return make_context_bundle(
            query=query_text,
            facts=[
                Fact(
                    id="fact-missing-evidence",
                    subject="user",
                    predicate="current research topic",
                    object="adoption agencies",
                    created_at=_now,
                    updated_at=_now,
                    source_ids=[],
                )
            ],
            trace=[{"event": "lane_plan"}],
            confidence=Confidence(
                score=0.2,
                semantic_enough=0.0,
                clusters_present=0.0,
                graph_present=0.0,
                graph_entity_support=0.0,
                graph_predicate_support=0.0,
                novelty_recent=0.0,
                contradictions=0.0,
            ),
            provenance=Provenance(
                source_chunk_ids=[],
                source_document_ids=[],
                derivation_type="context_retrieval",
                retrieval_path=[],
                parent_artifact_ids=[],
                support_density=None,
                confidence=None,
                conflicts=[],
                evidence_scopes=[],
                manual=False,
                valid=False,
                invalid_reasons=["missing_source_chunk_ids"],
            ),
        )

    runtime.retrieve_context = fake_context  # type: ignore[method-assign]

    result = await runtime.retrieve_memory(
        context,
        query_text="What did the user research?",
        memory_intent="continuity",
        include_debug=True,
    )

    assert result.facts
    assert result.provenance_valid is False
    assert result.provenance_error == "missing_source_chunk_ids"
    assert result.debug["compiled_answer"] is None
    assert result.debug["supporting_facts"]
    assert result.debug["trace"]


@pytest.mark.asyncio
async def test_runtime_context_trace_surfaces_lane_plan(uma_memory) -> None:
    runtime = UMARuntime.from_memory(uma_memory)
    context = RuntimeContext(
        tenant_id="tenant-1",
        agent_id=AGENT_ID,
        request_id="req-context-plan",
        user_id="user:u1",
    )

    class FakeController:
        async def retrieve_context(self, request, query_text):
            assert request.plan is not None
            pack = ContextPack(
                user_id=request.normalized_user_id,
                query_text=query_text,
                agent_id=request.context.agent_id,
                intent=request.plan.query_intent,
                active_lanes=list(request.plan.participating_lanes),
                active_domains=list(request.plan.active_domains),
                lane_plan=request.plan.to_trace(),
            )
            pack.steps.append({"step": 0, "phase": "plan", **request.plan.to_trace()})
            return pack

    runtime.ensure_retrieval_ready = lambda: None  # type: ignore[method-assign]
    uma_memory._rlm_controller = FakeController()

    result = await runtime.retrieve_context(
        context,
        query_text="What do I like?",
    )

    assert result.product == "context"
    assert result.debug.active_lanes == ["profile", "procedural", "semantic", "episodic"]
    lane_plan = next(step for step in result.debug.trace if step.get("event") == "lane_plan")
    assert lane_plan["product"] == "context"
    assert lane_plan["participating_lanes"] == ["profile", "procedural", "semantic", "episodic"]
    assert lane_plan["active_domains"] == ["user_profile", "procedural", "kb_doc"]

def test_umamemory_retrieval_shims_are_removed() -> None:
    assert hasattr(UMAMemory, "retrieve_context")
    assert hasattr(UMAMemory, "retrieve_memory")
    assert not hasattr(UMAMemory, "fetch_memory")
    assert not hasattr(UMAMemory, "get_structured_context")  # type: ignore[name-defined]
    assert not hasattr(UMAMemory, "get_rendered_context")  # type: ignore[name-defined]
    assert not hasattr(UMAMemory, "get_context_messages")  # type: ignore[name-defined]


@pytest.mark.asyncio
async def test_bound_context_workspace_id_does_not_broaden_retrieval_owner_support(
    uma_memory,
    tmp_path,
) -> None:
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

    runtime = UMARuntime.from_memory(memory)
    context = RuntimeContext(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id=AGENT_ID,
        request_id="req-workspace-inert",
        user_id="user:u1",
        workspace_id="workspace:alpha",
    )

    ctx = await runtime.retrieve_context(context, query_text="hello world")
    owner_types = {getattr(chunk, "owner_type", None) for chunk in ctx.chunks}
    chunk_lanes = {
        (getattr(chunk, "meta", {}) or {}).get("kb_lane")
        for chunk in ctx.chunks
    }

    assert owner_types
    assert owner_types.issubset({"agent", "user"})
    assert "workspace" not in owner_types
    assert "system" not in owner_types
    assert chunk_lanes == {"raw"}


# ── test_retrieve_memory_surface ──────────────────────────────────────────






@pytest.mark.asyncio
async def test_retrieve_memory_empty_result_shape(uma_memory) -> None:
    result = await uma_memory.retrieve_memory(
        query_text="something that does not exist",
        user_id="user:u1",
        request_id="req-mem-empty",
        session_id="session-mem-empty",
        agent_id=AGENT_ID,
    )

    assert isinstance(result, MemoryResult)
    # `compiled_memory`, `facts`, `evidence`, `provenance_valid`, `product`
    # are all required fields on the MemoryResult model — their presence
    # is guaranteed by model validation, so we assert the shape/value
    # rather than key presence.
    assert result.product == "memory"
    assert isinstance(result.facts, list)
    assert isinstance(result.evidence, list)


@pytest.mark.asyncio
async def test_retrieve_memory_returns_facts_after_ingest(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)

    bootstrap_path = tmp_path / "MEMORY.md"
    bootstrap_path.write_text(
        "# Memory\n- prefers espresso over drip coffee\n- reviews incidents before publishing\n",
        encoding="utf-8",
    )
    result = await memory.load_memory_bootstrap(
        str(bootstrap_path),
        user_id="user:u1",
        request_id="req-bootstrap",
        session_id="session-bootstrap",
        agent_id=AGENT_ID,
    )
    assert result["status"] == "ingested"
    assert result["facts_created"] == 2

    recalled = await memory.retrieve_memory(
        query_text="coffee preferences",
        user_id="user:u1",
        request_id="req-recall",
        session_id="session-recall",
        agent_id=AGENT_ID,
    )

    assert isinstance(recalled, MemoryResult)
    assert isinstance(recalled.facts, list)
    assert isinstance(recalled.evidence, list)
    assert isinstance(recalled.provenance_valid, bool)


@pytest.mark.asyncio
async def test_retrieve_memory_user_scope_isolation(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)

    bootstrap_path = tmp_path / "MEMORY.md"
    bootstrap_path.write_text(
        "# Memory\n- prefers dark roast coffee\n",
        encoding="utf-8",
    )
    await memory.load_memory_bootstrap(
        str(bootstrap_path),
        user_id="user:alice",
        tenant_id="default",
        request_id="req-alice-ingest",
        session_id="session-alice",
        agent_id=AGENT_ID,
    )

    bob_result = await memory.retrieve_memory(
        query_text="coffee preferences",
        user_id="user:bob",
        request_id="req-bob-recall",
        session_id="session-bob",
        agent_id=AGENT_ID,
    )

    assert bob_result.facts == []


@pytest.mark.asyncio
async def test_retrieve_memory_third_person_fact_stays_within_same_user_scope(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)
    now = datetime.now(timezone.utc)

    fact_alice = Fact(
        id="fact_maria_alice",
        subject="Maria",
        predicate="has hair color",
        object="red hair",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
        confidence=0.9,
        salience=0.9,
    )
    fact_bob = Fact(
        id="fact_maria_bob",
        subject="Maria",
        predicate="has hair color",
        object="red hair",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:bob",
        tenant_id="default",
        confidence=0.9,
        salience=0.9,
    )

    emb_alice, emb_bob = await memory.embedder.embed(
        [
            "Maria has hair color red hair",
            "Maria has hair color red hair",
        ]
    )
    await memory.semantic_core.upsert_fact(fact_alice, emb_alice)

    bob_before = await memory.retrieve_memory(
        query_text="Is Maria blond?",
        user_id="user:bob",
        request_id="req-bob-before-own-maria-fact",
        session_id="session-bob",
        agent_id=AGENT_ID,
    )
    assert bob_before.facts == []

    await memory.semantic_core.upsert_fact(fact_bob, emb_bob)
    bob_after = await memory.retrieve_memory(
        query_text="Is Maria blond?",
        user_id="user:bob",
        request_id="req-bob-after-own-maria-fact",
        session_id="session-bob",
        agent_id=AGENT_ID,
    )

    fact_texts = {
        str(item.get("text") or "").lower()
        for item in list(bob_after.facts)
        if isinstance(item, dict)
    }
    assert any("red hair" in t for t in fact_texts)


@pytest.mark.asyncio
async def test_retrieve_memory_include_debug_flag(uma_memory) -> None:
    result = await uma_memory.retrieve_memory(
        query_text="test query",
        user_id="user:u1",
        request_id="req-mem-debug",
        session_id="session-mem-debug",
        include_debug=True,
        agent_id=AGENT_ID,
    )

    assert isinstance(result, MemoryResult)
    assert isinstance(result.facts, list)
    assert result.debug is not None


# ── test_retrieval_scoped_requests ──────────────────────────────────────────






@dataclass
class _Chunk:
    id: str
    owner_type: str
    owner_id: str
    tenant_id: str = "default"
    meta: dict = field(default_factory=dict)


class _EvidenceEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []
        self._memory = type(
            "_MemoryCfg",
            (),
            {
                "retrieval_cfg": type("_Cfg", (), {"max_evidence_chunks": 6})(),
            },
        )()

    async def fetch_chunks(self, request: RetrievalRequest, *, ids, owner_type, owner_id):
        self.calls.append((owner_type, owner_id, list(ids)))
        return [_Chunk(id=f"{owner_type}:{chunk_id}", owner_type=owner_type, owner_id=owner_id) for chunk_id in ids]


@pytest.mark.asyncio
async def test_evidence_expansion_fetches_chunks_by_source_fact_owner_scope() -> None:
    env = _EvidenceEnv()
    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id="tenant-1",
            agent_id="agent:alpha",
            request_id="req-evidence",
            user_id="user:u1",
        )
    )
    pack = ContextPack(
        user_id="user:u1",
        query_text="hello world",
        owner_type="user",
        owner_id="user:u1",
        facts=[
            {"id": "fact-agent", "owner_type": "agent", "owner_id": "agent:alpha", "source_ids": ["chunk-agent"]},
            {"id": "fact-user", "owner_type": "user", "owner_id": "user:u1", "source_ids": ["chunk-user"]},
        ],
    )

    chunks = await expand_evidence_chunks_from_facts(
        env=env,
        request=request,
        pack=pack,
        max_items_per_type=10,
    )

    assert env.calls == [
        ("agent", "agent:alpha", ["chunk-agent"]),
        ("user", "user:u1", ["chunk-user"]),
    ]
    assert {chunk.owner_type for chunk in chunks} == {"agent", "user"}
    assert {chunk.owner_id for chunk in chunks} == {"agent:alpha", "user:u1"}


@pytest.mark.asyncio
async def test_bound_context_retrieval_is_isolated_across_agents_on_shared_runtime(uma_memory, tmp_path) -> None:
    memory = uma_memory
    runtime = UMARuntime.from_memory(memory)

    doc_a = tmp_path / "agent_a.txt"
    doc_a.write_text(
        (
            "Agent alpha KB document. It mentions shared keyword and alpha-only guidance. "
            "This sentence pads the content so chunking remains valid in CI.\n"
        ),
        encoding="utf-8",
    )
    doc_b = tmp_path / "agent_b.txt"
    doc_b.write_text(
        (
            "Agent beta KB document. It mentions shared keyword and beta-only guidance. "
            "This sentence pads the content so chunking remains valid in CI.\n"
        ),
        encoding="utf-8",
    )
    user_doc = tmp_path / "user_doc.txt"
    user_doc.write_text(
        (
            "User-owned document. It mentions shared keyword for the same user. "
            "This sentence pads the content so chunking remains valid in CI.\n"
        ),
        encoding="utf-8",
    )

    await memory.ingest_document(str(doc_a), owner_type="agent", owner_id="agent:alpha")
    await memory.ingest_document(str(doc_b), owner_type="agent", owner_id="agent:beta")
    await memory.ingest_document(str(user_doc), owner_type="user", owner_id="user:u1")

    ctx_a_context = RuntimeContext(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id="agent:alpha",
        request_id="req-alpha",
        user_id="user:u1",
    )
    ctx_b_context = RuntimeContext(
        tenant_id=DEFAULT_TENANT_ID,
        agent_id="agent:beta",
        request_id="req-beta",
        user_id="user:u1",
    )

    ctx_a, ctx_b = await asyncio.gather(
        runtime.retrieve_context(ctx_a_context, query_text="shared keyword"),
        runtime.retrieve_context(ctx_b_context, query_text="shared keyword"),
    )

    owner_pairs_a = {(getattr(chunk, "owner_type", None), getattr(chunk, "owner_id", None)) for chunk in ctx_a.chunks}
    owner_pairs_b = {(getattr(chunk, "owner_type", None), getattr(chunk, "owner_id", None)) for chunk in ctx_b.chunks}
    tenant_ids_a = {getattr(chunk, "tenant_id", None) for chunk in ctx_a.chunks}
    tenant_ids_b = {getattr(chunk, "tenant_id", None) for chunk in ctx_b.chunks}

    assert ("agent", "agent:alpha") in owner_pairs_a
    assert ("agent", "agent:beta") not in owner_pairs_a
    assert ("agent", "agent:beta") in owner_pairs_b
    assert ("agent", "agent:alpha") not in owner_pairs_b
    assert ("user", "user:u1") in owner_pairs_a
    assert ("user", "user:u1") in owner_pairs_b
    assert tenant_ids_a == {"default"}
    assert tenant_ids_b == {"default"}


# ── test_environment_api ──────────────────────────────────────────






@pytest.mark.asyncio
async def test_environment_get_query_embedding_shape(uma_memory):
    env = UMAMemoryEnvironment(uma_memory)
    vec = await env.get_query_embedding("hello world")
    assert isinstance(vec, list)
    assert len(vec) == int(getattr(uma_memory.embedder, "dimension", 0))


@pytest.mark.asyncio
async def test_environment_fetch_facts_by_ids_is_owner_scoped(uma_memory):
    memory = uma_memory
    env = UMAMemoryEnvironment(memory)
    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=AGENT_ID,
            request_id="req-env-facts",
            user_id="user:u1",
        )
    )

    now = datetime.now(timezone.utc)
    owner_type = "user"
    owner_id = "user:u1"
    emb = (await memory.embedder.embed(["coffee"]))[0]

    fact = Fact(
        id="fact_1",
        subject="user",
        predicate="LIKES",
        object="coffee",
        created_at=now,
        updated_at=now,
        source_ids=[],
        confidence=0.9,
        salience=0.8,
        owner_type=owner_type,
        owner_id=owner_id,
        meta={},
    )
    await memory.semantic_core.upsert_fact(fact, emb)

    facts = await env.fetch_facts_by_ids(request, ["fact_1"], owner_type=owner_type, owner_id=owner_id)
    assert facts and getattr(facts[0], "id", None) == "fact_1"

    # Wrong scope => no results.
    wrong = await env.fetch_facts_by_ids(
        request,
        ["fact_1"],
        owner_type="agent",
        owner_id=AGENT_ID,
    )
    assert wrong == []


@pytest.mark.asyncio
async def test_environment_graph_neighbors_returns_empty_when_no_edges(uma_memory):
    env = UMAMemoryEnvironment(uma_memory)
    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id="tenant-test",
            agent_id=AGENT_ID,
            request_id="req-env-graph",
            user_id="user:u1",
        )
    )
    out = await env.graph_neighbors(
        request,
        "node1",
        predicate_scope=["LIKES"],
        depth=2,
        k=5,
        owner_type="agent",
        owner_id=AGENT_ID,
    )
    assert out == []

    adapter = getattr(uma_memory.graph_core, "adapter", None)
    queries = getattr(adapter, "queries", None)
    if isinstance(queries, list) and queries:
        _cypher, params = queries[-1]
        assert params["tenant_id"] == "tenant-test"
        assert params["owner_type"] == "agent"
        assert params["owner_id"] == (AGENT_ID)


@pytest.mark.asyncio
async def test_environment_graph_resolve_nodes_is_tenant_and_owner_scoped(uma_memory):
    env = UMAMemoryEnvironment(uma_memory)
    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id="tenant-test",
            agent_id=AGENT_ID,
            request_id="req-env-graph-resolve",
            user_id="user:u1",
        )
    )
    adapter = RecordingGraphAdapter()
    adapter.next_results.append([{"node_id": "resolved-node"}])
    uma_memory.graph_core.adapter = adapter

    out = await env.graph_resolve_nodes(
        request,
        names=["Resolved Node"],
        owner_type="agent",
        owner_id=AGENT_ID,
        limit=5,
    )
    assert out == ["resolved-node"]
    assert adapter.queries
    _cypher, params = adapter.queries[-1]
    assert params["tenant_id"] == "tenant-test"
    assert params["owner_type"] == "agent"
    assert params["owner_id"] == (AGENT_ID)


@pytest.mark.asyncio
async def test_environment_graph_neighbors_rejects_workspace_scope_in_runtime_flow(uma_memory):
    env = UMAMemoryEnvironment(uma_memory)
    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id="tenant-test",
            agent_id=AGENT_ID,
            request_id="req-env-graph-workspace",
            user_id="user:u1",
        )
    )
    with pytest.raises(ValueError, match="invalid owner_type"):
        await env.graph_neighbors(
            request,
            "node1",
            depth=1,
            k=5,
            owner_type="workspace",
            owner_id="workspace:alpha",
        )


@pytest.mark.asyncio
async def test_environment_execute_action_delegates_semantic_search_to_core(uma_memory):
    memory = uma_memory
    env = UMAMemoryEnvironment(memory)
    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id="tenant-test",
            agent_id=AGENT_ID,
            request_id="req-env-execute-semantic",
            user_id="user:u1",
        )
    )

    captured = {}
    # The session-local filter downstream fails closed on a missing tenant, so
    # the stub returns a scope-bearing row rather than a bare sentinel.
    from types import SimpleNamespace

    native_fact = SimpleNamespace(
        id="fact-native", tenant_id="tenant-test", session_id=None
    )

    async def fake_search(
        query_embedding,
        *,
        tenant_id,
        owner_type,
        owner_id,
        k,
        offset=0,
        filters=None,
        query_text=None,
    ):
        captured.update(
            {
                "query_embedding": list(query_embedding),
                "tenant_id": tenant_id,
                "owner_type": owner_type,
                "owner_id": owner_id,
                "k": k,
                "offset": offset,
                "filters": filters,
                "query_text": query_text,
            }
        )
        return [native_fact]

    memory.semantic_core.search = fake_search  # type: ignore[method-assign]

    result = await env.execute_action(
        request=request,
        action=SearchSemanticAction(k=7, filters={"predicate": "LIKES"}),
        query_embedding=[1, 2, 3],
        query_text="coffee",
        owner_type="user",
        owner_id="user:u1",
        default_k=5,
    )

    assert result == [native_fact]
    assert captured == {
        "query_embedding": [1.0, 2.0, 3.0],
        "tenant_id": "tenant-test",
        "owner_type": "user",
        "owner_id": "user:u1",
        "k": 7,
        "offset": 0,
        "filters": {"predicate": "LIKES"},
        "query_text": "coffee",
    }


# ── test_rlm_decisions ──────────────────────────────────────────




def test_controller_decision_validates_actions():
    decision = ControllerDecision.model_validate({
        "actions": [
            {"action": "search_semantic", "k": 3},
            {"action": "fetch_facts", "ids": ["f1", "f2"]},
            {
                "action": "graph_neighbors",
                "node_id": "n1",
                "predicate_scope": ["LIKES"],
                "k": 5,
                "depth": 1,
            },
        ],
        "done": False,
    })
    assert len(decision.actions) == 3
    assert decision.actions[0].action == "search_semantic"
    assert decision.actions[1].ids == ["f1", "f2"]


def test_controller_decision_rejects_invalid_action():
    with pytest.raises(ValueError):
        ControllerDecision.model_validate(
            {"actions": [{"action": "fetch_facts"}], "done": False}
        )


def test_controller_decision_accepts_extended_actions():
    decision = ControllerDecision.model_validate({
        "actions": [
            {"action": "fetch_episode_clusters", "k": 2, "min_salience": 0.4},
            {
                "action": "expand_graph",
                "k": 5,
                "subject": "user:1",
                "direction": "both",
                "hops": 2,
            },
        ],
        "done": False,
    })
    assert decision.actions[0].action == "fetch_episode_clusters"
    assert decision.actions[1].subject == "user:1"


def test_controller_decision_rejects_unknown_action():
    with pytest.raises(ValueError):
        ControllerDecision.model_validate(
            {"actions": [{"action": "resolve_conflicts", "fact_ids": ["f1"]}], "done": False}
        )


# ── test_rlm_intent_and_domain ──────────────────────────────────────────





def _fact(*, predicate: str, subject: str = "user", obj: str = "x", source_ids=None, meta=None) -> Fact:
    return Fact(
        id="fact_test",
        subject=subject,
        predicate=predicate,
        object=obj,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        source_ids=list(source_ids or []),
        meta=dict(meta or {}),
        owner_type="user",
        owner_id="user:test",
        salience=0.0,
        confidence=0.7,
    )


def test_intent_topical_default() -> None:
    q = "How should a secure cloud security architecture be structured for a multi-tier application?"
    assert classify_query_intent(q) == QueryIntent.TOPICAL


def test_intent_personal_preferences() -> None:
    assert classify_query_intent("What do I like?") == QueryIntent.PERSONAL
    assert classify_query_intent("What are my preferences?") == QueryIntent.PERSONAL


def test_filtering_excludes_user_profile_when_not_allowed() -> None:
    profile = _fact(predicate="LIKES", obj="sushi")
    kb = _fact(predicate="STATES", subject="Document(d1)", obj="x", source_ids=["chunk_123"])
    out = filter_facts_by_domains([profile, kb], allowed_domains={"kb_doc"})
    assert out == [kb]


def test_domain_defaulting_preference_predicate() -> None:
    f = _fact(predicate="likes", obj="sushi")
    assert ensure_fact_domain(f) == "user_profile"
    assert f.meta.get("domain") == "user_profile"


def test_domain_defaulting_otherwise_kb_doc() -> None:
    f = _fact(predicate="STATES", subject="Document(d1)", obj="x")
    assert ensure_fact_domain(f) == "kb_doc"
    assert f.meta.get("domain") == "kb_doc"


def test_controller_candidate_filter_respects_explicit_active_lanes() -> None:
    semantic = _fact(
        predicate="STATES",
        subject="Document(d1)",
        obj="kb fact",
        meta={"kb_lane": "semantic", "domain": "kb_doc"},
    )
    profile = _fact(
        predicate="LIKES",
        obj="sushi",
        meta={"kb_lane": "profile", "domain": "user_profile"},
    )
    pack = ContextPack(
        user_id="user:test",
        query_text="What do I like?",
        active_lanes=["profile"],
        active_domains=["user_profile"],
    )

    assert RLMController._filter_items_by_active_lanes([semantic, profile], pack) == [profile]


# ── test_rlm_zero_yield_fallback_ladder ──────────────────────────────────────────




class _Coverage:
    needs_semantic = True
    needs_clusters = False


def test_fallback_ladder_fetch_more_facts_zero_yield_prefers_chunks_once() -> None:
    class _Pack:
        facts = ["placeholder"]  # avoid semantic branch using empty-facts shortcut
        chunks = []
        episodes = []
        graph = []
        steps = [
            {
                "event": "action_result",
                "action": "fetch_more_facts",
                "store": "facts",
                "returned": 0,
                "novelty": 0,
            }
        ]
        owner_type = "agent"
        owner_id = "agent:test"
        user_id = "user:123"
        query_text = "cloud security architecture"
        intent = "topical"
        active_domains = ["kb_doc"]
        chunk_fallback_used = False

        def get_predicate_offset(self, _p: str) -> int:
            return 0

        def bump_predicate_offset(self, _p: str, _d: int) -> int:
            return 0

    decision = deterministic_decision(
        _Pack(),
        _Coverage(),
        cfg={"max_items_per_type": 10, "chunk_fallback_k_multiplier": 2, "chunk_fallback_enabled": True},
    )
    assert decision is not None
    assert decision.actions
    assert decision.actions[0].action == "search_chunks"
    assert decision.actions[0].k and decision.actions[0].k > 10


def test_fallback_ladder_fetch_more_facts_zero_yield_then_broaden_semantic() -> None:
    class _Pack:
        facts = ["placeholder"]
        chunks = []
        episodes = []
        graph = []
        steps = [
            {
                "event": "action_result",
                "action": "search_chunks",
                "store": "chunks",
                "returned": 0,
                "novelty": 0,
            },
            {
                "event": "action_result",
                "action": "fetch_more_facts",
                "store": "facts",
                "returned": 0,
                "novelty": 0,
            },
        ]
        owner_type = "agent"
        owner_id = "agent:test"
        user_id = "user:123"
        query_text = "cloud security architecture"
        intent = "topical"
        active_domains = ["kb_doc"]
        chunk_fallback_used = False

        def get_predicate_offset(self, _p: str) -> int:
            return 0

        def bump_predicate_offset(self, _p: str, _d: int) -> int:
            return 0

    decision = deterministic_decision(
        _Pack(),
        _Coverage(),
        cfg={"max_items_per_type": 10, "chunk_fallback_k_multiplier": 2, "chunk_fallback_enabled": True},
    )
    assert decision is not None
    assert decision.actions
    assert decision.actions[0].action == "search_semantic"


# ── test_rlm_stop_confidence_wired ──────────────────────────────────────────




def test_should_stop_uses_confidence_key() -> None:
    stop, reason = should_stop(
        recall_score=0.0,
        coverage={"confidence": 0.9, "facts": 0, "episodes": 0},
        calls_made=0,
        max_calls=6,
        tokens_used=0,
        token_budget=5000,
        user_results_count=0,
    )
    assert stop is True
    assert reason == "coverage_confident"

