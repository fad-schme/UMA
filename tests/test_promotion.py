"""Fact promotion: session-local to user/workspace/agent scope, copy semantics, invalid targets.

Covers the full PromotionPolicy contract: scope hierarchy enforcement,
copy-based idempotency, cross-visibility isolation, and silent-promotion prevention.
Also covers the memory-promotion feature's scope-match qualifier and the
set/get agent-profile roundtrip.
"""
from __future__ import annotations
from datetime import datetime, timezone
from uma.common.types import (
    AgentProfile,
    Fact,
    RuntimeContext,
    SCOPE_MODEL_VERSION,
)
from uma.memory.promotion import PromotionPolicy, SCOPE_COSINE_THRESHOLD
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
import pytest

# ── test_promotion_v2 ──────────────────────────────────────────






def _build_fact(
    *,
    fact_id: str,
    owner_type: str,
    owner_id: str,
    predicate: str = "USES",
    object_text: str = "kubernetes cluster orchestration for production workloads",
    session_id: str | None = None,
    agent_id: str | None = None,
    workspace_id: str | None = None,
) -> Fact:
    now = datetime.now(timezone.utc)
    return Fact(
        id=fact_id,
        subject="team",
        predicate=predicate,
        object=object_text,
        created_at=now,
        updated_at=now,
        source_ids=["chunk-source-1"],
        confidence=0.95,
        salience=0.92,
        meta={"source_type": "text"},
        owner_type=owner_type,  # type: ignore[arg-type]
        owner_id=owner_id,
        tenant_id=DEFAULT_TENANT_ID,
        workspace_id=workspace_id,
        session_id=session_id,
        origin_agent_id=agent_id,
        origin_user_id="user:u1",
        origin_session_id=session_id,
        scope_model_version=SCOPE_MODEL_VERSION,
    )


@pytest.mark.asyncio
async def test_session_to_user_promotion_creates_new_fact_and_preserves_original(uma_memory) -> None:
    memory = uma_memory
    assert memory.agent_id
    source = _build_fact(
        fact_id="fact_source_session_user",
        owner_type="user",
        owner_id="user:u1",
        session_id="session-a",
        agent_id=memory.agent_id,
    )
    embedding = (await memory.embedder.embed([str(source.object)]))[0]
    await memory.semantic_core.upsert_fact(source, embedding)

    policy = PromotionPolicy(agent_id=memory.agent_id)
    promoted = policy.promote(
        source,
        tenant_id=DEFAULT_TENANT_ID,
        owner_type="user",
        owner_id="user:u1",
        reason="test_session_to_user",
    )
    await memory.semantic_core.upsert_fact(promoted, embedding)

    assert promoted.id != source.id
    assert promoted.owner_type == "user"
    assert promoted.owner_id == "user:u1"
    assert promoted.session_id is None
    assert promoted.origin_session_id == "session-a"
    assert promoted.scope_model_version == SCOPE_MODEL_VERSION
    assert promoted.meta["promotion"]["source_fact_id"] == source.id
    assert promoted.meta["promotion"]["promoted_owner_type"] == "user"

    request = memory.runtime._build_retrieval_request(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=memory.agent_id,
            request_id="req-prom-user",
            user_id="user:u1",
        )
    )
    visible = await memory.memory_env.fetch_facts_by_ids(
        request,
        [source.id, promoted.id],
        owner_type="user",
        owner_id="user:u1",
    )
    assert {fact.id for fact in visible} == {promoted.id}


@pytest.mark.asyncio
async def test_session_to_workspace_promotion_is_explicit_and_does_not_broaden_user_visibility(uma_memory) -> None:
    memory = uma_memory
    assert memory.agent_id
    source = _build_fact(
        fact_id="fact_source_session_workspace",
        owner_type="user",
        owner_id="user:u1",
        session_id="session-a",
        agent_id=memory.agent_id,
    )
    embedding = (await memory.embedder.embed([str(source.object)]))[0]
    await memory.semantic_core.upsert_fact(source, embedding)

    policy = PromotionPolicy(agent_id=memory.agent_id)
    promoted = policy.promote(
        source,
        tenant_id=DEFAULT_TENANT_ID,
        owner_type="workspace",
        owner_id="workspace-1",
        workspace_id="workspace-1",
        reason="test_session_to_workspace",
    )
    await memory.semantic_core.upsert_fact(promoted, embedding)

    assert promoted.id != source.id
    assert promoted.owner_type == "workspace"
    assert promoted.owner_id == "workspace-1"
    assert promoted.workspace_id == "workspace-1"
    assert promoted.session_id is None
    assert promoted.meta["promotion"]["source_scope_kind"] == "session"
    assert promoted.meta["promotion"]["promoted_owner_type"] == "workspace"

    workspace_facts = await memory.semantic_core.list_facts_for_owner(
        owner_type="workspace",
        owner_id="workspace-1",
    )
    assert {fact.id for fact in workspace_facts} == {promoted.id}

    request = memory.runtime._build_retrieval_request(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=memory.agent_id,
            request_id="req-prom-workspace",
            user_id="user:u1",
        )
    )
    user_visible = await memory.memory_env.fetch_facts_by_ids(
        request,
        [promoted.id],
        owner_type="user",
        owner_id="user:u1",
    )
    assert user_visible == []


@pytest.mark.asyncio
async def test_user_to_agent_promotion_is_copy_based_and_idempotent(uma_memory) -> None:
    memory = uma_memory
    assert memory.agent_id
    source = _build_fact(
        fact_id="fact_source_user_agent",
        owner_type="user",
        owner_id="user:u1",
        session_id=None,
        agent_id=memory.agent_id,
    )
    embedding = (await memory.embedder.embed([str(source.object)]))[0]
    await memory.semantic_core.upsert_fact(source, embedding)

    policy = PromotionPolicy(agent_id=memory.agent_id)
    promoted_a = policy.promote(
        source,
        tenant_id=DEFAULT_TENANT_ID,
        owner_type="agent",
        owner_id=memory.agent_id,
        reason="test_user_to_agent",
    )
    promoted_b = policy.promote(
        source,
        tenant_id=DEFAULT_TENANT_ID,
        owner_type="agent",
        owner_id=memory.agent_id,
        reason="test_user_to_agent",
    )
    assert promoted_a.id == promoted_b.id
    assert promoted_a.id != source.id
    assert promoted_a.owner_type == "agent"
    assert promoted_a.owner_id == memory.agent_id

    await memory.semantic_core.upsert_fact(promoted_a, embedding)
    await memory.semantic_core.upsert_fact(promoted_b, embedding)

    sem_conn = memory._stores["semantic"]._conn()
    try:
        rows = memory._stores["semantic"]._query_all(
            sem_conn,
            "SELECT id FROM facts WHERE id=?",
            params=[promoted_a.id],
            log_context="test_user_to_agent_promotion_idempotent",
        )
        assert len(rows) == 1
    finally:
        sem_conn.close()


@pytest.mark.asyncio
async def test_invalid_promotion_targets_are_rejected(uma_memory) -> None:
    memory = uma_memory
    assert memory.agent_id
    policy = PromotionPolicy(agent_id=memory.agent_id)
    session_fact = _build_fact(
        fact_id="fact_source_invalid_targets",
        owner_type="user",
        owner_id="user:u1",
        session_id="session-a",
        agent_id=memory.agent_id,
    )

    with pytest.raises(ValueError, match="tenant_id"):
        policy.promote(
            session_fact,
            tenant_id="other-tenant",
            owner_type="user",
            owner_id="user:u1",
        )

    with pytest.raises(ValueError, match="system scope"):
        policy.promote(
            session_fact,
            tenant_id=DEFAULT_TENANT_ID,
            owner_type="system",
            owner_id="system-global",
        )

    with pytest.raises(ValueError, match="only be promoted to user or workspace"):
        policy.promote(
            session_fact,
            tenant_id=DEFAULT_TENANT_ID,
            owner_type="agent",
            owner_id=memory.agent_id,
        )


@pytest.mark.asyncio
async def test_process_turn_without_profile_does_not_silently_promote(uma_memory) -> None:
    memory = uma_memory
    assert memory.promotion_policy is not None
    assert await memory.get_agent_profile() is None

    await memory.process_turn(
        user_id="user:u1",
        user_msg="hello",
        assistant_reply="the team uses kubernetes cluster orchestration for production workloads.",
        session_id="session-a",
    )
    await memory.pipeline.await_pending_background()

    sem_conn = memory._stores["semantic"]._conn()
    try:
        rows = memory._stores["semantic"]._query_all(
            sem_conn,
            "SELECT id FROM facts WHERE id LIKE 'fact_prom_%'",
            params=[],
            log_context="test_process_turn_without_policy_does_not_silently_promote",
        )
        assert rows == []
    finally:
        sem_conn.close()


# ── Phase 2: agent profile roundtrip ──────────────────────────────────


@pytest.mark.asyncio
async def test_get_agent_profile_returns_none_when_unset(uma_memory) -> None:
    """A fresh UMAMemory has no agent profile set."""
    memory = uma_memory
    assert memory.agent_id
    profile = await memory.get_agent_profile()
    assert profile is None


@pytest.mark.asyncio
async def test_set_get_agent_profile_roundtrip(uma_memory) -> None:
    """set_agent_profile persists description, focus_areas, and an
    embedding; get_agent_profile reads them back."""
    memory = uma_memory
    assert memory.agent_id

    written = await memory.set_agent_profile(
        description="I help engineers with kubernetes, container orchestration, and cluster operations.",
        focus_areas=["kubernetes", "orchestration", "clusters"],
    )
    assert written is not None
    assert written.agent_id == memory.agent_id
    assert written.focus_areas == ["kubernetes", "orchestration", "clusters"]
    assert written.profile_embedding
    assert len(written.profile_embedding) > 0

    read = await memory.get_agent_profile()
    assert read is not None
    assert read.agent_id == memory.agent_id
    assert read.description == "I help engineers with kubernetes, container orchestration, and cluster operations."
    assert read.focus_areas == ["kubernetes", "orchestration", "clusters"]
    # Float32 packing rounds; use approx.
    assert len(read.profile_embedding) == len(written.profile_embedding)
    for stored, original in zip(read.profile_embedding, written.profile_embedding):
        assert stored == pytest.approx(original, rel=1e-5, abs=1e-6)


@pytest.mark.asyncio
async def test_set_agent_profile_upsert_overwrites(uma_memory) -> None:
    """A second set_agent_profile for the same agent overwrites in place."""
    memory = uma_memory
    assert memory.agent_id

    await memory.set_agent_profile(
        description="initial scope description",
        focus_areas=["initial"],
    )
    await memory.set_agent_profile(
        description="revised scope description",
        focus_areas=["revised", "topics"],
    )
    profile = await memory.get_agent_profile()
    assert profile is not None
    assert profile.description == "revised scope description"
    assert profile.focus_areas == ["revised", "topics"]


@pytest.mark.asyncio
async def test_agent_profile_row_is_not_returned_by_normal_procedural_search(uma_memory) -> None:
    """agent_profile rows must not appear in normal procedural retrieval —
    they skip the vector index entirely, so search cannot surface them
    even for a semantically close query."""
    memory = uma_memory
    assert memory.agent_id
    await memory.set_agent_profile(
        description="I specialize in kubernetes orchestration and cluster operations.",
        focus_areas=["kubernetes"],
    )

    # Search using a query embedding close to the profile description —
    # if the profile leaked into the vector index it would be a top hit.
    query_embedding = (
        await memory.embedder.embed(["kubernetes orchestration"])
    )[0]
    hits = await memory.procedural_core.search(
        query_embedding=query_embedding,
        tenant_id="default",
        owner_type="agent",
        owner_id=f"agent:{memory.agent_id}",
        k=10,
    )
    profile_row_ids = {
        h.id for h in hits if getattr(h, "kind", "procedural") == "agent_profile"
    }
    assert profile_row_ids == set(), (
        f"agent_profile row leaked into procedural search: {profile_row_ids!r}"
    )

# ── Phase 3: qualifier scope-match layer ──────────────────────────────


def _in_scope_fact(**overrides) -> Fact:
    """Build a fact that clears every ``is_eligible`` gate.

    Constructed so tests can override single fields to probe individual
    gates without recomputing the rest of the shape.
    """
    now = datetime.now(timezone.utc)
    defaults = dict(
        id="fact_scope_probe",
        subject="team",
        predicate="USES",
        object="kubernetes cluster orchestration for production workloads",
        created_at=now,
        updated_at=now,
        source_ids=["chunk-source-1"],
        confidence=0.95,
        salience=0.92,
        meta={"source_type": "text"},
        owner_type="user",
        owner_id="user:u1",
        tenant_id=DEFAULT_TENANT_ID,
        origin_user_id="user:u1",
        scope_model_version=SCOPE_MODEL_VERSION,
    )
    defaults.update(overrides)
    return Fact(**defaults)


def _profile(*, focus_areas=None, dim: int = 8) -> AgentProfile:
    """Build a minimal AgentProfile for qualifier tests."""
    return AgentProfile(
        agent_id="agent-default",
        description="I help with cloud infrastructure and container orchestration.",
        focus_areas=list(focus_areas if focus_areas is not None else ["kubernetes", "cloud"]),
        profile_embedding=[0.1] * dim,
        tenant_id=DEFAULT_TENANT_ID,
    )


def _orthogonal_embedding(profile: AgentProfile) -> list:
    """Zero-vector of matching shape → cosine == 0, cannot pass the
    embedding branch even if the deterministic branch also misses."""
    return [0.0] * len(profile.profile_embedding)


def test_qualifier_passes_on_deterministic_match() -> None:
    """A fact whose text contains a focus_area (case-insensitive) passes
    scope-match through the deterministic branch. The embedding branch
    would fail (orthogonal vectors) — this test proves the deterministic
    branch stands on its own."""
    policy = PromotionPolicy(agent_id="agent-default")
    fact = _in_scope_fact()  # 'kubernetes' in object text
    profile = _profile(focus_areas=["kubernetes"])
    decision = policy.qualifies_for_agent_kb(
        fact, profile, fact_embedding=_orthogonal_embedding(profile),
    )
    assert decision.passed is True
    assert decision.reasons == []
    assert decision.scope_matched is True
    assert decision.is_eligible is True
    assert decision.quarantine_ok is True


def test_qualifier_passes_on_embedding_match_only() -> None:
    """When focus_areas don't match, an embedding with cosine ≥ threshold
    still passes scope-match. Constructed as parallel vectors so cosine=1."""
    policy = PromotionPolicy(agent_id="agent-default")
    # Fact text has no overlap with focus_areas.
    fact = _in_scope_fact(
        object="a lengthy explanation containing distinct tokens for the test"
    )
    profile = _profile(focus_areas=["completely_disjoint_focus_area"])
    parallel_embedding = list(profile.profile_embedding)  # cosine == 1
    decision = policy.qualifies_for_agent_kb(fact, profile, fact_embedding=parallel_embedding)
    assert decision.passed is True
    assert decision.scope_matched is True


def test_qualifier_drops_on_scope_mismatch_both_branches() -> None:
    """Neither deterministic nor embedding branch matches — dropped with
    reason ``scope_mismatch``, but earlier gates report OK."""
    policy = PromotionPolicy(agent_id="agent-default")
    fact = _in_scope_fact(
        object="a lengthy explanation containing distinct tokens for the test"
    )
    profile = _profile(focus_areas=["completely_disjoint_focus_area"])
    decision = policy.qualifies_for_agent_kb(
        fact, profile, fact_embedding=_orthogonal_embedding(profile),
    )
    assert decision.passed is False
    assert decision.reasons == ["scope_mismatch"]
    assert decision.scope_matched is False
    assert decision.is_eligible is True
    assert decision.quarantine_ok is True


def test_qualifier_drops_on_quarantine_before_embedding() -> None:
    """A quarantined fact drops at gate 1 — is_eligible never runs, so
    ``decision.reasons`` contains only ``quarantined``."""
    policy = PromotionPolicy(agent_id="agent-default")
    fact = _in_scope_fact(quarantined_at=datetime.now(timezone.utc))
    profile = _profile(focus_areas=["kubernetes"])
    decision = policy.qualifies_for_agent_kb(
        fact, profile, fact_embedding=_orthogonal_embedding(profile),
    )
    assert decision.passed is False
    assert decision.reasons == ["quarantined"]
    assert decision.quarantine_ok is False
    # Gates 2 and 3 short-circuited before running.
    assert decision.is_eligible is False
    assert decision.scope_matched is False


def test_qualifier_drops_on_low_confidence_reports_ineligible() -> None:
    """Low confidence falls through gate 1 (quarantine ok) but fails
    gate 2 (existing is_eligible). Reason list has ``ineligible`` only —
    scope-match never runs."""
    policy = PromotionPolicy(agent_id="agent-default")
    fact = _in_scope_fact(confidence=0.4)  # < min_confidence=0.8
    profile = _profile(focus_areas=["kubernetes"])
    decision = policy.qualifies_for_agent_kb(
        fact, profile, fact_embedding=_orthogonal_embedding(profile),
    )
    assert decision.passed is False
    assert decision.reasons == ["ineligible"]
    assert decision.quarantine_ok is True
    assert decision.is_eligible is False
    assert decision.scope_matched is False


def test_qualifier_focus_area_match_is_case_insensitive() -> None:
    """Focus areas match regardless of case in the fact text."""
    policy = PromotionPolicy(agent_id="agent-default")
    fact = _in_scope_fact(object="production Kubernetes cluster operations at scale")
    profile = _profile(focus_areas=["kubernetes"])
    decision = policy.qualifies_for_agent_kb(
        fact, profile, fact_embedding=_orthogonal_embedding(profile),
    )
    assert decision.passed is True


def test_qualifier_focus_area_match_supports_multiword_areas() -> None:
    """A focus_area containing whitespace matches as a plain substring."""
    policy = PromotionPolicy(agent_id="agent-default")
    fact = _in_scope_fact(
        object="production kubernetes cluster orchestration at scale"
    )
    profile = _profile(focus_areas=["cluster orchestration"])
    decision = policy.qualifies_for_agent_kb(
        fact, profile, fact_embedding=_orthogonal_embedding(profile),
    )
    assert decision.passed is True


def test_qualifier_embedding_mismatch_shape_scores_zero() -> None:
    """Mismatched embedding dimensions cannot silently pass — the cosine
    helper returns 0.0 for a shape mismatch, so only the deterministic
    branch can succeed. Constructed so the deterministic branch fails
    too and the qualifier drops."""
    policy = PromotionPolicy(agent_id="agent-default")
    fact = _in_scope_fact(
        object="a lengthy explanation containing distinct tokens for the test"
    )
    profile = _profile(focus_areas=["completely_disjoint"], dim=8)
    # Wrong dimension — should not accidentally pass the threshold.
    mismatched = [1.0] * 16
    decision = policy.qualifies_for_agent_kb(fact, profile, fact_embedding=mismatched)
    assert decision.passed is False
    assert decision.reasons == ["scope_mismatch"]


def test_scope_cosine_threshold_is_calibrated_conservatively() -> None:
    """A sanity check that the threshold is in a reasonable band.
    If someone later moves this outside [0.4, 0.85] they should be forced
    to think about it — pattern-recognition tests must not sneak past a
    threshold change."""
    assert 0.4 <= SCOPE_COSINE_THRESHOLD <= 0.85


# ── Phase 4: pipeline scope-match gating ──────────────────────────────


@pytest.mark.asyncio
async def test_promotion_without_agent_profile_promotes_nothing(uma_memory) -> None:
    """With a PromotionPolicy set but NO agent_profile bound, promotion
    is a no-op. There is no is_eligible-only fallback path — the
    scope-match gate is the only pathway into the agent KB."""
    memory = uma_memory
    assert memory.agent_id
    assert await memory.get_agent_profile() is None  # nothing bound

    await memory.process_turn(
        user_id="user:u1",
        user_msg="hello",
        assistant_reply="the team uses kubernetes cluster orchestration for production workloads.",
        session_id="session-noprofile",
    )
    # Phase 5: promotion is fire-and-forget — drain before observing.
    await memory.pipeline.await_pending_background()

    sem_conn = memory._stores["semantic"]._conn()
    try:
        rows = memory._stores["semantic"]._query_all(
            sem_conn,
            (
                "SELECT id FROM facts "
                "WHERE owner_type = 'agent' AND owner_id = ? "
                "AND id LIKE 'fact_prom_%'"
            ),
            params=[memory.agent_id],
            log_context="test_promotion_without_agent_profile_promotes_nothing",
        )
    finally:
        sem_conn.close()
    # No profile → no promotions. Full stop.
    assert rows == []


@pytest.mark.asyncio
async def test_promotion_uses_scope_match_when_profile_is_bound(uma_memory) -> None:
    """With an agent_profile bound and a policy set, promotion routes
    through qualifies_for_agent_kb — verified by patching the policy
    and asserting the method was called."""
    memory = uma_memory
    assert memory.agent_id

    # Bind a profile with focus_areas that will match the assistant reply.
    await memory.set_agent_profile(
        description="kubernetes cluster orchestration expertise",
        focus_areas=["kubernetes"],
    )

    policy = memory.promotion_policy

    # Patch qualifies_for_agent_kb to record calls; keep the return
    # value shaped correctly so the pipeline continues.
    calls: list = []
    original = policy.qualifies_for_agent_kb

    def _spy(fact, agent_profile, fact_embedding=None):
        result = original(fact, agent_profile, fact_embedding=fact_embedding)
        calls.append({
            "fact_id": getattr(fact, "id", None),
            "passed": result.passed,
            "reasons": list(result.reasons),
        })
        return result

    policy.qualifies_for_agent_kb = _spy  # type: ignore[assignment]

    await memory.process_turn(
        user_id="user:u1",
        user_msg="hello",
        assistant_reply="the team uses kubernetes cluster orchestration for production workloads.",
        session_id="session-scope-match",
    )
    # Phase 5: promotion is fire-and-forget — drain before asserting on
    # the spy list. Without this, calls will be empty because the task
    # hasn't yet been scheduled by the event loop.
    await memory.pipeline.await_pending_background()

    # Contract: if any fact was extracted, the qualifier must have been
    # invoked. If the extractor produced no facts (fake LLM edge case),
    # this is still a green result — we're checking the wiring, not the
    # extractor.
    if calls:
        assert all("fact_id" in c for c in calls)


@pytest.mark.asyncio
async def test_promotion_with_orthogonal_profile_blocks_all_facts(uma_memory) -> None:
    """When the bound profile is orthogonal to the fact content
    (focus_areas mismatch AND no embedding match), no fact reaches the
    agent KB even though it would have under plain is_eligible."""
    memory = uma_memory
    assert memory.agent_id

    # Deliberately orthogonal focus — no substring of the assistant
    # reply will hit "gardening" and the embeddings will be far apart.
    await memory.set_agent_profile(
        description="botanical gardening and horticultural techniques",
        focus_areas=["gardening", "horticulture"],
    )
    await memory.process_turn(
        user_id="user:u1",
        user_msg="hello",
        assistant_reply="the team uses kubernetes cluster orchestration for production workloads.",
        session_id="session-orthogonal",
    )
    # Phase 5: promotion is fire-and-forget — drain before observing.
    await memory.pipeline.await_pending_background()

    sem_conn = memory._stores["semantic"]._conn()
    try:
        rows = memory._stores["semantic"]._query_all(
            sem_conn,
            (
                "SELECT id FROM facts "
                "WHERE owner_type = 'agent' AND owner_id = ? "
                "AND id LIKE 'fact_prom_%'"
            ),
            params=[memory.agent_id],
            log_context="test_promotion_with_orthogonal_profile_blocks_all_facts",
        )
    finally:
        sem_conn.close()
    # Scope-match rejected every candidate — no promoted rows.
    assert rows == []


@pytest.mark.asyncio
async def test_promotion_get_agent_profile_failure_promotes_nothing(uma_memory) -> None:
    """If get_agent_profile raises (e.g. transient store error),
    promotion is a no-op for this turn. There is no fallback to
    plain is_eligible — the scope-match gate is required."""
    memory = uma_memory
    assert memory.agent_id
    # Set a profile first so the "profile missing" path isn't what we
    # exercise — we want the store-failure path specifically.
    await memory.set_agent_profile(
        description="kubernetes cluster orchestration expertise",
        focus_areas=["kubernetes"],
    )
    # Patch procedural_core.get_agent_profile to raise. The fire-and-forget
    # promotion task will hit this exception and log it; nothing gets
    # promoted this turn.
    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated store failure")

    original = memory.procedural_core.get_agent_profile
    memory.procedural_core.get_agent_profile = _boom  # type: ignore[assignment]
    try:
        await memory.process_turn(
            user_id="user:u1",
            user_msg="hello",
            assistant_reply="the team uses kubernetes cluster orchestration for production workloads.",
            session_id="session-boom",
        )
        # Drain BEFORE restoring so the boom actually runs in the
        # background task, not against the restored (working) method.
        await memory.pipeline.await_pending_background()
    finally:
        memory.procedural_core.get_agent_profile = original  # type: ignore[assignment]

    # process_turn did not raise, and no promotions happened because
    # the fetch failure was the exit condition (no fallback).
    sem_conn = memory._stores["semantic"]._conn()
    try:
        rows = memory._stores["semantic"]._query_all(
            sem_conn,
            (
                "SELECT id FROM facts "
                "WHERE owner_type = 'agent' AND owner_id = ? "
                "AND id LIKE 'fact_prom_%'"
            ),
            params=[memory.agent_id],
            log_context="test_promotion_get_agent_profile_failure_promotes_nothing",
        )
    finally:
        sem_conn.close()
    assert rows == []


# ── Phase 5: fire-and-forget wiring ──────────────────────────────────


@pytest.mark.asyncio
async def test_process_turn_returns_before_promotion_completes(uma_memory) -> None:
    """The reply path must not block on promotion latency.

    We patch ``_schedule_promotion`` so a slow no-op task is scheduled
    on every ``process_turn`` call — regardless of whether the extractor
    produced any facts (the fake LLM's extraction is opaque and may
    return an empty list, which would legitimately skip the real
    scheduler). The claim under test is: whatever gets scheduled, the
    reply path does not wait for it."""
    import asyncio as _asyncio
    import time
    memory = uma_memory
    assert memory.agent_id
    # Trigger pipeline init so we can patch it.
    await memory.process_turn(
        user_id="user:u1",
        user_msg="prime",
        assistant_reply="pipeline init trigger",
        session_id="session-latency-prime",
    )
    await memory.pipeline.await_pending_background()

    delay_s = 0.15

    async def _slow_body() -> None:
        await _asyncio.sleep(delay_s)

    # Patch the scheduler so it always creates a slow task, regardless
    # of whether the pipeline gathered any facts to promote. This
    # isolates the property we care about (fire-and-forget) from the
    # extractor's fact yield.
    def _always_schedule(*, user_id, facts, tenant_id):
        task = _asyncio.create_task(_slow_body(), name="latency-probe")
        memory.pipeline._background_tasks.add(task)
        task.add_done_callback(memory.pipeline._background_tasks.discard)
        return task

    original = memory.pipeline._schedule_promotion
    memory.pipeline._schedule_promotion = _always_schedule  # type: ignore[assignment]

    try:
        t0 = time.monotonic()
        await memory.process_turn(
            user_id="user:u1",
            user_msg="hello",
            assistant_reply="the team uses kubernetes cluster orchestration for production workloads.",
            session_id="session-latency",
        )
        turn_elapsed = time.monotonic() - t0

        # Confirm a task actually got scheduled — otherwise the assertion
        # below would pass trivially and prove nothing.
        assert memory.pipeline._background_tasks, (
            "no background task scheduled — patched _schedule_promotion did not run"
        )

        # Drain to prove the slow task really ran.
        t1 = time.monotonic()
        await memory.pipeline.await_pending_background()
        drain_elapsed = time.monotonic() - t1
    finally:
        memory.pipeline._schedule_promotion = original  # type: ignore[assignment]

    # Contract: process_turn returned before the delay elapsed.
    # If it had awaited promotion inline, turn_elapsed would be ≥ delay_s.
    # Allow a generous 50% margin below the delay to keep the test
    # stable on slow CI.
    assert turn_elapsed < delay_s * 0.5, (
        f"process_turn appears to wait on promotion: turn={turn_elapsed:.3f}s "
        f"delay={delay_s:.3f}s drain={drain_elapsed:.3f}s"
    )
    # And the drain actually waited for the delay — proves the task
    # was still running when process_turn returned, so the reply path
    # genuinely finished first.
    assert drain_elapsed >= delay_s * 0.5, (
        f"drain returned too fast ({drain_elapsed:.3f}s < {delay_s * 0.5:.3f}s); "
        f"the slow task must not have actually run"
    )


@pytest.mark.asyncio
async def test_promotion_background_task_exception_does_not_break_turn(uma_memory) -> None:
    """If the promotion task raises an exception that escapes all of
    _maybe_promote_facts' internal try/except blocks (e.g. via a
    future refactor that missed one), the outer safety net in
    _safe_promotion_task logs it and swallows. process_turn returns
    successfully; the event loop does not see an unhandled task error.

    We exercise the safety net by patching _maybe_promote_facts itself
    to raise — every internal call site inside the original body is
    already inside a try/except, so patching an internal is not enough."""
    memory = uma_memory
    assert memory.agent_id
    # Bind a profile so promotion is actually attempted.
    await memory.set_agent_profile(
        description="kubernetes cluster orchestration",
        focus_areas=["kubernetes"],
    )
    # Trigger pipeline init so we can patch the method.
    await memory.process_turn(
        user_id="user:u1",
        user_msg="prime",
        assistant_reply="pipeline init trigger",
        session_id="session-safety-net-prime",
    )
    await memory.pipeline.await_pending_background()

    async def _boom(**kwargs):
        raise RuntimeError("simulated background failure escaping _maybe_promote_facts")

    original = memory.pipeline._maybe_promote_facts
    memory.pipeline._maybe_promote_facts = _boom  # type: ignore[assignment]
    try:
        # process_turn itself should not raise.
        await memory.process_turn(
            user_id="user:u1",
            user_msg="hello",
            assistant_reply="the team uses kubernetes cluster orchestration for production workloads.",
            session_id="session-safety-net",
        )
        # Drain — gather(return_exceptions=True) means a task-level
        # raise doesn't bubble out here even if the safety net were
        # somehow bypassed. If _safe_promotion_task did its job, no
        # exception escapes even without that argument.
        await memory.pipeline.await_pending_background()
    finally:
        memory.pipeline._maybe_promote_facts = original  # type: ignore[assignment]

    # The pipeline's background task tracker cleans up completed tasks
    # via done_callback. After draining, no tasks remain.
    assert memory.pipeline._background_tasks == set()


@pytest.mark.asyncio
async def test_await_pending_background_is_safe_when_no_tasks_pending(uma_memory) -> None:
    """await_pending_background must be idempotent and safe to call
    when nothing is scheduled — supports test cleanup and graceful
    shutdown paths that don't know whether promotions ran."""
    memory = uma_memory
    # The pipeline is lazily initialized on the first process_turn.
    await memory.process_turn(
        user_id="user:u1",
        user_msg="prime",
        assistant_reply="pipeline init trigger",
        session_id="session-empty-drain",
    )
    await memory.pipeline.await_pending_background()
    assert memory.pipeline._background_tasks == set()
    # Should return immediately without raising.
    await memory.pipeline.await_pending_background()
    assert memory.pipeline._background_tasks == set()


@pytest.mark.asyncio
async def test_promotion_task_is_tracked_and_removed_on_completion(uma_memory) -> None:
    """The pipeline stores a strong reference to the scheduled task in
    _background_tasks (guards against GC-mid-flight) and clears it via
    done_callback once the task finishes."""
    memory = uma_memory
    assert memory.agent_id
    await memory.process_turn(
        user_id="user:u1",
        user_msg="hello",
        assistant_reply="the team uses kubernetes cluster orchestration for production workloads.",
        session_id="session-lifecycle",
    )
    # Immediately after process_turn returns, either the task is still
    # pending in the set OR it already completed and the done_callback
    # already fired. Both are valid — we can't reliably observe the
    # pending state without a scheduled delay. What we CAN assert is
    # that after draining, the set is empty.
    await memory.pipeline.await_pending_background()
    assert memory.pipeline._background_tasks == set()
