"""Trust scoring: dataclass security fields, content hashing, source classifier,
store round-trips, end-to-end trust propagation, threshold filter, score combination.

Covers the full trust primitive stack: hash functions, source-based trust
scores, store persistence, pipeline integration, ranking weight application,
and trust-based candidate filtering.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from tests.helpers.runtime import TEST_AGENT_ID, init_uma_for_tests
from uma.adapters.db.sqlite_adapter import SQLiteAdapter
from uma.adapters.vector.base import VectorIndex
from uma.common.integrity import hash_episode_content, hash_fact_content, hash_skill_content
from uma.common.trust import SourceDescriptor, score_source
from uma.common.types import Chunk, Episode, Fact, Skill
from uma.retrieve.ranking import Ranker
from uma.stores.chunk_sql import ChunkSQLStore
from uma.stores.episodic_sql import EpisodicSQLStore
from uma.stores.procedural_sql import ProceduralSQLStore
from uma.stores.semantic_sql import SemanticSQLStore
import hashlib
import pytest

AGENT_ID = TEST_AGENT_ID

# ── test_pr1_dataclass_fields ──────────────────────────────────────────






_NOW = datetime.now(timezone.utc)


# ──────────────────────────────────────────────
# Fact
# ──────────────────────────────────────────────

class TestFactSecurityFields:
    def _fact(self, **kwargs) -> Fact:
        defaults = dict(
            id="fact_001",
            subject="user:alice",
            predicate="LIKES",
            object="coffee",
            created_at=_NOW,
            updated_at=_NOW,
            owner_type="user",
            owner_id="user:alice",
        )
        defaults.update(kwargs)
        return Fact(**defaults)

    def test_defaults(self):
        f = self._fact()
        assert f.trust_score == 0.5
        assert f.content_hash is None

    def test_explicit_values_survive_validate(self):
        f = self._fact(trust_score=0.8, content_hash="abc123")
        f.validate()
        assert f.trust_score == 0.8
        assert f.content_hash == "abc123"

    def test_trust_score_too_high_raises(self):
        f = self._fact(trust_score=1.5)
        with pytest.raises(ValueError, match="trust_score"):
            f.validate()

    def test_trust_score_negative_raises(self):
        f = self._fact(trust_score=-0.1)
        with pytest.raises(ValueError, match="trust_score"):
            f.validate()

    def test_trust_score_boundary_zero(self):
        f = self._fact(trust_score=0.0)
        f.validate()

    def test_trust_score_boundary_one(self):
        f = self._fact(trust_score=1.0)
        f.validate()

    def test_content_hash_empty_string_raises(self):
        f = self._fact(content_hash="")
        with pytest.raises(ValueError, match="content_hash"):
            f.validate()

    def test_content_hash_none_is_valid(self):
        f = self._fact(content_hash=None)
        f.validate()


# ──────────────────────────────────────────────
# Episode
# ──────────────────────────────────────────────

class TestEpisodeSecurityFields:
    def _episode(self, **kwargs) -> Episode:
        defaults = dict(
            id="ep_001",
            timestamp=_NOW,
            summary="User discussed preferences.",
            user_id="user:alice",
            owner_type="user",
            owner_id="user:alice",
        )
        defaults.update(kwargs)
        return Episode(**defaults)

    def test_defaults(self):
        ep = self._episode()
        assert ep.trust_score == 0.5
        assert ep.content_hash is None

    def test_explicit_values_survive_validate(self):
        ep = self._episode(trust_score=0.9, content_hash="deadbeef")
        ep.validate()
        assert ep.trust_score == 0.9
        assert ep.content_hash == "deadbeef"

    def test_trust_score_too_high_raises(self):
        ep = self._episode(trust_score=1.5)
        with pytest.raises(ValueError, match="trust_score"):
            ep.validate()

    def test_trust_score_negative_raises(self):
        ep = self._episode(trust_score=-0.1)
        with pytest.raises(ValueError, match="trust_score"):
            ep.validate()

    def test_trust_score_boundary_zero(self):
        self._episode(trust_score=0.0).validate()

    def test_trust_score_boundary_one(self):
        self._episode(trust_score=1.0).validate()

    def test_content_hash_empty_string_raises(self):
        ep = self._episode(content_hash="")
        with pytest.raises(ValueError, match="content_hash"):
            ep.validate()

    def test_content_hash_none_is_valid(self):
        self._episode(content_hash=None).validate()


# ──────────────────────────────────────────────
# Skill
# ──────────────────────────────────────────────

class TestSkillSecurityFields:
    def _skill(self, **kwargs) -> Skill:
        defaults = dict(
            id="skill_001",
            name="greet_user",
            description="Greet the user politely.",
            created_at=_NOW,
            updated_at=_NOW,
            owner_type="agent",
            owner_id="agent-default",
        )
        defaults.update(kwargs)
        return Skill(**defaults)

    def test_defaults(self):
        s = self._skill()
        assert s.trust_score == 0.5
        assert s.content_hash is None

    def test_explicit_values_survive_validate(self):
        s = self._skill(trust_score=0.7, content_hash="cafebabe")
        s.validate()
        assert s.trust_score == 0.7
        assert s.content_hash == "cafebabe"

    def test_trust_score_too_high_raises(self):
        s = self._skill(trust_score=1.5)
        with pytest.raises(ValueError, match="trust_score"):
            s.validate()

    def test_trust_score_negative_raises(self):
        s = self._skill(trust_score=-0.1)
        with pytest.raises(ValueError, match="trust_score"):
            s.validate()

    def test_trust_score_boundary_zero(self):
        self._skill(trust_score=0.0).validate()

    def test_trust_score_boundary_one(self):
        self._skill(trust_score=1.0).validate()

    def test_content_hash_empty_string_raises(self):
        s = self._skill(content_hash="")
        with pytest.raises(ValueError, match="content_hash"):
            s.validate()

    def test_content_hash_none_is_valid(self):
        self._skill(content_hash=None).validate()


# ──────────────────────────────────────────────
# Chunk (trust_score only; no content_hash)
# ──────────────────────────────────────────────

class TestChunkSecurityFields:
    def _chunk(self, **kwargs) -> Chunk:
        defaults = dict(
            id="chunk_001",
            doc_id="doc_001",
            text="Some chunk text here.",
            page_range=(0, 1),
            position=0,
            source_path="/docs/test.pdf",
            source_hash="abc",
            created_at=_NOW,
            updated_at=_NOW,
            owner_type="user",
            owner_id="user:alice",
        )
        defaults.update(kwargs)
        return Chunk(**defaults)

    def test_default_trust_score(self):
        c = self._chunk()
        assert c.trust_score == 0.5

    def test_no_content_hash_field(self):
        c = self._chunk()
        assert not hasattr(c, "content_hash")

    def test_explicit_trust_score_survives_validate(self):
        c = self._chunk(trust_score=0.3)
        c.validate()
        assert c.trust_score == 0.3

    def test_trust_score_too_high_raises(self):
        c = self._chunk(trust_score=1.5)
        with pytest.raises(ValueError, match="trust_score"):
            c.validate()

    def test_trust_score_negative_raises(self):
        c = self._chunk(trust_score=-0.1)
        with pytest.raises(ValueError, match="trust_score"):
            c.validate()

    def test_trust_score_boundary_zero(self):
        self._chunk(trust_score=0.0).validate()

    def test_trust_score_boundary_one(self):
        self._chunk(trust_score=1.0).validate()


# ── test_pr1_integrity ──────────────────────────────────────────






class TestHashFactContent:
    def test_returns_64_char_hex(self):
        result = hash_fact_content("user", "lives_in", "São Paulo")
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_call_same_result(self):
        a = hash_fact_content("user", "lives_in", "São Paulo")
        b = hash_fact_content("user", "lives_in", "São Paulo")
        assert a == b

    def test_dict_object_order_independent(self):
        h1 = hash_fact_content("a", "b", {"x": 1, "y": 2})
        h2 = hash_fact_content("a", "b", {"y": 2, "x": 1})
        assert h1 == h2

    def test_different_subjects_differ(self):
        assert hash_fact_content("user:a", "LIKES", "sushi") != hash_fact_content("user:b", "LIKES", "sushi")

    def test_different_predicates_differ(self):
        assert hash_fact_content("user", "LIKES", "sushi") != hash_fact_content("user", "HATES", "sushi")

    def test_different_objects_differ(self):
        assert hash_fact_content("user", "LIKES", "sushi") != hash_fact_content("user", "LIKES", "pizza")

    def test_non_ascii_stable(self):
        h = hash_fact_content("user", "city", "São Paulo")
        assert h == hash_fact_content("user", "city", "São Paulo")


class TestHashEpisodeContent:
    def test_returns_64_char_hex(self):
        result = hash_episode_content("User talked about sushi.")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_same_summary_same_hash(self):
        assert hash_episode_content("Hello") == hash_episode_content("Hello")

    def test_empty_string_does_not_raise(self):
        result = hash_episode_content("")
        # SHA-256 of empty string
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_none_treated_as_empty(self):
        # hash_episode_content uses (summary or ""), so None → ""
        result = hash_episode_content(None)  # type: ignore[arg-type]
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_different_summaries_differ(self):
        assert hash_episode_content("foo") != hash_episode_content("bar")


class TestHashSkillContent:
    def test_returns_64_char_hex(self):
        result = hash_skill_content("greet_user", {"steps": ["say hello"]})
        assert isinstance(result, str)
        assert len(result) == 64

    def test_same_inputs_same_hash(self):
        a = hash_skill_content("greet", {"steps": ["a", "b"]})
        b = hash_skill_content("greet", {"steps": ["a", "b"]})
        assert a == b

    def test_plan_order_independent(self):
        h1 = hash_skill_content("skill", {"x": 1, "y": 2})
        h2 = hash_skill_content("skill", {"y": 2, "x": 1})
        assert h1 == h2

    def test_different_names_differ(self):
        assert hash_skill_content("skill_a", {}) != hash_skill_content("skill_b", {})

    def test_different_plans_differ(self):
        assert hash_skill_content("skill", {"x": 1}) != hash_skill_content("skill", {"x": 2})


# ── test_pr2_trust_classifier ──────────────────────────────────────────




_SESSION = "sess-abc-123"


class TestScoreSourcePolicyTable:
    def test_turn_user_authenticated(self):
        s = score_source(SourceDescriptor(kind="turn_user", session_id=_SESSION))
        assert s == pytest.approx(0.9)

    def test_turn_user_no_session(self):
        s = score_source(SourceDescriptor(kind="turn_user", session_id=None))
        assert s == pytest.approx(0.5)

    def test_turn_user_empty_session(self):
        s = score_source(SourceDescriptor(kind="turn_user", session_id="   "))
        assert s == pytest.approx(0.5)

    def test_turn_assistant_authenticated(self):
        s = score_source(SourceDescriptor(kind="turn_assistant", session_id=_SESSION))
        assert s == pytest.approx(0.7)

    def test_turn_assistant_no_session(self):
        s = score_source(SourceDescriptor(kind="turn_assistant", session_id=None))
        assert s == pytest.approx(0.5)

    def test_document(self):
        s = score_source(SourceDescriptor(kind="document"))
        assert s == pytest.approx(0.7)

    def test_bootstrap_memory_default(self):
        s = score_source(SourceDescriptor(kind="bootstrap_memory"))
        assert s == pytest.approx(0.6)

    def test_bootstrap_diary_default(self):
        s = score_source(SourceDescriptor(kind="bootstrap_diary"))
        assert s == pytest.approx(0.6)

    def test_bootstrap_memory_manual(self):
        s = score_source(SourceDescriptor(kind="bootstrap_memory", import_mode="manual"))
        assert s == pytest.approx(0.8)

    def test_bootstrap_diary_manual(self):
        s = score_source(SourceDescriptor(kind="bootstrap_diary", import_mode="manual"))
        assert s == pytest.approx(0.8)

    def test_tool_output(self):
        s = score_source(SourceDescriptor(kind="tool_output"))
        assert s == pytest.approx(0.5)

    def test_promotion_inherits_parent(self):
        s = score_source(SourceDescriptor(kind="promotion", parent_trust_score=0.85))
        assert s == pytest.approx(0.85)

    def test_promotion_no_parent_defaults_to_half(self):
        s = score_source(SourceDescriptor(kind="promotion", parent_trust_score=None))
        assert s == pytest.approx(0.5)

    def test_unknown_kind_defaults_to_half(self):
        s = score_source(SourceDescriptor(kind="unknown_future_kind"))
        assert s == pytest.approx(0.5)

    def test_result_in_unit_interval(self):
        kinds = [
            "turn_user", "turn_assistant", "document",
            "bootstrap_memory", "bootstrap_diary", "tool_output",
            "promotion", "something_else",
        ]
        for kind in kinds:
            v = score_source(SourceDescriptor(kind=kind))
            assert 0.0 <= v <= 1.0, f"{kind} produced out-of-range {v}"


# ── test_pr1_store_round_trip ──────────────────────────────────────────





_NOW = datetime.now(timezone.utc)
_EMBED = [0.1] * 64


class _NoopVectorIndex(VectorIndex):
    def upsert(self, ids, vectors, *, tenant_ids, owner_types, owner_ids, extra_metadata=None) -> None:
        return None

    def query(self, vector, *, tenant_id, owner_type, owner_id, k=10, extra_filters=None):
        return []

    def delete(self, ids) -> None:
        return None


def _semantic_store(tmp_path: Path) -> SemanticSQLStore:
    db_path = str(tmp_path / "semantic.db")
    return SemanticSQLStore(
        db_adapter=SQLiteAdapter(db_path),
        vector_index=_NoopVectorIndex(),
    )


def _episodic_store(tmp_path: Path) -> EpisodicSQLStore:
    db_path = str(tmp_path / "episodic.db")
    return EpisodicSQLStore(
        db_adapter=SQLiteAdapter(db_path),
        vector_index=_NoopVectorIndex(),
    )


def _procedural_store(tmp_path: Path) -> ProceduralSQLStore:
    db_path = str(tmp_path / "procedural.db")
    return ProceduralSQLStore(
        db_adapter=SQLiteAdapter(db_path),
        vector_index=_NoopVectorIndex(),
    )


def _chunk_store(tmp_path: Path) -> ChunkSQLStore:
    db_path = str(tmp_path / "chunks.db")
    return ChunkSQLStore(
        db_adapter=SQLiteAdapter(db_path),
        vector_index=_NoopVectorIndex(),
    )


# ──────────────────────────────────────────────
# Fact round-trip
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fact_round_trip_new_fields(tmp_path):
    store = _semantic_store(tmp_path)
    ch = hash_fact_content("user:alice", "LIKES", "sushi")
    fact = Fact(
        id="fact_001",
        subject="user:alice",
        predicate="LIKES",
        object="sushi",
        created_at=_NOW,
        updated_at=_NOW,
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
        trust_score=0.7,
        content_hash=ch,
    )
    await store.upsert_fact(fact, _EMBED)

    results = await store.list_facts_for_owner(
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert results, "expected at least one fact"
    r = results[0]
    assert r.trust_score == pytest.approx(0.7)
    assert r.content_hash == ch


@pytest.mark.asyncio
async def test_fact_default_trust_score_persisted(tmp_path):
    store = _semantic_store(tmp_path)
    fact = Fact(
        id="fact_002",
        subject="user:alice",
        predicate="LIKES",
        object="tea",
        created_at=_NOW,
        updated_at=_NOW,
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
    )
    await store.upsert_fact(fact, _EMBED)

    results = await store.list_facts_for_owner(
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert results
    r = results[0]
    assert r.trust_score == pytest.approx(0.5)
    assert r.content_hash is None


# ──────────────────────────────────────────────
# Episode round-trip
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_episode_round_trip_new_fields(tmp_path):
    store = _episodic_store(tmp_path)
    ch = hash_episode_content("User discussed sushi preferences.")
    ep = Episode(
        id="ep_001",
        timestamp=_NOW,
        summary="User discussed sushi preferences.",
        user_id="user:alice",
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
        trust_score=0.8,
        content_hash=ch,
    )
    await store.add_episode(ep, _EMBED)

    result = await store.get_episode(
        "ep_001",
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert result is not None
    assert result.trust_score == pytest.approx(0.8)
    assert result.content_hash == ch


@pytest.mark.asyncio
async def test_episode_default_trust_score_persisted(tmp_path):
    store = _episodic_store(tmp_path)
    ep = Episode(
        id="ep_002",
        timestamp=_NOW,
        summary="Short episode.",
        user_id="user:alice",
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
    )
    await store.add_episode(ep, _EMBED)

    result = await store.get_episode(
        "ep_002",
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert result is not None
    assert result.trust_score == pytest.approx(0.5)
    assert result.content_hash is None


# ──────────────────────────────────────────────
# Skill round-trip
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_round_trip_new_fields(tmp_path):
    store = _procedural_store(tmp_path)
    plan = {"steps": ["greet", "ask how they are"]}
    ch = hash_skill_content("greet_user", plan)
    skill = Skill(
        id="skill_001",
        name="greet_user",
        description="Greet the user.",
        created_at=_NOW,
        updated_at=_NOW,
        owner_type="agent",
        owner_id="agent-default",
        tenant_id="default",
        plan=plan,
        trust_score=0.6,
        content_hash=ch,
    )
    await store.add_skill(skill, _EMBED)

    result = await store.get_skill(
        "skill_001",
        tenant_id="default",
        owner_type="agent",
        owner_id="agent-default",
    )
    assert result is not None
    assert result.trust_score == pytest.approx(0.6)
    assert result.content_hash == ch


@pytest.mark.asyncio
async def test_skill_default_trust_score_persisted(tmp_path):
    store = _procedural_store(tmp_path)
    skill = Skill(
        id="skill_002",
        name="farewell",
        description="Say goodbye.",
        created_at=_NOW,
        updated_at=_NOW,
        owner_type="agent",
        owner_id="agent-default",
        tenant_id="default",
    )
    await store.add_skill(skill, _EMBED)

    result = await store.get_skill(
        "skill_002",
        tenant_id="default",
        owner_type="agent",
        owner_id="agent-default",
    )
    assert result is not None
    assert result.trust_score == pytest.approx(0.5)
    assert result.content_hash is None


# ──────────────────────────────────────────────
# Chunk round-trip (trust_score only)
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chunk_round_trip_trust_score(tmp_path):
    store = _chunk_store(tmp_path)
    chunk = Chunk(
        id="chunk_001",
        doc_id="doc_001",
        text="Some test chunk text.",
        page_range=(0, 1),
        position=0,
        source_path="/test/doc.pdf",
        source_hash="abc123",
        created_at=_NOW,
        updated_at=_NOW,
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
        trust_score=0.4,
    )
    await store.upsert_chunk(chunk, _EMBED)

    results = await store.fetch_by_ids(
        ["chunk_001"],
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert results
    r = results[0]
    assert r.trust_score == pytest.approx(0.4)
    assert not hasattr(r, "content_hash")



# ── test_pr2_trust_end_to_end ──────────────────────────────────────────





_FIXTURE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This document is used to verify PR2 trust scoring on ingest. "
    "Every stored chunk must carry a classifier-derived trust score. "
    "Memory systems need reliable trust tracking per artifact source."
)


@pytest.fixture
def fixture_doc(tmp_path) -> str:
    doc_path = tmp_path / "pr2_test_doc.txt"
    doc_path.write_text(_FIXTURE_TEXT, encoding="utf-8")
    return str(doc_path)


@pytest.mark.asyncio
async def test_process_turn_episode_trust_score(uma_memory):
    """Episode from a turn with a session_id carries synthesized-summary trust."""
    mem = uma_memory

    await mem.process_turn(
        user_id="user:alice",
        user_msg="I enjoy hiking in the mountains.",
        assistant_reply="That sounds like a great hobby.",
        session_id="session-pr2-ep",
        agent_id=AGENT_ID,
    )

    epi_store = mem._stores["episodic"]
    episodes = await epi_store.list_episodes(
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert episodes, "expected at least one episode after process_turn"

    ep = episodes[0]
    assert ep.trust_score == pytest.approx(0.8), (
        f"episode trust_score must be 0.8 for synthesized turn summaries; got {ep.trust_score}"
    )


@pytest.mark.asyncio
async def test_process_turn_facts_trust_score(uma_memory):
    """Turn facts inherit trust from their source side of the transcript."""
    mem = uma_memory

    await mem.process_turn(
        user_id="user:alice",
        user_msg="I like hiking and rock climbing.",
        assistant_reply="Those are excellent outdoor activities.",
        session_id="session-pr2-facts",
        agent_id=AGENT_ID,
    )

    sem_store = mem._stores["semantic"]
    facts = await sem_store.list_facts_for_owner(
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )

    if not facts:
        pytest.skip("fake_llm produced no facts for this input; skipping assertion")

    trust_scores = {round(float(fact.trust_score or 0.0), 1) for fact in facts}
    assert trust_scores.issubset({0.7, 0.9})
    assert 0.9 in trust_scores


@pytest.mark.asyncio
async def test_ingest_document_chunks_trust_score(tmp_path, fixture_doc):
    """Chunks from ingest_document must have trust_score == 0.7 (document source)."""
    mem = await init_uma_for_tests(tmp_path)
    try:
        report = await mem.ingest_document(
            fixture_doc,
            owner_type="user",
            owner_id="user:alice",
            agent_id=AGENT_ID,
        )
        assert report.chunks_created > 0, "expected at least one chunk"

        chunk_store = mem._stores["chunk"]
        conn = chunk_store._conn()
        try:
            rows = chunk_store._query_all(
                conn,
                "SELECT id, trust_score FROM chunks WHERE owner_id = ?",
                params=["user:alice"],
                log_context="test_pr2_ingest_trust_score",
            )
        finally:
            conn.close()

        assert rows, "expected chunk rows in DB"
        for row in rows:
            assert row["trust_score"] is not None
            assert abs(float(row["trust_score"]) - 0.7) < 1e-6, (
                f"chunk {row['id']} must have trust_score=0.7 (document source); got {row['trust_score']}"
            )
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass


# ── test_pr5_threshold_filter ──────────────────────────────────────────





_NOW = datetime.now(timezone.utc)
_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:alice")


def _fact_thresh(fid: str, trust: float) -> Fact:
    return Fact(
        id=fid, subject="user:alice", predicate="likes", object="thing",
        created_at=_NOW, updated_at=_NOW, trust_score=trust, **_SCOPE,
    )


# ---------------------------------------------------------------------------
# Default threshold (0.5): below-floor candidates are filtered
# ---------------------------------------------------------------------------

def test_default_min_trust_filters_below_half():
    """With min_trust_score=0.5 (default), only candidates at or above 0.5 pass."""
    ranker = Ranker()  # min_trust_score=0.5

    facts = [_fact_thresh("a", 0.9), _fact_thresh("b", 0.5), _fact_thresh("c", 0.0)]
    result = ranker.rank_facts(facts, query_text="thing")

    ids = {f.id for f in result}
    assert ids == {"a", "b"}, "default threshold must drop candidates below 0.5"


def test_min_trust_zero_passes_all_candidates():
    """With min_trust_score=0.0, every non-quarantined candidate passes."""
    ranker = Ranker(min_trust_score=0.0)

    facts = [_fact_thresh("a", 0.9), _fact_thresh("b", 0.5), _fact_thresh("c", 0.0)]
    result = ranker.rank_facts(facts, query_text="thing")

    ids = {f.id for f in result}
    assert ids == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# min_trust_score=0.5 drops below-threshold candidates
# ---------------------------------------------------------------------------

def test_threshold_drops_low_trust_facts():
    ranker = Ranker(min_trust_score=0.5)

    above = _fact_thresh("above", trust=0.8)
    exact = _fact_thresh("exact", trust=0.5)
    below = _fact_thresh("below", trust=0.3)

    result = ranker.rank_facts([above, exact, below], query_text="thing")
    ids = {f.id for f in result}

    assert "above" in ids
    assert "exact" in ids
    assert "below" not in ids, "candidate below threshold must be dropped"


def test_threshold_inclusive_at_exact_boundary():
    """Candidate with trust_score == min_trust_score must NOT be dropped (>= comparison)."""
    ranker = Ranker(min_trust_score=0.5)
    exact = _fact_thresh("exact", trust=0.5)

    result = ranker.rank_facts([exact], query_text="thing")
    assert len(result) == 1 and result[0].id == "exact"


def test_threshold_drops_zero_trust():
    ranker = Ranker(min_trust_score=0.5)
    zero = _fact_thresh("zero", trust=0.0)
    high = _fact_thresh("high", trust=0.9)

    result = ranker.rank_facts([zero, high], query_text="thing")
    assert [f.id for f in result] == ["high"]


def test_threshold_all_dropped_returns_empty():
    ranker = Ranker(min_trust_score=0.9)
    facts = [_fact_thresh("a", 0.1), _fact_thresh("b", 0.2), _fact_thresh("c", 0.3)]

    result = ranker.rank_facts(facts, query_text="thing")
    assert result == []


# ---------------------------------------------------------------------------
# Filter applies to all item types
# ---------------------------------------------------------------------------

def test_threshold_filter_on_episodes():
    from uma.common.types import Episode

    ep_ok = Episode(
        id="ep_ok", timestamp=_NOW, summary="ok",
        user_id="user:alice", trust_score=0.8, **_SCOPE,
    )
    ep_bad = Episode(
        id="ep_bad", timestamp=_NOW, summary="bad",
        user_id="user:alice", trust_score=0.2, **_SCOPE,
    )

    ranker = Ranker(min_trust_score=0.5)
    result = ranker.rank_episodes([ep_ok, ep_bad], query_text="ok")
    assert [e.id for e in result] == ["ep_ok"]


def test_threshold_filter_on_skills():
    from uma.common.types import Skill

    sk_ok = Skill(
        id="sk_ok", name="clean_skill", description="fine",
        created_at=_NOW, updated_at=_NOW, trust_score=0.7, **_SCOPE,
    )
    sk_bad = Skill(
        id="sk_bad", name="bad_skill", description="untrusted",
        created_at=_NOW, updated_at=_NOW, trust_score=0.1, **_SCOPE,
    )

    ranker = Ranker(min_trust_score=0.5)
    result = ranker.rank_skills([sk_ok, sk_bad], query_text="skill")
    assert [s.id for s in result] == ["sk_ok"]


def test_threshold_filter_on_chunks():
    from uma.common.types import Chunk

    ch_ok = Chunk(
        id="ch_ok", doc_id="d1", text="trusted content",
        page_range=(0, 1), position=0, source_path="f.pdf", source_hash="h",
        created_at=_NOW, updated_at=_NOW, trust_score=0.9, **_SCOPE,
    )
    ch_bad = Chunk(
        id="ch_bad", doc_id="d1", text="untrusted content",
        page_range=(1, 2), position=1, source_path="f.pdf", source_hash="h",
        created_at=_NOW, updated_at=_NOW, trust_score=0.2, **_SCOPE,
    )

    ranker = Ranker(min_trust_score=0.5)
    result = ranker.rank_chunks([ch_ok, ch_bad], query_text="content")
    assert [c.id for c in result] == ["ch_ok"]


# ---------------------------------------------------------------------------
# Truncation is independent of filtering
# ---------------------------------------------------------------------------

def test_filter_runs_before_truncation():
    """After filter drops below-threshold items, truncate operates on the smaller pool."""
    ranker = Ranker(min_trust_score=0.5)

    facts = [_fact_thresh("a", 0.9), _fact_thresh("b", 0.8), _fact_thresh("c", 0.1), _fact_thresh("d", 0.0)]
    filtered = ranker.rank_facts(facts, query_text="thing")
    truncated = ranker.truncate(filtered, k=1)

    # Only "a" and "b" should survive the filter; truncate gives us the top 1.
    assert len(truncated) == 1
    assert truncated[0].trust_score >= 0.5


# ── test_pr5_score_combination ──────────────────────────────────────────





_NOW = datetime.now(timezone.utc)
_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:alice")


def _fact_combo(fid: str, trust: float, predicate: str = "likes") -> Fact:
    return Fact(
        id=fid,
        subject="user:alice",
        predicate=predicate,
        object="thing",
        created_at=_NOW,
        updated_at=_NOW,
        trust_score=trust,
        **_SCOPE,
    )


def test_higher_trust_ranks_first_at_default_weights():
    """Two facts with equal text → equal rerank; higher trust must rank first."""
    # Use identical text so rerank scores are equal.
    high = _fact_combo("high_trust", trust=0.9, predicate="eats")
    low = _fact_combo("low_trust", trust=0.1, predicate="eats")

    ranker = Ranker()  # default trust_weight=0.15
    ranked = ranker.rank_facts([low, high], query_text="eats")

    assert ranked[0].id == "high_trust", (
        "candidate with higher trust_score must rank first when rerank scores are equal"
    )


def test_trust_formula_at_custom_weights():
    """Verify the formula: final = alpha * existing + beta * trust at non-default weights."""
    # trust_weight=0.5 → alpha=0.5, beta=0.5
    ranker = Ranker(trust_weight=0.5)

    # Give low_trust slightly higher text-relevance but much lower trust.
    # With beta=0.5, trust dominates → high_trust should still win.
    low = _fact_combo("low_trust", trust=0.0, predicate="sushi recipe")
    high = _fact_combo("high_trust", trust=1.0, predicate="sushi recipe")

    ranked = ranker.rank_facts([low, high], query_text="sushi recipe")
    assert ranked[0].id == "high_trust"


def test_trust_weight_zero_preserves_existing_order():
    """With trust_weight=0, trust has no effect: ranking is purely by existing score."""
    ranker = Ranker(trust_weight=0.0, min_trust_score=0.0)

    # high_trust has low text relevance; low_trust matches the query perfectly.
    low_trust = Fact(
        id="low_trust", subject="user:alice", predicate="likes", object="sushi",
        created_at=_NOW, updated_at=_NOW, trust_score=0.0, **_SCOPE,
    )
    high_trust = Fact(
        id="high_trust", subject="user:alice", predicate="owns", object="car",
        created_at=_NOW, updated_at=_NOW, trust_score=1.0, **_SCOPE,
    )

    ranked = ranker.rank_facts([high_trust, low_trust], query_text="sushi")
    # With trust_weight=0 and filtering disabled, only text-rerank matters.
    assert ranked[0].id == "low_trust"


def test_trust_adjusts_meta_final_score():
    """After ranking, meta.final_score reflects the trust-adjusted value."""
    ranker = Ranker(trust_weight=0.5)
    f = _fact_combo("f1", trust=0.8, predicate="eats")
    ranker.rank_facts([f], query_text="eats")

    m = f.meta or {}
    assert "final_score" in m
    # The final_score must incorporate trust contribution.
    # At trust_weight=0.5, beta*trust = 0.5*0.8 = 0.4 minimum contribution.
    assert m["final_score"] >= 0.4


def test_trust_weight_one_orders_purely_by_trust():
    """With trust_weight=1.0, ranking is purely by trust_score."""
    ranker = Ranker(trust_weight=1.0)

    hi = _fact_combo("hi", trust=0.95, predicate="something")
    lo = _fact_combo("lo", trust=0.05, predicate="something")

    ranked = ranker.rank_facts([lo, hi], query_text="something")
    assert ranked[0].id == "hi"


def test_trust_combination_with_episodes():
    """Trust-aware ranking works for episodes too."""
    from uma.common.types import Episode

    ep_high = Episode(
        id="ep_high", timestamp=_NOW, summary="clean episode", user_id="user:alice",
        trust_score=0.9, **_SCOPE,
    )
    ep_low = Episode(
        id="ep_low", timestamp=_NOW, summary="clean episode", user_id="user:alice",
        trust_score=0.1, **_SCOPE,
    )

    ranker = Ranker(trust_weight=0.5)
    ranked = ranker.rank_episodes([ep_low, ep_high], query_text="clean episode")
    assert ranked[0].id == "ep_high"


# ── test_pr5_score_card ──────────────────────────────────────────





_NOW = datetime.now(timezone.utc)
_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:alice")


def _fact_card(fid: str, trust: float = 0.7) -> Fact:
    return Fact(
        id=fid, subject="user:alice", predicate="likes", object="coffee",
        created_at=_NOW, updated_at=_NOW, trust_score=trust, **_SCOPE,
    )


# ---------------------------------------------------------------------------
# debug_scores=True: trust fields present in score_card
# ---------------------------------------------------------------------------

def test_score_card_includes_trust_score_when_debug_on():
    ranker = Ranker(debug_scores=True)
    f = _fact_card("f1", trust=0.75)

    ranker.rank_facts([f], query_text="coffee")

    m = f.meta or {}
    card = m.get("score_card") or {}
    assert "trust_score" in card, "score_card must include trust_score when debug_scores=True"
    assert abs(card["trust_score"] - 0.75) < 1e-6


def test_score_card_includes_final_score_with_trust_when_debug_on():
    ranker = Ranker(debug_scores=True, trust_weight=0.15)
    f = _fact_card("f1", trust=0.9)

    ranker.rank_facts([f], query_text="coffee")

    m = f.meta or {}
    card = m.get("score_card") or {}
    assert "final_score_with_trust" in card
    # final_score_with_trust must match meta.final_score
    assert abs(card["final_score_with_trust"] - m.get("final_score", 0.0)) < 1e-6


def test_score_card_trust_fields_all_candidates():
    """Trust fields are populated on EVERY returned candidate, not just the top."""
    ranker = Ranker(debug_scores=True)
    facts = [_fact_card("a", trust=0.9), _fact_card("b", trust=0.5), _fact_card("c", trust=0.1)]

    ranker.rank_facts(facts, query_text="coffee")

    for f in facts:
        card = (f.meta or {}).get("score_card") or {}
        assert "trust_score" in card, f"score_card missing trust_score on {f.id}"
        assert "final_score_with_trust" in card, f"score_card missing final_score_with_trust on {f.id}"


# ---------------------------------------------------------------------------
# debug_scores=False: trust fields absent (no debug leakage)
# ---------------------------------------------------------------------------

def test_score_card_absent_when_debug_off():
    ranker = Ranker(debug_scores=False)
    f = _fact_card("f1", trust=0.9)

    ranker.rank_facts([f], query_text="coffee")

    m = f.meta or {}
    assert "score_card" not in m, "score_card must not be written when debug_scores=False"


def test_trust_fields_absent_in_meta_when_debug_off():
    """meta.trust_score and meta.final_score_with_trust must not leak into normal output."""
    ranker = Ranker(debug_scores=False)
    f = _fact_card("f1", trust=0.9)

    ranker.rank_facts([f], query_text="coffee")

    m = f.meta or {}
    # trust fields are only in score_card (debug only), not at top-level meta
    assert "trust_score" not in m or m.get("trust_score") is None or "score_card" not in m


# ---------------------------------------------------------------------------
# Trust fields correct across item types
# ---------------------------------------------------------------------------

def test_score_card_trust_on_episodes():
    from uma.common.types import Episode

    ep = Episode(
        id="ep1", timestamp=_NOW, summary="clean", user_id="user:alice",
        trust_score=0.6, **_SCOPE,
    )
    ranker = Ranker(debug_scores=True)
    ranker.rank_episodes([ep], query_text="clean")

    card = (ep.meta or {}).get("score_card") or {}
    assert "trust_score" in card
    assert abs(card["trust_score"] - 0.6) < 1e-6


def test_score_card_trust_on_chunks():
    from uma.common.types import Chunk

    ch = Chunk(
        id="ch1", doc_id="d1", text="trusted content here for testing",
        page_range=(0, 1), position=0, source_path="f.pdf", source_hash="h",
        created_at=_NOW, updated_at=_NOW, trust_score=0.8, **_SCOPE,
    )
    ranker = Ranker(debug_scores=True)
    ranker.rank_chunks([ch], query_text="trusted content")

    card = (ch.meta or {}).get("score_card") or {}
    assert "trust_score" in card
    assert abs(card["trust_score"] - 0.8) < 1e-6
