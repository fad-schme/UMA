"""
PR1 — dataclass field contracts for trust_score and content_hash.

Tests:
- default values (trust_score=0.5, content_hash=None)
- explicit valid values survive validate()
- out-of-range trust_score raises ValueError
- boundary values (0.0, 1.0) are valid
- content_hash="" raises ValueError; content_hash=None is OK
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.common.types import Chunk, Episode, Fact, Skill


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
