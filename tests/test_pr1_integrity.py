"""
PR1 — content integrity hash stability contract.

Tests:
- hash functions return 64-char hex strings
- same inputs always produce the same hash (stability)
- dict-valued objects hash identically regardless of key insertion order
- hash_episode_content("") does not raise
- hash_skill_content covers name + plan
"""

from __future__ import annotations

import hashlib

import pytest

from uma.common.integrity import hash_episode_content, hash_fact_content, hash_skill_content


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
