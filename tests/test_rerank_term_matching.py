"""Query-term matching in the reranker.

A query term is credited for a genuine occurrence of that word, not for any
longer word that merely contains it (`far` in "fare", `sent` in "consent").
Simple plurals and hyphenated compound parts still match, so ordinary
inflection does not cost a hit. Text here is deliberately neutral: nothing in
this file comes from a benchmark or an e2e corpus.
"""
from __future__ import annotations

import pytest

from uma.common.text import term_in_text
from uma.retrieve.ranking import compute_rerank_score


def _score(query: str, text: str) -> float:
    return compute_rerank_score(query_text=query, candidate={"text": text, "meta": {}})


@pytest.mark.parametrize(
    "term,text",
    [
        ("far", "the standard fare applies on every route"),
        ("sent", "written consent is required before the trial"),
        ("where", "you can park anywhere on the estate"),
        ("pass", "each passenger receives a ticket"),
        ("used", "the request was refused twice"),
        ("time", "the timetable changes at midnight"),
        ("run", "a dead-running vehicle is towed"),
    ],
)
def test_term_is_not_credited_for_a_longer_word_that_contains_it(term: str, text: str) -> None:
    assert term_in_text(term, text) is False


@pytest.mark.parametrize(
    "term,text",
    [
        ("depot", "the depots open at dawn"),
        ("vehicle", "each vehicle is inspected"),
        ("driver", "drivers must rest between shifts"),
        ("shifts", "the shift pattern is fixed"),
        ("card", "a card-only policy applies"),
        ("single", "a single-journey ticket expires"),
        ("night", "Night service runs hourly."),
    ],
)
def test_plurals_hyphen_parts_and_case_still_match(term: str, text: str) -> None:
    assert term_in_text(term, text) is True


def test_phrase_needs_word_boundaries() -> None:
    assert term_in_text("night network", "the night network runs hourly") is True
    assert term_in_text("night network", "the midnight network runs hourly") is False


def test_rerank_score_does_not_reward_a_look_alike_word() -> None:
    query = "how far is the depot"
    exact = _score(query, "the depot is far from the station")
    look_alike = _score(query, "the standard fare from the depot station")
    assert exact > look_alike


def test_rerank_score_still_rewards_a_plural_form() -> None:
    query = "where is the depot"
    plural = _score(query, "the depots are on the east side")
    absent = _score(query, "the station is on the east side")
    assert plural > absent

