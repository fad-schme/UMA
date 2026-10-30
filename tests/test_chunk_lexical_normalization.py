"""Chunk lexical scores reach the reranker on a comparable [0, 1] scale.

`lexical_search` returns raw BM25, which is unbounded; the reranker clamps its
lexical input to 1.0, so every chunk whose raw score reached 1 tied regardless
of how much higher it was. Scores are now made relative to the best lexical hit
of the same query's pool, which keeps their order and needs no constant.
"""
from __future__ import annotations

from types import SimpleNamespace

from uma.memory.chunk.core import ChunkCore
from uma.retrieve.ranking import compute_rerank_score


def _chunk(chunk_id: str, *, confidence: float | None = None, text: str = "neutral text") -> SimpleNamespace:
    meta = {} if confidence is None else {"lexical_confidence": confidence}
    return SimpleNamespace(id=chunk_id, text=text, meta=meta)


def _tag(chunks: list[SimpleNamespace], lexical_ids: set[str]) -> None:
    ChunkCore._tag_retrieval_metadata(None, chunks, lexical_ids)  # type: ignore[arg-type]


def test_lexical_scores_are_relative_to_the_best_lexical_hit() -> None:
    strong, middling, weak = _chunk("a", confidence=4.0), _chunk("b", confidence=2.0), _chunk("c", confidence=0.5)
    _tag([strong, middling, weak], {"a", "b", "c"})

    assert strong.meta["lexical_score"] == 1.0
    assert middling.meta["lexical_score"] == 0.5
    assert weak.meta["lexical_score"] == 0.125


def test_scores_above_the_old_clamp_no_longer_tie_in_the_reranker() -> None:
    strong, middling = _chunk("a", confidence=4.0), _chunk("b", confidence=2.0)
    _tag([strong, middling], {"a", "b"})

    query = "unrelated query words"
    assert compute_rerank_score(query_text=query, candidate=strong) > compute_rerank_score(
        query_text=query, candidate=middling
    )


def test_a_single_lexical_hit_scores_one_and_vector_only_chunks_are_untouched() -> None:
    only_lexical, vector_only = _chunk("a", confidence=7.3), _chunk("b")
    _tag([only_lexical, vector_only], {"a"})

    assert only_lexical.meta["lexical_score"] == 1.0
    assert "lexical_score" not in vector_only.meta
    assert vector_only.meta["retrieval_method"] == "vector"


def test_missing_or_zero_confidence_keeps_the_existing_fallback() -> None:
    no_conf, zero = _chunk("a"), _chunk("b", confidence=0.0)
    _tag([no_conf, zero], {"a", "b"})

    assert no_conf.meta["lexical_score"] == 0.2
    assert zero.meta["lexical_score"] == 0.2
