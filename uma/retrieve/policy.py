"""
uma.retrieve.retrieval_policy
==================================

RetrievalPolicy — intent-aware ranking adjustments for UMA retrieval.

This module implements *query-time* policies that influence how retrieved
memory items are ranked and selected, without affecting storage or retrieval.

Added: should_stop() helper for RLM controller stopping decisions.
"""

from __future__ import annotations
from typing import Set, Dict, Any, Tuple

# -- existing keyword set --
RECALL_KEYWORDS: Set[str] = {
    "remember",
    "recall",
    "previous",
    "earlier",
    "last time",
    "before",
    "we talked",
    "you said",
    "you told me",
    "yesterday",
    "last week",
    "last month",
    "our conversation",
    "prior",
}

# ----------------------------------------------------------
# Trust-aware ranking defaults (PR5)
DEFAULT_TRUST_WEIGHT = 0.15      # beta: weight of trust_score in the final score
DEFAULT_MIN_TRUST_SCORE = 0.0    # candidates with trust_score < this are dropped

# ----------------------------------------------------------
# RLM stopping thresholds
DEFAULT_MAX_RLM_CALLS = 6
DEFAULT_TOKEN_BUDGET = 5000  # tokens (soft)
COVERAGE_CONFIDENCE_THRESHOLD = 0.8  # coverage "good enough"
MIN_USER_RESULTS_FOR_RECALL = 1  # if recall intent but no user results, keep going
# ----------------------------------------------------------


class RetrievalPolicy:
    """
    Intent-aware retrieval policy.

    Parameters
    ----------
    query_text : str
        Raw user query text.
    """

    def __init__(self, query_text: str) -> None:
        self.query_text = query_text or ""
        self._recall_score = self._compute_recall_score(self.query_text)

    @property
    def recall_score(self) -> float:
        """
        Recall intent score in [0.0, 1.0].

        0.0 → no recall intent
        1.0 → strong recall intent
        """
        return self._recall_score

    def _compute_recall_score(self, query_text: str) -> float:
        """
        Compute recall intent score using keyword matching.

        This is intentionally simple and conservative.
        """
        q = query_text.lower()

        hits = 0
        for kw in RECALL_KEYWORDS:
            if kw in q:
                hits += 1

        # Saturate quickly: 1–2 hits → strong recall
        if hits >= 2:
            return 1.0
        if hits == 1:
            return 0.6
        return 0.0


# ---------------------------------------------------------------------
# RLM stopping helper
# ---------------------------------------------------------------------
def should_stop(
    *,
    recall_score: float,
    coverage: Dict[str, Any],
    calls_made: int = 0,
    max_calls: int = DEFAULT_MAX_RLM_CALLS,
    tokens_used: int = 0,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    user_results_count: int = 0,
) -> Tuple[bool, str]:
    """
    Decide whether the RLM controller should stop recursion.

    Parameters
    ----------
    recall_score : float
        Recall intent score in [0.0, 1.0].
    coverage : dict
        Coverage metrics computed by the controller. Expected keys (optional):
          - 'confidence' : float (0..1) estimated coverage confidence
          - 'facts' : int number of facts retrieved
          - 'episodes' : int number of episodes retrieved
    calls_made : int
        How many environment calls have been executed so far.
    max_calls : int
        Maximum allowed environment calls (hard limit).
    tokens_used : int
        Estimated tokens consumed by RLM so far.
    token_budget : int
        Allowed token budget (soft limit).
    user_results_count : int
        Number of user-owned items retrieved (for recall path).

    Returns
    -------
    Tuple[bool, str]
        (should_stop, reason)
    """
    # Safety: hard limits first
    if calls_made >= max_calls:
        return True, "max_calls_reached"

    if tokens_used >= token_budget:
        return True, "token_budget_exhausted"

    # If recall intent is strong, ensure we actually have user memory results
    if recall_score >= 0.75:
        # If policy expects user memory but none present → don't stop
        if user_results_count < MIN_USER_RESULTS_FOR_RECALL:
            return False, "recall_expected_but_no_user_results"

        # If coverage confidence is strong, we can stop
        conf = float(coverage.get("confidence", 0.0) or 0.0)
        if conf >= COVERAGE_CONFIDENCE_THRESHOLD:
            return True, "coverage_confident_and_recall_satisfied"
        # otherwise continue
        return False, "recall_present_but_coverage_low"

    # For non-recall queries: use coverage confidence primarily
    conf = float(coverage.get("confidence", 0.0) or 0.0)
    if conf >= COVERAGE_CONFIDENCE_THRESHOLD:
        return True, "coverage_confident"

    # If coverage low but we have many facts/episodes and recall score low, stop to avoid overfetch
    facts = int(coverage.get("facts", 0) or 0)
    episodes = int(coverage.get("episodes", 0) or 0)
    if facts + episodes >= 10:
        return True, "sufficient_items_collected"

    # Default: continue
    return False, "continue_search"