"""
Procedural Skill Matcher for UMA-3.

This module implements the logic for determining *which* skills apply
to a given user query based on:

1. Embedding similarity (handled by ProceduralSQLStore)
2. Trigger phrases (e.g. "reset database", "optimize SQL")
3. Regex patterns (e.g. r"(?i).*error.*timeout.*")

Coding agent instructions
-------------------------
- Extend phrase and pattern rules to match your domain.
- Maintain a clear, explainable scoring model.
- DO NOT perform DB operations here; only inspect candidate skills.
"""

from __future__ import annotations

import re
import logging
from typing import List

from ...types_skill import Skill

logger = logging.getLogger(__name__)


class SkillMatcher:
    """
    Hybrid matcher for procedural skills.

    Inputs:
        - User query text (string)
        - Candidate skills from semantic search

    Outputs:
        - Subset of skills with strong rule-based matches
    """

    def match_skills(
        self,
        query: str,
        candidate_skills: List[Skill],
        min_phrase_score: float = 0.5,
    ) -> List[Skill]:
        """
        Determine which skills match the query.

        Parameters
        ----------
        query : str
            Text of the user's request.
        candidate_skills : List[Skill]
            Skills returned by semantic search.
        min_phrase_score : float
            Minimum score for rule-based phrase/regex matches.

        Returns
        -------
        List[Skill]
            Filtered skills that match the query.
        """
        matched: List[Skill] = []
        query_lower = (query or "").lower()

        for skill in candidate_skills:
            try:
                phrase_score = self._score_phrases(query_lower, skill.trigger_phrases)
                regex_score = self._score_patterns(query_lower, skill.trigger_patterns)

                if phrase_score >= min_phrase_score or regex_score >= min_phrase_score:
                    matched.append(skill)
            except Exception:
                logger.exception("SkillMatcher: error while scoring skill id=%s", skill.id)

        logger.debug("SkillMatcher: matched %d skills for query=%r", len(matched), query)
        return matched

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _score_phrases(self, query: str, phrases: List[str]) -> float:
        """
        Return a simple phrase-matching score.

        Current behavior:
            - Returns 1.0 if any phrase is a substring of the query.
            - Returns 0.0 otherwise.

        Extend this to use partial similarity if needed.
        """
        score = 0.0
        for phrase in phrases or []:
            phrase_clean = phrase.lower().strip()
            if phrase_clean and phrase_clean in query:
                score = max(score, 1.0)
        return score

    def _score_patterns(self, query: str, patterns: List[str]) -> float:
        """
        Return 1.0 if any regex matches, otherwise 0.0.

        Logs invalid regex patterns instead of raising.
        """
        for pat in patterns or []:
            try:
                if re.search(pat, query):
                    return 1.0
            except re.error:
                logger.error("SkillMatcher: invalid regex pattern in Skill: %s", pat)
        return 0.0