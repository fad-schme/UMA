"""
Semantic Fact Conflict Resolution — UMA Core
==============================================

Provides resolution strategies for semantic fact conflicts during upsert.

This module is used exclusively by SemanticSQLStore.

Two strategies included:
    • LatestWinsFactResolver
    • ConfidenceFactResolver

Coding Agent Instructions
-------------------------
- Keep resolvers deterministic.
- ALWAYS log decisions for observability.
- NEVER throw errors—exceptions bubble up through the store.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

from uma.common.types import Fact

logger = logging.getLogger(__name__)


# ======================================================================
#  Abstract Base
# ======================================================================

class FactResolver(ABC):
    """
    Abstract base class for UMA semantic conflict resolution.
    """

    @abstractmethod
    def resolve(
        self, existing: Iterable[Fact], new: Fact
    ) -> Tuple[Fact, List[Fact]]:
        """
        Determine canonical Fact and archived ones.

        Returns
        -------
        canonical : Fact
        archived : List[Fact]
        """
        raise NotImplementedError


# ======================================================================
#  Latest-Wins Strategy
# ======================================================================

class LatestWinsFactResolver(FactResolver):
    """
    Time-based conflict resolution.

    Canonical fact = one with the latest updated_at timestamp.
    """

    def resolve(
        self, existing: Iterable[Fact], new: Fact
    ) -> Tuple[Fact, List[Fact]]:
        facts = list(existing) + [new]

        if not facts:
            logger.debug("LatestWins: empty list; new fact is canonical.")
            return new, []

        canonical = max(facts, key=lambda f: f.updated_at)
        archived = [f for f in facts if f is not canonical]

        logger.info(
            "LatestWins: chosen id=%s as canonical; %d archived.",
            canonical.id,
            len(archived),
        )
        return canonical, archived


# ======================================================================
#  Confidence Strategy
# ======================================================================

class ConfidenceFactResolver(FactResolver):
    """
    Confidence-based conflict resolution.

    - Pick highest confidence
    - If tie, pick most recent updated_at
    - Missing confidence defaults to 0.5
    """

    def resolve(
        self, existing: Iterable[Fact], new: Fact
    ) -> Tuple[Fact, List[Fact]]:
        facts = list(existing) + [new]

        def score(f: Fact):
            conf = f.confidence if f.confidence is not None else 0.5
            return (conf, f.updated_at)

        canonical = max(facts, key=score)
        archived = [f for f in facts if f is not canonical]

        logger.info(
            "ConfidenceResolver: canonical id=%s (confidence=%s), %d archived.",
            canonical.id,
            canonical.confidence,
            len(archived),
        )
        return canonical, archived
