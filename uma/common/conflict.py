"""
Semantic Fact Conflict Resolution — UMA Core

Used exclusively by SemanticSQLStore.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable

from uma.common.types import Fact

logger = logging.getLogger(__name__)


class FactResolver(ABC):
    @abstractmethod
    def resolve(
        self, existing: Iterable[Fact], new: Fact
    ) -> tuple[Fact, list[Fact]]:
        raise NotImplementedError


class LatestWinsFactResolver(FactResolver):
    """Canonical fact = one with the latest updated_at timestamp."""

    def resolve(
        self, existing: Iterable[Fact], new: Fact
    ) -> tuple[Fact, list[Fact]]:
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
