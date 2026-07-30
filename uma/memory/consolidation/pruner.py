"""
UMA Memory Pruner (Forgetting Mechanism)
=========================================

This module removes low-value or outdated memory items from:
- Episodic Memory
- Semantic Memory

Design Goals:
-------------
• Must NEVER delete mission-critical memory.
• Must degrade gracefully (never break consolidator).
• Deterministic and timestamp-based.
• Backend-agnostic (SQL / Postgres / cloud-native).
• Provide clear, structured logging for auditability.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta

from uma.common.types import Fact
from uma.common.types import Episode

logger = logging.getLogger(__name__)


class Pruner:
    """
    Implements UMA forgetting rules.

    Episodic Forgetting:
    --------------------
    • Age-based: remove episodes older than `max_episode_age_days`
    • Safety window: never prune episodes younger than `min_episode_age_hours`
    • Mission-critical guard: never remove episodes with meta["critical"] == True

    Semantic Forgetting:
    ---------------------
    • Salience-based: remove facts below `min_fact_salience`
    • Safety window: never prune recently updated facts
    """

    def __init__(
        self,
        max_episode_age_days: int = 90,
        min_episode_age_hours: int = 24,
        min_fact_salience: float = 0.2,
        min_fact_age_hours: int = 12,
    ):
        self.max_episode_age = timedelta(days=max_episode_age_days)
        self.min_episode_age = timedelta(hours=min_episode_age_hours)
        self.min_fact_salience = float(min_fact_salience)
        self.min_fact_age = timedelta(hours=min_fact_age_hours)

        logger.info(
            "Pruner initialized: max_episode_age=%dd, "
            "safety_window=%dh, min_fact_salience=%.2f, fact_safety=%dh",
            max_episode_age_days,
            min_episode_age_hours,
            min_fact_salience,
            min_fact_age_hours,
        )

    # ------------------------------------------------------------------
    # Episodic Filtering
    # ------------------------------------------------------------------
    def filter_episodes(self, episodes: list[Episode]) -> list[Episode]:
        """
        Return episodes to KEEP.

        Rules:
        1. Never prune episodes marked mission-critical.
        2. Never prune episodes younger than min_episode_age.
        3. Prune episodes older than max_episode_age.
        """
        now = datetime.now(timezone.utc)
        keep: list[Episode] = []
        pruned = 0

        for ep in episodes:
            # Mission-critical protection
            if ep.meta.get("critical") is True:
                keep.append(ep)
                continue

            age = now - ep.timestamp.replace(tzinfo=timezone.utc)

            # Safety window: do not prune very recent episodes
            if age <= self.min_episode_age:
                keep.append(ep)
                continue

            # Drop old episodes
            if age > self.max_episode_age:
                pruned += 1
                continue

            # Otherwise keep
            keep.append(ep)

        logger.info(
            "Pruner: kept %d/%d episodes (pruned=%d)",
            len(keep),
            len(episodes),
            pruned,
        )
        return keep

    # ------------------------------------------------------------------
    # Semantic Filtering
    # ------------------------------------------------------------------
    def filter_facts(self, facts: list[Fact]) -> list[Fact]:
        """
        Return facts to KEEP.

        Rules:
        1. Keep all mission-critical facts.
        2. Drop facts below salience threshold.
        3. Do not prune very recent facts (safety window).
        """
        now = datetime.now(timezone.utc)
        keep: list[Fact] = []
        pruned = 0

        for fact in facts:
            # Mission-critical
            if fact.meta.get("critical") is True:
                keep.append(fact)
                continue

            salience = fact.meta.get("salience", fact.confidence or 0.0)
            age = now - fact.updated_at.replace(tzinfo=timezone.utc)

            # Safety window: do not prune recently updated facts
            if age <= self.min_fact_age:
                keep.append(fact)
                continue

            # Drop low-salience facts
            if salience < self.min_fact_salience:
                pruned += 1
                continue

            keep.append(fact)

        logger.info(
            "Pruner: kept %d/%d semantic facts (pruned=%d)",
            len(keep),
            len(facts),
            pruned,
        )
        return keep
