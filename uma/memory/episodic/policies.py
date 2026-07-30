"""
EpisodicRetentionPolicy
------------------------

Defines retention rules for episodic memory.

This allows UMA to:
- prune old episodes
- implement TTL (time-to-live)
- enforce size limits
- apply salience-based retention
- cluster episodes in the future

Coding Agent Instructions
-------------------------
- Policies must NOT modify the store directly.
- They return a list of episodes to prune.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any

import logging

logger = logging.getLogger(__name__)


class EpisodicRetentionPolicy:
    """
    Production-Grade Episodic Retention Policy for UMA
    ====================================================

    This class implements a comprehensive retention strategy for episodic
    memory. Its purpose is to determine which episodes should be pruned
    (deleted) based on:

        • TTL (time-to-live): remove episodes older than X hours
        • Max episode capacity: keep recent episodes, prune oldest overflow
        • Salience-based retention: retain more important episodes
          if metadata provides a salience score

    Episodes passed to `select_prunable()` MUST be objects with:
        • episode.timestamp : datetime
        • episode.meta.get("salience", float) : optional salience metadata

    Coding Agent Instructions
    -------------------------
    - This policy MUST NOT delete episodes directly. It only returns a list.
    - ALWAYS log decisions and errors.
    - MUST gracefully skip malformed episode entries.
    - The caller (EpisodicCore) handles actual deletion.
    """

    def __init__(
        self,
        ttl_hours: int = 72,
        max_episodes: int = 200,
        minimum_salience: float = 0.0,
        prefer_high_salience: bool = True,
    ):
        """
        Parameters
        ----------
        ttl_hours : int
            Any episode older than this TTL is prunable.
        max_episodes : int
            Maximum number of episodes to retain after TTL filtering.
        minimum_salience : float
            Minimum salience score to retain. Episodes with lower salience
            are prunable when capacity is exceeded.
        prefer_high_salience : bool
            When pruning overflow episodes, sort by:
                True  → keep higher salience first
                False → ignore salience (purely timestamp based)
        """

        self.ttl_hours = ttl_hours
        self.max_episodes = max_episodes
        self.minimum_salience = minimum_salience
        self.prefer_high_salience = prefer_high_salience

        logger.info(
            "EpisodicRetentionPolicy initialized (ttl=%dh, max=%d, min_salience=%.2f, prefer_high_salience=%s)",
            ttl_hours,
            max_episodes,
            minimum_salience,
            prefer_high_salience,
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def select_prunable(self, episodes: list[Any]) -> list[Any]:
        """
        Determine which episodes to prune.

        Rules applied in order:
        -----------------------
        1. Remove episodes older than TTL.
        2. Among remaining episodes, remove those below minimum salience.
        3. If count still above max_episodes:
            a. Sort remaining by (salience desc, timestamp desc)
               if prefer_high_salience=True
            b. Else sort purely by timestamp desc (keep newest)
            c. Prune the oldest overflow

        Parameters
        ----------
        episodes : List[Any]
            Episode objects. Each must have:
                - timestamp: datetime
                - meta: dict with optional "salience" key

        Returns
        -------
        List[Any]
            The episodes that should be deleted.
        """

        if not episodes:
            logger.debug("RetentionPolicy: no episodes provided.")
            return []

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self.ttl_hours)

        prunable = []
        survivors = []

        # --------------------------------------------------------------
        # 1. TTL Filtering
        # --------------------------------------------------------------
        for ep in episodes:
            try:
                ts = getattr(ep, "timestamp", None)
                if ts and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts and ts < cutoff:
                    prunable.append(ep)
                else:
                    survivors.append(ep)
            except Exception:
                logger.exception("RetentionPolicy: invalid episode timestamp; marking prunable.")
                prunable.append(ep)

        # --------------------------------------------------------------
        # 2. Minimum Salience Filtering
        # --------------------------------------------------------------
        high_value = []
        for ep in survivors:
            try:
                sal = 0.0
                meta = getattr(ep, "meta", None)
                if isinstance(meta, dict):
                    sal = float(meta.get("salience", 0.0))
                if sal < self.minimum_salience:
                    prunable.append(ep)
                else:
                    high_value.append(ep)
            except Exception:
                logger.exception("RetentionPolicy: error reading salience; pruning episode.")
                prunable.append(ep)

        # --------------------------------------------------------------
        # 3. Capacity-Based Pruning
        # --------------------------------------------------------------
        survivors = high_value
        if len(survivors) > self.max_episodes:
            overflow_count = len(survivors) - self.max_episodes

            try:
                if self.prefer_high_salience:
                    # Sort by salience DESC, then timestamp DESC
                    def _salience(ep):
                        meta = getattr(ep, "meta", None)
                        if isinstance(meta, dict):
                            return float(meta.get("salience", 0.0))
                        return 0.0

                    survivors_sorted = sorted(
                        survivors,
                        key=lambda ep: (
                            _salience(ep),
                            getattr(ep, "timestamp", datetime.min),
                        ),
                        reverse=True,
                    )
                else:
                    # Sort only by timestamp DESC
                    survivors_sorted = sorted(
                        survivors,
                        key=lambda ep: getattr(ep, "timestamp", datetime.min),
                        reverse=True,
                    )

                # The lowest-ranked episodes (overflow) become prunable
                overflow = survivors_sorted[-overflow_count:]
                prunable.extend(overflow)

            except Exception:
                logger.exception("RetentionPolicy: overflow pruning failed; skipping capacity pruning.")

        # --------------------------------------------------------------
        # Final Result
        # --------------------------------------------------------------
        prunable_set = []
        seen = set()
        for ep in prunable:
            ep_id = getattr(ep, "id", None)
            key = ep_id if isinstance(ep_id, str) and ep_id else id(ep)
            if key in seen:
                continue
            seen.add(key)
            prunable_set.append(ep)
        logger.debug(
            "EpisodicRetentionPolicy: TTL=%d, min_salience=%.2f, max=%d → %d prunable episodes.",
            self.ttl_hours,
            self.minimum_salience,
            self.max_episodes,
            len(prunable_set),
        )

        return prunable_set
