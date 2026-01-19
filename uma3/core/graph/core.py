"""
TemporalGraphCore
=================

High-level interface for UMA-3 temporal graph memory.

Responsibilities
----------------
- Initialize the GraphAdapter backend
- Provide a clean API to:
    • add_episode(episode)
    • update_with_facts(episode, facts)
    • maintain_temporal_links(episode, previous_episode)
- Ensure errors NEVER crash UMA-3

Coding Agent Instructions
-------------------------
- This is a CORE subsystem (required if graph.backend != "disabled").
- The underlying adapter must be a GraphAdapter instance.
"""

from __future__ import annotations
import logging
from typing import Any, Optional, List

from ...adapters.graph.base import GraphAdapter
from .updater import GraphUpdater

logger = logging.getLogger(__name__)


class TemporalGraphCore:
    """
    Core temporal graph subsystem.
    """

    def __init__(self, adapter: GraphAdapter):
        # Type validation for safer initialization
        if not isinstance(adapter, GraphAdapter):
            raise TypeError(
                f"TemporalGraphCore requires a GraphAdapter instance, got {type(adapter).__name__}"
            )

        self.adapter = adapter
        self.updater = GraphUpdater(adapter)
        logger.info(
            "TemporalGraphCore initialized with adapter=%s",
            adapter.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def add_episode(self, episode: Any):
        """Insert an Episode node."""
        try:
            self.updater.add_episode_node(episode)
        except Exception:
            logger.exception("TemporalGraphCore.add_episode failed.")

    def add_facts(self, facts: List[Any]):
        """Insert Fact nodes."""
        for fact in facts:
            try:
                self.updater.add_fact_node(fact)
            except Exception:
                logger.exception("TemporalGraphCore.add_facts failed.")

    def link_episode_to_facts(self, episode: Any, facts: List[Any]):
        """Link Episode to its facts."""
        try:
            self.updater.link_episode_to_facts(episode, facts)
        except Exception:
            logger.exception("TemporalGraphCore.link_episode_to_facts failed.")

    def link_temporal(self, prev_ep: Any, next_ep: Any):
        """Link episodes in a temporal sequence."""
        try:
            self.updater.link_temporal(prev_ep, next_ep)
        except Exception:
            logger.exception("TemporalGraphCore.link_temporal failed.")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self):
        """Close graph backend."""
        try:
            self.adapter.close()
        except Exception:
            logger.exception("TemporalGraphCore.close failed.")