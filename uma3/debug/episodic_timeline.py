"""
Episodic Memory Timeline Visualization.

Generates:
- Sorted chronological view of episodes
- Optional color-coded tagging
- Optional ASCII-based timeline rendering

Coding agent instructions:
--------------------------
- You can replace ASCII timeline with matplotlib timeline if desired.
- Keep this read-only.
"""

from __future__ import annotations

import logging
from typing import List
from datetime import datetime

from ..types_episode import Episode

logger = logging.getLogger(__name__)


class EpisodicTimeline:
    """Timeline renderer for episodic memory."""

    @staticmethod
    def chronological_view(episodes: List[Episode]) -> str:
        """
        Return a chronological listing of episodes.

        Format:
            [2025-01-01 12:00] ep123  summary...
        """

        episodes = sorted(episodes, key=lambda e: e.timestamp)

        lines = [
            f"[{ep.timestamp}] {ep.id}  {ep.summary}" for ep in episodes
        ]
        return "\n".join(lines)

    @staticmethod
    def ascii_timeline(episodes: List[Episode]) -> str:
        """
        Render an ASCII timeline:

        user ----|---|---------|-------->
                 e1  e2        e3
        """

        if not episodes:
            return "No episodes."

        episodes = sorted(episodes, key=lambda e: e.timestamp)
        start = episodes[0].timestamp

        def offset(ep: Episode):
            delta = (ep.timestamp - start).total_seconds()
            return int(delta // 60)  # 1 char = 1 minute

        timeline = ""
        for ep in episodes:
            timeline += (" " * offset(ep)) + "*"
        return timeline