"""
Working Memory Viewer

Provides utilities to inspect UMA-3 working memory, including:
- Step-by-step listing
- Token usage display
- Condensed view with summaries highlighted

Coding agent instructions:
--------------------------
- You must plug this into UMA3Memory that has WorkingMemoryFeature enabled.
- This is a pure debug tool; it must NEVER modify memory.
"""

from __future__ import annotations
import logging
from typing import List, Dict

from ..core.working_memory.buffer import WorkingMemoryMessage

logger = logging.getLogger(__name__)


class WorkingMemoryViewer:
    """Debugging utility for inspecting working memory."""

    @staticmethod
    def render_context(messages: List[WorkingMemoryMessage]) -> str:
        """
        Render working memory messages into a readable multi-line string.

        Summaries are highlighted for debugging.

        Returns
        -------
        str
        """
        lines = []
        for idx, msg in enumerate(messages):
            role = msg.role.upper()
            content = msg.content
            token_info = f"(tokens≈{msg.token_estimate})"

            if msg.role == "summary":
                lines.append(f"--- SUMMARY[{idx}] {token_info} ---\n{content}\n")
            else:
                lines.append(f"[{idx}] {role} {token_info}: {content}")

        return "\n".join(lines)