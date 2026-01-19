"""
Semantic Graph Viewer

Uses simple ASCII indentation or pretty tables to show:
- Facts grouped by subject
- Fact salience
- Confidence
- Timestamps

Coding agent instructions:
--------------------------
- You can later convert this to an interactive HTML render.
"""

from __future__ import annotations

import logging
from typing import List, Dict
from datetime import datetime

from ..types_fact import Fact

logger = logging.getLogger(__name__)


class SemanticGraphViewer:
    """Pretty-printer for semantic memory."""

    @staticmethod
    def render_subject_facts(facts: List[Fact]) -> str:
        """
        Group facts by subject, indent predicates.

        Example:
            user:123
              prefers_tone = concise  (salience 0.92)
              likes = database tools
        """
        groups: Dict[str, List[Fact]] = {}
        for f in facts:
            groups.setdefault(f.subject, []).append(f)

        lines = []
        for subject, items in groups.items():
            lines.append(subject)
            for f in items:
                sal = f.meta.get("salience", "n/a")
                lines.append(f"  {f.predicate} = {f.object}  [sal={sal}]")
            lines.append("")  # spacing

        return "\n".join(lines)