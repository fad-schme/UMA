"""
Skill Viewer for Procedural Memory

Displays:
- Skill name
- Trigger phrases
- Trigger patterns
- Plan steps
- Tools
- Example

Coding agent instructions:
--------------------------
- Keep viewer read-only.
"""

from __future__ import annotations

from typing import List
from ..types_skill import Skill


class SkillViewer:

    @staticmethod
    def render(skills: List[Skill]) -> str:
        """Return human-readable representation of skills."""
        lines = []
        for s in skills:
            lines.append(f"Skill: {s.name} (id={s.id})")
            if s.trigger_phrases:
                lines.append("  Trigger phrases:")
                for p in s.trigger_phrases:
                    lines.append(f"    - {p}")
            if s.trigger_patterns:
                lines.append("  Trigger patterns:")
                for p in s.trigger_patterns:
                    lines.append(f"    - {p}")
            if s.plan:
                lines.append("  Plan:")
                for step in s.plan.get("steps", []):
                    lines.append(f"    • {step}")
            if s.tools:
                lines.append(f"  Tools: {', '.join(s.tools)}")
            if s.example:
                lines.append("  Example:")
                lines.append(f"    {s.example}")
            lines.append("")
        return "\n".join(lines)