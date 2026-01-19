"""Memory Exporter

Exports UMA-3 internal memory as JSON for:
- offline inspection
- dataset building
- reproducibility audits

Coding agent instructions:
--------------------------
- Use this in admin tooling, not end-user flows.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import List, Dict, Any
from datetime import datetime

from ..types_fact import Fact
from ..types_episode import Episode
from ..types_skill import Skill


class MemoryExporter:

    @staticmethod
    def export_facts(facts: List[Fact]) -> str:
        """Return pretty JSON for a list of `Fact` objects."""
        objs = [asdict(f) for f in facts]
        return json.dumps(objs, indent=2, default=str)

    @staticmethod
    def export_episodes(episodes: List[Episode]) -> str:
        """Return pretty JSON for a list of `Episode` objects."""
        out = []
        for ep in episodes:
            d = asdict(ep)
            # ensure timestamp is ISO formatted
            if isinstance(d.get("timestamp"), datetime):
                d["timestamp"] = d["timestamp"].isoformat()
            out.append(d)
        return json.dumps(out, indent=2, default=str)

    @staticmethod
    def export_skills(skills: List[Skill]) -> str:
        """Return pretty JSON for a list of `Skill` objects."""
        objs = [asdict(s) for s in skills]
        return json.dumps(objs, indent=2, default=str)