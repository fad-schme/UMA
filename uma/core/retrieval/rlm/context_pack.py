# uma3/core/retrieval/rlm/context_pack.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ContextPack:
    """
    Immutable, developer-facing context bundle.

    This is NOT a prompt.
    This is structured memory data that downstream agents
    can inject into prompts however they choose.
    """

    user_id: str
    query_text: str

    # Memory layers
    working_memory: List[Any] = field(default_factory=list)
    episodes: List[Any] = field(default_factory=list)
    facts: List[Any] = field(default_factory=list)
    skills: List[Any] = field(default_factory=list)
    graph: List[Any] = field(default_factory=list)

    # Controller trace (for debugging & observability)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        """Safe summary for logs / telemetry."""
        return {
            "user_id": self.user_id,
            "query": self.query_text,
            "counts": {
                "wm": len(self.working_memory),
                "episodes": len(self.episodes),
                "facts": len(self.facts),
                "skills": len(self.skills),
                "graph": len(self.graph),
            },
            "steps": len(self.steps),
            "warnings": self.warnings,
        }