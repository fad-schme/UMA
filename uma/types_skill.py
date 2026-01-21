"""
Skill model for Procedural Memory in UMA.

A Skill represents a reusable workflow or capability that the agent can use
to solve tasks. Skills are similar to "procedures" or "subroutines" in
classical AI architectures.

Typical examples:
- "SQL debugging workflow"
- "Explain a concept to a child"
- "Generate a project status report"

Coding agent instructions
-------------------------
- Treat Skill as a stable schema: do not add arbitrary fields without
  also updating the store and indexer.
- Ensure all fields are JSON-serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Skill:
    """
    Represents a procedural memory unit.

    Attributes
    ----------
    id: str
        Unique identifier for this skill (UUID recommended).
    name: str
        Human-readable name of the skill.
    trigger_phrases: List[str]
        Literal phrases that hint this skill should be used.
    trigger_patterns: List[str]
        Optional regex patterns for more flexible matching.
    plan: Dict[str, Any]
        Structured step-by-step plan, e.g.:
            {"steps": ["Inspect logs", "Run test query", "Check indexes"]}
    tools: List[str]
        Set of tool names (external functions/APIs) required by the skill.
    example: str
        Example conversation or usage demonstration.
    embedding: List[float] | None
        Semantic vector representing this skill.
    meta: Dict[str, Any]
        Free-form metadata (e.g. domain, tags, confidence).
    """

    id: str
    name: str
    trigger_phrases: List[str] = field(default_factory=list)
    trigger_patterns: List[str] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    tools: List[str] = field(default_factory=list)
    example: str = ""
    embedding: List[float] | None = None
    meta: Dict[str, Any] = field(default_factory=dict)