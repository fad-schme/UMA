"""
cot_memory_builder.py
======================

Builds "Structured CoT Memory Sections" from UMA long-term memory.

Purpose
-------
UMA stores *structured* knowledge (facts, episodes, skills, graph).
This module converts retrieval output into a reusable chain-of-thought 
memory artifact that the agent can use for:
    • planning
    • tool orchestration
    • multi-step reasoning
    • workflow generation
    • debugging and interpretability

This is NOT an LLM-generated CoT. It is a deterministic, structured 
knowledge template extracted from UMA memory.
"""

from __future__ import annotations
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class CoTMemoryBuilder:
    """
    Convert UMA retrieved memory (episodic, semantic, procedural, graph)
    into a structured CoT reasoning scaffold.
    """

    @staticmethod
    def build(ctx: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Build a structured CoT memory representation.

        Parameters
        ----------
        ctx : dict
            Context object from UMAMemory.get_user_context().

        Returns
        -------
        dict
            {
                "reasoning_goals": [...],
                "known_facts": [...],
                "relevant_events": [...],
                "available_skills": [...],
                "graph_context": [...],
                "planning_scaffold": [...],
            }
        """

        # Prepare output
        cot = {
            "reasoning_goals": [],
            "known_facts": [],
            "relevant_events": [],
            "available_skills": [],
            "graph_context": [],
            "planning_scaffold": [],
        }

        # -------------------------------
        # Known Facts (Semantic Memory)
        # -------------------------------
        for fact in ctx.get("semantic", []):
            try:
                subj = _get_attr_or_key(fact, "subject")
                pred = _get_attr_or_key(fact, "predicate")
                obj = _get_attr_or_key(fact, "object")
                cot["known_facts"].append(f"{subj} {pred} {obj}")
            except Exception:
                logger.exception("Failed to convert fact for CoT memory.")

        # -------------------------------
        # Relevant Events (Episodic Memory)
        # -------------------------------
        for ep in ctx.get("episodic", []):
            try:
                summary = _get_attr_or_key(ep, "summary") or repr(ep)
                cot["relevant_events"].append(summary)
            except Exception:
                logger.exception("Failed to convert episode for CoT memory.")

        # -------------------------------
        # Available Skills (Procedural Memory)
        # -------------------------------
        for skill in ctx.get("procedural", []):
            try:
                name = _get_attr_or_key(skill, "name")
                plan = _get_attr_or_key(skill, "plan", {})
                steps = plan.get("steps") if isinstance(plan, dict) else None

                cot["available_skills"].append(
                    {
                        "skill_name": name,
                        "steps": steps or [],
                    }
                )
            except Exception:
                logger.exception("Failed to convert procedural skill to CoT.")

        # -------------------------------
        # Graph Context
        # -------------------------------
        for node in ctx.get("graph", []):
            cot["graph_context"].append(repr(node))

        # -------------------------------
        # *Planning Scaffold*
        #
        # UMA can synthesize a rationale plan by combining:
        #   - skills
        #   - facts
        #   - recent events
        #
        # This is not an LLM CoT. It is a structured *template* that an
        # external reasoning engine fills in.
        # -------------------------------
        cot["planning_scaffold"] = [
            "1. Understand the user's intent.",
            "2. Identify relevant known facts.",
            "3. Detect applicable skills or multi-step routines.",
            "4. Combine episodic context (past situations).",
            "5. Use graph context (entities, preferences).",
            "6. Propose pathways or solutions.",
            "7. Evaluate trade-offs based on memory.",
        ]

        logger.info("CoTMemoryBuilder: Built structured CoT memory section.")
        return cot


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Normalize object/dict access for RLM snippets and domain models."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
