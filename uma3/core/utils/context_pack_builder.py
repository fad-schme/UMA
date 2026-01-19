"""
context_pack_builder.py
=======================

Transforms UMA-3 memory (from UMA3Memory.get_user_context) into a
RAG-ready structured context pack.

This module does NOT generate prompts. It produces structured, 
machine-readable artifacts for:
    • RAG input pipelines
    • multi-document retrieval re-ranking
    • agent planning
    • debugging / observability
"""

from __future__ import annotations
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class ContextPackBuilder:
    """
    Convert UMA-3 memory into a standardized, RAG-ready context pack.
    
    The output is deterministic, structured, and LLM-agnostic.
    """

    @staticmethod
    def build(query: str, ctx: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Create a structured context pack.

        Parameters
        ----------
        query : str
            The natural-language query used to retrieve memory.
        
        ctx : dict
            Full memory context from UMA3Memory.get_user_context(), e.g.:

                {
                    "working_memory": [...],
                    "episodic": [...],
                    "semantic": [...],
                    "procedural": [...],
                    "graph": [...],
                }

        Returns
        -------
        dict
            A fully structured context pack suitable for RAG pipelines.
        """
        pack = {
            "query": query,
            "working_memory": [],
            "episodic": [],
            "semantic": [],
            "procedural": [],
            "graph": [],
        }

        # -------------------------------
        # Working Memory (WM + LT nodes)
        # -------------------------------
        for msg in ctx.get("working_memory", []):
            pack["working_memory"].append(
                {
                    "role": msg.role,
                    "text": msg.content,
                    "metadata": msg.metadata or {},
                    "tokens": msg.token_estimate,
                }
            )

        # -------------------------------
        # Episodic
        # -------------------------------
        for ep in ctx.get("episodic", []):
            try:
                pack["episodic"].append(
                    {
                        "id": getattr(ep, "id", None) or None,
                        "timestamp": getattr(ep, "timestamp", None),
                        "summary": getattr(ep, "summary", repr(ep)),
                        "tags": getattr(ep, "tags", []),
                        "meta": getattr(ep, "meta", {}),
                    }
                )
            except Exception:
                logger.exception("Failed to pack episodic memory entry.")

        # -------------------------------
        # Semantic Facts
        # -------------------------------
        for fact in ctx.get("semantic", []):
            try:
                pack["semantic"].append(
                    {
                        "subject": getattr(fact, "subject", None)
                        or fact.get("subject", "unknown"),
                        "predicate": getattr(fact, "predicate", None)
                        or fact.get("predicate", "related_to"),
                        "object": getattr(fact, "object", None)
                        or fact.get("object", None),
                        "confidence": getattr(fact, "confidence", None)
                        or fact.get("confidence", 0.0),
                        "meta": getattr(fact, "meta", {})
                        or fact.get("meta", {}),
                    }
                )
            except Exception:
                logger.exception("Failed to pack semantic fact.")

        # -------------------------------
        # Procedural Skills
        # -------------------------------
        for skill in ctx.get("procedural", []):
            try:
                pack["procedural"].append(
                    {
                        "name": getattr(skill, "name", None)
                        or skill.get("name", "Unnamed Skill"),
                        "description": getattr(skill, "description", None)
                        or skill.get("description", None),
                        "plan": getattr(skill, "plan", None)
                        or skill.get("plan", {}),
                        "tools": getattr(skill, "tools", None)
                        or skill.get("tools", []),
                    }
                )
            except Exception:
                logger.exception("Failed to pack procedural skill.")

        # -------------------------------
        # Graph Items
        # -------------------------------
        for node in ctx.get("graph", []):
            pack["graph"].append({"node": repr(node)})

        logger.info("ContextPackBuilder: Built RAG-ready context pack.")
        return pack