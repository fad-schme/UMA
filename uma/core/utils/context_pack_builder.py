"""
context_pack_builder.py
=======================

Transforms UMA-3 memory (from UMAMemory.get_user_context) into a
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
            Full memory context from UMAMemory.get_user_context(), e.g.:

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
            try:
                role = getattr(msg, "role", None)
                if role is None and isinstance(msg, dict):
                    role = msg.get("role")
                text = getattr(msg, "content", None)
                if text is None and isinstance(msg, dict):
                    text = msg.get("text", "")
                metadata = getattr(msg, "metadata", None)
                if metadata is None and isinstance(msg, dict):
                    metadata = msg.get("metadata", {})
                tokens = getattr(msg, "token_estimate", None)
                if tokens is None and isinstance(msg, dict):
                    tokens = msg.get("tokens", 0)

                pack["working_memory"].append(
                    {
                        "role": role,
                        "text": text,
                        "metadata": metadata or {},
                        "tokens": tokens if tokens is not None else 0,
                    }
                )
            except Exception:
                logger.exception("Failed to pack working memory entry.")

        # -------------------------------
        # Episodic
        # -------------------------------
        for ep in ctx.get("episodic", []):
            try:
                pack["episodic"].append(
                    {
                        "id": _get_attr_or_key(ep, "id"),
                        "timestamp": _get_attr_or_key(ep, "timestamp"),
                        "summary": _get_attr_or_key(ep, "summary") or repr(ep),
                        "tags": _get_attr_or_key(ep, "tags", []),
                        "meta": _get_attr_or_key(ep, "meta", {}),
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
                        "subject": _get_attr_or_key(fact, "subject", "unknown"),
                        "predicate": _get_attr_or_key(fact, "predicate", "related_to"),
                        "object": _get_attr_or_key(fact, "object"),
                        "confidence": _get_attr_or_key(fact, "confidence", 0.0),
                        "meta": _get_attr_or_key(fact, "meta", {}),
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
                        "name": _get_attr_or_key(skill, "name", "Unnamed Skill"),
                        "description": _get_attr_or_key(skill, "description"),
                        "plan": _get_attr_or_key(skill, "plan", {}),
                        "tools": _get_attr_or_key(skill, "tools", []),
                    }
                )
            except Exception:
                logger.exception("Failed to pack procedural skill.")

        # -------------------------------
        # Graph Items
        # -------------------------------
        for node in ctx.get("graph", []):
            try:
                if isinstance(node, dict):
                    pack["graph"].append(node)
                else:
                    pack["graph"].append({"node": repr(node)})
            except Exception:
                logger.exception("Failed to pack graph node.")

        logger.info("ContextPackBuilder: Built RAG-ready context pack.")
        return pack


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Normalize object/dict access for RLM snippets and domain models."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
