"""
context_pack_builder.py
=======================

Transforms UMA memory (from UMAMemory.get_user_context) into a
RAG-ready structured context pack.

This module does NOT generate prompts. It produces structured, 
machine-readable artifacts for:
    • RAG input pipelines
    • multi-document retrieval re-ranking
    • agent planning
    • debugging / observability
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from .config_types import RetrievalContextConfig
    from .text import extract_query_terms
except Exception:  # pragma: no cover
    RetrievalContextConfig = None
    extract_query_terms = None

class ContextPackBuilder:
    """
    Convert UMA memory into a standardized, RAG-ready context pack.
    
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

    @staticmethod
    def render_snippet(
        pack: Dict[str, Any],
        context_cfg: Optional["RetrievalContextConfig"] = None,
    ) -> str:
        """
        Render a compact, human-readable snippet for LLM prompts.
        """
        lines: List[str] = []
        cfg = context_cfg or RetrievalContextConfig()
        query_text = (pack.get("query") or "").lower()
        semantic_present = bool(pack.get("semantic"))
        prefer_semantic = bool(cfg.prefer_semantic_only)
        semantic_only = (
            prefer_semantic
            and semantic_present
            and not any(k in query_text for k in ("remember", "recall", "previous", "earlier", "last time"))
        )

        wm = pack.get("working_memory", [])
        if wm and not semantic_only:
            lines.append("Working memory:")
            for msg in wm[-cfg.max_working_messages:]:
                role = msg.get("role")
                text = (msg.get("text") or "").strip()
                if text:
                    lines.append(f"- {role}: {text}")

        episodic = pack.get("episodic", [])
        if episodic and not semantic_only:
            lines.append("\nEpisodic:")
            for ep in episodic[: cfg.max_episodic]:
                summary = (ep.get("summary") or "").strip()
                if summary:
                    lines.append(f"- {summary}")

        semantic = pack.get("semantic", [])
        allowed_topics = cfg.allowed_topics or []
        if allowed_topics:
            semantic = [
                fact
                for fact in semantic
                if (fact.get("meta") or {}).get("topic") in allowed_topics
            ]
        semantic = _filter_semantic_by_query(semantic, query_text)
        if semantic:
            if any((fact.get("predicate") == "document_snippet") for fact in semantic):
                semantic = [fact for fact in semantic if fact.get("predicate") == "document_snippet"]
            deduped = []
            seen = set()
            for fact in semantic:
                obj = fact.get("object")
                text = ""
                if isinstance(obj, dict):
                    text = (obj.get("text") or "").strip()
                key = text or str(obj)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(fact)
            semantic = deduped
        if semantic:
            lines.append("\nSemantic facts:")
            for fact in semantic[: cfg.max_semantic]:
                subject = fact.get("subject", "unknown")
                predicate = fact.get("predicate", "related_to")
                obj = fact.get("object")
                if isinstance(obj, dict):
                    title = obj.get("title") or obj.get("text") or str(obj)
                    snippet = (obj.get("text") or "").strip()
                    snippet = snippet[:400] if snippet else ""
                    lines.append(f"- {subject} {predicate} {title}")
                    if snippet:
                        lines.append(f"  excerpt: {snippet}")
                else:
                    lines.append(f"- {subject} {predicate} {obj}")

        procedural = pack.get("procedural", [])
        if procedural:
            lines.append("\nProcedural skills:")
            for skill in procedural[: cfg.max_procedural]:
                name = skill.get("name") or "Unnamed"
                desc = (skill.get("description") or "").strip()
                if desc:
                    lines.append(f"- {name}: {desc}")
                else:
                    lines.append(f"- {name}")

        graph = pack.get("graph", [])
        if graph:
            lines.append("\nGraph:")
            for node in graph[: cfg.max_graph]:
                lines.append(f"- {node}")

        return "\n".join(lines).strip()


def _filter_semantic_by_query(semantic: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
    if not semantic or not query_text:
        return semantic
    if extract_query_terms:
        terms = extract_query_terms(query_text)
    else:
        terms = []
    if not terms:
        return semantic
    scored: List[tuple[int, Dict[str, Any]]] = []
    for fact in semantic:
        obj = fact.get("object")
        haystack = ""
        if isinstance(obj, dict):
            haystack = " ".join(
                str(v) for v in (obj.get("title"), obj.get("text"), obj.get("path")) if v
            )
        else:
            haystack = str(obj or "")
        haystack = f"{fact.get('subject','')} {fact.get('predicate','')} {haystack}".lower()
        score = sum(1 for t in terms if t in haystack)
        if score:
            scored.append((score, fact))
    if not scored:
        return semantic
    scored.sort(key=lambda item: item[0], reverse=True)
    return [fact for _, fact in scored]


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Normalize object/dict access for RLM snippets and domain models."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
