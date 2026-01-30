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
    from .user_query_helper import extract_query_terms, expand_query_terms
except Exception:  # pragma: no cover
    RetrievalContextConfig = None
    extract_query_terms = None
    expand_query_terms = None

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
            "chunks": [],
            "procedural": [],
            "graph": [],
            "trace": [],
            "confidence": {},
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
        # Document Chunks
        # -------------------------------
        for chunk in ctx.get("chunks", []):
            try:
                pack["chunks"].append(
                    {
                        "id": _get_attr_or_key(chunk, "id"),
                        "doc_id": _get_attr_or_key(chunk, "doc_id"),
                        "text": _get_attr_or_key(chunk, "text", ""),
                        "page_range": _get_attr_or_key(chunk, "page_range"),
                        "position": _get_attr_or_key(chunk, "position", 0),
                        "meta": _get_attr_or_key(chunk, "meta", {}),
                    }
                )
            except Exception:
                logger.exception("Failed to pack chunk.")

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

        # Best-effort trace/confidence if present on ctx
        try:
            trace = ctx.get("trace") if isinstance(ctx, dict) else None
            if isinstance(trace, list):
                pack["trace"] = trace
        except Exception:
            logger.exception("Failed to pack retrieval trace.")
        try:
            conf = ctx.get("confidence") if isinstance(ctx, dict) else None
            if isinstance(conf, dict):
                pack["confidence"] = conf
        except Exception:
            logger.exception("Failed to pack confidence metadata.")

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
        # Note: WM/episodic/chunks/procedural/graph are always allowed per design.

        wm = pack.get("working_memory", [])
        lines.append("Working memory:")
        if wm:
            for msg in wm[-cfg.max_working_messages:]:
                role = msg.get("role")
                text = (msg.get("text") or "").strip()
                if text:
                    lines.append(f"- {role}: {text}")
        else:
            lines.append("- (empty)")

        episodic = pack.get("episodic", [])
        if episodic:
            lines.append("\nEpisodic:")
            for ep in episodic[: cfg.max_episodic]:
                summary = (ep.get("summary") or "").strip()
                if summary:
                    lines.append(f"- {summary}")

        semantic = pack.get("semantic", [])
        allowed_topics = cfg.allowed_topics or []
        if allowed_topics:
            filtered = [
                fact
                for fact in semantic
                if any(
                    t in allowed_topics
                    for t in _semantic_topics(fact)
                )
            ]
            if filtered:
                semantic = filtered
        semantic = _filter_semantic_by_query(semantic, query_text)
        if semantic:
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
                    raw_text = (obj.get("text") or "").strip()
                    snippet = _extract_relevant_excerpt(raw_text, query_text, max_chars=320)
                    lines.append(f"- {subject} {predicate} {title}")
                    if snippet:
                        lines.append(f"  excerpt: {snippet}")
                else:
                    lines.append(f"- {subject} {predicate} {obj}")

        # -------------------------------
        # Chunks (render snippets)
        # -------------------------------
        chunks = pack.get("chunks", [])
        if chunks:
            lines.append("\nDocument chunks:")
            seen_chunk_text = set()
            for ch in chunks[: cfg.max_chunks]:
                text = (ch.get("text") or "").strip()
                key = " ".join(text.split()).lower()
                if not text or key in seen_chunk_text:
                    continue
                seen_chunk_text.add(key)
                snippet = _extract_relevant_excerpt(text, query_text, max_chars=320)
                if snippet:
                    lines.append(f"- {snippet}")

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
    if expand_query_terms:
        terms = expand_query_terms(query_text)
    elif extract_query_terms:
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


def _extract_relevant_excerpt(text: str, query_text: str, max_chars: int = 320) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if not query_text or not extract_query_terms:
        return cleaned[:max_chars]
    if expand_query_terms:
        terms = expand_query_terms(query_text)
    else:
        terms = extract_query_terms(query_text)
    if not terms:
        return cleaned[:max_chars]

    lowered = cleaned.lower()
    # Find first occurrence of any term
    positions = [lowered.find(t.lower()) for t in terms if t]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return cleaned[:max_chars]

    idx = min(positions)
    # Try sentence window around match
    before = lowered.rfind(". ", 0, idx)
    after = lowered.find(". ", idx)
    if before == -1:
        before = 0
    else:
        before = before + 2
    if after == -1:
        after = len(cleaned)
    else:
        after = after + 1
    snippet = cleaned[before:after].strip()
    if len(snippet) <= max_chars:
        return snippet
    # Fallback to centered window
    start = max(0, idx - max_chars // 3)
    end = min(len(cleaned), start + max_chars)
    return cleaned[start:end].strip()


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Normalize object/dict access for RLM snippets and domain models."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _semantic_topics(fact: Dict[str, Any]) -> List[str]:
    meta = fact.get("meta") or {}
    if not isinstance(meta, dict):
        return []
    topics = meta.get("topics")
    if isinstance(topics, list):
        return [str(t) for t in topics if t]
    topic = meta.get("topic")
    if topic:
        return [str(topic)]
    return []
