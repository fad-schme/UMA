"""
context_pack_builder.py
=======================

Transforms UMA memory (from UMAMemory.get_structured_context) into a
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
import json
import re
import logging

from uma.core.utils.dedupe import dedupe_by_id
from uma.core.utils.accessors import get_attr_or_key
from uma.core.utils.serialization import chunk_to_dict
from uma.core.utils.identity import ensure_user_subject

logger = logging.getLogger(__name__)

# --------------------------------
# SnippetRefiner import
# --------------------------------
from uma.core.retrieval.rlm.snippet_refiner import SnippetRefiner
from .config_types import RetrievalContextConfig
from .user_query_helper import extract_keywords_and_phrases

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
            Full memory context from UMAMemory.get_structured_context(), e.g.:

                {
                    "working_memory": [...],
                    "episodic": [...],
                    "facts": [...],
                    "skills": [...],
                    "graph": [...],
                }

        Returns
        -------
        dict
            A fully structured context pack suitable for RAG pipelines.
        """
        owner_type = ctx.get("owner_type") if isinstance(ctx, dict) else None
        owner_id = ctx.get("owner_id") if isinstance(ctx, dict) else None
        trace_id = None
        if isinstance(ctx, dict):
            trace = ctx.get("trace")
            if isinstance(trace, list):
                for item in trace:
                    if isinstance(item, dict) and item.get("trace_id"):
                        trace_id = item.get("trace_id")
                        break

        pack = {
            "query": query,
            "working_memory": [],
            "episodic": [],
            "facts": [],
            "chunks": [],
            "skills": [],
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
                logger.exception(
                    "ContextPackBuilder: failed to pack working memory entry owner_type=%s owner_id=%s trace_id=%s",
                    owner_type,
                    owner_id,
                    trace_id,
                )

        # -------------------------------
        # Episodic
        # -------------------------------
        for ep in ctx.get("episodic", []):
            try:
                pack["episodic"].append(
                    {
                        "id": get_attr_or_key(ep, "id"),
                        "timestamp": get_attr_or_key(ep, "timestamp"),
                        "summary": get_attr_or_key(ep, "summary") or repr(ep),
                        "tags": get_attr_or_key(ep, "tags", []),
                        "meta": get_attr_or_key(ep, "meta", {}),
                    }
                )
            except Exception:
                logger.exception(
                    "ContextPackBuilder: failed to pack episodic memory entry owner_type=%s owner_id=%s trace_id=%s",
                    owner_type,
                    owner_id,
                    trace_id,
                )

        # -------------------------------
        # Facts
        # -------------------------------
        for fact in ctx.get("facts", []):
            try:
                meta = get_attr_or_key(fact, "meta", {})
                pack["facts"].append(
                    {
                        "id": get_attr_or_key(fact, "id"),
                        "subject": get_attr_or_key(fact, "subject", "unknown"),
                        "predicate": get_attr_or_key(fact, "predicate", "related_to"),
                        "object": get_attr_or_key(fact, "object"),
                        "confidence": get_attr_or_key(fact, "confidence", 0.0),
                        "source_ids": get_attr_or_key(fact, "source_ids", []),
                        "meta": meta,
                        "fact_text": meta.get("fact_text") if isinstance(meta, dict) else None,
                    }
                )
            except Exception:
                logger.exception(
                    "ContextPackBuilder: failed to pack semantic fact owner_type=%s owner_id=%s trace_id=%s",
                    owner_type,
                    owner_id,
                    trace_id,
                )

        # -------------------------------
        # Document Chunks
        # -------------------------------
        for chunk in ctx.get("chunks", []):
            try:
                pack["chunks"].append(
                    {
                        "id": get_attr_or_key(chunk, "id"),
                        "doc_id": get_attr_or_key(chunk, "doc_id"),
                        "text": get_attr_or_key(chunk, "text", ""),
                        "page_range": get_attr_or_key(chunk, "page_range"),
                        "position": get_attr_or_key(chunk, "position", 0),
                        "meta": get_attr_or_key(chunk, "meta", {}),
                    }
                )
            except Exception:
                logger.exception(
                    "ContextPackBuilder: failed to pack chunk owner_type=%s owner_id=%s trace_id=%s",
                    owner_type,
                    owner_id,
                    trace_id,
                )

        # -------------------------------
        # Skills
        # -------------------------------
        for skill in ctx.get("skills", []):
            try:
                pack["skills"].append(
                    {
                        "id": get_attr_or_key(skill, "id"),
                        "name": get_attr_or_key(skill, "name", "Unnamed Skill"),
                        "description": get_attr_or_key(skill, "description"),
                        "plan": get_attr_or_key(skill, "plan", {}),
                        "tools": get_attr_or_key(skill, "tools", []),
                    }
                )
            except Exception:
                logger.exception(
                    "ContextPackBuilder: failed to pack procedural skill owner_type=%s owner_id=%s trace_id=%s",
                    owner_type,
                    owner_id,
                    trace_id,
                )

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
                logger.exception(
                    "ContextPackBuilder: failed to pack graph node owner_type=%s owner_id=%s trace_id=%s",
                    owner_type,
                    owner_id,
                    trace_id,
                )

        # Best-effort trace/confidence if present on ctx
        try:
            trace = ctx.get("trace") if isinstance(ctx, dict) else None
            if isinstance(trace, list):
                pack["trace"] = trace
        except Exception:
            logger.exception(
                "ContextPackBuilder: failed to pack retrieval trace owner_type=%s owner_id=%s trace_id=%s",
                owner_type,
                owner_id,
                trace_id,
            )
        try:
            conf = ctx.get("confidence") if isinstance(ctx, dict) else None
            if isinstance(conf, dict):
                pack["confidence"] = conf
        except Exception:
            logger.exception(
                "ContextPackBuilder: failed to pack confidence metadata owner_type=%s owner_id=%s trace_id=%s",
                owner_type,
                owner_id,
                trace_id,
            )

        # Pack-level hygiene: prevent duplicates wasting budgets downstream.
        pack["episodic"] = dedupe_by_id(pack.get("episodic", []))
        pack["facts"] = dedupe_by_id(pack.get("facts", []))
        pack["chunks"] = dedupe_by_id(pack.get("chunks", []))
        pack["skills"] = dedupe_by_id(pack.get("skills", []))
        pack["graph"] = dedupe_by_id(pack.get("graph", []))

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
        cfg = context_cfg or RetrievalContextConfig()
        query_text = (pack.get("query") or "").lower()

        lines, facts = _render_common_sections(pack, cfg, query_text)
        _append_chunk_snippets(lines, pack, cfg, query_text, facts, heading="Document chunks:")
        _append_skills_and_graph(lines, pack, cfg)
        return "\n".join(lines).strip()

    @staticmethod
    async def render_snippet_async(
        pack: Dict[str, Any],
        context_cfg: Optional["RetrievalContextConfig"] = None,
        llm: Any = None,
    ) -> str:
        """
        Async variant that can optionally refine snippets with an LLM.
        """
        # --------------------------------------------------
        # Normalize pack to dict if a ContextPack object was passed
        # RLMController returns a ContextPack instance, while this
        # builder operates on dict-like structures.
        # --------------------------------------------------
        orig_pack = pack
        owner_type = None
        owner_id = None
        trace_id = None
        if not isinstance(pack, dict):
            try:
                pack = pack.__dict__
            except Exception:
                logger.exception(
                    "ContextPackBuilder.render_snippet_async: failed to normalize pack object"
                )
                return ""

        if isinstance(pack, dict):
            owner_type = pack.get("owner_type")
            owner_id = pack.get("owner_id")
            trace = pack.get("trace")
            if isinstance(trace, list):
                for item in trace:
                    if isinstance(item, dict) and item.get("trace_id"):
                        trace_id = item.get("trace_id")
                        break

        cfg = context_cfg or RetrievalContextConfig()
        query_text = (pack.get("query") or "").lower()

        lines, facts = _render_common_sections(pack, cfg, query_text)

        # Final evidence snippets (via SnippetRefiner)
        final_snippets = []
        refiner_failed = False
        if cfg.snippet_refiner_enabled:
            try:
                refiner = SnippetRefiner(llm=llm, cfg=cfg)
                final_snippets = await refiner.refine(
                    query_text=query_text,
                    facts=facts,
                    chunks=pack.get("chunks", []),
                )
            except Exception:
                logger.exception(
                    "ContextPackBuilder: SnippetRefiner failed owner_type=%s owner_id=%s trace_id=%s",
                    owner_type,
                    owner_id,
                    trace_id,
                )
                refiner_failed = True
                final_snippets = []

        if final_snippets:
            # Enforce context config: bound snippet length and count here as a final gate.
            # SnippetRefiner focuses on quality; ContextPackBuilder enforces budgets.
            bounded: List[Dict[str, Any]] = []
            for sn in final_snippets:
                if not isinstance(sn, dict):
                    continue
                text = (sn.get("text") or "").strip()
                if not text:
                    continue
                sn = dict(sn)
                sn["text"] = _trim_to_sentence_boundary(text, max_chars=int(cfg.snippet_max_chars or 240))
                bounded.append(sn)
                if len(bounded) >= int(cfg.max_chunks or 3):
                    break
            final_snippets = bounded

        if not final_snippets:
            final_snippets = _build_fallback_snippets(pack, cfg, query_text, facts, refiner_failed)

        # Always render final_snippets as a single consistent evidence section.
        if final_snippets:
            lines.append("\nDocument snippets:")
            for snip in final_snippets[: int(cfg.max_chunks or 3)]:
                if not isinstance(snip, dict):
                    continue
                text = (snip.get("text") or "").strip()
                if text:
                    lines.append(f"- {text}")

        # Attach final_snippets to pack for downstream use (gold runner expects this field)
        try:
            if isinstance(pack, dict):
                pack["final_snippets"] = final_snippets
            else:
                setattr(pack, "final_snippets", final_snippets)
            if orig_pack is not pack and not isinstance(orig_pack, dict):
                setattr(orig_pack, "final_snippets", final_snippets)
        except Exception:
            logger.exception(
                "ContextPackBuilder: failed to attach final_snippets owner_type=%s owner_id=%s trace_id=%s",
                owner_type,
                owner_id,
                trace_id,
            )

        _append_skills_and_graph(lines, pack, cfg)

        return "\n".join(lines).strip()


async def get_rendered_context(
    memory: Any,
    *,
    user_id: str,
    query_text: str,
) -> str:
    """
    Retrieve context and render a production-ready snippet.

    This path is shared by the app and tests to avoid divergence.
    """
    if not getattr(memory, "_rlm_controller", None):
        pack = await build_context_pack(memory, user_id=user_id, query_text=query_text)
        ctx_cfg = getattr(getattr(memory, "retrieval_cfg", None), "context", None)
        if getattr(ctx_cfg, "snippet_refiner_enabled", False):
            return await ContextPackBuilder.render_snippet_async(pack, ctx_cfg, llm=getattr(memory, "llm", None))
        return ContextPackBuilder.render_snippet(pack, ctx_cfg)

    pack = await memory._rlm_controller.retrieve_context(
        user_id=ensure_user_subject(user_id),
        query_text=query_text,
    )
    ctx_cfg = getattr(getattr(memory, "retrieval_cfg", None), "context", None)
    return await ContextPackBuilder.render_snippet_async(pack, ctx_cfg, llm=getattr(memory, "llm", None))


async def build_context_pack(
    memory: Any,
    *,
    user_id: str,
    query_text: str,
) -> Dict[str, Any]:
    """
    Build a RAG-ready structured context pack using UMA memory.

    Convenience wrapper around:
    - UMAMemory.get_structured_context()
    - ContextPackBuilder.build()
    """
    ctx = await memory.get_structured_context(user_id, query_text)
    return ContextPackBuilder.build(query_text, ctx)


async def build_prompt_messages(
    memory: Any,
    *,
    user_id: str,
    query_text: str,
) -> list:
    """
    Backward-compatible prompt helper (deprecated): wraps retrieval + formatting
    into a single LLM-style messages array.
    """
    pack = await build_context_pack(memory, user_id=user_id, query_text=query_text)
    ctx_cfg = getattr(getattr(memory, "retrieval_cfg", None), "context", None)
    if getattr(ctx_cfg, "snippet_refiner_enabled", False):
        snippet = await ContextPackBuilder.render_snippet_async(pack, ctx_cfg, llm=getattr(memory, "llm", None))
    else:
        snippet = ContextPackBuilder.render_snippet(pack, ctx_cfg)

    user_content = f"{query_text}\n\nRelevant memory:\n{snippet}" if snippet else query_text
    return [{"role": "user", "content": user_content}]


def _filter_facts_by_query(facts: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
    if not facts or not query_text:
        return facts
    extracted = extract_keywords_and_phrases(query_text)
    terms = (extracted.get("keywords") or []) + (extracted.get("keyphrases") or [])
    terms = [t for t in terms if isinstance(t, str) and t]
    if not terms:
        return facts
    scored: List[tuple[int, Dict[str, Any]]] = []
    for fact in facts:
        obj = get_attr_or_key(fact, "object")
        haystack = ""
        if isinstance(obj, dict):
            haystack = " ".join(
                str(v) for v in (obj.get("title"), obj.get("text"), obj.get("path")) if v
            )
        else:
            haystack = str(obj or "")
        haystack = f"{get_attr_or_key(fact,'subject','')} {get_attr_or_key(fact,'predicate','')} {haystack}".lower()
        score = sum(1 for t in terms if t in haystack)
        if score:
            scored.append((score, fact))
    if not scored:
        return facts
    scored.sort(key=lambda item: item[0], reverse=True)
    return [fact for _, fact in scored]


def _extract_relevant_excerpt(text: str, query_text: str, max_chars: int = 240) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    # Sentence-aware clipping to avoid fragments.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s and s.strip()]
    if not sentences:
        return cleaned[:max_chars]

    terms = []
    if query_text:
        extracted = extract_keywords_and_phrases(query_text)
        terms = (extracted.get("keywords") or []) + (extracted.get("keyphrases") or [])
    terms = [t for t in terms if isinstance(t, str) and t]

    if terms:
        scored = []
        for i, s in enumerate(sentences):
            if _starts_like_fragment(s):
                continue
            s_lower = s.lower()
            score = sum(1 for t in terms if t.lower() in s_lower)
            if score:
                scored.append((score, i))
        scored.sort(reverse=True)
        if scored:
            _, idx = scored[0]
            start = max(0, idx - 1)
            end = min(len(sentences), idx + 2)
            excerpt = " ".join(sentences[start:end]).strip()
            return _trim_to_sentence_boundary(excerpt, max_chars=max_chars)

    # Fallback: first complete sentences
    # Fallback: first complete sentences that don't look like fragments.
    kept = [s for s in sentences if not _starts_like_fragment(s)]
    excerpt = " ".join((kept or sentences)[:3]).strip()
    return _trim_to_sentence_boundary(excerpt, max_chars=max_chars)


def _trim_to_sentence_boundary(text: str, *, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text.strip()
    cut = text[:max_chars]
    m = re.search(r"[.!?](?!.*[.!?])", cut)
    if m:
        return cut[: m.end()].strip()
    return cut.strip()


def _snippet_quality_ok(snippet: str, terms: List[str], *, require_terms: bool) -> bool:
    s = (snippet or "").strip()
    if not s:
        return False
    if _starts_like_fragment(s):
        return False
    if len(s) < 40:
        return False
    words = [w for w in s.split() if w]
    if len(words) < 12:
        return False
    # Starts mid-word or punctuation-heavy start/end
    if re.match(r"^[^\w]", s) or re.match(r".*[^\w]$", s):
        return False
    # Reject common sentence fragments (lowercase starts with no clear sentence start).
    if re.match(r"^(and|or|but|so|because|that|which|with)\b", s.lower()):
        return False
    # Mostly punctuation
    letters = sum(1 for c in s if c.isalnum())
    if letters < max(10, len(s) // 5):
        return False
    # Require a verb-like token to avoid fragments.
    if not re.search(r"\b(is|are|was|were|be|been|being|has|have|had|do|does|did|can|could|should|would|will|may|might|must|include|includes|including|uses|use|ensure|ensures|required|requires|allow|allows|prevent|prevents|provide|provides|protect|protects)\b", s.lower()):
        return False
    if require_terms and terms:
        s_lower = s.lower()
        if not any(t.lower() in s_lower for t in terms if t):
            return False
    return True


def _starts_like_fragment(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    # Allow quotes, parentheses, or brackets at the start.
    s = re.sub(r"^[\"'“”‘’\\(\\[\\{\\s]+", "", s)
    if not s:
        return True
    # Allow common lowercase starters like e.g. / i.e.
    lowered = s.lower()
    if lowered.startswith(("e.g.", "i.e.", "etc.")):
        return False
    # If the first alpha character is lowercase, treat as fragment.
    m = re.search(r"[A-Za-z]", s)
    if not m:
        return True
    return s[m.start()].islower()


def _collect_chunk_snippets(
    pack: Dict[str, Any],
    cfg: "RetrievalContextConfig",
    query_text: str,
    facts: List[Dict[str, Any]],
) -> List[str]:
    chunks = pack.get("chunks", []) or []
    if not chunks:
        return []

    chunks = _group_adjacent_chunks(chunks)
    preferred_ids: List[str] = []
    for fact in (facts or []):
        src_ids = get_attr_or_key(fact, "source_ids")
        if isinstance(src_ids, list):
            for sid in src_ids:
                if sid:
                    preferred_ids.append(str(sid))
    preferred_ids = list(dict.fromkeys(preferred_ids))

    chunk_by_id = {}
    for ch in chunks:
        cid = ch.get("id") if isinstance(ch, dict) else None
        if cid:
            chunk_by_id[str(cid)] = ch
    ordered_chunks: List[dict] = []
    for cid in preferred_ids:
        ch = chunk_by_id.get(cid)
        if ch is not None:
            ordered_chunks.append(ch)
    # If we have fact-linked chunks, use only those; otherwise fall back.
    if not ordered_chunks:
        ordered_chunks = list(chunks)

    terms = []
    if query_text:
        extracted = extract_keywords_and_phrases(query_text)
        terms = (extracted.get("keywords") or []) + (extracted.get("keyphrases") or [])
    terms = [t for t in terms if isinstance(t, str) and t]

    seen_chunk_text = set()
    snippets: List[str] = []
    seen_snippets: set[str] = set()
    added = 0
    for ch in ordered_chunks[: cfg.max_chunks]:
        text = (ch.get("text") or "").strip()
        key = " ".join(text.split()).lower()
        if not text or key in seen_chunk_text:
            continue
        seen_chunk_text.add(key)
        snippet = _extract_relevant_excerpt(text, query_text, max_chars=cfg.snippet_max_chars)
        if not snippet:
            continue
        require_terms = bool(terms) and added > 0
        if not _snippet_quality_ok(snippet, terms, require_terms=require_terms):
            continue
        norm = " ".join(snippet.split()).lower()
        if norm in seen_snippets:
            continue
        seen_snippets.add(norm)
        snippets.append(snippet)
        added += 1

    if not snippets:
        for ch in ordered_chunks[: cfg.max_chunks]:
            text = (ch.get("text") or "").strip()
            key = " ".join(text.split()).lower()
            if not text or key in seen_chunk_text:
                continue
            seen_chunk_text.add(key)
            snippet = _extract_relevant_excerpt(text, query_text, max_chars=cfg.snippet_max_chars)
            if not snippet:
                continue
            if _snippet_quality_ok(snippet, terms, require_terms=False):
                norm = " ".join(snippet.split()).lower()
                if norm in seen_snippets:
                    continue
                seen_snippets.add(norm)
                snippets.append(snippet)
                break

    return snippets


def _collect_raw_chunk_texts(chunks: List[Any], limit: int) -> List[str]:
    if not chunks:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for ch in chunks:
        text = get_attr_or_key(ch, "text") or get_attr_or_key(ch, "chunk") or get_attr_or_key(ch, "content")
        if not text:
            continue
        s = " ".join(str(text).split())
        key = s.lower()
        if not s or key in seen:
            continue
        seen.add(key)
        out.append(s)
        if limit and len(out) >= limit:
            break
    return out


def _render_common_sections(
    pack: Dict[str, Any],
    cfg: "RetrievalContextConfig",
    query_text: str,
) -> tuple[List[str], List[Dict[str, Any]]]:
    lines: List[str] = []

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

    # -------------------------------
    # Facts (single consistent format)
    # -------------------------------
    facts = pack.get("facts", []) or []

    # Normalize facts into dicts (some paths may pass objects)
    norm_facts: List[Dict[str, Any]] = []
    for f in facts:
        if not f:
            continue
        if isinstance(f, dict):
            norm_facts.append(f)
        else:
            try:
                norm_facts.append(f.__dict__)
            except Exception:
                norm_facts.append({"fact": repr(f)})

    facts = norm_facts

    if facts:
        lines.append("\nFacts:")

        max_facts = int(getattr(cfg, "max_facts", 0) or 0)
        facts_to_render = facts[:max_facts] if max_facts > 0 else facts

        for f in facts_to_render:
            subj = get_attr_or_key(f, "subject", "")
            pred = get_attr_or_key(f, "predicate", "")
            obj = get_attr_or_key(f, "object", "")
            conf = get_attr_or_key(f, "confidence", None)
            src_ids = get_attr_or_key(f, "source_ids", None)

            # Normalize object text (may be dict in some store backends)
            if isinstance(obj, dict):
                obj_text = " ".join(
                    str(v) for v in (obj.get("title"), obj.get("text"), obj.get("path")) if v
                )
            else:
                obj_text = str(obj or "")
            obj_text = " ".join(obj_text.split())

            # Prefer the first source chunk id as evidence tag.
            src = ""
            if isinstance(src_ids, list) and src_ids:
                src = str(src_ids[0] or "")
            elif isinstance(src_ids, str) and src_ids:
                src = src_ids

            # Confidence rendering (compact)
            conf_s = ""
            try:
                if conf is not None:
                    conf_s = f" conf={float(conf):.2f}"
            except Exception:
                conf_s = ""

            # Final deterministic bullet line.
            # Format: - [src:<chunk_id>] <subject> <predicate> <object> conf=0.xx
            parts: List[str] = []
            if src:
                parts.append(f"[src:{src}]")
            if subj:
                parts.append(str(subj))
            if pred:
                parts.append(str(pred))
            if obj_text:
                parts.append(obj_text)

            line = " ".join(parts).strip()
            if not line:
                continue
            lines.append(f"- {line}{conf_s}")

    return lines, facts


def _build_fallback_snippets(
    pack: Dict[str, Any],
    cfg: "RetrievalContextConfig",
    query_text: str,
    facts: List[Dict[str, Any]],
    refiner_failed: bool,
) -> List[Dict[str, Any]]:
    chunk_snippets = _collect_chunk_snippets(pack, cfg, query_text, facts)
    final_snippets = [{"text": s} for s in chunk_snippets]
    if cfg.snippet_refiner_enabled and refiner_failed:
        logger.warning("ContextPackBuilder: SnippetRefiner failed; using fallback snippets")
    if not final_snippets and pack.get("chunks"):
        raw_snippets = _collect_raw_chunk_texts(pack.get("chunks", []), cfg.max_chunks)
        final_snippets = [{"text": s} for s in raw_snippets]
        if raw_snippets:
            logger.warning(
                "ContextPackBuilder: fallback produced no snippets; using raw chunk texts (%d)",
                len(raw_snippets),
            )
    return final_snippets


def _append_chunk_snippets(
    lines: List[str],
    pack: Dict[str, Any],
    cfg: "RetrievalContextConfig",
    query_text: str,
    facts: List[Dict[str, Any]],
    *,
    heading: str,
) -> None:
    chunk_snippets = _collect_chunk_snippets(pack, cfg, query_text, facts)
    if not chunk_snippets:
        return
    lines.append(f"\n{heading}")
    for snip in chunk_snippets[: cfg.max_chunks]:
        lines.append(f"- {snip}")


def _append_skills_and_graph(
    lines: List[str],
    pack: Dict[str, Any],
    cfg: "RetrievalContextConfig",
) -> None:
    skills = pack.get("skills", [])
    if skills:
        lines.append("\nSkills:")
        for skill in skills[: cfg.max_procedural]:
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


def _normalize_chunk(chunk: Any) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk
    # Internal UMA runtime expects Chunk objects; this is the serialization boundary.
    try:
        return chunk_to_dict(chunk)  # type: ignore[arg-type]
    except Exception:
        text = get_attr_or_key(chunk, "text")
        if not text:
            text = get_attr_or_key(chunk, "chunk") or get_attr_or_key(chunk, "content")
        return {
            "id": get_attr_or_key(chunk, "id"),
            "doc_id": get_attr_or_key(chunk, "doc_id"),
            "position": get_attr_or_key(chunk, "position", 0),
            "page_range": get_attr_or_key(chunk, "page_range"),
            "text": text or "",
            "meta": get_attr_or_key(chunk, "meta", {}),
        }


def _group_adjacent_chunks(chunks: List[Any]) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    normalized = [_normalize_chunk(c) for c in chunks]
    # Sort by doc_id then position to enable adjacency grouping.
    def _pos(ch: Dict[str, Any]) -> int:
        try:
            return int(ch.get("position") or 0)
        except Exception:
            return 0

    sorted_chunks = sorted(
        normalized,
        key=lambda c: (str(c.get("doc_id") or ""), _pos(c)),
    )
    grouped: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    current_ids: List[str] = []
    current_positions: List[int] = []

    for ch in sorted_chunks:
        doc_id = ch.get("doc_id")
        pos = _pos(ch)
        section_id = None
        meta = ch.get("meta") or {}
        if isinstance(meta, dict):
            section_id = meta.get("section_id")

        if current is None:
            current = dict(ch)
            current_ids = [str(ch.get("id"))] if ch.get("id") else []
            current_positions = [pos]
            current["meta"] = dict(meta) if isinstance(meta, dict) else {}
            current["meta"]["merged_ids"] = list(current_ids)
            current["meta"]["merged_positions"] = list(current_positions)
            current["meta"]["section_id"] = section_id or current["meta"].get("section_id")
            continue

        same_doc = current.get("doc_id") == doc_id
        prev_pos = current_positions[-1] if current_positions else None
        adjacent = prev_pos is not None and pos == prev_pos + 1
        current_section = (current.get("meta") or {}).get("section_id")
        same_section = bool(section_id) and section_id == current_section

        if same_doc and (adjacent or same_section):
            # Merge
            cur_text = (current.get("text") or "").strip()
            new_text = (ch.get("text") or "").strip()
            if new_text:
                current["text"] = f"{cur_text} {new_text}".strip() if cur_text else new_text
            current_ids.append(str(ch.get("id")) if ch.get("id") else "")
            current_positions.append(pos)
            meta_cur = current.get("meta") or {}
            if isinstance(meta_cur, dict):
                meta_cur["merged_ids"] = [i for i in current_ids if i]
                meta_cur["merged_positions"] = list(current_positions)
                current["meta"] = meta_cur
            continue

        grouped.append(current)
        current = dict(ch)
        current_ids = [str(ch.get("id"))] if ch.get("id") else []
        current_positions = [pos]
        meta = ch.get("meta") or {}
        current["meta"] = dict(meta) if isinstance(meta, dict) else {}
        current["meta"]["merged_ids"] = list(current_ids)
        current["meta"]["merged_positions"] = list(current_positions)
        current["meta"]["section_id"] = section_id or current["meta"].get("section_id")

    if current is not None:
        grouped.append(current)

    return grouped


def _fact_topics(fact: Dict[str, Any]) -> List[str]:
    meta = get_attr_or_key(fact, "meta") or {}
    if not isinstance(meta, dict):
        return []
    topics = meta.get("topics")
    if isinstance(topics, list):
        return [str(t) for t in topics if t]
    topic = meta.get("topic")
    if topic:
        return [str(topic)]
    return []
