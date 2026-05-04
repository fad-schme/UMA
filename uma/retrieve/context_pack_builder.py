"""
context_pack_builder.py
=======================

Transforms UMA retrieval output from the bound runtime/request-handle path into a
RAG-ready structured context pack.

This module does NOT generate prompts. It produces structured, 
machine-readable artifacts for:
    • RAG input pipelines
    • multi-document retrieval re-ranking
    • agent planning
    • debugging / observability
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
import re
import logging

from uma.common.dedupe import dedupe_by_id
from uma.common.accessors import get_attr_or_key
from uma.common.serialization import chunk_to_dict
from uma.common.identity import normalize_user_id
from uma.common.storage_metadata import (
    shared_metadata_view,
)
from uma.common.text_bounds import trim_to_sentence_boundary

logger = logging.getLogger(__name__)

# --------------------------------
# SnippetRefiner import
# --------------------------------
from uma.retrieve.rlm.snippet_refiner import SnippetRefiner
from uma.common.config_types import RetrievalContextConfig
from uma.retrieve.user_query_helper import extract_keywords_and_phrases

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
            Full memory context returned by `UMAMemory.retrieve_context(...)`, e.g.:

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

        for ep in ctx.get("episodic", []):
            try:
                _meta, metadata = _artifact_metadata(
                    ep,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    created_at=get_attr_or_key(ep, "created_at") or get_attr_or_key(ep, "timestamp"),
                    updated_at=get_attr_or_key(ep, "updated_at") or get_attr_or_key(ep, "timestamp"),
                    session_id=get_attr_or_key(ep, "session_id"),
                )
                pack["episodic"].append(
                    {
                        "id": get_attr_or_key(ep, "id"),
                        "timestamp": get_attr_or_key(ep, "timestamp"),
                        "summary": get_attr_or_key(ep, "summary") or repr(ep),
                        "tags": get_attr_or_key(ep, "tags", []),
                        "kind": metadata["kind"],
                        "kb_lane": metadata["kb_lane"],
                        "provenance": dict(metadata.get("provenance") or {}),
                        "meta": metadata,
                    }
                )
            except Exception:
                logger.exception(
                    "ContextPackBuilder: failed to pack episodic memory entry owner_type=%s owner_id=%s trace_id=%s",
                    owner_type,
                    owner_id,
                    trace_id,
                )

        for fact in ctx.get("facts", []):
            try:
                meta, metadata = _artifact_metadata(
                    fact,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    created_at=get_attr_or_key(fact, "created_at"),
                    updated_at=get_attr_or_key(fact, "updated_at"),
                    session_id=get_attr_or_key(fact, "session_id"),
                )
                pack["facts"].append(
                    {
                        "id": get_attr_or_key(fact, "id"),
                        "subject": get_attr_or_key(fact, "subject", "unknown"),
                        "predicate": get_attr_or_key(fact, "predicate", "related_to"),
                        "object": get_attr_or_key(fact, "object"),
                        "confidence": get_attr_or_key(fact, "confidence", 0.0),
                        "source_ids": get_attr_or_key(fact, "source_ids", []),
                        "kind": metadata["kind"],
                        "kb_lane": metadata["kb_lane"],
                        "provenance": dict(metadata.get("provenance") or {}),
                        "meta": metadata,
                        "fact_text": meta.get("fact_text"),
                    }
                )
            except Exception:
                logger.exception(
                    "ContextPackBuilder: failed to pack semantic fact owner_type=%s owner_id=%s trace_id=%s",
                    owner_type,
                    owner_id,
                    trace_id,
                )

        for chunk in ctx.get("chunks", []):
            try:
                _meta, metadata = _artifact_metadata(
                    chunk,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    created_at=get_attr_or_key(chunk, "created_at"),
                    updated_at=get_attr_or_key(chunk, "updated_at"),
                )
                pack["chunks"].append(
                    {
                        "id": get_attr_or_key(chunk, "id"),
                        "doc_id": get_attr_or_key(chunk, "doc_id"),
                        "source_path": get_attr_or_key(chunk, "source_path"),
                        "text": get_attr_or_key(chunk, "text", ""),
                        "page_range": get_attr_or_key(chunk, "page_range"),
                        "position": get_attr_or_key(chunk, "position", 0),
                        "kind": metadata["kind"],
                        "kb_lane": metadata["kb_lane"],
                        "provenance": dict(metadata.get("provenance") or {}),
                        "meta": metadata,
                    }
                )
            except Exception:
                logger.exception(
                    "ContextPackBuilder: failed to pack chunk owner_type=%s owner_id=%s trace_id=%s",
                    owner_type,
                    owner_id,
                    trace_id,
                )

        for skill in ctx.get("skills", []):
            try:
                _meta, metadata = _artifact_metadata(
                    skill,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    created_at=get_attr_or_key(skill, "created_at"),
                    updated_at=get_attr_or_key(skill, "updated_at"),
                )
                pack["skills"].append(
                    {
                        "id": get_attr_or_key(skill, "id"),
                        "name": get_attr_or_key(skill, "name", "Unnamed Skill"),
                        "description": get_attr_or_key(skill, "description"),
                        "plan": get_attr_or_key(skill, "plan", {}),
                        "tools": get_attr_or_key(skill, "tools", []),
                        "kind": metadata["kind"],
                        "kb_lane": metadata["kb_lane"],
                        "provenance": dict(metadata.get("provenance") or {}),
                        "meta": metadata,
                    }
                )
            except Exception:
                logger.exception(
                    "ContextPackBuilder: failed to pack procedural skill owner_type=%s owner_id=%s trace_id=%s",
                    owner_type,
                    owner_id,
                    trace_id,
                )

        _pack_graph(pack, ctx, owner_type=owner_type, owner_id=owner_id, trace_id=trace_id)
        _pack_trace_and_confidence(pack, ctx, owner_type=owner_type, owner_id=owner_id, trace_id=trace_id)

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
        _append_sources(lines, pack, final_snippets=pack.get("final_snippets"))
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
        owner_type = pack.get("owner_type")
        owner_id = pack.get("owner_id")
        trace_id = None
        trace = pack.get("trace")
        if isinstance(trace, list):
            for item in trace:
                if isinstance(item, dict) and item.get("trace_id"):
                    trace_id = item.get("trace_id")
                    break

        cfg = context_cfg or RetrievalContextConfig()
        query_text = (pack.get("query") or "").lower()

        lines, facts = _render_common_sections(pack, cfg, query_text)
        final_snippets = await _compute_final_snippets(
            pack,
            cfg,
            query_text,
            facts,
            llm=llm,
            owner_type=owner_type,
            owner_id=owner_id,
            trace_id=trace_id,
        )
        if final_snippets:
            lines.append("\nDocument snippets:")
            for snip in final_snippets[: int(cfg.max_chunks or 3)]:
                if not isinstance(snip, dict):
                    continue
                text = (snip.get("text") or "").strip()
                if text:
                    lines.append(f"- {text}")
        pack["final_snippets"] = final_snippets
        _append_sources(lines, pack, final_snippets=final_snippets)
        _append_skills_and_graph(lines, pack, cfg)

        return "\n".join(lines).strip()


def _artifact_metadata(
    artifact: Any,
    *,
    owner_type: Any,
    owner_id: Any,
    created_at: Any,
    updated_at: Any,
    session_id: Any = None,
) -> tuple[dict, dict]:
    meta = dict(get_attr_or_key(artifact, "meta", {}) or {})
    metadata = shared_metadata_view(
        meta=meta,
        owner_type=str(get_attr_or_key(artifact, "owner_type") or owner_type),
        owner_id=str(get_attr_or_key(artifact, "owner_id") or owner_id),
        created_at=created_at,
        updated_at=updated_at,
        session_id=session_id,
    )
    return meta, metadata


def _pack_graph(
    pack: Dict[str, Any],
    ctx: Dict[str, Any],
    *,
    owner_type: Any,
    owner_id: Any,
    trace_id: Any,
) -> None:
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


def _pack_trace_and_confidence(
    pack: Dict[str, Any],
    ctx: Dict[str, Any],
    *,
    owner_type: Any,
    owner_id: Any,
    trace_id: Any,
) -> None:
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


def _basename(path: Any) -> str:
    if isinstance(path, str) and path.strip():
        return os.path.basename(path.strip())
    return ""


def _collect_source_filenames(pack: Dict[str, Any], final_snippets: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """
    Collect unique source filenames (basenames only) in stable order.
    Preference:
      1) snippet_refiner output (final_snippets[*].source.file_name / source_path)
      2) pack chunks (chunks[*].source_path)
    """
    seen: set[str] = set()
    out: List[str] = []

    # 1) from refined snippets (most accurate)
    for sn in final_snippets or []:
        if not isinstance(sn, dict):
            continue
        src = sn.get("source")
        if not isinstance(src, dict):
            continue

        name = src.get("file_name")
        if not (isinstance(name, str) and name.strip()):
            name = _basename(src.get("source_path"))

        if isinstance(name, str) and name.strip():
            key = name.lower()
            if key not in seen:
                seen.add(key)
                out.append(name)

    # 2) fallback from chunks
    if not out:
        for ch in pack.get("chunks", []) or []:
            if not isinstance(ch, dict):
                continue
            name = _basename(ch.get("source_path"))
            if name:
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(name)

    return out


def _append_sources(
    lines: List[str],
    pack: Dict[str, Any],
    *,
    final_snippets: Optional[List[Dict[str, Any]]] = None,
) -> None:
    sources = _collect_source_filenames(pack, final_snippets=final_snippets)
    if not sources:
        return
    lines.append("\nSources:")
    for i, source_name in enumerate(sources, start=1):
        lines.append(f"{i}. {source_name}")

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
            return trim_to_sentence_boundary(excerpt, max_chars=max_chars)

    # Fallback: first complete sentences
    # Fallback: first complete sentences that don't look like fragments.
    kept = [s for s in sentences if not _starts_like_fragment(s)]
    excerpt = " ".join((kept or sentences)[:3]).strip()
    return trim_to_sentence_boundary(excerpt, max_chars=max_chars)


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


async def _compute_final_snippets(
    pack: Dict[str, Any],
    cfg: "RetrievalContextConfig",
    query_text: str,
    facts: List[Dict[str, Any]],
    *,
    llm: Any,
    owner_type: Any,
    owner_id: Any,
    trace_id: Any,
) -> List[Dict[str, Any]]:
    final_snippets: List[Dict[str, Any]] = []
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
        bounded: List[Dict[str, Any]] = []
        for sn in final_snippets:
            if not isinstance(sn, dict):
                continue
            text = (sn.get("text") or "").strip()
            if not text:
                continue
            snippet = dict(sn)
            snippet["text"] = trim_to_sentence_boundary(text, max_chars=int(cfg.snippet_max_chars or 240))
            bounded.append(snippet)
            if len(bounded) >= int(cfg.max_chunks or 3):
                break
        return bounded
    return _build_fallback_snippets(pack, cfg, query_text, facts, refiner_failed)


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
