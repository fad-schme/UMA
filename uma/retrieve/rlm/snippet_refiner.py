# uma/retrieve/rlm/snippet_refiner.py

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from uma.common.accessors import get_attr_or_key
from uma.common.text_bounds import trim_to_sentence_boundary
from uma.adapters.llm.controller import LLMCallContext, generate_text

logger = logging.getLogger(__name__)


class SnippetRefiner:
    """
    SnippetRefiner

    Transforms raw retrieved document chunks into final, coherent
    evidence snippets suitable for agent context injection.

    Design principles:
    ------------------
    - Retrieval is greedy; presentation is ruthless.
    - Chunks are retrieval units, snippets are evidence units.
    - LLMs are used ONLY for evaluation/refinement, never generation.
    - All outputs are bounded, traceable, and deterministic.
    """

    def __init__(self, llm: Optional[Any], cfg: Any):
        """
        Parameters
        ----------
        llm : optional LLM adapter
            Used ONLY for snippet evaluation / light rewrite.
            If None, SnippetRefiner runs in deterministic-only mode.
        cfg : RetrievalContextConfig
            Context configuration with snippet limits.
        """
        self.llm = llm
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def refine(
        self,
        *,
        query_text: str,
        facts: List[Any],
        chunks: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Produce final evidence snippets.

        Input:
            - query_text: user question
            - facts: semantic facts already selected by RLM
            - chunks: raw retrieved chunks (large, overlapping)

        Output:
            - List of final snippet dicts
        """
        if not chunks:
            logger.debug("SnippetRefiner.refine: no chunks provided")
            return []

        # Presentation-only: normalize and group already-retrieved chunks.
        candidates = [self._normalize_chunk(ch) for ch in (chunks or [])]
        candidates = [c for c in candidates if (c.get("text") or "").strip()]
        grouped = self._group_chunks(candidates)
        logger.debug(
            "SnippetRefiner.refine: candidates=%d grouped=%d facts=%d",
            len(candidates or []),
            len(grouped),
            len(facts or []),
        )

        # Keep input ordering: no reranking, no relevance filtering.
        # Enforce only strict presentation budgets (count + length).
        max_out = int(getattr(self.cfg, "max_chunks", 3) or 3)
        max_out = max(0, max_out)
        max_chars = int(getattr(self.cfg, "snippet_max_chars", 240) or 240)
        max_chars = max(1, max_chars)

        out: List[Dict[str, Any]] = []
        seen_text: set[str] = set()
        for cand in grouped:
            if max_out and len(out) >= max_out:
                break
            refined, _score = await self._refine_single(query_text, cand, max_chars=max_chars)
            if refined is None:
                continue
            text = (refined.get("text") or "").strip()
            if not text:
                continue
            refined = dict(refined)
            refined["text"] = trim_to_sentence_boundary(text, max_chars=max_chars)
            final_text = (refined.get("text") or "").strip()
            if not final_text:
                continue
            key = " ".join(final_text.lower().split())
            if key in seen_text:
                continue
            seen_text.add(key)
            out.append(refined)
        logger.debug("SnippetRefiner.refine: final_snippets=%d", len(out))
        return out

    # ------------------------------------------------------------------
    # Step 1 — group adjacent chunks
    # ------------------------------------------------------------------

    def _group_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Presentation-only grouping for snippet coherence.

        Contract:
        - This function MUST NOT fetch/expand neighbors.
        - Deterministic neighbor expansion (pos-window fetch) happens upstream in
          retrieval (ChunkCore.search_chunks(expand_neighbors=True)).
        - Here we only merge already-retrieved chunks into coherent snippet candidates.

        Groups by doc_id and adjacency (position ±1) among the provided chunks.
        """
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []

        for ch in chunks:
            if not current:
                current = [ch]
                continue

            prev = current[-1]
            if (
                ch.get("doc_id") == prev.get("doc_id")
                and abs(int(ch.get("position", 0)) - int(prev.get("position", 0))) <= 1
            ):
                current.append(ch)
            else:
                groups.append(current)
                current = [ch]

        if current:
            groups.append(current)

        return [self._merge_group(g) for g in groups]

    def _merge_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        group = sorted(group, key=lambda g: int(g.get("position", 0) or 0))
        texts = [g.get("text", "").strip() for g in group if g.get("text")]
        merged_text = " ".join(texts)
        return {
            "doc_id": group[0].get("doc_id"),
            "chunk_ids": [g.get("id") for g in group if g.get("id")],
            "page_range": self._merge_page_ranges(group),
            "source_path": group[0].get("source_path"),
            "text": merged_text,
        }

    def _normalize_chunk(self, chunk: Any) -> Dict[str, Any]:
        if isinstance(chunk, dict):
            return chunk
        return {
            "id": get_attr_or_key(chunk, "id"),
            "doc_id": get_attr_or_key(chunk, "doc_id"),
            "position": get_attr_or_key(chunk, "position", 0),
            "page_range": get_attr_or_key(chunk, "page_range"),
            "source_path": get_attr_or_key(chunk, "source_path"),
            "text": get_attr_or_key(chunk, "text", ""),
            "meta": get_attr_or_key(chunk, "meta") or {},
        }

    # ------------------------------------------------------------------
    # Step 2 — scoring
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 4 — per-snippet refinement
    # ------------------------------------------------------------------

    async def _refine_single(
        self,
        query_text: str,
        candidate: Dict[str, Any],
        *,
        max_chars: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        text = trim_to_sentence_boundary(str(candidate.get("text") or ""), max_chars=max_chars)
        if not self.llm:
            return self._build_snippet(candidate, text), 1.0

        prompt = self._single_prompt(query_text, text)
        try:
            raw = await generate_text(
                llm=self.llm,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                ctx=LLMCallContext(op="snippet_refine_single"),
            )
            result = self._parse_single_response(raw)
        except Exception:
            logger.exception("SnippetRefiner.refine_single failed")
            return self._build_snippet(candidate, text), 1.0

        score_val = result.get("score")
        try:
            score = float(score_val) if score_val is not None else None
        except Exception:
            score = None
        if score is None:
            logger.debug("SnippetRefiner.refine_single: missing score; keeping")
            score = 1.0

        refined_text = result.get("rewritten_text") or text
        refined_text = trim_to_sentence_boundary(str(refined_text or ""), max_chars=max_chars)
        return self._build_snippet(candidate, refined_text), score

    # ------------------------------------------------------------------
    # Snippet construction
    # ------------------------------------------------------------------

    def _build_snippet(self, candidate: Dict[str, Any], text: str) -> Dict[str, Any]:
        sid = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        source_path = candidate.get("source_path")
        file_name = ""
        if isinstance(source_path, str) and source_path.strip():
            file_name = os.path.basename(source_path.strip())

        return {
            "id": f"snippet:{sid}",
            "source": {
                "type": "document",
                "doc_id": candidate.get("doc_id"),
                "chunk_ids": candidate.get("chunk_ids", []),
                "page_range": candidate.get("page_range"),
                "source_path": source_path,
                "file_name": file_name,  # <-- added (basename only)
            },
            "text": text.strip(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _merge_page_ranges(self, group: List[Dict[str, Any]]) -> Optional[str]:
        pages = [g.get("page_range") for g in group if g.get("page_range")]
        return pages[0] if pages else None

    # ------------------------------------------------------------------
    # LLM prompts (safe, bounded)
    # ------------------------------------------------------------------

    def _single_prompt(self, query_text: str, text: str) -> str:
        return f"""
        You are a memory quality evaluator.

        User question:
        \"\"\"{query_text}\"\"\"

        Candidate excerpt:
        \"\"\"{text}\"\"\"

        You are a retrieval assistant that scores excerpt relevance.
        Given a user question and excerpts, evaluate how relevant this excerpt is to the user question.
        
        Relevance definition:
        An excerpt is relevant if it directly helps answer the question or provides essential context needed to interpret or extend another relevant excerpts.

        Score definition (0.0-1.0):
        - 1.0 = directly answers the question
        - 0.7 = strongly relevant supporting evidence
        - 0.4 = somewhat relevant background
        - 0.0 = irrelevant

        If score >= 0.7, you MAY lightly rewrite for coherence only (no new facts).
        If score < 0.7, do not rewrite; return rewritten_text as an empty string.

        Return STRICT JSON only:
        {{\"score\": 0.0-1.0, \"rewritten_text\": \"...\"}}

        """

    def _parse_single_response(self, raw: str) -> Dict[str, Any]:
        raw = (raw or "").strip()
        if not raw:
            logger.debug("SnippetRefiner._parse_single_response: raw is empty")
            return {"score": 1.0}
        try:
            return json.loads(raw)
        except Exception:
            logger.exception("SnippetRefiner._parse_single_response: strict json.loads failed")
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                logger.exception("SnippetRefiner._parse_single_response: salvaged json.loads failed")
        # Fallback: keep original snippet if model returns non-JSON
        return {"score": 1.0}
    
