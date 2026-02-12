# uma/core/retrieval/rlm/snippet_refiner.py

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from uma.core.utils.user_query_helper import extract_keywords_and_phrases
from uma.core.utils.accessors import get_attr_or_key
from uma.core.llm.controller import LLMCallContext, generate_text

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

        # Step 0 — cheap deterministic prefilter
        candidates = self._prefilter_chunks(chunks, query_text)
        if not candidates:
            logger.debug("SnippetRefiner.refine: prefilter removed all candidates")
            return []

        # Step 1 — group adjacent chunks
        grouped = self._group_chunks(candidates)
        logger.debug(
            "SnippetRefiner.refine: candidates=%d grouped=%d facts=%d",
            len(candidates),
            len(grouped),
            len(facts or []),
        )

        # Step 2 — score + shortlist
        facts_norm = [self._normalize_fact(f) for f in (facts or [])]
        scored = self._score_candidates(grouped, query_text, facts_norm)
        shortlist_k = max(1, int(self.cfg.snippet_refiner_top_k or 6))
        shortlist = scored[:shortlist_k]
        logger.debug(
            "SnippetRefiner.refine: scored=%d shortlist_k=%d kept_for_refine=%d",
            len(scored),
            shortlist_k,
            len(shortlist),
        )

        # Step 3 — keep shortlist deterministically (LLM does not gate selection)
        kept = shortlist

        # Step 4 — per-snippet refinement (bounded)
        max_out = int(self.cfg.max_chunks or 3)
        refined_scored: List[Tuple[float, Dict[str, Any]]] = []
        for cand in kept[:max(1, shortlist_k)]:
            refined, score = await self._refine_single(query_text, cand)
            if refined is not None and score is not None:
                refined_scored.append((float(score), refined))

        kept_scored = sorted(
            [(s, snip) for (s, snip) in refined_scored if s >= 0.7],
            key=lambda x: x[0],
            reverse=True,
        )

        if kept_scored:
            final = [snip for _, snip in kept_scored[:max_out]]
            logger.debug("SnippetRefiner.refine: final_snippets=%d", len(final))
            return final

        # Top-K fallback: if everything is below threshold, keep the best N anyway.
        if refined_scored:
            refined_scored.sort(key=lambda x: x[0], reverse=True)
            final = [snip for _, snip in refined_scored[:max_out]]
            logger.debug(
                "SnippetRefiner.refine: fallback_keep_top_k=%d best_score=%.2f",
                len(final),
                refined_scored[0][0],
            )
            return final

        logger.debug("SnippetRefiner.refine: no snippets survived refinement")
        return []

    # ------------------------------------------------------------------
    # Step 0 — deterministic prefilter
    # ------------------------------------------------------------------

    def _prefilter_chunks(self, chunks: List[Any], query_text: str) -> List[Dict[str, Any]]:
        terms = self._extract_terms(query_text)
        out: List[Dict[str, Any]] = []

        for ch in chunks:
            norm = self._normalize_chunk(ch)
            text = (norm.get("text") or "").strip()
            if not text:
                continue
            if not self._contains_terms(text, terms):
                continue
            out.append(norm)

        logger.debug("SnippetRefiner.prefilter: %d → %d", len(chunks), len(out))
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
        chunks = sorted(
            chunks,
            key=lambda c: (c.get("doc_id"), int(c.get("position", 0))),
        )

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

    def _normalize_fact(self, fact: Any) -> Dict[str, Any]:
        if isinstance(fact, dict):
            return fact
        return {
            "id": get_attr_or_key(fact, "id"),
            "subject": get_attr_or_key(fact, "subject"),
            "predicate": get_attr_or_key(fact, "predicate"),
            "object": get_attr_or_key(fact, "object"),
            "confidence": get_attr_or_key(fact, "confidence", 0.5),
            "meta": get_attr_or_key(fact, "meta") or {},
        }

    # ------------------------------------------------------------------
    # Step 2 — scoring
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query_text: str,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        terms = self._extract_terms(query_text)
        if not terms and not facts:
            return candidates
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for c in candidates:
            text = c["text"].lower()
            relevance = sum(1 for t in terms if t in text)
            fact_support = sum(
                1
                for f in facts
                if any(x in text for x in self._fact_terms(f))
            )
            length_penalty = max(0.0, (len(text) - 1200) / 1200)

            score = relevance * 1.0 + fact_support * 1.5 - length_penalty
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        kept = [c for score, c in scored if score > 0]
        return kept or candidates

    # ------------------------------------------------------------------
    # Step 4 — per-snippet refinement
    # ------------------------------------------------------------------

    async def _refine_single(
        self,
        query_text: str,
        candidate: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        text = candidate["text"]
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
        return self._build_snippet(candidate, refined_text), score

    # ------------------------------------------------------------------
    # Snippet construction
    # ------------------------------------------------------------------

    def _build_snippet(self, candidate: Dict[str, Any], text: str) -> Dict[str, Any]:
        sid = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        return {
            "id": f"snippet:{sid}",
            "source": {
                "type": "document",
                "doc_id": candidate.get("doc_id"),
                "chunk_ids": candidate.get("chunk_ids", []),
                "page_range": candidate.get("page_range"),
                "source_path": candidate.get("source_path"),
            },
            "text": text.strip(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_terms(self, query_text: str) -> List[str]:
        extracted = extract_keywords_and_phrases(query_text)
        terms = (extracted.get("keywords") or []) + (extracted.get("keyphrases") or [])
        return [t.lower() for t in terms if isinstance(t, str) and len(t) > 2]

    def _contains_terms(self, text: str, terms: List[str]) -> bool:
        if not terms:
            return True
        low = text.lower()
        return any(t in low for t in terms)

    def _fact_terms(self, fact: Any) -> List[str]:
        out: List[str] = []
        for k in ("subject", "predicate", "object"):
            v = get_attr_or_key(fact, k)
            if isinstance(v, str):
                out.append(v.lower())
        return out

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
            raise
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                logger.exception("SnippetRefiner._parse_single_response: salvaged json.loads failed")
                raise
        # Fallback: keep original snippet if model returns non-JSON
        return {"score": 1.0}
    
