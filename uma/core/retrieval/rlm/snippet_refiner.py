# uma/core/retrieval/rlm/snippet_refiner.py

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from uma.core.utils.user_query_helper import extract_query_terms, expand_query_terms
from uma.core.utils.accessors import get_attr_or_key

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
        facts: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
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
            return []

        # Step 0 — cheap deterministic prefilter
        candidates = self._prefilter_chunks(chunks, query_text)
        if not candidates:
            return []

        # Step 1 — group adjacent chunks
        grouped = self._group_chunks(candidates)

        # Step 2 — score + shortlist
        scored = self._score_candidates(grouped, query_text, facts)
        shortlist = scored[: max(1, int(self.cfg.snippet_refiner_top_k or 6))]

        # Step 3 — keep shortlist deterministically (LLM does not gate selection)
        kept = shortlist

        # Step 4 — per-snippet refinement (bounded)
        final: List[Dict[str, Any]] = []
        for cand in kept[: int(self.cfg.max_chunks or 3)]:
            refined = await self._refine_single(query_text, cand)
            if refined:
                final.append(refined)

        return final

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
        Group chunks by doc_id and adjacency (position ±1).
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
            "text": get_attr_or_key(chunk, "text", ""),
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
    # Step 3 — batch classification
    # ------------------------------------------------------------------

    async def _batch_classify(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.llm:
            return candidates

        excerpts = [c["text"] for c in candidates]
        prompt = self._batch_prompt(query_text, excerpts)

        try:
            raw = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
            )
            parsed = self._parse_batch_response(raw)
        except Exception:
            logger.exception("SnippetRefiner.batch_classify failed; keeping all")
            return candidates

        kept: List[Dict[str, Any]] = []
        for idx, keep in parsed.items():
            if keep and idx < len(candidates):
                kept.append(candidates[idx])

        return kept

    # ------------------------------------------------------------------
    # Step 4 — per-snippet refinement
    # ------------------------------------------------------------------

    async def _refine_single(
        self,
        query_text: str,
        candidate: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        text = candidate["text"]
        if not self.llm:
            return self._build_snippet(candidate, text)

        prompt = self._single_prompt(query_text, text)
        try:
            raw = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
            )
            result = self._parse_single_response(raw)
        except Exception:
            logger.exception("SnippetRefiner.refine_single failed")
            return self._build_snippet(candidate, text)

        confidence = result.get("confidence")
        try:
            conf = float(confidence) if confidence is not None else 0.0
        except Exception:
            conf = 0.0
        # Require high confidence for relevance; low confidence can be discarded.
        if conf < 0.8:
            return None
        if result.get("keep") is False:
            return None

        refined_text = result.get("rewritten_text") or text
        return self._build_snippet(candidate, refined_text)

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
            },
            "text": text.strip(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_terms(self, query_text: str) -> List[str]:
        terms = expand_query_terms(query_text) or extract_query_terms(query_text) or []
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

    def _starts_like_fragment(self, text: str) -> bool:
        return text[:1].islower() and not text.strip().endswith((".", "!", "?"))

    # ------------------------------------------------------------------
    # LLM prompts (safe, bounded)
    # ------------------------------------------------------------------

    def _batch_prompt(self, query_text: str, snippets: List[str]) -> str:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(snippets))
        return f"""
You are a memory quality evaluator.

User question:
\"\"\"{query_text}\"\"\"

Candidate excerpts:
{numbered}

For each excerpt, decide if it is meaningful on its own.

Return STRICT JSON:
{{\"keep\": [true|false, ...]}}
"""

    def _parse_batch_response(self, raw: str) -> Dict[int, bool]:
        data = json.loads(raw)
        keeps = data.get("keep", [])
        return {i: bool(v) for i, v in enumerate(keeps)}

    def _single_prompt(self, query_text: str, text: str) -> str:
        return f"""
You are a memory quality evaluator.

User question:
\"\"\"{query_text}\"\"\"

Candidate excerpt:
\"\"\"{text}\"\"\"

Decide if the excerpt is meaningful.
If yes, lightly rewrite for coherence only.

Return STRICT JSON:
{{\"keep\": true|false, \"confidence\": 0.0-1.0, \"rewritten_text\": \"...\"}}

Confidence definition:
- High confidence means you are very sure the excerpt is relevant.
- Low confidence means you are not sure and it can be discarded.
"""

    def _parse_single_response(self, raw: str) -> Dict[str, Any]:
        raw = (raw or "").strip()
        if not raw:
            return {"keep": True}
        try:
            return json.loads(raw)
        except Exception:
            pass
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # Fallback: keep original snippet if model returns non-JSON
        return {"keep": True}
    
