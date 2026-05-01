from __future__ import annotations

"""
uma.memory.semantic.extractor
==========================

Canonical API for semantic fact extraction.

This module defines a single class: `FactExtractor`.

It supports two related but distinct extraction surfaces:

1) User/turn facts
   - Extract stable, long-term facts about a user from interaction text.
   - Short facts are allowed (e.g., "user likes sushi"), so `min_fact_words` defaults to 0.

2) Document-chunk facts (ingestion)
   - Extract KB-grade facts from document chunks.
   - Chunk facts are expected to be more descriptive, so ingestion passes a
     higher `min_fact_words` (e.g., 10-15).
   - Batching is used to keep cost predictable.
   - Enforces hard caps in code: up to 4 facts per chunk, object <= 50 words,
     max_fact_tokens ~120 (approx).

Robustness & invariants
-----------------------
- Ingestion must not fail if a single chunk or a single batch fails.
- For eligible chunks (>= MIN_EXTRACT_CHUNK_CHARS), we do NOT allow 0 facts:
  - First attempt to salvage by re-parsing the SAME payload with relaxed min_fact_words,
    so we avoid deterministic fallback whenever possible.
  - If still empty, use deterministic fallback as a last resort (data-agnostic).

Observability
-------------
- INFO for start/end totals.
- WARNING for partial failures / forced fallbacks.
- DEBUG for fine-grained drop reasons and previews (capped).

Implementation notes
--------------------
- Chunk selection, batching, prompt building, parsing, and enforcement helpers
  live in `uma.memory.semantic.extractor_utils`.

"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from uma.adapters.llm.base import LLMInterface
from uma.common.types import Fact
from uma.ingest.types import DocumentChunk
from uma.adapters.llm.controller import LLMCallContext, generate_json
from uma.common.json_utils import try_parse_json_object
from .scorer import SalienceScorer
from . import extractor_utils as utils
from uma.common.storage_metadata import normalize_fact_metadata

logger = logging.getLogger(__name__)


class FactExtractor:
    """Canonical fact extraction surface for both user facts and chunk facts."""

    def __init__(self, llm: LLMInterface, scorer: Optional[SalienceScorer] = None) -> None:
        if llm is None or not hasattr(llm, "generate"):
            raise ValueError("FactExtractor: llm with .generate() required")
        self.llm = llm
        self.scorer = scorer or SalienceScorer()
        logger.debug("FactExtractor initialized.")

    # ---------------------------------------------------------------------
    # User facts (turn/text)
    # ---------------------------------------------------------------------
    async def extract_user_facts(
        self,
        *,
        subject: str,
        text: str,
        owner_type: str,
        owner_id: str,
        min_fact_words: int = 0,
        max_facts: int = 6,
        max_tokens: int = 450,
        extra_meta: Optional[dict] = None,
    ) -> List[Fact]:
        """
        Extract stable, long-term facts about the user.

        - Short objects are allowed by default (min_fact_words=0).
        - Returns List[Fact] with consistent Fact.salience.
        """
        if not isinstance(subject, str) or not subject.strip():
            logger.warning("FactExtractor.extract_user_facts: invalid subject=%r", subject)
            return []
        if not isinstance(text, str) or not text.strip():
            return []
        if owner_type not in ("user", "agent"):
            raise ValueError(f"FactExtractor.extract_user_facts: invalid owner_type={owner_type!r}")
        if not owner_id or not isinstance(owner_id, str):
            raise ValueError("FactExtractor.extract_user_facts: owner_id must be a non-empty string")

        min_fact_words = max(0, int(min_fact_words))
        max_facts = max(0, int(max_facts))
        if max_facts == 0:
            return []

        system_prompt = (
            "Extract LONG-TERM, STABLE facts about the USER from the text.\n"
            "Do NOT include ephemeral or turn-specific details.\n"
            "Do NOT paraphrase the whole message—extract only durable user facts.\n\n"
            "Return ONLY valid JSON in this schema:\n"
            "{\n"
            '  "facts": [\n'
            "    {\n"
            '      "predicate": "likes",\n'
            '      "object": "sushi",\n'
            '      "confidence": 0.0-1.0,\n'
            '      "source_ids": []\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"Rules: return AT MOST {max_facts} facts. "
            f"Each object must be at least {min_fact_words} words long.\n"
            "No prose. No markdown."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"SUBJECT: {subject}\nTEXT:\n{text}\n"},
        ]

        try:
            raw = await self.llm.generate(messages, max_tokens=int(max_tokens), temperature=0.0)
        except Exception:
            logger.exception("FactExtractor.extract_user_facts: LLM generate failed.")
            return []

        data = try_parse_json_object(raw)
        if data is None:
            logger.error("FactExtractor.extract_user_facts: invalid JSON output (unsalvageable). RAW=%r", raw)
            return []

        facts_payload = data.get("facts")
        if not isinstance(facts_payload, list):
            logger.warning("FactExtractor.extract_user_facts: JSON missing list 'facts'.")
            return []

        now = datetime.now(timezone.utc)
        out: List[Fact] = []

        turn_id = None
        if isinstance(extra_meta, dict) and extra_meta.get("turn_id"):
            turn_id = str(extra_meta["turn_id"])

        kept = 0
        for item in facts_payload:
            if kept >= max_facts:
                break
            if not isinstance(item, dict):
                continue

            predicate = item.get("predicate")
            obj = item.get("object")
            conf = item.get("confidence", 0.7)
            source_ids = item.get("source_ids", [])

            if not predicate or obj is None:
                continue

            obj_text = utils.coerce_object_text(obj)
            if not obj_text:
                continue
            if min_fact_words and utils.word_count(obj_text) < min_fact_words:
                continue

            confidence = utils.safe_confidence(conf, default=0.7)

            # Keep user facts reasonably bounded (still generic).
            subj_n, pred_n, obj_n = utils.enforce_fact_limits(
                subj=subject,
                pred=str(predicate),
                obj_text=obj_text,
                object_max_words=utils.DEFAULT_OBJECT_MAX_WORDS,
                max_fact_tokens=utils.DEFAULT_MAX_FACT_TOKENS_USER,
            )
            if not obj_n:
                continue

            sid_list: List[str] = []
            if isinstance(source_ids, list):
                sid_list = [str(s) for s in source_ids if s is not None]

            fact = Fact(
                id=f"fact_{utils.uuid_from_text(f'userfact:v1:{owner_type}:{owner_id}:{subj_n}:{pred_n}:{obj_n}')}",
                subject=subj_n,
                predicate=pred_n,
                object=obj_n,
                created_at=now,
                updated_at=now,
                source_ids=sid_list,
                confidence=confidence,
                salience=0.0,
                owner_type=owner_type,
                owner_id=owner_id,
                meta=normalize_fact_metadata(
                    {"domain": "user_profile"},
                    fact_id=f"fact_{utils.uuid_from_text(f'userfact:v1:{owner_type}:{owner_id}:{subj_n}:{pred_n}:{obj_n}')}",
                    owner_type=owner_type,
                    owner_id=owner_id,
                    created_at=now,
                    updated_at=now,
                    source_ids=sid_list,
                    session_id=None,
                ),
            )
            if turn_id:
                fact.meta["turn_id"] = turn_id

            fact.salience = float(self.scorer.score(fact))
            out.append(fact)
            kept += 1

        logger.info("FactExtractor.extract_user_facts: extracted=%d subject=%s", len(out), subject)
        return out

    # ---------------------------------------------------------------------
    # Chunk selection (generic, deterministic)
    # ---------------------------------------------------------------------
    @staticmethod
    def select_chunks_for_fact_extraction(
        chunks: List[DocumentChunk],
        *,
        max_chunks: Optional[int] = None,
        max_per_page: Optional[int] = None,
    ) -> List[DocumentChunk]:
        return utils.select_chunks_for_fact_extraction(chunks, max_chunks=max_chunks, max_per_page=max_per_page)

    # ---------------------------------------------------------------------
    # Chunk facts (ingestion)
    # ---------------------------------------------------------------------
    async def extract_chunk_facts_batch(
        self,
        chunks: List[DocumentChunk],
        *,
        owner_type: str,
        owner_id: str,
        source_path: str,
        source_hash: str,
        doc_id: str,
        min_fact_words: int,
        batch_size_chunks: int = utils.DEFAULT_BATCH_SIZE_CHUNKS,
        max_chars: int = utils.DEFAULT_BATCH_MAX_CHARS,
        max_facts_per_chunk: int = utils.DEFAULT_MAX_FACTS_PER_CHUNK,
        object_max_words: int = utils.DEFAULT_OBJECT_MAX_WORDS,
        max_fact_tokens: int = utils.DEFAULT_MAX_FACT_TOKENS,
    ) -> List[Fact]:
        """
        Extract KB-grade facts from chunks using batch calls.

        Returns List[Fact] where:
        - fact.source_ids = [source_chunk_id]
        - fact.owner_type/owner_id set by caller (document owner)
        - fact.meta includes doc/source fields used downstream

        Invariant (eligible chunk): never returns 0 facts for a chunk with text >= MIN_EXTRACT_CHUNK_CHARS.
        """
        if not chunks:
            return []
        if owner_type not in ("user", "agent", "workspace"):
            raise ValueError(f"extract_chunk_facts_batch: invalid owner_type={owner_type!r}")
        if not owner_id or not isinstance(owner_id, str):
            raise ValueError("extract_chunk_facts_batch: owner_id must be a non-empty string")

        min_fact_words = max(0, int(min_fact_words))

        logger.info(
            "FactExtractor.extract_chunk_facts_batch: start chunks=%d batch_size_chunks=%d max_chars=%d min_fact_words=%d",
            len(chunks),
            int(batch_size_chunks),
            int(max_chars),
            min_fact_words,
        )

        batches = utils.partition_batches_by_chars(
            chunks,
            batch_size_chunks=int(batch_size_chunks),
            max_chars=int(max_chars),
        )

        now = datetime.now(timezone.utc)
        out: List[Fact] = []
        batch_failures = 0
        forced_fallbacks = 0

        for batch_idx, batch in enumerate(batches):
            batch_for_llm: List[DocumentChunk] = []
            for c in batch:
                t = (c.text or "").strip()
                if not t:
                    continue
                if len(t) < utils.MIN_EXTRACT_CHUNK_CHARS:
                    logger.debug(
                        "FactExtractor.extract_chunk_facts_batch: skip tiny chunk chunk_id=%s chars=%d",
                        c.chunk_id,
                        len(t),
                    )
                    continue
                batch_for_llm.append(c)

            if not batch_for_llm:
                continue

            items = [{"chunk_id": c.chunk_id, "text": (c.text or "")} for c in batch_for_llm]

            data: Optional[Dict[str, Any]] = None
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    approx_chars = sum(len((it.get("text") or "")) for it in items)
                    logger.debug(
                        "FactExtractor.extract_chunk_facts_batch: batch_idx=%d send chunks=%d approx_chars=%d ids=%s",
                        batch_idx,
                        len(items),
                        approx_chars,
                        [it["chunk_id"] for it in items[:3]],
                    )

                data = await generate_json(
                    llm=self.llm,
                    messages=utils.build_prompt(
                        mode="batch",
                        items=items,
                        min_fact_words=min_fact_words,
                        max_facts_per_chunk=int(max_facts_per_chunk),
                    ),
                    max_tokens=utils.DEFAULT_BATCH_CALL_MAX_TOKENS,
                    ctx=LLMCallContext(op="ingest_fact_extract_batch"),
                    repair_messages_fn=lambda bad: utils.batch_repair_messages(
                        bad,
                        max_facts_per_chunk=int(max_facts_per_chunk),
                    ),
                )
            except Exception:
                logger.exception("FactExtractor.extract_chunk_facts_batch: llm failed batch_idx=%d", batch_idx)
                data = None
                batch_failures += 1

            chunks_payload = data.get("chunks") if isinstance(data, dict) else None
            if not isinstance(chunks_payload, dict):
                chunks_payload = {}

            if logger.isEnabledFor(logging.DEBUG):
                returned_keys = list(chunks_payload.keys())
                logger.debug(
                    "FactExtractor.extract_chunk_facts_batch: batch_idx=%d payload_keys=%d missing=%d",
                    batch_idx,
                    len(returned_keys),
                    max(0, len(items) - len(returned_keys)),
                )

            for c in batch_for_llm:
                payload = chunks_payload.get(c.chunk_id)
                extracted_for_chunk: List[Fact] = []

                # 1) parse batch payload if present
                if isinstance(payload, dict):
                    extracted_for_chunk = utils.parse_chunk_payload_into_facts(
                        chunk=c,
                        payload=payload,
                        min_fact_words=min_fact_words,
                        scorer=self.scorer,
                        max_facts_per_chunk=int(max_facts_per_chunk),
                        object_max_words=int(object_max_words),
                        max_fact_tokens=int(max_fact_tokens),
                        owner_type=owner_type,
                        owner_id=owner_id,
                        now=now,
                        doc_id=doc_id,
                        source_path=source_path,
                        source_hash=source_hash,
                    )

                # 2) if missing/invalid payload, fallback to per-chunk LLM call
                if not extracted_for_chunk and not isinstance(payload, dict):
                    extracted_for_chunk = await self.extract_chunk_facts_one(
                        c,
                        owner_type=owner_type,
                        owner_id=owner_id,
                        source_path=source_path,
                        source_hash=source_hash,
                        doc_id=doc_id,
                        min_fact_words=min_fact_words,
                        max_facts=int(max_facts_per_chunk),
                        object_max_words=int(object_max_words),
                        max_fact_tokens=int(max_fact_tokens),
                        now=now,
                    )

                # 3) invariant: eligible chunk must never yield 0 facts
                if not extracted_for_chunk:
                    forced_fallbacks += 1
                    logger.warning(
                        "FactExtractor.extract_chunk_facts_batch: forcing deterministic fallback chunk_id=%s",
                        c.chunk_id,
                    )
                    extracted_for_chunk = [
                        utils.fallback_fact_for_chunk(
                            c,
                            owner_type=owner_type,
                            owner_id=owner_id,
                            doc_id=doc_id,
                            source_path=source_path,
                            source_hash=source_hash,
                            now=now,
                            object_max_words=int(object_max_words),
                            max_fact_tokens=int(max_fact_tokens),
                            scorer=self.scorer,
                        )
                    ]

                out.extend(extracted_for_chunk)

        if batch_failures:
            logger.warning(
                "FactExtractor.extract_chunk_facts_batch: completed with batch_failures=%d total_chunks=%d",
                batch_failures,
                len(chunks),
            )
        if forced_fallbacks:
            logger.warning("FactExtractor.extract_chunk_facts_batch: forced_fallbacks=%d", forced_fallbacks)

        logger.info(
            "FactExtractor.extract_chunk_facts_batch: done facts=%d chunks=%d",
            len(out),
            len(chunks),
        )
        return out

    async def extract_chunk_facts_one(
        self,
        chunk: DocumentChunk,
        *,
        owner_type: str,
        owner_id: str,
        source_path: str,
        source_hash: str,
        doc_id: str,
        min_fact_words: int,
        max_facts: int = utils.DEFAULT_MAX_FACTS_PER_CHUNK,
        object_max_words: int = utils.DEFAULT_OBJECT_MAX_WORDS,
        max_fact_tokens: int = utils.DEFAULT_MAX_FACT_TOKENS,
        now: Optional[datetime] = None,
    ) -> List[Fact]:
        if chunk is None:
            return []
        if owner_type not in ("user", "agent", "workspace"):
            raise ValueError(f"extract_chunk_facts_one: invalid owner_type={owner_type!r}")
        if not owner_id or not isinstance(owner_id, str):
            raise ValueError("extract_chunk_facts_one: owner_id must be a non-empty string")

        text = (chunk.text or "").strip()
        if not text:
            return []
        if len(text) < utils.MIN_EXTRACT_CHUNK_CHARS:
            logger.debug(
                "FactExtractor.extract_chunk_facts_one: skip tiny chunk chunk_id=%s chars=%d",
                chunk.chunk_id,
                len(text),
            )
            return []

        now = now or datetime.now(timezone.utc)
        min_fact_words = max(0, int(min_fact_words))

        try:
            data = await generate_json(
                llm=self.llm,
                messages=utils.build_prompt(
                    mode="single",
                    chunk_text=text,
                    min_fact_words=min_fact_words,
                    max_facts_per_chunk=int(max_facts),
                ),
                max_tokens=utils.DEFAULT_SINGLE_CALL_MAX_TOKENS,
                ctx=LLMCallContext(op="ingest_fact_extract_one"),
                repair_messages_fn=lambda bad: utils.single_repair_messages(bad, max_facts=int(max_facts)),
            )
        except Exception:
            logger.exception("FactExtractor.extract_chunk_facts_one: LLM failed chunk_id=%s", chunk.chunk_id)
            return [
                utils.fallback_fact_for_chunk(
                    chunk,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    doc_id=doc_id,
                    source_path=source_path,
                    source_hash=source_hash,
                    now=now,
                    object_max_words=int(object_max_words),
                    max_fact_tokens=int(max_fact_tokens),
                    scorer=self.scorer,
                )
            ]

        facts_payload = data.get("facts") if isinstance(data, dict) else None

        out = utils.parse_facts_list_into_facts(
            facts_payload=facts_payload,
            chunk=chunk,
            min_fact_words=min_fact_words,
            scorer=self.scorer,
            max_facts_per_chunk=int(max_facts),
            object_max_words=int(object_max_words),
            max_fact_tokens=int(max_fact_tokens),
            predicate_default="STATES",
            owner_type=owner_type,
            owner_id=owner_id,
            now=now,
            doc_id=doc_id,
            source_path=source_path,
            source_hash=source_hash,
        )

        # Salvage: same payload but relaxed min_fact_words
        # if not out and min_fact_words > 0:
        #     out = utils.parse_facts_list_into_facts(
        #         facts_payload=facts_payload,
        #         chunk=chunk,
        #         min_fact_words=0,
        #         scorer=self.scorer,
        #         max_facts_per_chunk=int(max_facts),
        #         object_max_words=int(object_max_words),
        #         max_fact_tokens=int(max_fact_tokens),
        #         predicate_default="STATES",
        #         owner_type=owner_type,
        #         owner_id=owner_id,
        #         now=now,
        #         doc_id=doc_id,
        #         source_path=source_path,
        #         source_hash=source_hash,
        #     )

        # if not out:
        #     logger.warning(
        #         "FactExtractor.extract_chunk_facts_one: no facts after parsing; forcing fallback chunk_id=%s",
        #         chunk.chunk_id,
        #     )
        #     out = [
        #         utils.fallback_fact_for_chunk(
        #             chunk,
        #             owner_type=owner_type,
        #             owner_id=owner_id,
        #             doc_id=doc_id,
        #             source_path=source_path,
        #             source_hash=source_hash,
        #             now=now,
        #             object_max_words=int(object_max_words),
        #             max_fact_tokens=int(max_fact_tokens),
        #             scorer=self.scorer,
        #         )
        #     ]
        return out

    async def extract_chunk_facts(
        self,
        chunks: List[DocumentChunk],
        *,
        owner_type: str,
        owner_id: str,
        source_path: str,
        source_hash: str,
        doc_id: str,
        min_fact_words: int,
        max_facts_per_chunk: int = utils.DEFAULT_MAX_FACTS_PER_CHUNK,
        object_max_words: int = utils.DEFAULT_OBJECT_MAX_WORDS,
        max_fact_tokens: int = utils.DEFAULT_MAX_FACT_TOKENS,
    ) -> List[Fact]:
        """
        Deterministic per-chunk extraction (no batching).
        Intended primarily for debugging or narrow use cases.
        """
        if not chunks:
            return []
        now = datetime.now(timezone.utc)
        out: List[Fact] = []
        ordered = sorted(chunks, key=lambda c: (int(getattr(c, "position", 0) or 0), getattr(c, "chunk_id", "")))
        for ch in ordered:
            out.extend(
                await self.extract_chunk_facts_one(
                    ch,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    source_path=source_path,
                    source_hash=source_hash,
                    doc_id=doc_id,
                    min_fact_words=min_fact_words,
                    max_facts=int(max_facts_per_chunk),
                    object_max_words=int(object_max_words),
                    max_fact_tokens=int(max_fact_tokens),
                    now=now,
                )
            )
        return out
