"""
semantic/core.py
================

SemanticCore — The unified interface for UMA semantic memory.

Ownership model (vNext)
-----------------------
Retrieval MUST be ownership-scoped ONLY via (owner_type, owner_id).

- owner_type: "user" | "agent"
- owner_id: scope identifier (for user scope: canonical "user:<id>" string)

Fact.subject is treated as OPTIONAL metadata and MUST NOT gate retrieval.

Ingestion/extract APIs accept a `user_id` string and normalize it, but persisted facts
MUST have explicit ownership (owner_type/owner_id).

Coding Agent Instructions
-------------------------
- Keep this interface simple and stable.
- Retrieval must NOT require or accept "subject" gating.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import Fact, RuntimeContext, SCOPE_MODEL_VERSION
from uma.retrieve.ranking import fuse_candidates
from uma.common.identity import normalize_user_id
from uma.common.dedupe import dedupe_by_id
from uma.common.trust import SourceDescriptor, score_source
from uma.adapters.scanner.injection_scan import apply_scan, quarantine_enabled, scan_content
from .ingestor import SemanticIngestor

logger = logging.getLogger(__name__)


class SemanticCore:
    """
    High-level interface for UMA semantic memory.

    Wraps:
        - SemanticIngestor
        - semantic_store (SQL + vector)

    IMPORTANT:
    - Retrieval is ownership-only.
    - No subject-driven retrieval is allowed.
    """

    def __init__(
        self,
        llm: Any,
        embedder: Any,
        semantic_store: Any,
        salience_threshold: float = 0.3,
        salience_decay_days: float = 180.0,
        *,
        memory: Optional[Any] = None,
    ) -> None:
        self.ingestor = SemanticIngestor(
            llm=llm,
            embedder=embedder,
            semantic_store=semantic_store,
            salience_threshold=salience_threshold,
            salience_decay_days=salience_decay_days,
        )
        logger.debug("SemanticCore initialized.")

        self._memory = memory
        self.store = getattr(self.ingestor, "semantic_store", None)
        if self.store is None:
            logger.error("SemanticCore: store missing or unsupported")
            raise RuntimeError("SemanticCore: store missing or unsupported")


    # ------------------------------------------------------------------
    # PUBLIC API (WRITE / INGEST)
    # ------------------------------------------------------------------

    async def upsert_fact(self, fact: Fact, embedding: list[float]) -> None:
        """
        Persist a fact + embedding.

        Ownership consistency:
        - owner_type and owner_id MUST be present.
        - We do not infer owner_id from fact.subject.
        """
        if not hasattr(self.store, "upsert_fact"):
            logger.error("SemanticCore.upsert_fact: semantic_store missing")
            raise RuntimeError("SemanticCore.upsert_fact: semantic_store missing")

        fact_id = getattr(fact, "id", None)
        if not fact_id or not isinstance(fact_id, str) or not fact_id.startswith("fact_"):
            logger.error("SemanticCore.upsert_fact: invalid fact id=%r (must start with 'fact_')", fact_id)
            raise ValueError("SemanticCore.upsert_fact: fact.id must start with 'fact_'")

        owner_type = getattr(fact, "owner_type", None)
        owner_id = getattr(fact, "owner_id", None)

        if not owner_type or not owner_id:
            logger.error(
                "SemanticCore.upsert_fact: missing ownership on fact id=%r owner_type=%r owner_id=%r",
                getattr(fact, "id", None),
                owner_type,
                owner_id,
            )
            raise ValueError("SemanticCore.upsert_fact: fact must include owner_type and owner_id")

        try:
            await self.store.upsert_fact(fact, embedding)
        except Exception:
            logger.exception("SemanticCore.upsert_fact failed")
            raise

    async def durable_fact_exists(
        self,
        fact: Fact,
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
    ) -> bool:
        """Return True if an equivalent fact already exists in the target scope.

        Used by the promotion path to avoid minting a second durable copy of
        content that is already in the agent KB. Fails **open** (returns
        False) when the store cannot answer: promotion is best-effort, and a
        guard failure must not silently stop legitimate promotions.
        """
        if not hasattr(self.store, "durable_fact_exists"):
            logger.debug(
                "SemanticCore.durable_fact_exists: store does not implement the guard; treating as novel."
            )
            return False
        try:
            return await self.store.durable_fact_exists(
                fact,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception(
                "SemanticCore.durable_fact_exists failed for fact id=%r; treating as novel.",
                getattr(fact, "id", None),
            )
            return False

    def vector_index(self):
        """Return the underlying ``VectorIndex`` instance for this store."""
        return getattr(self.store, "vector_index", None)

    async def extract(
        self,
        user_id: str,
        text: str,
        *,
        extra_meta: dict | None = None,
    ) -> list[Fact]:
        """
        Extract semantic facts (not persisted).
        Normalizes user_id to the canonical identity format.
        """
        try:
            normalized_user_id = normalize_user_id(user_id)
        except Exception:
            logger.exception("SemanticCore.extract: invalid user_id=%r", user_id)
            raise

        return await self.ingestor.extract(normalized_user_id, text, extra_meta=extra_meta)

    async def ingest(
        self,
        user_id: str,
        text: str,
        *,
        extra_meta: dict | None = None,
        turn_context: Optional[RuntimeContext] = None,
        source_kind: str = "turn_user",
    ) -> list[Fact]:
        """
        Extract + ingest facts into the semantic self.store.

        Ownership consistency:
        - Facts extracted from user-scoped turns MUST be persisted as:
          owner_type="user", owner_id="user:<id>"

        Write-time scanning:
        - The raw input text is always scanned at the boundary; the verdict
          (trust adjustment, security metadata, quarantine timestamp) is
          inherited by every fact derived from it. The earlier behavior of
          gating the scan on `turn_context is not None` is removed: a caller
          providing text without a turn_context is still introducing text
          into UMA, and the boundary scan must run.
        """
        try:
            normalized_user_id = normalize_user_id(user_id)
        except Exception:
            logger.exception("SemanticCore.ingest: invalid user_id=%r", user_id)
            raise

        # Scan input text once at the boundary; result is captured by closure
        # and applied per-fact before storage. Option A discipline: scan at
        # the boundary; everything derived inherits.
        _text_scan = scan_content(text or "")
        _source_ids = [
            str(item)
            for item in list((extra_meta or {}).get("source_ids") or [])
            if item
        ]

        def _apply_turn_scope(f: Fact) -> None:
            try:
                f.owner_type = "user"
                f.owner_id = normalized_user_id
                if _source_ids and not list(getattr(f, "source_ids", None) or []):
                    f.source_ids = list(_source_ids)
                if turn_context is not None:
                    f.tenant_id = turn_context.tenant_id
                    f.workspace_id = turn_context.workspace_id
                    f.session_id = turn_context.session_id
                    f.origin_agent_id = turn_context.agent_id
                    f.origin_user_id = normalized_user_id
                    f.origin_session_id = turn_context.session_id
                    f.scope_model_version = SCOPE_MODEL_VERSION
                    f.trust_score = score_source(SourceDescriptor(kind=source_kind, session_id=turn_context.session_id))
                # Apply the boundary scan verdict to the derived fact. This
                # runs whether or not turn_context was provided — the scan
                # ran at the boundary regardless. Trust starts from whatever
                # was set above (turn-context branch) or the extractor's
                # default; apply_scan adjusts it according to severity.
                f.trust_score, f.meta = apply_scan(
                    f.trust_score,
                    f.meta or {},
                    _text_scan,
                    log_context=f"semantic/user:{normalized_user_id}",
                )
                if _text_scan.severity == "high" and quarantine_enabled():
                    from datetime import datetime, timezone
                    f.quarantined_at = datetime.now(timezone.utc)
            except Exception:
                logger.exception("SemanticCore.ingest: failed to set owner on fact id=%s", getattr(f, "id", None))
                raise

        facts = await self.ingestor.ingest(
            normalized_user_id,
            text,
            extra_meta=extra_meta,
            fact_transform=_apply_turn_scope,
        )

        return self._dedup_facts(facts)

    # ------------------------------------------------------------------
    # RETRIEVAL API (OWNERSHIP-ONLY)
    # ------------------------------------------------------------------

    async def list_facts_for_owner(
        self,
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        limit: Optional[int] = None,
    ) -> list[Fact]:
        """
        List facts for an owner scope (ownership-only).
        """

        if not hasattr(self.store, "list_facts_for_owner"):
            logger.error("SemanticCore.list_facts_for_owner: store missing or unsupported")
            raise RuntimeError("SemanticCore.list_facts_for_owner: store missing or unsupported")
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticCore.list_facts_for_owner requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticCore.list_facts_for_owner requires tenant_id, owner_type and owner_id")
        try:
            return await self.store.list_facts_for_owner(
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                limit=limit,
            )
        except Exception:
            logger.exception("SemanticCore.list_facts_for_owner failed")
            raise

    async def delete_fact(
        self,
        fact_id: str,
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
    ) -> None:
        """
        Delete a fact by id, scoped by ownership.

        Notes
        -----
        - This is a maintenance/pruning API, not a retrieval filter.
        - Store must enforce ownership in the delete path.
        """

        if not fact_id or not isinstance(fact_id, str):
            logger.error("SemanticCore.delete_fact requires fact_id as a non-empty string")
            raise ValueError("SemanticCore.delete_fact requires fact_id as a non-empty string")
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticCore.delete_fact requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticCore.delete_fact requires tenant_id, owner_type and owner_id")
        try:
            await self.store.delete_fact(
                fact_id,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("SemanticCore.delete_fact failed")
            raise

    async def update_trust(
        self,
        fact_id: str,
        new_score: float,
        *,
        reason: str,
        ctx: RuntimeContext,
    ) -> None:
        """Update the trust score of a fact and append an audit entry to its metadata."""
        if not hasattr(self.store, "update_trust"):
            logger.error("SemanticCore.update_trust: semantic_store missing")
            raise RuntimeError("SemanticCore.update_trust: semantic_store missing")
        if not fact_id or not isinstance(fact_id, str):
            logger.error("SemanticCore.update_trust requires fact_id as a non-empty string")
            raise ValueError("SemanticCore.update_trust requires fact_id as a non-empty string")
        if not isinstance(ctx, RuntimeContext):
            logger.error("SemanticCore.update_trust requires a RuntimeContext")
            raise TypeError("SemanticCore.update_trust requires a RuntimeContext")
        try:
            await self.store.update_trust(
                fact_id,
                new_score,
                reason=reason,
                ctx=ctx,
            )
        except Exception:
            logger.exception("SemanticCore.update_trust failed")
            raise

    async def search(
        self,
        query_embedding: list[float],
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        offset: int = 0,
        filters: Optional[dict[str, Any]] = None,
        query_text: Optional[str] = None,
    ) -> list[Fact]:
        """
        Unified semantic retrieval entry point (ownership-only).

        - Vector search is primary.
        - Optional topic/predicate filtering is applied after retrieval.
        - Lexical fallback is allowed but must also be ownership-only.
        """
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticCore.search requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticCore.search requires tenant_id, owner_type and owner_id")

        requested_topic = filters.get("topic") if isinstance(filters, dict) else None
        requested_predicate = filters.get("predicate") if isinstance(filters, dict) else None
        if requested_predicate:
            requested_predicate = str(requested_predicate).upper()

        memory = getattr(self, "_memory", None)
        retrieval_cfg = getattr(memory, "retrieval_cfg", None) if memory is not None else None
        hybrid_cfg = getattr(retrieval_cfg, "hybrid", None) if retrieval_cfg is not None else None
        hybrid_enabled = bool(getattr(hybrid_cfg, "enabled", True)) if hybrid_cfg is not None else True
        fusion_strategy = str(getattr(hybrid_cfg, "fusion_strategy", "rrf") or "rrf") if hybrid_cfg is not None else "rrf"
        try:
            top_k_dense = int(getattr(hybrid_cfg, "top_k_dense", 0) or 0) if hybrid_cfg is not None else 0
        except Exception:
            top_k_dense = 0
        try:
            top_k_sparse = int(getattr(hybrid_cfg, "top_k_sparse", 15) or 15) if hybrid_cfg is not None else 15
        except Exception:
            top_k_sparse = 15
        dense_k = int(k) if top_k_dense <= 0 else max(0, int(top_k_dense))

        dense_facts: list[Any] = []
        try:
            logger.debug("SemanticCore.search: path=vector owner=%s:%s", owner_type, owner_id)
            found = await self.store.search(
                query_embedding=query_embedding,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(dense_k),
                offset=int(offset),
            )
            if found:
                dense_facts.extend(found)
        except Exception:
            logger.exception("SemanticCore.search failed owner=%s:%s", owner_type, owner_id)
            raise

        sparse_facts: list[Any] = []
        if (
            hybrid_enabled
            and top_k_sparse > 0
            and query_text
            and isinstance(query_text, str)
            and query_text.strip()
            and hasattr(self.store, "lexical_search")
        ):
            try:
                logger.debug(
                    "SemanticCore.search: lexical=enabled owner=%s:%s k_sparse=%s strategy=%s",
                    owner_type,
                    owner_id,
                    top_k_sparse,
                    fusion_strategy,
                )
                found = await self.store.lexical_search(
                    query_text=query_text,
                    tenant_id=tenant_id,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(top_k_sparse),
                )
                if found:
                    # Feature annotation for downstream ranking (rank-only, provider-agnostic).
                    for i, f in enumerate(found, start=1):
                        if isinstance(f, dict):
                            meta = f.get("meta") or {}
                            if not isinstance(meta, dict):
                                meta = {}
                            meta.setdefault("lexical_score", 1.0 / float(60 + i))
                            meta.setdefault("retrieval_method", "lexical")
                            f["meta"] = meta
                        else:
                            meta = getattr(f, "meta", None) or {}
                            if not isinstance(meta, dict):
                                meta = {}
                            meta.setdefault("lexical_score", 1.0 / float(60 + i))
                            meta.setdefault("retrieval_method", "lexical")
                            f.meta = meta  # type: ignore[attr-defined]
                    sparse_facts = list(found)
            except Exception:
                logger.exception("SemanticCore.search: lexical search failed owner=%s:%s", owner_type, owner_id)
                raise

        facts: list[Any] = (
            fuse_candidates(dense=dense_facts, sparse=sparse_facts, strategy=fusion_strategy)
            if sparse_facts
            else dense_facts
        )

        # Optional topic filtering (soft)
        if requested_topic:
            filtered = [f for f in facts if requested_topic in _fact_topics(f)]
            if filtered:
                facts = filtered

        if requested_predicate:
            facts = [f for f in facts if getattr(f, "predicate", "").upper() == requested_predicate]

        # Lexical fallback ONLY if vector yielded nothing.
        if (not facts) and query_text and hasattr(self.store, "lexical_search"):
            try:
                logger.debug("SemanticCore.search: lexical fallback owner=%s:%s", owner_type, owner_id)
                # Ownership-only lexical search (no subject param)
                found = await self.store.lexical_search(
                    query_text=query_text,
                    tenant_id=tenant_id,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(k),
                )
                if found:
                    facts = list(found)
            except Exception:
                logger.exception("SemanticCore.search: lexical fallback failed owner=%s:%s", owner_type, owner_id)
                raise

        return dedupe_by_id(facts)

    async def fetch_more_facts(
        self,
        predicate: str,
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        k: int,
        offset: int = 0,
    ) -> list[Fact]:
        """
        Fetch additional facts for an owner scope, filtered by predicate using deterministic paging.

        Ownership-only:
        - No subject parameter allowed.
        - Store must provide stable ordering for list_facts_for_owner (e.g., updated_at desc).
        """
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticCore.fetch_more_facts requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticCore.fetch_more_facts requires tenant_id, owner_type and owner_id")

        predicate_u = (predicate or "").upper()
        try:
            if hasattr(self.store, "list_facts_for_owner"):
                facts = await self.store.list_facts_for_owner(
                    tenant_id=tenant_id,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    limit=None,
                )
            else:
                facts = []
        except Exception:
            logger.exception(
                "SemanticCore.fetch_more_facts: list_facts_for_owner failed owner=%s:%s",
                owner_type,
                owner_id,
            )
            raise

        filtered: list[Any] = []
        for f in facts or []:
            pred_val = getattr(f, "predicate", None) if hasattr(f, "predicate") else (
                f.get("predicate") if isinstance(f, dict) else None
            )
            if pred_val and str(pred_val).upper() == predicate_u:
                filtered.append(f)

        offset_i = max(0, int(offset))
        k_i = max(0, int(k))
        windowed = filtered[offset_i : offset_i + k_i] if k_i else filtered[offset_i:]
        return dedupe_by_id(windowed)

    async def fetch_by_ids(
        self,
        ids: list[str],
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
    ) -> list[Fact]:
        """
        Fetch facts by IDs (authoritative payload) with ownership enforcement.
        """

        if not hasattr(self.store, "fetch_by_ids"):
            return []
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticCore.fetch_by_ids requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticCore.fetch_by_ids requires tenant_id, owner_type and owner_id")
        try:
            facts = await self.store.fetch_by_ids(
                ids=ids,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
            missing = max(0, len(ids or []) - len(facts or []))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "SemanticCore.fetch_by_ids: ids=%d returned=%d owner=%s:%s",
                    len(ids or []),
                    len(facts or []),
                    owner_type,
                    owner_id,
                )
            if missing:
                logger.warning(
                    "SemanticCore.fetch_by_ids: missing=%d owner=%s:%s",
                    missing,
                    owner_type,
                    owner_id,
                )
            # Defensive filter in case upstream store misbehaves.
            filtered = [
                f for f in (facts or [])
                if getattr(f, "tenant_id", None) == tenant_id
                and getattr(f, "owner_type", None) == owner_type
                and getattr(f, "owner_id", None) == owner_id
            ]
            if filtered and len(filtered) != len(facts or []):
                logger.warning(
                    "SemanticCore.fetch_by_ids: dropped %d cross-scope facts owner=%s:%s",
                    len(facts or []) - len(filtered),
                    owner_type,
                    owner_id,
                )
            return filtered
        except Exception:
            logger.exception("SemanticCore.fetch_by_ids failed")
            raise

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def _dedup_facts(self, facts: list[Fact]) -> list[Fact]:
        return dedupe_by_id(facts or [])


def _fact_topics(f: Any) -> list[str]:
    meta = getattr(f, "meta", None) or {}
    if isinstance(f, dict):
        meta = f.get("meta") or {}
    if not isinstance(meta, dict):
        return []
    topics = meta.get("topics") or []
    if isinstance(topics, list):
        return [str(t) for t in topics if t]
    if isinstance(topics, str):
        return [topics]
    return []
