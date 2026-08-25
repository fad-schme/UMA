"""
Shared runtime and bound request handle for UMA.

UMARuntime is the canonical internal execution surface for request-scoped
retrieval. UMAMemory remains the public-primary API and delegates to this
runtime internally.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from uma.common.results import ContextBundle, MemoryResult

from uma.common.compiled_memory import (
    build_compiled_memory_artifact,
    build_compiled_memory_index_entry,
    build_compiled_memory_log_event,
)
from uma.common.ownership import validate_explicit_owner
from uma.common.provenance import (
    build_provenance,
    collect_parent_artifact_ids,
    collect_direct_source_chunk_ids,
    collect_source_chunk_ids,
    collect_transitive_source_chunk_ids,
    provenance_for_artifact,
)
from uma.common.dedupe import dedupe_evidence_by_text
from uma.retrieve.gaps import assess_gaps, gap_thresholds
from uma.common.identity import normalize_user_id
from uma.common.types import RuntimeContext
from uma.common.storage_metadata import (
    EPISODIC_LANE,
    KB_LANES,
    PROCEDURAL_LANE,
    PROFILE_LANE,
    RAW_LANE,
    SEMANTIC_LANE,
    WIKI_LANE,
    shared_metadata_view,
)
from uma.retrieve.planner import build_retrieval_plan
from uma.retrieve.rlm.request import RetrievalRequest, RetrievalScope
from uma.memory.working_memory.core import session_scope_from_runtime_context

logger = logging.getLogger(__name__)


class UMARuntime:
    """Shared UMA infrastructure container.

    Request scope must never be stored on this object. Retrieval-only
    dependencies (stores, cores, the RLM controller, config) are passed in
    explicitly at construction — this runtime holds no reference back to a
    `UMAMemory` instance and is constructible standalone. `UMAMemory.runtime`
    is the sole place that builds one, snapshotting its own already-ready
    dependencies.
    """

    def __init__(
        self,
        *,
        config: Any = None,
        stores: Optional[Mapping[str, Any]] = None,
        llm: Any = None,
        retrieval_cfg: Any = None,
        chunk_core: Any = None,
        semantic_core: Any = None,
        episodic_core: Any = None,
        procedural_core: Any = None,
        working_memory: Any = None,
        rlm_controller: Any = None,
        memory_env: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = config
        self.stores: dict[str, Any] = dict(stores or {})
        self.llm = llm
        self.retrieval_cfg = retrieval_cfg
        self.chunk_core = chunk_core
        self.semantic_core = semantic_core
        self.episodic_core = episodic_core
        self.procedural_core = procedural_core
        self.working_memory = working_memory
        self.rlm_controller = rlm_controller
        self.memory_env = memory_env
        self.metadata: dict[str, Any] = dict(metadata or {})

    def ensure_retrieval_ready(self) -> None:
        """No-op: dependencies are guaranteed ready by `UMAMemory.runtime`
        before this object is ever constructed. Kept so retrieval call sites
        that call it need no changes."""
        return None

    # ------------------------------------------------------------------
    # CR3 — Retrieval audit log
    # ------------------------------------------------------------------
    #
    # The audit store is lazy: a single instance per runtime, created on
    # first use, gated on a config flag (default ON). Failures during
    # initialization or write are logged at WARNING and swallowed —
    # retrieval results are returned regardless. This is observability,
    # not a correctness dependency.

    def _retrieval_audit_enabled(self) -> bool:
        """Config gate. Default ON.

        Disable by setting `security.retrieval_audit_enabled: false` in
        the YAML config. Returns False if the config can't be read,
        which makes the audit log opt-in for any environment where the
        config surface is exotic.
        """
        try:
            sec = getattr(self.config, "security", None)
            if sec is None:
                return True  # default ON when no security section configured
            # support both dict-style and attribute access
            if hasattr(sec, "get"):
                return bool(sec.get("retrieval_audit_enabled", True))
            return bool(getattr(sec, "retrieval_audit_enabled", True))
        except Exception:
            logger.debug("retrieval audit: failed to read config flag; defaulting to enabled")
            return True

    def _retrieval_audit_db_path(self) -> Optional[str]:
        """Resolve the audit DB path.

        Order of precedence:
        1. `security.retrieval_audit_db_path` if set.
        2. `<storage.db_dir>/retrieval_audit.db` if storage.db_dir is set.
        3. `.uma/db/retrieval_audit.db` as the embedded-profile default.
        """
        try:
            sec = getattr(self.config, "security", None)
            if sec is not None:
                if hasattr(sec, "get"):
                    explicit = sec.get("retrieval_audit_db_path")
                else:
                    explicit = getattr(sec, "retrieval_audit_db_path", None)
                if isinstance(explicit, str) and explicit.strip():
                    return explicit.strip()
        except Exception:  # nosec B110 — failure means fall through to next config key
            pass

        # Fall back to storage.db_dir if available.
        try:
            storage = getattr(self.config, "storage", None)
            if storage is not None:
                if hasattr(storage, "get"):
                    db_dir = storage.get("db_dir")
                else:
                    db_dir = getattr(storage, "db_dir", None)
                if isinstance(db_dir, str) and db_dir.strip():
                    import os as _os
                    return _os.path.join(db_dir.strip(), "retrieval_audit.db")
        except Exception:  # nosec B110 — failure means fall through to hardcoded default
            pass

        # Embedded-profile default.
        return ".uma/db/retrieval_audit.db"

    def _get_retrieval_audit_store(self) -> Optional[Any]:
        """Lazy accessor. Returns None when audit is disabled or unavailable."""
        if not self._retrieval_audit_enabled():
            return None
        existing = getattr(self, "_retrieval_audit_store_cached", None)
        if existing is not None:
            return existing
        try:
            from uma.stores.retrieval_audit import RetrievalAuditStore
            path = self._retrieval_audit_db_path()
            if not path:
                return None
            store = RetrievalAuditStore(db_path=path)
            self._retrieval_audit_store_cached = store
            return store
        except Exception:
            logger.warning(
                "retrieval audit: failed to initialize store; subsequent retrievals will not be audited",
                exc_info=True,
            )
            # Cache the None so we don't retry initialization every call.
            self._retrieval_audit_store_cached = None
            return None

    async def _append_retrieval_audit_row(
        self,
        *,
        runtime_context: RuntimeContext,
        query_text: str,
        scan_severity: str,
        lane_filter: list[str],
        result: "ContextBundle",
    ) -> None:
        """Append one audit row for this retrieval call.

        Fail-soft: any exception logs at WARNING and is swallowed. The
        retrieval result has already been built; an audit-log failure
        does not invalidate it.

        The sqlite write is dispatched to a worker thread so the event
        loop is not blocked by the audit-log I/O.
        """
        try:
            store = self._get_retrieval_audit_store()
            if store is None:
                return

            # Import here to keep startup paths free of audit-only imports.
            from uma.retrieve.rlm.controller import _hash_and_preview
            from uma.stores.retrieval_audit import RetrievalAuditRow

            query_hash, query_preview = _hash_and_preview(query_text or "")
            result_count = len(result.chunks) + len(result.facts)
            active_lanes = result.debug.active_lanes or lane_filter or []

            row = RetrievalAuditRow(
                request_id=str(getattr(runtime_context, "request_id", None) or ""),
                tenant_id=str(getattr(runtime_context, "tenant_id", None) or ""),
                user_id=str(normalize_user_id(runtime_context.user_id)),
                agent_id=getattr(runtime_context, "agent_id", None),
                query_hash=query_hash,
                query_preview=query_preview,
                scan_severity=str(scan_severity or "none"),
                lanes=list(active_lanes),
                result_count=int(result_count),
                # refined_via_llm: refinement happens in render_context,
                # not retrieve_context — always False at this audit point.
                refined_via_llm=False,
                pruned_via_llm=result.debug.pruned_via_llm,
            )
            await asyncio.to_thread(store.append, row)
        except Exception:
            logger.warning(
                "retrieval audit: failed to write row (continuing)",
                exc_info=True,
            )

    def _available_retrieval_lanes(self) -> list[str]:
        """Advertise the retrieval lanes this runtime can execute today.

        `wiki` currently means the runtime can participate in the compiled-memory
        lane policy using the existing retrievable document/evidence stack. It is
        not a separate wiki engine guarantee here; planner/runtime still surface
        `wiki` explicitly so memory-path lane selection is honest about the
        compiled-memory-first intent.
        """
        stores = self.stores or {}
        ctx_cfg = getattr(self.retrieval_cfg, "context", None)
        lanes: list[str] = []
        if self.chunk_core is not None or stores.get("chunk") is not None:
            lanes.extend([RAW_LANE, WIKI_LANE])
        if self.semantic_core is not None or stores.get("semantic") is not None:
            lanes.extend([SEMANTIC_LANE, PROFILE_LANE])
        if (
            (self.episodic_core is not None or stores.get("episodic") is not None)
            and bool(getattr(ctx_cfg, "include_episodic", True))
        ):
            lanes.append(EPISODIC_LANE)
        if (
            (self.procedural_core is not None or stores.get("procedural") is not None)
            and bool(getattr(ctx_cfg, "include_procedural", True))
        ):
            lanes.append(PROCEDURAL_LANE)
        seen: set[str] = set()
        return [lane for lane in lanes if not (lane in seen or seen.add(lane))]

    @staticmethod
    def _build_retrieval_request(
        context: RuntimeContext,
        *,
        plan: Optional[Any] = None,
        query_scan_severity: Optional[str] = None,
        debug: bool = False,
    ) -> RetrievalRequest:
        """Convert a RuntimeContext into a RetrievalRequest."""
        return RetrievalRequest.from_runtime_context(
            context,
            trace_id=context.request_id,
            plan=plan,
            query_scan_severity=query_scan_severity,
            debug=debug,
        )

    @staticmethod
    def _working_memory_scope_set_context(context: RuntimeContext) -> Optional[Any]:
        """Build the working-memory scope for a runtime context."""
        return session_scope_from_runtime_context(context)

    @staticmethod
    def _normalize_lane_filter(lane_filter: Optional[list[str]]) -> list[str]:
        if lane_filter is None:
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for lane in lane_filter:
            candidate = str(lane or "").strip().lower()
            if not candidate:
                continue
            if candidate not in KB_LANES:
                raise ValueError(f"UMARuntime.retrieve_context: invalid lane_filter value {lane!r}")
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        return normalized

    @staticmethod
    def _item_lane(item: Any) -> Optional[str]:
        meta = getattr(item, "meta", None)
        if isinstance(item, dict):
            meta = item.get("meta", meta)
        if not isinstance(meta, dict):
            return None
        raw = meta.get("kb_lane")
        if raw is None:
            return None
        lane = str(raw).strip().lower()
        return lane or None

    @classmethod
    def _filter_items_by_lanes(cls, items: list[Any], lane_filter: list[str]) -> list[Any]:
        if not lane_filter:
            return list(items or [])
        allowed = set(lane_filter)
        return [item for item in (items or []) if cls._item_lane(item) in allowed]

    @staticmethod
    def _filter_items_by_scope(
        items: list[Any],
        allowed_scopes: tuple[RetrievalScope, ...],
        tenant_id: str,
    ) -> list[Any]:
        """Keep only items in the request's own tenant and retrieval scopes.

        Defense in depth at the public boundary. Every store read is already
        scoped, so anything dropped here is a plumbing defect upstream rather
        than routine filtering — hence the warning on each drop.

        The admissible set is the request's own ``tenant_id`` plus its own
        ``RetrievalScope`` tuple, so this cannot drift from what retrieval
        actually asked for: an item is admissible only if it carries the
        calling tenant AND belongs to the calling agent's shared KB or to the
        calling user. ``RetrievalScope`` carries no tenant of its own, which
        is why the tenant is passed in alongside it rather than folded into
        the scope set.

        It fails closed — an item whose tenant or owner cannot be read is
        dropped, because every domain object these lanes return carries a
        validated tenant and owner, so a missing one means the item did not
        come from a store.

        Applied to the four lanes whose items carry owner fields (facts,
        chunks, episodic, skills). Graph rows carry no top-level owner — they
        are scoped inside the Cypher predicate — and working memory is keyed
        per user by the buffer, so neither is filtered here.
        """
        allowed = {
            (
                scope.owner_type,
                normalize_user_id(scope.owner_id)
                if scope.owner_type == "user"
                else scope.owner_id,
            )
            for scope in allowed_scopes
        }
        out = []
        for item in items or []:
            item_tenant_id = _artifact_value(item, "tenant_id")
            owner_type = _artifact_value(item, "owner_type")
            owner_id = _artifact_value(item, "owner_id")
            if owner_type == "user" and owner_id:
                owner_id = normalize_user_id(str(owner_id))
            if item_tenant_id == tenant_id and (owner_type, owner_id) in allowed:
                out.append(item)
            else:
                logger.warning(
                    "UMARuntime: dropped out-of-scope item tenant_id=%s owner=%s:%s "
                    "allowed_tenant_id=%s allowed_owners=%s",
                    item_tenant_id,
                    owner_type,
                    owner_id,
                    tenant_id,
                    sorted(allowed),
                )
        return out

    def _load_working_memory_for_context(
        self,
        runtime_context: RuntimeContext,
    ) -> list[Any]:
        try:
            wm_scope = self._working_memory_scope_set_context(runtime_context)
            wm_core = self.working_memory
            return (
                wm_core.get_context(wm_scope)
                if wm_core is not None and wm_scope is not None
                else []
            )
        except Exception:
            logger.exception(
                "UMARuntime.retrieve_context: failed to load WM "
                "tenant=%s agent=%s session=%s",
                runtime_context.tenant_id,
                runtime_context.agent_id,
                runtime_context.session_id,
            )
            return []

    def _assemble_public_context_result(
        self,
        *,
        query_text: str,
        lane_filter: list[str],
        plan: Any,
        working_memory: list[Any],
        pack: Any,
        allowed_scopes: tuple[RetrievalScope, ...],
        tenant_id: str,
    ) -> "ContextBundle":
        from uma.common.results import ContextBundle, Confidence, DebugInfo, Provenance
        from uma.retrieve.rlm.coverage import compute_confidence

        coverage = getattr(pack, "coverage", None)
        active_lanes = list(plan.participating_lanes)
        episodic = self._filter_items_by_scope(
            self._filter_items_by_lanes(pack.episodes, active_lanes),
            allowed_scopes,
            tenant_id,
        )
        facts = self._filter_items_by_scope(
            self._filter_items_by_lanes(pack.facts or [], active_lanes),
            allowed_scopes,
            tenant_id,
        )
        chunks = self._filter_items_by_scope(
            self._filter_items_by_lanes(getattr(pack, "chunks", []), active_lanes),
            allowed_scopes,
            tenant_id,
        )
        skills = self._filter_items_by_scope(
            self._filter_items_by_lanes(pack.skills, active_lanes),
            allowed_scopes,
            tenant_id,
        )
        trace = list(getattr(pack, "steps", []) or [])
        confidence_data = compute_confidence(coverage) if coverage is not None else None

        provenance_dict = build_provenance(
            source_chunk_ids=[getattr(chunk, "id", None) for chunk in chunks],
            source_document_ids=[getattr(chunk, "doc_id", None) for chunk in chunks],
            derived_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            derivation_type="context_retrieval",
            retrieval_path=trace,
            support_density=(1.0 if chunks else 0.0),
            confidence=(confidence_data.get("score") if confidence_data else None),
            conflicts=[],
            require_source_chunks=True,
        )

        return ContextBundle(
            query=query_text,
            working_memory=working_memory,
            episodic=episodic,
            facts=facts,
            chunks=chunks,
            documents=self._group_memory_artifacts(chunks),
            skills=skills,
            graph=list(pack.graph or []),
            confidence=Confidence(**confidence_data) if confidence_data else None,
            provenance=Provenance(**provenance_dict),
            query_scan_severity="none",  # overwritten by the caller before return
            debug=DebugInfo(
                lane_filter=list(lane_filter),
                active_lanes=active_lanes,
                trace=trace,
                pruned_via_llm=bool(getattr(pack, "pruned_via_llm", False)),
            ),
        )


    async def retrieve_context(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
        lane_filter: Optional[list[str]] = None,
        include_debug: bool = False,
    ) -> "ContextBundle":
        """Retrieve curated evidence-oriented UMA context for one explicit request scope.

        Contract:
        - context retrieval is the canonical RAG/context path
        - a small lane planner decides which persisted lanes participate before RLM runs
        - chunks/documents remain the primary evidence product
        - wiki/compiled memory state is not required by default
        - `lane_filter` applies only to persisted retrieval lanes, not working memory
        - `include_debug` attaches a per-candidate `score_card` to each
          artifact's `meta` (vector/lexical/rerank/route/method/final/trust)
          so callers can see why an artifact ranked where it did

        Security:
        - The query text is scanned for injection patterns at the runtime
          boundary. The resulting severity is propagated to downstream LLM
          hops (snippet refiner, fact pruner) which skip amplification on
          "medium" / "high". High-severity queries are NOT blocked — the
          scan is advisory; the caller still gets retrieval results, just
          without LLM-refined snippets. This preserves the developer's
          right to handle false positives.
        """
        from uma.adapters.observability.metrics import increment, timed
        from uma.adapters.scanner.injection_scan import scan_content

        if not isinstance(runtime_context, RuntimeContext):
            raise TypeError("UMARuntime retrieval requires a RuntimeContext instance.")
        if not runtime_context.user_id:
            raise ValueError("UMARuntime retrieval requires RuntimeContext.user_id.")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(
                "UMARuntime.retrieve_context: query_text must be a non-empty string."
            )
        normalized_query_text = query_text.strip()

        # CR3: boundary scan on query_text. Severity flows through to the
        # controller via RetrievalRequest.query_scan_severity, and is
        # stamped on the returned public dict. Non-blocking; the caller
        # decides what to do with the signal.
        scan_result = scan_content(normalized_query_text)
        query_scan_severity = scan_result.severity
        if query_scan_severity != "none":
            logger.warning(
                "UMARuntime.retrieve_context: scan flagged query severity=%s rules=%s tenant=%s user=%s",
                query_scan_severity,
                scan_result.matched_rules,
                runtime_context.tenant_id,
                normalize_user_id(runtime_context.user_id),
            )

        self.ensure_retrieval_ready()
        normalized_lane_filter = self._normalize_lane_filter(lane_filter)
        plan = build_retrieval_plan(
            product="context",
            query_text=normalized_query_text,
            available_lanes=self._available_retrieval_lanes(),
            lane_filter=normalized_lane_filter,
        )

        with timed("uma.get_structured_context.latency"):
            working_memory = self._load_working_memory_for_context(runtime_context)
            controller = self.rlm_controller
            if controller is None:
                raise RuntimeError(
                    "UMARuntime.retrieve_context: RLM controller not initialized."
                )
            request = self._build_retrieval_request(
                runtime_context,
                plan=plan,
                query_scan_severity=query_scan_severity,
                debug=include_debug,
            )
            pack = await controller.retrieve_context(
                request=request,
                query_text=normalized_query_text,
            )
            increment("uma.get_structured_context.calls", tags={"path": "rlm"})
            result = self._assemble_public_context_result(
                query_text=normalized_query_text,
                lane_filter=normalized_lane_filter,
                plan=plan,
                working_memory=working_memory,
                pack=pack,
                allowed_scopes=request.scopes,
                tenant_id=request.context.tenant_id,
            )
            # Surface the scan severity on the bundle so callers and
            # downstream consumers can act on it. Always present as a
            # string ("none" when nothing matched) so consumers don't
            # have to disambiguate "scan ran" from "scan was skipped."
            result.query_scan_severity = query_scan_severity
            # CR3: write a structured audit log row for this retrieval.
            await self._append_retrieval_audit_row(
                runtime_context=runtime_context,
                query_text=normalized_query_text,
                scan_severity=query_scan_severity,
                lane_filter=normalized_lane_filter,
                result=result,
            )
            return result

    @staticmethod
    def _group_memory_artifacts(chunks: list[Any]) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for chunk in chunks or []:
            chunk_id = getattr(chunk, "id", None)
            doc_id = str(getattr(chunk, "doc_id", None) or chunk_id or "")
            if not doc_id:
                continue
            meta = dict(getattr(chunk, "meta", None) or {})
            artifact = grouped.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "kind": meta.get("kind"),
                    "kb_lane": meta.get("kb_lane"),
                    "owner_type": getattr(chunk, "owner_type", None),
                    "owner_id": getattr(chunk, "owner_id", None),
                    "source_path": getattr(chunk, "source_path", None),
                    "metadata": shared_metadata_view(
                        meta=meta,
                        owner_type=str(getattr(chunk, "owner_type", None) or ""),
                        owner_id=str(getattr(chunk, "owner_id", None) or ""),
                        created_at=getattr(chunk, "created_at", None),
                        updated_at=getattr(chunk, "updated_at", None),
                    ),
                    "page_ranges": [],
                    "chunk_ids": [],
                    "evidence": [],
                },
            )
            page_range = getattr(chunk, "page_range", None)
            if page_range and page_range not in artifact["page_ranges"]:
                artifact["page_ranges"].append(page_range)
            if chunk_id and chunk_id not in artifact["chunk_ids"]:
                artifact["chunk_ids"].append(chunk_id)
            artifact["evidence"].append(chunk)
        out = list(grouped.values())
        for artifact in out:
            metadata = dict(artifact.get("metadata") or {})
            artifact["provenance"] = build_provenance(
                existing=metadata.get("provenance"),
                source_chunk_ids=artifact.get("chunk_ids") or [],
                source_document_ids=[artifact.get("doc_id")],
                derived_at=metadata.get("updated_at") or metadata.get("created_at"),
                derivation_type="memory_source_group",
                support_density=(1.0 if artifact.get("chunk_ids") else 0.0),
                conflicts=metadata.get("provenance", {}).get("conflicts") if isinstance(metadata.get("provenance"), dict) else [],
                require_source_chunks=(metadata.get("kb_lane") == WIKI_LANE),
            )
            if metadata.get("kb_lane") == WIKI_LANE:
                page_title = metadata.get("page_title") or metadata.get("title") or artifact.get("doc_id")
                compiled_artifact = build_compiled_memory_artifact(
                    artifact_id=str(artifact.get("doc_id") or ""),
                    title=str(page_title or artifact.get("doc_id") or ""),
                    owner_type=str(artifact.get("owner_type") or ""),
                    owner_id=str(artifact.get("owner_id") or ""),
                    artifact_kind=str(artifact.get("kind") or "wiki_page"),
                    summary=metadata.get("summary"),
                    topic_key=metadata.get("page_slug") or page_title,
                    derived_at=artifact["provenance"].get("derived_at"),
                    derivation_type="wiki_artifact_created",
                    direct_source_chunk_ids=artifact.get("chunk_ids") or [],
                    direct_source_document_ids=[artifact.get("doc_id")] if artifact.get("doc_id") else [],
                    related_artifact_ids=metadata.get("related_artifact_ids") or [],
                    retrieval_tags=[
                        item
                        for item in [
                            metadata.get("page_slug"),
                            metadata.get("category"),
                            metadata.get("page_title"),
                        ]
                        if item
                    ],
                    retrieval_path=artifact["provenance"].get("retrieval_path") or [],
                    support_density=artifact["provenance"].get("support_density"),
                    confidence=artifact["provenance"].get("confidence"),
                    conflicts=artifact["provenance"].get("conflicts") if isinstance(artifact["provenance"].get("conflicts"), list) else [],
                    manual=bool(artifact["provenance"].get("manual")),
                    metadata=metadata,
                    event_type="wiki_artifact_created",
                )
                compiled_artifact["evidence"] = list(artifact.get("evidence") or [])
                compiled_artifact["page_ranges"] = list(artifact.get("page_ranges") or [])
                compiled_artifact["source_path"] = artifact.get("source_path")
                artifact.clear()
                artifact.update(compiled_artifact)
        return out

    def compile_memory_artifact(
        self,
        *,
        artifact_id: str,
        title: str,
        owner_type: str,
        owner_id: str,
        text: str | None = None,
        summary: str | None = None,
        topic_key: str | None = None,
        direct_source_chunk_ids: Optional[list[str]] = None,
        direct_source_document_ids: Optional[list[str]] = None,
        parent_artifacts: Optional[list[Any]] = None,
        related_artifact_ids: Optional[list[str]] = None,
        retrieval_tags: Optional[list[str]] = None,
        retrieval_path: Optional[list[Mapping[str, Any]]] = None,
        support_density: float | None = None,
        confidence: float | None = None,
        conflicts: Optional[list[Mapping[str, Any]]] = None,
        existing_artifact: Any | None = None,
        manual: bool = False,
        operation: str = "wiki_artifact_created",
        metadata: Mapping[str, Any] | None = None,
        status: str = "active",
    ) -> dict[str, Any]:
        owner = validate_explicit_owner(
            owner_type=owner_type,
            owner_id=owner_id,
        )
        return build_compiled_memory_artifact(
            artifact_id=artifact_id,
            title=title,
            owner_type=str(owner["owner_type"]),
            owner_id=str(owner["owner_id"]),
            artifact_kind="wiki_page",
            text=text,
            summary=summary,
            topic_key=topic_key,
            derivation_type=operation,
            direct_source_chunk_ids=direct_source_chunk_ids or [],
            direct_source_document_ids=direct_source_document_ids or [],
            parent_artifacts=parent_artifacts or [],
            related_artifact_ids=related_artifact_ids or [],
            retrieval_tags=retrieval_tags or [],
            retrieval_path=retrieval_path or [],
            support_density=support_density,
            confidence=confidence,
            conflicts=conflicts or [],
            status=status,
            metadata=metadata,
            existing_artifact=existing_artifact,
            manual=manual,
            event_type=operation,
        )

    async def expand_evidence(
        self,
        runtime_context: RuntimeContext,
        artifact: Any,
    ) -> dict[str, Any]:
        self.ensure_retrieval_ready()
        request = self._build_retrieval_request(runtime_context)
        env = self.memory_env
        if env is None:
            raise RuntimeError("UMARuntime.expand_evidence: memory environment is not initialized.")

        artifacts = artifact if isinstance(artifact, list) else [artifact]
        by_scope: dict[tuple[str, str], list[str]] = {}
        expanded_provenance: list[dict[str, Any]] = []
        lineage: list[dict[str, Any]] = []
        direct_chunk_ids: list[str] = []
        unresolved_parent_artifact_ids: list[str] = []

        for item in artifacts:
            provenance = provenance_for_artifact(item)
            chunk_ids = collect_transitive_source_chunk_ids(item)
            direct_ids = collect_direct_source_chunk_ids(item)
            parent_artifact_ids = collect_parent_artifact_ids(item)
            if not chunk_ids and parent_artifact_ids and not bool(provenance.get("manual")):
                for parent_artifact_id in parent_artifact_ids:
                    if parent_artifact_id not in unresolved_parent_artifact_ids:
                        unresolved_parent_artifact_ids.append(parent_artifact_id)
            owner_type = _artifact_value(item, "owner_type")
            owner_id = _artifact_value(item, "owner_id")
            for chunk_id in direct_ids:
                if chunk_id not in direct_chunk_ids:
                    direct_chunk_ids.append(chunk_id)
            if chunk_ids and owner_type and owner_id:
                by_scope.setdefault((str(owner_type), str(owner_id)), [])
                for chunk_id in chunk_ids:
                    if chunk_id not in by_scope[(str(owner_type), str(owner_id))]:
                        by_scope[(str(owner_type), str(owner_id))].append(chunk_id)
            normalized = build_provenance(
                existing=provenance,
                source_chunk_ids=chunk_ids,
                derivation_type=str(provenance.get("derivation_type") or _artifact_value(item, "kind") or "artifact"),
                derived_at=provenance.get("derived_at") or _artifact_value(item, "updated_at") or _artifact_value(item, "created_at"),
                support_density=provenance.get("support_density"),
                confidence=provenance.get("confidence"),
                conflicts=provenance.get("conflicts") if isinstance(provenance.get("conflicts"), list) else [],
                parent_artifact_ids=parent_artifact_ids,
                require_source_chunks=not bool(provenance.get("manual")),
            )
            if parent_artifact_ids and not chunk_ids and not bool(provenance.get("manual")):
                invalid_reasons = list(normalized.get("invalid_reasons") or [])
                if "unreachable_raw_source_chunks" not in invalid_reasons:
                    invalid_reasons.append("unreachable_raw_source_chunks")
                normalized["invalid_reasons"] = invalid_reasons
                normalized["valid"] = False
            expanded_provenance.append(normalized)
            lineage.append(
                {
                    "artifact_id": _artifact_value(item, "id") or _artifact_value(item, "doc_id"),
                    "kind": _artifact_value(item, "kind"),
                    "parent_artifact_ids": parent_artifact_ids,
                    "direct_source_chunk_ids": direct_ids,
                    "transitive_source_chunk_ids": chunk_ids,
                }
            )

        evidence: list[dict[str, Any]] = []
        missing_chunk_ids: list[str] = []
        for (owner_type, owner_id), ids in by_scope.items():
            chunks = await env.fetch_chunks(
                request,
                ids=ids,
                owner_type=owner_type,
                owner_id=owner_id,
            )
            found_ids: list[str] = []
            for chunk in chunks or []:
                payload = _chunk_payload(chunk)
                payload["retrieval_score"] = ((payload.get("meta") or {}).get("vector_score"))
                evidence.append(payload)
                if payload.get("id"):
                    found_ids.append(str(payload["id"]))
            for chunk_id in ids:
                if chunk_id not in found_ids and chunk_id not in missing_chunk_ids:
                    missing_chunk_ids.append(chunk_id)

        event = build_compiled_memory_log_event(
            event_type="evidence_expanded",
            artifact=artifacts[0] if len(artifacts) == 1 else None,
            source_chunk_ids=[item["id"] for item in evidence if item.get("id")],
            parent_artifact_ids=unresolved_parent_artifact_ids,
        )

        return {
            "artifacts": artifacts,
            "provenance": expanded_provenance,
            "evidence": evidence,
            "direct_chunk_ids": direct_chunk_ids,
            "transitive_chunk_ids": [item["id"] for item in evidence if item.get("id")],
            "chunk_ids": [item["id"] for item in evidence if item.get("id")],
            "missing_chunk_ids": missing_chunk_ids,
            "unresolved_parent_artifact_ids": unresolved_parent_artifact_ids,
            "lineage": lineage,
            "mode": (
                "manual_audit"
                if not evidence and expanded_provenance and all(bool(item.get("manual")) for item in expanded_provenance)
                else ("raw_chunk_evidence" if evidence else "invalid")
            ),
            "compiled_memory_log": [event],
        }

    async def retrieve_memory(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
        memory_intent: str = "continuity",
        include_debug: bool = False,
    ) -> "MemoryResult":
        """Retrieve compiled, evidence-backed memory results for one request scope.

        This path uses its own lane plan, may reuse context-path candidate
        gathering beneath that plan, and must not silently collapse into
        context retrieval. If compiled memory artifacts are unavailable, the
        result surfaces an explicit evidence-only fallback.
        """
        if not isinstance(memory_intent, str) or not memory_intent.strip():
            raise ValueError("UMARuntime.retrieve_memory: memory_intent must be a non-empty string.")
        self.ensure_retrieval_ready()
        plan = build_retrieval_plan(
            product="memory",
            query_text=query_text.strip(),
            available_lanes=self._available_retrieval_lanes(),
            memory_intent=memory_intent.strip(),
        )
        context = await self.retrieve_context(
            runtime_context,
            query_text=query_text,
            lane_filter=list(plan.participating_lanes),
            include_debug=include_debug,
        )
        chunks = list(context.chunks)
        memory_sources = self._group_memory_artifacts(chunks)
        support_density = 1.0 if chunks else 0.0
        confidence_score = context.confidence.score if context.confidence is not None else None
        confidence_dict = context.confidence.model_dump() if context.confidence is not None else {}
        trace = list(context.debug.trace)
        compiled_conflicts = [
            dict(conflict)
            for artifact in memory_sources
            for conflict in (artifact.get("conflicts") or [])
            if isinstance(conflict, Mapping)
        ]
        supporting_evidence = [_chunk_payload(chunk) for chunk in chunks]
        fallback_used = not chunks
        fallback_reason = "no_compiled_memory_available" if fallback_used else None
        compiled_answer = None
        if not fallback_used:
            compiled_answer = build_compiled_memory_artifact(
                artifact_id=f"memory:{runtime_context.request_id}:{memory_intent.strip()}",
                title=f"Memory {memory_intent.strip()}",
                owner_type="user",
                owner_id=normalize_user_id(runtime_context.user_id),
                artifact_kind="compiled_memory_answer",
                text=None,
                summary=None,
                topic_key=memory_intent.strip(),
                derived_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                derivation_type="memory_compiled",
                direct_source_chunk_ids=[item["id"] for item in supporting_evidence if item.get("id")],
                direct_source_document_ids=[item.get("doc_id") for item in supporting_evidence if item.get("doc_id")],
                parent_artifacts=[artifact for artifact in memory_sources if artifact.get("artifact_type") == "compiled_memory_artifact"],
                parent_artifact_ids=[artifact.get("id") or artifact.get("doc_id") for artifact in memory_sources if artifact.get("id") or artifact.get("doc_id")],
                related_artifact_ids=[artifact.get("id") or artifact.get("doc_id") for artifact in memory_sources if artifact.get("id") or artifact.get("doc_id")],
                retrieval_tags=[memory_intent.strip()],
                retrieval_path=trace,
                support_density=support_density,
                confidence=confidence_score,
                conflicts=compiled_conflicts,
                manual=False,
                metadata={"memory_intent": memory_intent.strip()},
                event_type="memory_compiled",
            )
            compiled_answer["memory_intent"] = memory_intent.strip()
            compiled_answer["supporting_evidence"] = supporting_evidence
        memories: list[dict[str, Any]] = [compiled_answer] if compiled_answer is not None else []
        if fallback_used:
            logger.info(
                "UMARuntime.retrieve_memory: explicit evidence-only fallback tenant=%s agent=%s user=%s intent=%s chunks=%d",
                runtime_context.tenant_id,
                runtime_context.agent_id,
                runtime_context.user_id,
                memory_intent,
                len(chunks),
            )
        # Gap analysis runs on the raw domain objects, not the serialized
        # projections: `_chunk_payload` drops `created_at` and `trust_score`,
        # which are exactly the two signals this needs.
        max_support_age_days, min_support_trust = gap_thresholds(self.retrieval_cfg)
        gaps = assess_gaps(
            facts=context.facts,
            chunks=chunks,
            max_support_age_days=max_support_age_days,
            min_support_trust=min_support_trust,
        )
        if gaps:
            logger.info(
                "UMARuntime.retrieve_memory: %d gap(s) flagged tenant=%s user=%s intent=%s",
                len(gaps),
                runtime_context.tenant_id,
                runtime_context.user_id,
                memory_intent,
            )

        semantic_retrieved_event = build_compiled_memory_log_event(
            event_type="semantic_retrieved",
            artifact=compiled_answer,
            source_chunk_ids=[item["id"] for item in supporting_evidence if item.get("id")],
            retrieval_path=trace,
        )
        compiled_memory_index = [
            build_compiled_memory_index_entry(artifact)
            for artifact in [*memory_sources, *([compiled_answer] if compiled_answer is not None else [])]
            if artifact.get("artifact_type") == "compiled_memory_artifact"
        ]
        compiled_memory_log = [semantic_retrieved_event]
        for artifact in [*memory_sources, *([compiled_answer] if compiled_answer is not None else [])]:
            compiled_memory_log.extend(list(artifact.get("compiled_memory_log") or []))
        detailed_result = {
            "query": query_text.strip(),
            "product": "memory",
            "memory_intent": memory_intent.strip(),
            "memories": memories,
            "compiled_answer": compiled_answer,
            "evidence": chunks,
            "supporting_evidence": supporting_evidence,
            "supporting_facts": list(context.facts),
            "gaps": gaps,
            "supporting_skills": list(context.skills),
            "conflicts": [],
            "support_density": support_density,
            "fallback": {
                "used": fallback_used,
                "mode": "evidence_only" if fallback_used else "none",
                "reason": fallback_reason,
                "message": (
                    "No compiled memory artifacts were available; returning evidence-only fallback."
                    if fallback_used
                    else ""
                ),
            },
            "memory_sources": memory_sources,
            "compiled_memory_index": compiled_memory_index,
            "compiled_memory_log": compiled_memory_log,
            "confidence": confidence_dict,
            "provenance": (
                compiled_answer["provenance"]
                if compiled_answer is not None
                else context.provenance.model_dump()
            ),
            "trace": trace
            + [
                {
                    "event": "memory_plan",
                    "product": "memory",
                    "memory_intent": memory_intent.strip(),
                    "requested_lanes": list(plan.requested_lanes),
                    "participating_lanes": list(plan.participating_lanes),
                    "excluded_lanes": [dict(item) for item in plan.excluded_lanes],
                    "fallback_used": fallback_used,
                }
            ],
        }
        return self._build_public_memory_result(
            detailed_result,
            include_debug=include_debug,
        )

    @staticmethod
    def _serialize_memory_fact(fact: Any) -> dict[str, Any]:
        provenance = provenance_for_artifact(fact)
        text = ""
        if isinstance(fact, Mapping):
            subject = str(fact.get("subject") or "").strip()
            predicate = str(fact.get("predicate") or "").strip()
            object_text = str(fact.get("object") or "").strip()
            confidence = fact.get("confidence")
            salience = fact.get("salience")
        else:
            subject = str(getattr(fact, "subject", "") or "").strip()
            predicate = str(getattr(fact, "predicate", "") or "").strip()
            object_text = str(getattr(fact, "object", "") or "").strip()
            confidence = getattr(fact, "confidence", None)
            salience = getattr(fact, "salience", None)
        text = " ".join(part for part in (subject, predicate, object_text) if part).strip()
        return {
            "text": text,
            "confidence": confidence,
            "salience": salience,
            "source_chunk_ids": collect_source_chunk_ids(fact) or list(provenance.get("source_chunk_ids") or []),
        }

    @staticmethod
    def _serialize_memory_evidence(chunk: Any) -> dict[str, Any]:
        if isinstance(chunk, Mapping):
            payload = dict(chunk)
        else:
            payload = _chunk_payload(chunk)

        source_path = payload.get("source_path")
        source_name = payload.get("source")
        if not source_name and source_path:
            source_name = str(source_path).rsplit("/", 1)[-1]

        return {
            "id": payload.get("id"),
            "text": payload.get("text"),
            "source": source_name,
            "source_document_id": payload.get("doc_id") or payload.get("source_document_id"),
        }

    def _build_public_memory_result(
        self,
        detailed_result: Mapping[str, Any],
        *,
        include_debug: bool,
    ) -> "MemoryResult":
        from uma.common.results import CompiledMemory, MemoryResult

        facts = [
            self._serialize_memory_fact(fact)
            for fact in list(detailed_result.get("supporting_facts") or [])
        ]
        evidence = dedupe_evidence_by_text([
            item
            for item in (
                self._serialize_memory_evidence(chunk)
                for chunk in list(detailed_result.get("supporting_evidence") or [])
            )
            if any(item.get(key) for key in ("id", "text", "source", "source_document_id"))
        ])
        provenance = dict(detailed_result.get("provenance") or {})
        ca = detailed_result.get("compiled_answer")
        compiled_memory = (
            CompiledMemory(
                memory_intent=ca.get("memory_intent"),
                provenance_valid=bool((ca.get("provenance") or {}).get("valid", True)),
            )
            if ca is not None
            else None
        )
        invalid_reasons = list(provenance.get("invalid_reasons") or [])
        provenance_error = str(invalid_reasons[0]) if invalid_reasons else None
        debug = dict(detailed_result) if include_debug else None

        return MemoryResult(
            query=detailed_result.get("query") or "",
            compiled_memory=compiled_memory,
            facts=facts,
            evidence=evidence,
            provenance_valid=bool(provenance.get("valid")),
            provenance_error=provenance_error,
            gaps=[dict(item) for item in (detailed_result.get("gaps") or []) if isinstance(item, Mapping)],
            debug=debug,
        )


def _artifact_value(artifact: Any, field_name: str) -> Any:
    if isinstance(artifact, dict):
        return artifact.get(field_name)
    return getattr(artifact, field_name, None)


def _chunk_payload(chunk: Any) -> dict[str, Any]:
    return {
        "id": getattr(chunk, "id", None),
        "doc_id": getattr(chunk, "doc_id", None),
        "text": getattr(chunk, "text", None),
        "page_range": getattr(chunk, "page_range", None),
        "position": getattr(chunk, "position", None),
        "owner_type": getattr(chunk, "owner_type", None),
        "owner_id": getattr(chunk, "owner_id", None),
        "source_path": getattr(chunk, "source_path", None),
        "meta": dict(getattr(chunk, "meta", None) or {}),
        "provenance": provenance_for_artifact(chunk),
    }
