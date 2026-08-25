r"""
  _   _ __  __   _      ___ _    __  __
 | | | |  \/  | /_\ ___| _ \ |  |  \/  |
 | |_| | |\/| |/ _ \___|   / |__| |\/| |
  \___/|_|  |_/_/ \_\  |_|_\____|_|  |_|

    UMAMemory — Core UMA Memory Runtime
    =====================================

UMA is a **memory SDK**, not an autonomous agent.

This class owns and orchestrates all UMA memory subsystems:

    • Working Memory (short-term conversational state)
    • Episodic Memory (indexed event history)
    • Fact Memory (facts, preferences, domain knowledge)
    • Skill Memory (skills, routines)
    • Temporal Graph (optional knowledge graph)

UMA does not generate assistant replies and does not perform agent reasoning.
Developers bring their own LLM or agent loop and use UMA strictly for memory
management.

Typical developer workflow
--------------------------

1. Initialize UMA:

    memory = UMAMemory.from_yaml("uma.yaml")

2. Retrieve context with explicit per-call scope. UMA is single-tenant,
   multi-agent and multi-user: `agent_id` and `user_id` identify the caller
   on every call, and `tenant_id` falls back to the single-tenant default
   when the caller does not set it.

    context = await memory.retrieve_context(
        query_text="the user's current question or task",
        agent_id=agent_id,
        user_id=user_id,
        request_id="req-1",
        session_id="session-1",
    )

3. Ingest conversation turns or documents through the public APIs on UMAMemory.

One `UMAMemory` instance serves every agent and every user. Identity is never
held on the instance — passing it per call is what makes concurrent agents
safe on a shared runtime.

Design Philosophy
-----------------
- UMA provides *memory operations*, not agent behaviors.
- All reasoning, tool use, and final reply generation happen outside UMA.
- UMA focuses on correctness, retrieval quality, summarization, and
  long-term memory structure.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Union

if TYPE_CHECKING:
    from uma.common.results import (
        ContextBundle,
        DerivedRebuildReport,
        HealthStatus,
        MemoryResult,
        VectorRebuildReport,
    )
    from uma.ingest.types import IngestReport

from uma.common.config import UMAConfig
from uma.common.config_types import RuntimeConfig, SecretsProviderConfig, parse_plugin_spec
from uma.common.hooks import UMAHooks
from uma.common.identity import normalize_user_id
from uma.common.types.types_scope import DEFAULT_TENANT_ID
from uma.adapters.secrets import SecretsProvider
from uma.memory.working_memory.core import WorkingMemoryCore
from uma.memory.episodic.core import EpisodicCore
from uma.memory.episodic.indexer import EpisodeIndexer
from uma.memory.episodic.policies import EpisodicRetentionPolicy
from uma.memory.semantic.core import SemanticCore
from uma.memory.procedural.core import ProceduralCore
from uma.memory.chunk.core import ChunkCore
from uma.memory.graph import GraphCore
from uma.common.initializers.runtime import (
    init_retrieval_ready,
    init_ingestion_ready,
    schedule_ingestion_warmup,
)
from uma.common.registry import FeatureLoader, FeaturePolicy, default_feature_registry
from .runtime import UMARuntime
from uma.common.types import AgentProfile, RuntimeContext
from uma.common.types.types_scope import validate_agent_id, validate_tenant_id
from uma.adapters.scanner.injection_scan import scan_content, InjectionDetectedError

logger = logging.getLogger(__name__)


# M6 — Rate-limit hook (SDK-level)
# ---------------------------------------------------------------------------
# Operators may register a single hook to throttle expensive UMA operations.
# The hook is invoked at the top of each public async entry point and is
# expected to RAISE to refuse the call. Returning normally allows the call.
#
# Hook signature (sync OR async):
#     def rate_limit_hook(operation: str, ctx: Optional[RuntimeContext]) -> None
#     async def rate_limit_hook(operation: str, ctx: Optional[RuntimeContext]) -> None
#
# Operation strings (free-form, match the public method name):
#     "retrieve_context"
#     "retrieve_memory"
#     "process_turn"
#     "ingest_document"
#
# `ctx` is the RuntimeContext built from the call's user_id / tenant_id /
# session_id when available. For `ingest_document`, no RuntimeContext is
# constructed (the ingest API takes owner_type/owner_id, not a user/session
# scope), so `ctx` is None — the hook may still throttle by operation name
# alone or fall back to allowing the call.
#
# UMA does NOT provide a default rate limiter. This is intentional: UMA
# doesn't know your deployment topology, your tenancy model, or your
# rate-limit storage backend. Plug in whatever you already use.
RateLimitHook = Union[
    Callable[[str, Optional["RuntimeContext"]], None],
    Callable[[str, Optional["RuntimeContext"]], Awaitable[None]],
]


class UMAMemory:
    """UMA Memory Runtime Container.

    This class:
        - initializes all core subsystems lazily
        - provides public APIs for retrieval, ingestion, and maintenance
        - loads optional features via direct attachment


    UMARuntime remains internal and canonical for retrieval execution.
    """

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "UMAMemory":
        """Create a UMAMemory instance from a YAML config file.

        Parameters
        ----------
        path : str
            Filesystem path to a UMA config YAML. May be absolute or
            relative to the process working directory.

        Storage-path resolution
        -----------------------
        By default, relative `storage.db_root` values are resolved
        relative to the CONFIG FILE'S directory (``db_root_base:
        "config"``). This keeps the database in a stable location no
        matter which directory the process is launched from.

        .. warning::
           If your YAML sets ``storage.db_root_base: "cwd"``, relative
           db_root values are resolved from the process working
           directory instead. Launching the process from a different
           directory will silently create a fresh empty database — the
           existing memories will appear "lost" until you return to the
           original working directory. Prefer absolute paths or
           ``db_root_base: "config"`` for production deployments.
        """
        cfg = UMAConfig.load_yaml(path)
        mem = cls(cfg, config_path=path)

        # Predictable startup cost: make retrieval instant thereafter.
        init_retrieval_ready(mem)
        mem._retrieval_ready = True
        mem.initialized = True

        # Background warmup: ingestion subsystems and optional memory features.
        schedule_ingestion_warmup(mem)

        return mem

    def __init__(self, config: UMAConfig, config_path: Optional[str] = None) -> None:
        """Initialize the UMA memory container from validated config."""
        self.raw_config = config
        self._config_path = config_path or getattr(config, "_source_path", None)
        self._config_dir = getattr(config, "_source_dir", None)
        self.initialized: bool = False
        self._features_initialized: bool = False
        self._base_ready: bool = False
        self._retrieval_ready: bool = False
        self._ingestion_ready: bool = False
        self._warmup_scheduled: bool = False
        self._lifecycle_lock = threading.RLock()
        self._init_condition = threading.Condition(self._lifecycle_lock)
        self._init_inflight: set[str] = set()

        # RLM components (initialized later)
        self.memory_env = None
        self._rlm_controller = None

        # Unified runtime config
        self.cfg = RuntimeConfig.from_uma_config(config)
        from uma.adapters.scanner.injection_scan import configure_security
        configure_security(self.cfg.security)
        self.llm_cfg = self.cfg.llm
        self.agent_llm_cfg = self.cfg.agent_llm
        self.embedding_cfg = self.cfg.embedding
        self.working_memory_cfg = self.cfg.working_memory
        self.retrieval_cfg = self.cfg.retrieval
        self.features_cfg = self.cfg.features
        self.consolidation_cfg = self.cfg.consolidation
        self.semantic_salience_threshold = self.cfg.semantic_salience_threshold
        self._secrets_cfg = self.cfg.secrets
        self._secrets_provider: Optional[SecretsProvider] = self._build_secrets_provider(self._secrets_cfg)
        self._runtime: Optional[UMARuntime] = None

        # Internal store registry (core-only access; no direct store usage outside cores)
        self._stores: dict[str, Any] = {}

        # Hooks + Feature attachment registry
        self.hooks = UMAHooks()
        self.features: dict[str, Any] = {}
        self._feature_policy = FeaturePolicy()

        # M6: optional SDK-level rate-limit hook. See module-level
        # `RateLimitHook` documentation. Default None = no throttling.
        # Set via `set_rate_limit_hook(...)`.
        self._rate_limit_hook: Optional[RateLimitHook] = None

        # Core runtime components (initialized later)
        self.llm: Any = None
        self.agent_llm: Any = None
        self.embedder: Any = None
        self.pipeline: Optional[Any] = None
        self.document_store: Optional[Any] = None

        self.working_memory: Optional[WorkingMemoryCore] = None
        self.semantic_core: Optional[SemanticCore] = None
        self.episodic_core: Optional[EpisodicCore] = None
        self.graph_core: Optional[GraphCore] = None
        self.procedural_core: Optional[ProceduralCore] = None
        self.chunk_core: Optional[ChunkCore] = None

        logger.debug("UMAMemory instance created with unified storage config.")

    # ----------------------------------------------------------------------
    # Internal lazy initialization
    # ----------------------------------------------------------------------

    @property
    def runtime(self) -> UMARuntime:
        """Return the shared internal runtime view over this memory instance.

        Ensures retrieval readiness before snapshotting dependencies into the
        runtime, since `UMARuntime` holds them as plain attributes rather than
        reading them live off this instance.
        """
        if self._runtime is None:
            with self._lifecycle_lock:
                if self._runtime is None:
                    self._ensure_retrieval_ready()
                    self._runtime = UMARuntime(
                        config=self.cfg,
                        stores=self._stores,
                        llm=self.llm,
                        retrieval_cfg=self.retrieval_cfg,
                        chunk_core=self.chunk_core,
                        semantic_core=self.semantic_core,
                        episodic_core=self.episodic_core,
                        procedural_core=self.procedural_core,
                        working_memory=self.working_memory,
                        rlm_controller=self._rlm_controller,
                        memory_env=self.memory_env,
                        metadata={"source": "UMAMemory"},
                    )
        return self._runtime

    def _build_secrets_provider(
        self,
        secrets_cfg: Optional[SecretsProviderConfig],
    ) -> Optional[SecretsProvider]:
        if secrets_cfg is None:
            return None

        provider_path = str(secrets_cfg.provider).strip()
        module_path, _, attr = provider_path.replace(":", ".").rpartition(".")
        if not module_path or not attr:
            raise ValueError(
                "Invalid config at 'secrets.provider': expected an import path like "
                "'uma.adapters.secrets.EnvVarProvider'."
            )

        try:
            module = importlib.import_module(module_path)
            provider_cls = getattr(module, attr)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resolve config at 'secrets.provider': {provider_path!r}"
            ) from exc

        if not inspect.isclass(provider_cls):
            raise TypeError(
                f"Invalid config at 'secrets.provider': {provider_path!r} is not a class."
            )
        if not issubclass(provider_cls, SecretsProvider):
            raise TypeError(
                "Invalid config at 'secrets.provider': "
                f"{provider_path!r} must subclass SecretsProvider."
            )

        try:
            provider = provider_cls(**dict(secrets_cfg.options))
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize config at 'secrets.options' for provider "
                f"{provider_path!r}."
            ) from exc

        return provider

    def _resolve_runtime_context(
        self,
        *,
        agent_id: Optional[str],
        user_id: Optional[str],
        tenant_id: Optional[str] = DEFAULT_TENANT_ID,
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> RuntimeContext:
        """Resolve request scope for public APIs from explicit per-call values only.

        UMA is single-tenant, multi-agent and multi-user. ``agent_id`` and
        ``user_id`` are always the caller's to supply — no instance-level
        fallback exists, because one runtime serves every agent concurrently.
        ``tenant_id`` falls back to ``DEFAULT_TENANT_ID`` when unset.

        ``RuntimeContext`` validates every field it is given; this only fills
        in the two values UMA derives rather than requires.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("UMAMemory requires an explicit user_id.")

        return RuntimeContext(
            tenant_id=tenant_id or DEFAULT_TENANT_ID,
            agent_id=agent_id,
            request_id=request_id or f"request:{user_id.strip()}",
            user_id=user_id.strip(),
            workspace_id=workspace_id,
            session_id=session_id,
        )

    def _ensure_retrieval_ready(self) -> None:
        """Ensure retrieval-only dependencies are ready."""
        if not getattr(self, "_retrieval_ready", False):
            init_retrieval_ready(self)
        self.initialized = True

    def _ensure_ingestion_ready(self) -> None:
        """Ensure ingestion dependencies are ready."""
        if not getattr(self, "_ingestion_ready", False):
            init_ingestion_ready(self)
        self.initialized = True

    # ----------------------------------------------------------------------
    # M6 — Rate-limit hook
    # ----------------------------------------------------------------------

    def set_rate_limit_hook(self, hook: Optional[RateLimitHook]) -> "UMAMemory":
        """Register a single hook for SDK-level throttling of expensive ops.

        See module-level `RateLimitHook` documentation for the full contract.

        Pass `None` to clear an existing hook. Replacing a previously-set
        hook is allowed.

        Returns self for fluent-style chaining: `mem.set_rate_limit_hook(...)`.
        """
        if hook is not None and not callable(hook):
            raise TypeError(
                "UMAMemory.set_rate_limit_hook: hook must be callable or None; "
                f"got {type(hook).__name__}"
            )
        self._rate_limit_hook = hook
        if hook is None:
            logger.debug("UMAMemory: rate_limit_hook cleared")
        else:
            logger.debug(
                "UMAMemory: rate_limit_hook registered async=%s",
                inspect.iscoroutinefunction(hook),
            )
        return self

    async def _invoke_rate_limit_hook(
        self,
        operation: str,
        ctx: Optional[RuntimeContext],
    ) -> None:
        """Invoke the registered rate-limit hook, if any.

        Both sync and async hooks are supported. If the hook raises, the
        exception propagates to the caller and the operation is refused.
        If no hook is registered, returns silently.

        Operation strings are free-form; UMA passes the public method name
        ("retrieve_context", "retrieve_memory", "process_turn",
        "ingest_document").
        """
        hook = self._rate_limit_hook
        if hook is None:
            return
        result = hook(operation, ctx)
        # Async hook: coroutine returned, await it. Sync hook: result is None
        # (or whatever the hook returned, which we ignore).
        if inspect.iscoroutine(result):
            await result

    # ----------------------------------------------------------------------
    # -------------------- Core Public APIs -------------------------------
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Agent Profile (memory-promotion feature)
    # ----------------------------------------------------------------------

    async def set_agent_profile(
        self,
        *,
        agent_id: str,
        description: str,
        focus_areas: list[str],
        tenant_id: Optional[str] = DEFAULT_TENANT_ID,
    ) -> AgentProfile:
        """Upsert the profile for ``agent_id``.

        The profile is consulted by ``PromotionPolicy.qualifies_for_agent_kb``
        to decide which user-owned facts qualify for elevation into the
        agent's global KB during ``process_turn``. Without a profile, automatic
        promotion is a no-op; there is no eligibility-only fallback.

        Computes the profile embedding from ``description`` via the configured
        embedder. Idempotent — a second call for the same agent overwrites.
        The row is persisted in the procedural store with
        ``kind='agent_profile'`` and never enters the vector index.
        """
        resolved_agent_id = validate_agent_id(agent_id)
        resolved_tenant_id = validate_tenant_id(tenant_id or DEFAULT_TENANT_ID)
        if self.procedural_core is None:
            raise RuntimeError(
                "UMAMemory.set_agent_profile: procedural_core is not initialized."
            )
        if self.embedder is None:
            raise RuntimeError(
                "UMAMemory.set_agent_profile: embedder is not initialized."
            )
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(focus_areas, list):
            raise ValueError("focus_areas must be a list of strings")

        embeddings = await self.embedder.embed([description])
        if not embeddings or not embeddings[0]:
            raise RuntimeError(
                "UMAMemory.set_agent_profile: embedder returned no vector for description."
            )
        embedding = list(embeddings[0])

        profile = await self.procedural_core.upsert_agent_profile(
            agent_id=resolved_agent_id,
            description=description,
            focus_areas=list(focus_areas),
            embedding=embedding,
            tenant_id=resolved_tenant_id,
        )
        logger.info(
            "UMAMemory.set_agent_profile: upserted tenant_id=%s agent_id=%s focus_areas=%d",
            resolved_tenant_id,
            resolved_agent_id,
            len(profile.focus_areas),
        )
        return profile

    async def get_agent_profile(
        self,
        *,
        agent_id: str,
        tenant_id: Optional[str] = DEFAULT_TENANT_ID,
    ) -> Optional[AgentProfile]:
        """Return the profile for ``agent_id``, or None if unset."""
        resolved_agent_id = validate_agent_id(agent_id)
        resolved_tenant_id = validate_tenant_id(tenant_id or DEFAULT_TENANT_ID)
        if self.procedural_core is None:
            return None
        return await self.procedural_core.get_agent_profile(
            agent_id=resolved_agent_id,
            tenant_id=resolved_tenant_id,
        )

    # ----------------------------------------------------------------------
    # Core API: Retrieval
    # ----------------------------------------------------------------------
    async def retrieve_context(
        self,
        *,
        query_text: str,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = DEFAULT_TENANT_ID,
        lane_filter: Optional[list[str]] = None,
        include_debug: bool = False,
    ) -> "ContextBundle":
        """Return the curated LLM context bundle for the explicit request scope.

        Contract:
        - intended for LLM context assembly, not durable memory projection
        - returns a `ContextBundle` — attribute access, not dict access
        - chunks/documents remain the primary retrieval product
        - optional `lane_filter` narrows persisted retrieval lanes without requiring wiki state
        - observability lives on `bundle.debug` (lane plan, trace, pruned_via_llm)
        - `include_debug=True` additionally attaches a per-candidate
          `score_card` to each artifact's `meta`, exposing the score
          attribution (vector, lexical, rerank, route, method, final, trust)
          behind its ranking. Off by default — it is diagnostic detail, not
          part of the normal retrieval product.
        """
        runtime_context = self._resolve_runtime_context(
            agent_id=agent_id,
            user_id=user_id,
            tenant_id=tenant_id,
            request_id=request_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        await self._invoke_rate_limit_hook("retrieve_context", runtime_context)
        return await self.runtime.retrieve_context(
            runtime_context,
            query_text=query_text,
            lane_filter=lane_filter,
            include_debug=include_debug,
        )

    async def retrieve_memory(
        self,
        *,
        query_text: str,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = DEFAULT_TENANT_ID,
        memory_intent: str = "continuity",
        include_debug: bool = False,
    ) -> "MemoryResult":
        """Return compiled, evidence-backed memory results for the explicit request scope.

        Contract:
        - returns a `MemoryResult` — attribute access, not dict access
        - `compiled_memory` is the primary compiled-memory field; `None` on
          the explicit evidence-only fallback path
        - `facts` and `evidence` are serialized projections (dicts, not
          `Fact` / `Chunk` domain types) — the memory API is a narrower
          projection than `retrieve_context`
        - `provenance_valid` / `provenance_error` surface provenance status
          without exposing the full provenance sub-tree
        - full retrieval detail available on `result.debug` only when
          `include_debug=True`
        """
        runtime_context = self._resolve_runtime_context(
            agent_id=agent_id,
            user_id=user_id,
            tenant_id=tenant_id,
            request_id=request_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        await self._invoke_rate_limit_hook("retrieve_memory", runtime_context)
        return await self.runtime.retrieve_memory(
            runtime_context,
            query_text=query_text,
            memory_intent=memory_intent,
            include_debug=include_debug,
        )

    # ----------------------------------------------------------------------
    # Core API: Data Ingestion
    # ----------------------------------------------------------------------

    def scan_user_input(self, text: str) -> dict:
        """Scan user input for injection patterns before forwarding to an LLM.

        Call this at the top of your agent loop, before retrieve_context and
        before any LLM call. On a high-severity result, do not forward the
        message to the LLM and do not call process_turn.

        Returns a dict with keys: severity, matched_rules, score.
        Raises nothing — the caller decides what to do with the result.
        """
        result = scan_content(text or "")
        return {"severity": result.severity, "matched_rules": result.matched_rules, "score": result.score}

    async def process_turn(
        self,
        *,
        agent_id: str,
        user_id: str,
        user_msg: str,
        assistant_reply: str,
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        extra_meta: Optional[dict[str, Any]] = None,
        skip_scan: bool = False,
        tenant_id: Optional[str] = DEFAULT_TENANT_ID,
    ) -> None:
        """Public turn-ingest entrypoint.

        This is the supported memory-surface wrapper over the canonical
        pipeline turn processor. Session scope is explicit at this boundary;
        `extra_meta` remains optional non-session metadata.

        Raises InjectionDetectedError if a high-severity injection pattern is
        detected in user_msg. Pass skip_scan=True only when the caller has
        already validated the input and explicitly accepts responsibility.
        """
        self._ensure_ingestion_ready()

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("UMAMemory.process_turn requires a non-empty session_id.")

        # M6: rate-limit hook fires before the injection scan and before any
        # work happens. Build a synthetic RuntimeContext from process_turn
        # args so the hook sees the same shape as it does for retrieval.
        rl_ctx = self._resolve_runtime_context(
            agent_id=agent_id,
            user_id=user_id,
            tenant_id=tenant_id,
            request_id=None,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        await self._invoke_rate_limit_hook("process_turn", rl_ctx)

        if not skip_scan:
            scan = scan_content(user_msg or "")
            if scan.severity == "high":
                logger.warning(
                    "UMAMemory.process_turn: high-severity injection in user_msg "
                    "user_id=%s session_id=%s rules=%s",
                    user_id,
                    session_id,
                    scan.matched_rules,
                )
                raise InjectionDetectedError(
                    severity=scan.severity,
                    matched_rules=scan.matched_rules,
                    score=scan.score,
                )
            if scan.severity != "none":
                logger.warning(
                    "UMAMemory.process_turn: injection scan severity=%s user_id=%s session_id=%s rules=%s",
                    scan.severity,
                    user_id,
                    session_id,
                    scan.matched_rules,
                )

        if self.pipeline is None:
            from uma.ingest.pipeline import MemoryPipeline

            self.pipeline = MemoryPipeline(
                memory_client=self,
                hooks=self.hooks,
            )
            logger.debug("UMAMemory.process_turn: MemoryPipeline initialized lazily.")

        normalized_user_id = normalize_user_id(user_id)
        await self.pipeline.process_turn(
            agent_id=rl_ctx.agent_id,
            user_id=normalized_user_id,
            user_msg=user_msg,
            assistant_reply=assistant_reply,
            session_id=session_id.strip(),
            tenant_id=rl_ctx.tenant_id,
            workspace_id=workspace_id,
            extra_meta=extra_meta,
        )



    async def ingest_document(
        self,
        file_path: str,
        *,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        tenant_id: Optional[str] = DEFAULT_TENANT_ID,
        config: Optional[Any] = None,
    ) -> "IngestReport":
        """Ingest an unstructured document into UMA memory.

        Documents are scoped by ``(tenant_id, owner_type, owner_id)`` — the
        DAT invariant — not by a user/session request scope, so this API takes
        neither ``user_id`` nor ``agent_id``. Uploading a file is a user
        action; the owner tuple alone decides who can read the document back,
        and ``owner_type``/``owner_id`` are required and validated by the
        ingest layer. ``tenant_id`` defaults to the single-tenant value.
        """
        if not file_path or not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path is required and cannot be empty")
        import os as _os
        if not _os.path.exists(file_path):
            raise FileNotFoundError(f"file not found: {file_path}")
        if not _os.path.isfile(file_path):
            raise ValueError("file_path must point to a regular file")
        self._ensure_ingestion_ready()

        # M6: ingest_document has no user_id / session_id scope (the API
        # takes owner_type/owner_id, not a user scope), so we cannot
        # construct a full RuntimeContext. Pass None to the hook; the
        # hook may still throttle by operation name plus any
        # owner-derived heuristic in its closure.
        await self._invoke_rate_limit_hook("ingest_document", None)

        resolved_tenant_id = validate_tenant_id(tenant_id or DEFAULT_TENANT_ID)

        from uma.ingest.ingest_service import ingest_document as _ingest
        return await _ingest(
            file_path,
            owner_type=owner_type,
            owner_id=owner_id,
            tenant_id=resolved_tenant_id,
            workspace_id=workspace_id,
            config=config,
            memory=self,
        )

    async def load_memory_bootstrap(
        self,
        file_path: str,
        *,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = DEFAULT_TENANT_ID,
        config: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Bootstrap long-term memory facts from MEMORY.md through the ingest layer."""
        runtime_context = self._resolve_runtime_context(
            agent_id=agent_id,
            user_id=user_id,
            tenant_id=tenant_id,
            request_id=request_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        self._ensure_ingestion_ready()

        from uma.ingest.ingest_service import ingest_memory_bootstrap

        return await ingest_memory_bootstrap(
            file_path,
            memory=self,
            runtime_context=runtime_context,
            config=config,
        )

    async def load_daily_diary_bootstrap(
        self,
        file_path: str,
        *,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = DEFAULT_TENANT_ID,
        config: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Bootstrap a daily diary file through the ingest layer.

        Pass an IngestConfig via `config` to override defaults like
        max_file_bytes and pdf_max_pages. If omitted, the default
        IngestConfig is used (50 MiB file cap, 5000-page PDF cap).
        """
        runtime_context = self._resolve_runtime_context(
            agent_id=agent_id,
            user_id=user_id,
            tenant_id=tenant_id,
            request_id=request_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        self._ensure_ingestion_ready()

        from uma.ingest.ingest_service import ingest_daily_diary_bootstrap

        return await ingest_daily_diary_bootstrap(
            file_path,
            memory=self,
            runtime_context=runtime_context,
            config=config,
        )

    # ----------------------------------------------------------------------
    # Core API: Health and maintenance
    # ----------------------------------------------------------------------

    def health_check(self) -> "HealthStatus":
        """Run basic dependency readiness checks."""
        from uma.common.health import HealthCheck
        from uma.common.results import HealthStatus

        if not self.initialized:
            return HealthStatus(
                status="error",
                checks={
                    "memory": HealthCheck(
                        name="memory",
                        status="error",
                        detail="UMAMemory not initialized",
                        latency_ms=None,
                    )
                },
            )

        from uma.common.health import run_health_checks

        return run_health_checks(self)

    async def rebuild_vector_indexes(
        self,
        *,
        tenant_id: Optional[str] = DEFAULT_TENANT_ID,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        batch_size: int = 32,
    ) -> "VectorRebuildReport":
        """Rebuild vector indexes from SQL-backed data.

        Maintenance is scoped by tenant plus the durable owner tuple; an
        agent's own rows are addressed as ``owner_type="agent"``,
        ``owner_id=<agent_id>``.
        """
        from uma.common.maintenance import rebuild_vector_indexes

        return await rebuild_vector_indexes(
            self,
            tenant_id=validate_tenant_id(tenant_id or DEFAULT_TENANT_ID),
            owner_type=owner_type,
            owner_id=owner_id,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
            include_procedural=include_procedural,
            batch_size=batch_size,
        )

    async def rebuild_derived_indexes(
        self,
        *,
        tenant_id: Optional[str] = DEFAULT_TENANT_ID,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        include_graph: bool = True,
        batch_size: int = 32,
    ) -> "DerivedRebuildReport":
        """Rebuild derived vector and graph indexes from authoritative SQL-backed data.

        Maintenance is scoped by tenant plus the durable owner tuple; an
        agent's own rows are addressed as ``owner_type="agent"``,
        ``owner_id=<agent_id>``.
        """
        from uma.common.maintenance import rebuild_derived_indexes

        return await rebuild_derived_indexes(
            self,
            tenant_id=validate_tenant_id(tenant_id or DEFAULT_TENANT_ID),
            owner_type=owner_type,
            owner_id=owner_id,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
            include_procedural=include_procedural,
            include_graph=include_graph,
            batch_size=batch_size,
        )

    # ----------------------------------------------------------------------
    # Core API: Shutdown
    # ----------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean up backend resources."""
        if self.graph_core:
            try:
                self.graph_core.close()
            except Exception:
                logger.exception("Error shutting down GraphCore.")


    # ----------------------------------------------------------------------
    # Core subsystem initialization
    # ----------------------------------------------------------------------

    def _init_core_subsystems(self) -> None:
        """Initialize core UMA subsystems that depend on LLM/embedder/stores."""
        if self.llm is None or self.embedder is None:
            raise RuntimeError(
                "UMAMemory._init_core_subsystems: LLM and embedder must "
                "be initialized before core subsystems."
            )

        if not self._stores:
            raise RuntimeError(
                "UMAMemory._init_core_subsystems: stores must be initialized "
                "before core subsystems."
            )

        # Build all core subsystems first so a failed init cannot leave a
        # partially-ready core graph on the shared memory instance.
        working_memory: Optional[WorkingMemoryCore] = None
        episodic_core: Optional[EpisodicCore] = None
        semantic_core: Optional[SemanticCore] = None
        procedural_core: Optional[ProceduralCore] = None
        chunk_core: Optional[ChunkCore] = None

        # ---------------------- Working Memory Core ----------------------
        try:
            working_memory = WorkingMemoryCore(
                llm=self.llm,
                memory_client=self,
            )
            logger.debug("WorkingMemoryCore initialized.")
        except Exception:
            logger.exception("UMAMemory: failed to initialize WorkingMemoryCore.")
            raise

        # ---------------------- Episodic Core ---------------------------
        try:
            indexer = EpisodeIndexer(
                llm=self.llm,
                embedder=self.embedder,
            )

            retention = EpisodicRetentionPolicy(
                max_episodes=self.consolidation_cfg.max_episodes_per_cycle
            )

            episodic_core = EpisodicCore(
                episodic_store=self._stores["episodic"],
                episode_indexer=indexer,
                retention_policy=retention,
            )
            logger.debug("EpisodicCore initialized.")
        except Exception:
            logger.exception("UMAMemory: failed to initialize EpisodicCore.")
            raise

        # ---------------------- Semantic Core ---------------------------
        try:
            salience = self.semantic_salience_threshold
            decay_days = self.cfg.semantic_salience_decay_days
            semantic_core = SemanticCore(
                llm=self.llm,
                embedder=self.embedder,
                semantic_store=self._stores["semantic"],
                salience_threshold=salience,
                salience_decay_days=decay_days,
                memory=self,
            )
            logger.info(
                "SemanticCore initialized (salience_threshold=%.2f, decay_days=%.0f).",
                salience,
                decay_days,
            )
        except Exception:
            logger.exception("UMAMemory: failed to initialize SemanticCore.")
            raise

        # ---------------------- Procedural Core -------------------------
        try:
            procedural_core = ProceduralCore(self._stores["procedural"])
            logger.debug("ProceduralCore initialized.")
        except Exception:
            logger.exception("UMAMemory: failed to initialize ProceduralCore.")
            raise

        # ---------------------- Chunk Core ------------------------------
        try:
            chunk_core = ChunkCore(self._stores["chunk"], memory=self)
            logger.debug("ChunkCore initialized.")
        except Exception:
            logger.exception("UMAMemory: failed to initialize ChunkCore.")
            raise

        self.working_memory = working_memory
        self.episodic_core = episodic_core
        self.semantic_core = semantic_core
        self.procedural_core = procedural_core
        self.chunk_core = chunk_core

    # ----------------------------------------------------------------------
    # Graph subsystem initialization
    # ----------------------------------------------------------------------

    def _init_graph_core(self) -> None:
        """Initialize the graph subsystem using the unified storage config."""
        storage_cfg = self.raw_config.storage
        backend = storage_cfg.graph_backend

        # --------------------------------------------------------------
        # 1) Disabled backend → skip cleanly
        # --------------------------------------------------------------
        if backend == "disabled":
            self.graph_core = None
            logger.info("Graph subsystem disabled via config.storage.graph_backend.")
            return

        # --------------------------------------------------------------
        # 2) Load graph config block
        # --------------------------------------------------------------
        if "graph_config" not in storage_cfg:
            raise ValueError(
                "Missing required 'storage.graph_config' section in config when "
                f"graph_backend is '{backend}'."
            )

        graph_cfg = storage_cfg.get("graph_config") or {}

        # --------------------------------------------------------------
        # 3) Backend selection
        # --------------------------------------------------------------
        if backend in {"neo4j", "memgraph"}:
            raise ValueError(
                "Graph backends must be loaded via a plugin spec. "
                "Set storage.graph_backend to a plugin spec 'module:callable'."
            )

        if ":" not in str(backend):
            raise ValueError(
                f"Unsupported storage.graph_backend={backend!r}. "
                "Expected: 'disabled' or plugin spec 'module:callable'."
            )

        if not isinstance(graph_cfg, dict):
            raise ValueError("'storage.graph_config' must be a mapping for plugin graph backends")

        try:
            adapter_factory = parse_plugin_spec(backend)
            if not callable(adapter_factory):
                raise TypeError("storage.graph_backend plugin must be a callable 'module:attr'")
            adapter = adapter_factory(**graph_cfg)
        except Exception as exc:
            logger.exception("Graph adapter initialization failed.")
            raise RuntimeError(
                "Failed to initialize graph adapter. "
                "Verify plugin path and configuration."
            ) from exc

        # --------------------------------------------------------------
        # 4) Connect adapter to GraphCore
        # --------------------------------------------------------------
        try:
            self.graph_core = GraphCore(adapter)
            logger.info(
                "GraphCore initialized (backend=%s, uri=%s).",
                backend,
                graph_cfg.get("uri"),
            )
        except Exception as exc:
            logger.exception("Failed to initialize GraphCore.")
            raise RuntimeError(
                "Graph core initialization failed. "
                "Verify graph adapter dependencies and configuration."
            ) from exc

    # ----------------------------------------------------------------------
    # Optional Features
    # ----------------------------------------------------------------------

    def _init_optional_features(self) -> None:
        """Attach optional UMA features from config."""
        policy_cfg = self.features_cfg.policy or {}
        self._feature_policy = FeaturePolicy(
            on_attach_error=str(policy_cfg.get("on_attach_error", "log_and_skip")),
            allow_method_override=bool(policy_cfg.get("allow_method_override", False)),
        )

        registry = default_feature_registry()
        registry.register_entry_points()
        loader = FeatureLoader(registry, self._feature_policy)

        services = {
            "procedural_core": self.procedural_core,
            "episodic_core": self.episodic_core,
            "semantic_core": self.semantic_core,
            "llm": self.llm,
            "embedder": self.embedder,
            "hooks": self.hooks,
            "graph_core": self.graph_core,
            "cluster_similarity": self.consolidation_cfg.cluster_similarity,
            "max_episodes_per_cycle": self.consolidation_cfg.max_episodes_per_cycle,
        }

        loader.load_from_config(
            memory_client=self,
            feature_cfgs=self.features_cfg.load,
            services=services,
        )

    def _register_methods(
        self,
        feature_name: str,
        methods: dict[str, Any],
        allow_override: Optional[bool] = None,
    ) -> None:
        """Attach internal feature methods to UMAMemory with collision checks."""
        allow_override = (
            self._feature_policy.allow_method_override
            if allow_override is None
            else allow_override
        )
        for name, func in methods.items():
            if not allow_override and hasattr(self, name):
                raise ValueError(
                    f"Feature '{feature_name}' attempted to override '{name}'"
                )
            setattr(self, name, func)

