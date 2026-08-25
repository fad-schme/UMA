# Changelog

All notable changes to UMA are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [unreleased]

### Added
- **`uma maintenance consolidate --user ... [--tenant ...]`** CLI command and
  `uma.api.management.consolidate(memory, user_id=..., tenant_id=...)`, the
  first reachable entry point for consolidation (clustering, fact extraction,
  pruning). Consolidation remains caller-invoked only — UMA core still never
  schedules it (ADR-0004) — this just makes the existing capability callable.
  Destructive (deletes episodes and facts), so it's gated behind the same
  `--yes` / interactive confirmation as `quarantine purge`.
- **MCP server** (`uma-mcp` console script) exposes UMA's public API as MCP
  tools. Install with `pip install 'uma-mem[mcp]'`. Stdio mode wires Claude
  Code, Codex, Cursor, Windsurf, and Claude Desktop (local bridge) via a
  single JSON snippet or one `mcp add` command per client. Five tools
  exposed: `retrieve_context`, `retrieve_memory`, `process_turn`,
  `ingest_document`, `health_check`. Every response is a well-formed JSON
  string matching the corresponding `uma.common.results` model
  (`.model_dump_json()`); `IngestReport` (frozen dataclass) serializes via
  `dataclasses.asdict`. See [`docs/mcp/STDIO_CLIENTS.md`](docs/mcp/STDIO_CLIENTS.md).
- **HTTP transport for `uma-mcp`** with opaque bearer-token auth via
  `--http --port --bind --public-url` flags. Unlocks Claude Desktop
  (remote connector), Claude Cowork, and Perplexity. SQLite-backed token
  store at `.uma/db/mcp_tokens.db` (override with `--tokens-db`); tokens
  are SHA-256 hashed and the raw token is shown exactly once at issue
  time. Managed via new `uma auth create/list/revoke` CLI subcommands.
  `UMATokenVerifier` (`uma.mcp.auth`) is the `mcp.server.auth.provider.TokenVerifier`
  subclass that consults the store. See [`docs/mcp/DEPLOY.md`](docs/mcp/DEPLOY.md).
- **OAuth 2.1 JWT verification for `uma-mcp --http`** via `--oauth-issuer`,
  `--oauth-audience`, `--oauth-jwks-uri`, `--oauth-tenant`,
  `--oauth-required-scope` flags. Enables ChatGPT as a remote MCP client.
  UMA acts as a pure OAuth 2.1 resource server per the current MCP spec —
  the authorization server is any RFC 8414-compliant IdP (Auth0, Microsoft
  Entra ID, Google, Okta, Keycloak, Authentik). UMA never issues JWTs.
  Requires the new `oauth` extra (`pip install 'uma-mem[mcp,oauth]'`)
  which pins `PyJWT>=2.10.0` for RFC 7519 §4.1.2 strict-`sub` validation.
  `UMAJWTVerifier` uses an explicit `["RS256", "ES256"]` algorithm
  allowlist; HS256 rejected to close the algorithm-confusion attack. See
  [`docs/mcp/CHATGPT.md`](docs/mcp/CHATGPT.md).
- **`mcp` and `oauth` optional install extras** in `pyproject.toml`.
  Base install stays lean; only stdio/HTTP mode pulls the MCP SDK, only
  OAuth mode pulls PyJWT + cryptography. Bearer verifier and JWT verifier
  are mutually exclusive at CLI level (`--oauth-issuer` conflicts with
  `--tokens-db`).
- **`scripts/cleandir.py --verify`** pre-release / CI gate. Fails with a
  listed diff if `__MACOSX/`, `.DS_Store`, orphaned `.pyc`, `*.egg-info/`,
  `build/`, `dist/`, `.coverage`, `htmlcov/`, `.tox/`, or macOS AppleDouble
  files (`._*`) exist in the tree. Auto-detects repo root via `.git` /
  `pyproject.toml` / `setup.py` markers and refuses to run outside a
  project root.

### Removed
- **`agent_id` on `UMAMemory.ingest_document` and on `uma ingest document`.**
  Uploading a file is a user action, and a document is scoped by its durable
  `(tenant_id, owner_type, owner_id)` tuple — that tuple alone decides who can
  read it back. The argument was validated and then discarded, so it recorded
  nothing while implying an agent-level authorization that did not exist. The
  CLI `--agent` flag for this subcommand is gone with it. The `ingest_document`
  **MCP tool** keeps its `agent_id` parameter: there it resolves the owner
  tuple for an agent-owned document rather than travelling with the ingest.
- **`set_context()` and `ScopedUMAMemory`.** Agent identity is no longer bound
  to an instance. UMA is multi-agent on one shared runtime, so binding an agent
  to an object meant one `UMAMemory` per agent and made the identity of a call
  depend on which object you happened to hold. `UMAMemory.agent_id` and
  `UMAMemory.promotion_policy` are gone with it.
- **`--agent` on `uma health` / `uma doctor`.** `health_check()` probes
  dependency readiness, which is shared by every agent on the runtime, so the
  flag scoped nothing.
- **USER.md / SOUL.md profile overlay** (`load_userprofile`, `load_agentprofile`,
  `AnimusProfileProvider`). Loading was public but consumption was internal-only,
  so no supported caller could read the overlay back. `set_agent_profile` and the
  `load_*_bootstrap` methods are unaffected.
- **`ContextPackBuilder`** (`uma.retrieve.context_pack_builder`) and the
  `UMARuntime.render_context` / `get_context_messages` methods that used it.
  This was a second, unused context-assembly path: no production caller
  remained, and every bundled example already hand-rolled its own renderer
  over `ContextBundle` instead. Anyone importing `ContextPackBuilder` from
  `uma.retrieve` must migrate to their own `ContextBundle → str` rendering.
- **`consolidation_run` / `consolidation_health`** on `UMAMemory`. These were
  `setattr`-attached by `ConsolidationFeature`, invisible to static readers
  and mypy, and reachable by nothing except a test workaround — no CLI and no
  documented caller ever existed. Replaced by
  `uma.api.management.consolidate(memory, user_id=..., tenant_id=...)`, in
  the same shape as `purge_quarantined`/`reinstate_quarantined` rather than a
  `UMAMemory` method, since consolidation is destructive like those
  operations. See **Added** for the new entry point.

### Changed
- **`agent_id` is a required per-call argument** on every request-scoped
  public API: `retrieve_context`, `retrieve_memory`, `process_turn`,
  `load_memory_bootstrap`, `load_daily_diary_bootstrap`, `set_agent_profile`,
  `get_agent_profile`, and the `explain_result` / `lint_memory_drift`
  management functions. Omitting it raises `ValueError`; there is no
  instance-level or default fallback. `UMAMemory.ingest_document` is
  deliberately not on that list — see Removed.
- **MCP tools require `agent_id` on every call** — a required parameter in the
  tool schema for `retrieve_context`, `retrieve_memory`, `process_turn`, and
  `ingest_document`, with no default and no environment fallback.
  `UMA_CONFIG_PATH` is the only environment variable the server reads; one
  `uma-mcp` process serves every agent.
- **`tenant_id` is a parameter on every public method**, defaulting to
  `DEFAULT_TENANT_ID`. UMA Lite is single-tenant, but the value is carried
  explicitly alongside `agent_id` and `user_id` rather than assumed at the
  storage boundary.
- **`DEFAULT_TENANT_ID` lives in `uma.common.types.types_scope`** with the rest
  of the identity vocabulary, and every `"default"` tenant literal across the
  public API, MCP server, and domain types references it.
- **Promotion policy is bound to the turn's `agent_id`** inside the pipeline
  instead of being carried on the memory instance, so concurrent agents cannot
  promote into each other's KB.
- **`uma ingest document` requires `--agent`** in addition to `--owner-type` /
  `--owner-id`. The agent records who performed the ingest; the owner tuple
  still determines what the document is scoped to.
- Trimmed the README security section and moved the OWASP/ASI mappings to
  [`SECURITY.md`](SECURITY.md).

### Security
- **Promotion no longer carries the originating user into a shared scope.**
  A fact promoted from a user into an agent's KB — which every user of that
  agent reads — travelled with `origin_user_id` and `origin_session_id` on the
  `Fact` itself and with `promotion.source_owner_id` / `source_session_id` in
  its metadata, which surfaces on `ContextBundle.facts` and in
  `provenance.derivation`. Content promotion is gated against user-identifying
  subjects; the lineage reinstated the identity that gate refuses. Both
  carriers are now redacted whenever a promotion lands on an owner other than
  the source's. `source_fact_id` is preserved, so the audit chain still
  resolves to the source row and its owner. Session-to-user promotion keeps
  the same principal and is unaffected.

### Fixed
- **Retrieval's public-boundary scope filter ignored `tenant_id`.**
  `UMARuntime._filter_items_by_scope` — the defense-in-depth pass over the
  facts, chunks, episodic, and skills lanes — matched only `(owner_type,
  owner_id)`, so an item from another tenant with the same owner tuple would
  have passed it. Store reads were already tenant-scoped, so nothing leaked in
  practice; the boundary check now verifies the tenant too and fails closed on
  a missing one.
- **Session-local retrieval filtering failed open.** Items missing `tenant_id`,
  or session-local items missing `origin_agent_id`, were admitted rather than
  dropped — the opposite convention to every other isolation check. Both now
  fail closed.
- **`uma health` / `uma doctor` exited 4 on a healthy install.** A `skipped`
  check forced overall `degraded`, and graph — disabled by default — always
  reported `skipped`, so container healthchecks and CI gates failed everywhere.
  Disabled optional subsystems now report a neutral `disabled` status.
- **Health reported `[ok]` for unreachable LLM and embedding providers.** Both
  checks only asserted the adapter object existed. Health now calls the
  providers and reports `error` with provider, host, and cause. `--offline` is
  unchanged.
- **The MCP docs were never committed.** `.gitignore` excluded `docs/`, so
  `docs/mcp/*` never appeared in `git status` while README linked to it. Docs
  rewritten, `.gitignore` narrowed, and the `docs-lint` CI job is now blocking
  for repo-relative links.
- **Examples crashed on first use.** They called `.get()` on `ContextBundle` and
  `MemoryResult`, which are Pydantic models. All rewritten against the typed
  API; added `examples/README.md`.
- **`CODEOWNERS` was inert** — it lived in `.github/workflows/`, which GitHub
  does not read. Moved to `.github/`.
- **`[ollama]` install extra pulled the wrong package.** It declared the
  `ollama` client, which UMA never imports. Ollama is reached over its
  OpenAI-compatible HTTP API through `OpenAICompatibleLLM` /
  `OpenAICompatibleEmbedder`, so the extra now declares `openai>=1.0.0`.
  Previously `pip install uma-mem[ollama]` still failed at initialization
  with "requires the 'openai' package to be installed"; `uma doctor` now
  reports `[ok]` for both the LLM and embedding provider on a fresh install.
- **Source distribution shipped an uncollectable test suite.** setuptools
  auto-included only top-level `tests/test_*.py`, omitting `tests/__init__.py`,
  `conftest.py`, `helpers/`, `fixtures/`, and `e2e/` — 16 of 27 test modules
  failed to import on `No module named 'tests.helpers'`. `MANIFEST.in` now
  grafts `tests/` and `config/` (the reference config two tests read as the
  source of truth for the documented lite profile). The sdist suite collects
  and runs: 767 passed.
- **README links were dead on the package page.** All 18 repo-relative links
  (`ARCHITECTURE.md`, `CONTRIBUTING.md`, `LICENSE`, `tests/e2e/README.md`, and
  the nine `.claude/skills/*.md` guides) plus the architecture diagram resolved
  against nothing once rendered as the PyPI long description. They are now
  absolute GitHub URLs — the diagram via `raw.githubusercontent.com` so it
  actually renders. Added `Source` and `Changelog` project URLs, which the
  package page previously lacked entirely.
- **`pip install -e '.[dev]'` could not run the test suite.** The parser tests
  need beautifulsoup4/markdown and `tests/test_cli.py` runs `doctor` against an
  ollama-provider config, but neither was reachable from `dev`. It now pulls
  `uma-mem[parsers]` plus `openai`, so a single `[dev]` install is sufficient.
  Parser libraries stay in the `parsers` extra rather than the base install —
  `uma/ingest/parser.py` guards those imports with install hints, which only
  holds if they remain optional.

---

## [0.2.0] — 2026-08-10

First release published to PyPI.

### Added
- **Operational CLI** with stable `uma` and `python -m uma.cli` entry points,
  text/JSON output, secret-redacted configuration inspection, offline and
  runtime diagnostics, injection scanning, CI-aligned development checks,
  scoped retrieval and ingestion, audit/quarantine listing, and guarded
  quarantine, index, and integrity administration. Destructive operations
  require an exact resolved target and interactive confirmation or `--yes`.
- **`google-re2` regex backend for the injection scanner** via a new
  `pip install uma-mem[security]` install extra. When installed, `scan_content`
  and every rule-function scorer compile their patterns through RE2, which
  is linear-time by construction — ReDoS is impossible regardless of what
  patterns future contributors add. Base install falls back to Python's
  `re` with a one-time WARNING at first import. All 200 shipped patterns
  are already RE2-compatible (verified); no pattern rewrites required.
- **`uma/common/_regex_backend.py`** — single canonical selector for the
  security-critical regex engine. Exposes `compile`, `MULTILINE`,
  `IGNORECASE`, `DOTALL`, `error`, and `USING_RE2`. Scoped deliberately to
  the write-time attack surface; other UMA code paths that use regex are
  unchanged.
- **`tests/test_injection_scan_perf.py`** — ReDoS-defense benchmark. A
  100 KB realistic input must scan under a backend-aware ceiling (200 ms
  under RE2, 1000 ms under the `re` fallback). A crafted adversarial
  input with repeated tokens must scan under 50 ms. CI ratchet against
  future pathological patterns.

### Fixed
- **`import uma.memory` no longer fails in a fresh interpreter.** The
  re-exports in `uma/retrieve/rlm/__init__.py` turned any leaf import into a
  whole-subsystem import, closing two cycles: `uma.memory.chunk.core` →
  `uma.retrieve.ranking` → (rlm package) → `controller` →
  `uma.memory.chunk.core`, and `planner` → `rlm.intent` → (rlm package) →
  `controller` → `evidence` → `request` → `planner`. Either one raised
  ImportError whenever `uma.memory` (or `uma.memory.chunk.core`, or
  `uma.retrieve.planner`) was the first UMA import; the SDK entry points
  happened to import in an order that hid it. The package initializer is now
  import-free — every caller already imported the submodules directly, so no
  call site changed.
- **Stale `uma.core.*` imports in the consolidation and procedural features**
  (`from ...core.semantic.extractor import FactExtractor` and four siblings)
  pointed at a package layout that no longer exists, so
  `import uma.memory.consolidation` raised
  `ModuleNotFoundError: No module named 'uma.core'`. Repointed at
  `uma.memory.*`.
- **`tests/test_package_imports.py`** — new guard importing every shipped
  subpackage in its own subprocess. The rest of the suite imports `uma.api`
  first via conftest, which masks this entire class of bug.
- **Skill filenames in documentation.** `ARCHITECTURE.md` and four skills
  referenced `.claude/skills/uma-*.md`; `uma-` is the skill *name* in
  frontmatter, not part of the filename, so all 18 references pointed at
  files that do not exist. `ARCHITECTURE.md` also no longer implies `lancedb`
  is the base install's only dependency.

### Changed
- **Distribution renamed to `uma-mem`.** `pip install uma` installs an
  unrelated project that already owns that name on PyPI; install UMA with
  `pip install uma-mem` (extras: `pip install 'uma-mem[security]'`). Only the
  distribution name changed — the import package is still `uma`
  (`import uma`, `from uma import UMAMemory`) and the CLI is still `uma`, so
  no source change is required in dependent code.
- **Version is now plain PEP 440 without a pre-release suffix** (`0.2.0`,
  previously `0.1.5-beta`). `uma version`, the wheel filename, and PyPI now
  all report the same string; `0.1.5-beta` normalized to `0.1.5b0` in
  packaging tools while the CLI printed the unnormalized form. UMA remains
  beta-quality software — see `SECURITY.md § Supported Versions`.
- **Agent identity binding is immutable.** `set_context(agent_id=...)` now
  returns a distinct per-agent `ScopedUMAMemory` view and never mutates the
  source runtime. Each scoped instance owns its turn pipeline and promotion
  policy, preventing concurrent agents from overwriting ambient identity.
- **Promotion is a documented public feature.** Every scoped instance has a
  bound default `PromotionPolicy`; `set_agent_profile` opts the agent into
  background, profile-gated promotion, while no profile remains a safe no-op.
- **Feature method registration is internal-only.** `register_methods` is now
  `_register_methods`, and built-in features use the private attachment path.
- **Packaging license metadata migrated to PEP 639.** The build backend now
  requires setuptools 77+, uses the `Apache-2.0` SPDX expression, declares
  `LICENSE` and `NOTICE` explicitly, and removes the deprecated license
  classifier.
- **`uma/common/rule_functions.py` patterns precompiled at module load**
  instead of recompiled on every scorer call. Behaviour identical; removes
  per-call `re.compile` overhead on the write-time hot path.
- **Packaging consolidated into `pyproject.toml`.** `setup.py` deleted;
  every field (runtime dependencies, optional-dependencies, project
  metadata, package-data, dynamic version) now lives in `pyproject.toml`
  as the single source of truth. Supported extras are
  (`llm`, `openai`, `ollama`, `e2e`, `vector`, `graph`, `security`,
  `parsers`, `dev`); the `vector` extra installs
  FAISS. Build with `python -m build`; install with
  `pip install .` or `pip install -e '.[dev]'`. Anyone invoking
  `python setup.py <cmd>` directly will need to switch to `pip` or
  `python -m build` — the file is gone.
- **OWASP LLM10 posture corrected from "In scope" to "Partial"** across
  `README.md`, `ARCHITECTURE.md`, `.claude/skills/overview.md`, and
  `.claude/skills/security.md` (row + frontmatter description). UMA-owned
  defenses cover only the ingest side (`max_file_bytes`, `pdf_max_pages`).
  The retrieval side is a caller-owned plug-point: `set_rate_limit_hook`
  registers a single hook, but UMA ships no default limiter and owns no
  throttling policy — accounting, storage, timeouts, and refusal semantics
  are the caller's. No code changes; documentation now reflects what the
  code actually does.

### Removed
- **`setup.py`** — merged into `pyproject.toml` (see above). The custom
  `build_py` cmdclass that cleared stale generated package trees is not
  ported: `python -m build` runs in an isolated environment where the
  stale-tree problem does not occur. If you hit it, `git clean -fdx`
  before building.
- **Unsupported backend dependencies** — removed Weaviate, Pinecone, and
  FastEmbed from the `vector` extra and removed the `postgres` extra.
  UMA Lite has no corresponding adapters; custom adapters must declare
  their own client dependencies.

---

## [0.1.5-beta] — 2026-07-23

### Added
- **Qdrant vector adapter** moved to the Enterprise tier. The Lite core now
  ships three adapters: LanceDB, FAISS (in-process), and InMemory (testing and CI).
- **Multilingual injection catalog** (`injection_patterns.l10n.yaml`) is now
  loaded automatically alongside the English catalog. Covers French, Spanish,
  German, and Simplified Chinese. CJK fast-path skips non-CJK content on the
  write-time hot path.
- **`consolidation_trigger.py`** — scaffolding for the configurable
  consolidation auto-trigger hook (wiring to `ingest_service` in progress).
- **`retrieve/__init__.py`** now exports `ContextPackBuilder`, `Ranker`,
  `RetrievalPolicy`, `fuse_candidates`, `rerank_candidates`, and `should_stop`
  as stable public surface.
- **GitHub Actions CI** workflow added (`.github/workflows/ci.yml`). Runs the
  full test suite on Python 3.9–3.12 on every push and pull request. Separate
  Bandit security scan job.
- **CHANGELOG** (this file).

### Changed
- **`openai` is now an optional dependency** (`pip install -e '.[openai]'`).
  The base install no longer requires the OpenAI SDK. Every OpenAI import in
  the codebase was already lazily guarded; this change makes the packaging
  consistent with that design.
- **LanceDB version pin relaxed** from `>=0.25.3,<0.26` to `>=0.25.3`. The
  UMA adapter uses only stable LanceDB APIs (`connect`, `open_table`,
  `create_table`, `table.search().where().limit().to_list()`, `table.add()`,
  `table.delete()`) that are unchanged in 0.26+.
- **RLM documentation corrected.** The iterative retrieval loop (RLM) is
  accurately described as coverage-driven and deterministic at the navigation
  level. The LLM participates only in post-loop fact pruning, not in deciding
  what to retrieve next. Affected files: `ARCHITECTURE.md`,
  `.claude/skills/overview.md`.
- **`from_yaml` path clarified** in all documentation. The path argument
  accepts any absolute or relative filesystem path; `config/uma.yaml` is a
  convention used in examples, not a requirement.
- **Features config validation** (`common/config.py`) — `features: null` in
  `uma.yaml` no longer raises a false validation error; it now falls through
  to the default feature registry.
- **`REVIEW.md` removed** from the distributed package.
- **`.DS_Store` files** excluded from all future release archives.

### Fixed
- `CONTRIBUTING.md` added — previously missing from the repo root.
- Placeholder GitHub URL (`fad-schme/UMA`) replaced with the real
  repository URL in `pyproject.toml` and `setup.py`.

---

## [0.1.4-beta] — 2026-07

### Added
- **B608 SQL-injection nosec annotations** across all seven store files.
  Every f-string SQL construction is either guarded by a frozenset whitelist
  (schema identifiers), uses `?,?,?` bound-parameter expansion only, or
  operates on module-level constants. Annotations suppress Bandit false
  positives while explaining the rationale inline.
- **B110/B112 exception-handler annotations** across eight files. Best-effort
  processing loops now carry `# nosec B110/B112` with `logger.debug(...,
  exc_info=True)`. Intentional fallback handlers carry `# nosec` with an
  explanatory comment.
- **SHA-256 in snippet refiner** — replaced `hashlib.sha1` with
  `hashlib.sha256` (B324 fix).
- **OWASP coverage table** added to `security.md` — full 10-row table
  with explicit in-scope / out-of-scope statements for all LLM Top 10 2025
  categories.
- **ASI mapping** added to `ARCHITECTURE.md` — ASI06 (Memory Poisoning
  primary), ASI03 (Identity & Privilege Abuse), ASI05 (Unexpected Code
  Execution, ingest path).
- **Seven security primitives** named and described consistently across
  `overview.md`, `security.md`, README, and ARCHITECTURE:
  Provenance, Write-time trust scoring, Cryptographic integrity, Injection
  pattern detection, Two-layer injection gate, Quarantine, Ingest boundary
  hardening.

### Changed
- **`LatestWinsFactResolver` description corrected** in `lanes.md`. The
  resolver picks by `max(updated_at)` across all facts including quarantined
  ones; quarantine exclusion is at the SQL retrieval layer
  (`AND quarantined_at IS NULL`), not in the resolver.
- **Dead code removed** — `defer_post_turn` queue drainer, consolidation
  hardcoded-off wrapper, `CoTMemoryBuilder`, orphaned legacy graph-update path,
  13 confirmed-dead functions, 34 unused imports.
- **`M5` store deduplication** — `quarantine_record` and
  `reinstate_quarantined_record` lifted into `BaseVectorSQLStore`; scope-
  helper names unified.
- **ARCHITECTURE.md lane table** corrected to match `planner.py` (PERSONAL
  intent activates profile + procedural + semantic + episodic; MIXED activates
  five lanes).
- **`set_context` ambient-scope warning** documented in `overview.md`
  (L3).

---

## [0.1.3-beta] — 2026-06

### Added
- **C1 vector isolation contract** — `VectorIndex` base class requires
  `tenant_ids`, `owner_types`, `owner_ids` as mandatory parallel-list
  parameters on `upsert`, and `tenant_id`, `owner_type`, `owner_id` as
  mandatory keyword arguments on `query`. All three adapters (LanceDB, FAISS,
  InMemory) enforce isolation before the k-nearest cap.
- **LanceDB push-down filter** — isolation filter pushed into LanceDB's DuckDB
  `WHERE` clause before `.limit()` so the cap applies after tenant narrowing.
  Cross-tenant leakage is impossible by construction.
- **FAISS oversample compensation** — `_oversample_multiplier = 4` provides
  headroom for post-filter recall under moderate cross-tenant load.
- **`_validated_table_name()` / `_validated_id_column()`** frozenset
  whitelists in `BaseVectorSQLStore` — schema identifiers validated before
  any f-string SQL construction.
- **`_QUARANTINE_FILTER` / `_NO_FILTER` module constants** in all four leaf
  stores — quarantine toggle uses named constants, not inline strings.
- **Write-time injection scanning** wired at all five storage boundaries:
  document chunks, turn chunks, episodes, working memory messages, bootstrap
  writers.
- **Retrieval audit log** — every `retrieve_context` and `retrieve_memory`
  call records a SHA-256-hashed query preview, scope, severity, and result
  counts. Disable via `security.retrieval_audit_enabled: false`.

### Changed
- **`quarantine_record` / `reinstate_quarantined_record`** lifted to
  `BaseVectorSQLStore` (completed in 0.1.4).
- **`InjectionDetectedError`** exported from `uma.__init__` as public surface.

---

## [0.1.0-beta] — 2026-04

Initial public beta release.

### Features
- Six typed memory lanes: working memory, semantic facts, raw chunks,
  episodic, procedural, compiled wiki.
- SQLite (authoritative) + LanceDB (rebuildable vector accelerator) embedded
  profile. No external storage service required; model providers are configured
  separately and may run locally or remotely.
- Ownership-scoped retrieval: `tenant_id / owner_type / owner_id` enforced at
  every SQL and vector read boundary.
- `UMAMemory` public API: `retrieve_context`, `retrieve_memory`,
  `process_turn`, `ingest_document`, `scan_user_input`, `health_check`.
- Management API: `explain_result`, `lint_memory_drift`, `verify_integrity`,
  `list_quarantined`, `reinstate_quarantined`, `purge_quarantined`,
  `list_retrieval_audit`.
- Provider-agnostic: Ollama, OpenAI, Anthropic LLM support; Ollama and OpenAI
  embedding support.
- Apache 2.0 license.
