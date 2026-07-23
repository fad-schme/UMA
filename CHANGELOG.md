# Changelog

All notable changes to UMA are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.5-beta] — 2026-07-23

### Added
- **Qdrant vector adapter** moved to the Enterprise tier. The Lite core now
  ships three adapters: LanceDB (default, recommended for multi-tenant),
  FAISS (single-tenant, in-process), and InMemory (testing and CI).
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
- Placeholder GitHub URL (`your-org/uma-rlm`) replaced with the real
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
- **OWASP coverage table** added to `uma-security.md` — full 10-row table
  with explicit in-scope / out-of-scope statements for all LLM Top 10 2025
  categories.
- **ASI mapping** added to `ARCHITECTURE.md` — ASI06 (Memory Poisoning
  primary), ASI03 (Identity & Privilege Abuse), ASI05 (Unexpected Code
  Execution, ingest path).
- **Seven security primitives** named and described consistently across
  `uma-overview.md`, `uma-security.md`, README, and ARCHITECTURE:
  Provenance, Write-time trust scoring, Cryptographic integrity, Injection
  pattern detection, Two-layer injection gate, Quarantine, Ingest boundary
  hardening.

### Changed
- **`LatestWinsFactResolver` description corrected** in `uma-lanes.md`. The
  resolver picks by `max(updated_at)` across all facts including quarantined
  ones; quarantine exclusion is at the SQL retrieval layer
  (`AND quarantined_at IS NULL`), not in the resolver.
- **Dead code removed** — `defer_post_turn` queue drainer, consolidation
  hardcoded-off wrapper, `CoTMemoryBuilder`, orphaned `GraphUpdater` class,
  13 confirmed-dead functions, 34 unused imports.
- **`M5` store deduplication** — `quarantine_record` and
  `reinstate_quarantined_record` lifted into `BaseVectorSQLStore`; scope-
  helper names unified.
- **ARCHITECTURE.md lane table** corrected to match `planner.py` (PERSONAL
  intent activates profile + procedural + semantic + episodic; MIXED activates
  five lanes).
- **`set_context` ambient-scope warning** documented in `uma-overview.md`
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
  profile. No external services required.
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
