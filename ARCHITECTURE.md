# ARCHITECTURE.md — UMA

UMA is a memory and context runtime SDK for AI agents. It ingests data, stores it across typed memory lanes, and exposes two thin retrieval products; the calling application owns replies, reasoning, and tool use.

This document covers the architectural invariants. For developer-facing references, see `.claude/skills/overview.md`, `.claude/skills/api.md`, and the topic-specific skills under `.claude/skills/`.

---

## Core Products

| API | Purpose |
|-----|---------|
| `retrieve_context(...)` | Evidence-oriented RAG context for LLM prompting |
| `retrieve_memory(...)` | Compiled, evidence-backed memory for continuity — returns `compiled_memory`, `facts` (full subject-predicate-object triples), and `evidence` |

Both products are owned by `UMAMemory`, initialized from a config file, and operated with explicit per-call request scope.

---

## Memory Lanes

UMA exposes six public `lane_filter` lanes. The planner also uses `profile` (a semantic-store projection) and optional `graph`, for eight architectural lane names in total. Lane selection is explicit — ownership scope alone does not determine which lane to query.

| Lane | Store | Role | Scope default |
|------|-------|------|---------------|
| Working Memory | In-memory buffer | Recent message continuity within a session | Per user, per session |
| Raw Chunks (`raw`) | SQLite + vector index | Immutable source evidence from ingested documents | Durable |
| Semantic Facts (`semantic`) | SQLite + vector index | Structured statements extracted from chunks/turns | Session-local; promotable |
| Profile Facts (`profile`) | SQLite + vector index (shared with `semantic`) | User-profile facts; `kind=profile_fact` rows in the semantic store surfaced as a distinct retrieval lane | Durable |
| Episodic (`episodic`) | SQLite + vector index | Time-ordered interaction history | Cross-session; session_id is provenance metadata only |
| Procedural / Skills (`procedural`) | SQLite + vector index | Named skills and how-to knowledge | Durable |
| Compiled Wiki (`wiki`) | SQLite (document store) | Mutable, evidence-backed synthesis artifacts | Durable |
| Graph (optional) | Plugin backend | Relationship routing and entity expansion | — |

`profile` and `semantic` share the same SQLite store and vector index. `profile` is a distinct retrieval lane at query time — the planner selects it independently of `semantic` — but physically both read from `semantic_sql`. There is no double-count risk because each query targets one lane and the store filters by `kind`.

`trace` (`decision_trace`) exists as a kind in `KB_LANES` but is debug metadata only. The planner unconditionally excludes it with reason `trace_is_debug_metadata_not_a_retrieval_lane`; it never participates in retrieval.

Graph is disabled in all public profiles. It is a supporting lane for relationship traversal.

---

## Storage Model

```
.uma/db/
  chunks.db            — authoritative chunk text and metadata
  documents.db         — ingested document manifests
  episodic.db          — episodic event history
  semantic.db          — extracted facts
  procedural.db        — skills and routines
  retrieval_audit.db   — retrieval audit log

.uma/vectors/
  vectors_chunks
  vectors_episodic
  vectors_semantic
  vectors_procedural
```

**Invariant:** SQL is always authoritative. Vector stores hold only ids, vectors, isolation columns (`tenant_id`, `owner_type`, `owner_id`), and filterable metadata — never full chunk text. They are rebuildable from SQL at any time:

```python
await memory.rebuild_vector_indexes(tenant_id="default")
```

## Adapter Interfaces

Storage-facing extension seams live under `uma.adapters`. The secrets interface is `uma.adapters.secrets.SecretsProvider`, with the Lite reference implementation `EnvVarProvider` beside it; storage adapters that need credentials depend on that stable contract instead of inventing their own lookup shape.

`UMAMemory.from_yaml(path)` accepts an optional top-level `secrets:` block whose `provider` resolves to a `SecretsProvider` class and whose `options` are passed through as constructor kwargs. Lite's embedded SQLite profile does not use a secrets provider, but the interface is part of the architecture so community and Enterprise storage adapters can share one credential boundary.

---

## Ownership and Tenancy (DAT Invariants)

Every stored and retrieved artifact must carry explicit ownership. These invariants are non-negotiable.

| Field | Requirement |
|-------|-------------|
| `owner_type` | One of: `agent`, `user`, `workspace`, `system` |
| `owner_id` | Required; non-empty; consistent with the lane |
| `tenant_id` | Required for durable artifacts; preserved end-to-end |
| `session_id` + `agent_id` | Required for session-local artifacts |

`agent_id` and `user_id` are supplied on every public call and are never held
on the `UMAMemory` instance; `tenant_id` is supplied per call too and falls
back to `DEFAULT_TENANT_ID`. One instance therefore serves every agent and
user concurrently without ambient scope.

UMA is single-tenant, multi-agent and multi-user. Cross-owner access — one
agent or user reading another's artifacts — is prevented **at the storage
layer** (not by application-layer convention):

- The LanceDB adapter promotes `owner_type` / `owner_id` to first-class indexed columns and pushes them into every query's `WHERE` clause **before** the candidate cap is applied — the k-nearest cap cannot starve one owner under load from another.
- SQL stores filter by `tenant_id AND owner_type AND owner_id` in every read path, with `AND quarantined_at IS NULL` appended.
- Vector adapters refuse empty isolation values at write time (`ValueError`).

Note: SQL stores bind isolation values as parameters; the LanceDB predicate is a
string the adapter escapes itself (`_sql_escape`), so that path depends on the
escaping rather than on engine-side binding.

**Practical rule:** If you cannot answer "which user/agent/project is allowed to see this row?" the design is wrong.

---

## Runtime Scope Invariants

- No shared mutable object stores current request scope. Patterns like `memory.user_id` or `controller.current_scope` are forbidden.
- Every API entry point operates from an explicit, immutable `RuntimeContext` built at the call boundary.
- Working memory is keyed by `(tenant_id, agent_id, user_id, session_id)`; a shared `session_id` does not merge two users' buffers. Episodic turns are session-local by default. Semantic facts extracted from turns are also session-local by default and must be explicitly promoted to become durable.

---

## Security Architecture

Security in UMA is not an overlay — it is the shape of every code path that touches an artifact. Five primitives compose at every write boundary and every read boundary:

1. **Two-layer injection scanning** — pre-LLM advisory gate (`scan_user_input`) + write-time defense-in-depth on every storage boundary
2. **Trust scoring + quarantine** — every artifact carries `trust_score ∈ [0, 1]` (classifier-derived via `uma.common.trust.score_source`) and a nullable `quarantined_at` timestamp
3. **Content hashing + integrity verification** — every Fact / Episode / Skill / Chunk carries a canonical SHA-256 `content_hash`; `verify_integrity` re-derives and quarantines on mismatch
4. **Ingest gating** — MIME consistency, file size limits, PDF page caps, HTML/Markdown sanitization
5. **Retrieval audit log** — every retrieval call records a query digest plus a bounded 80-character preview (never the full query), scope, severity, and result counts

Coverage highlights: primitives 1/2/4 address **LLM01 (Prompt Injection)**; the C1 vector isolation contract addresses **LLM08**; ingest size caps address the ingest side of **LLM10**; the hashed-preview audit log addresses part of **LLM02**; provenance and quarantine exclusion give partial coverage of **LLM09**. `set_rate_limit_hook` is a caller-owned plug-point for the retrieval side of LLM10 — UMA ships no default limiter. Full OWASP LLM Top 10 and ASI mapping, including scope boundaries: [`SECURITY.md`](SECURITY.md).

### Injection Scanning

At every write boundary, `uma.adapters.scanner.injection_scan.scan_artifact_text` checks incoming text against three bundled YAML catalogs: the English prompt-injection rules in `uma/adapters/scanner/injection_patterns.yaml`, French, Spanish, German, and Simplified Chinese variants in `uma/adapters/scanner/injection_patterns.l10n.yaml`, and compound SQL/NoSQL injection rules in `uma/adapters/scanner/sqli_patterns.yaml`:

| Severity | Action | Trust effect |
|---|---|---|
| `none` | Proceed | Unchanged |
| `low` | Logged, proceed | Reduced by 20% |
| `medium` | Logged, proceed | Reduced by 50% |
| `high` | Quarantine (or raise `InjectionDetectedError` for `process_turn` user_msg) | Set to `0.0` |

Scan results are recorded in `meta["security"]["injection_scan"]` with rule names, severity, and score.

## Multi-language support
The pattern scanners support English (EN), French (FR), Spanish (ES), German (DE), and Simplified Chinese (ZH). Localized patterns are loaded automatically alongside the base English catalog, with no additional configuration required.

### Quarantine

When a high-severity scan hit is detected and `SecurityConfig.quarantine_enabled=True`, the artifact is stored with `quarantined_at` set to the current UTC time. Quarantined artifacts are excluded from all normal retrieval queries (`AND quarantined_at IS NULL`) but remain in the database for forensic review.

Cross-lane quarantine awareness:

- All SQL search paths in `chunk_sql.py`, `semantic_sql.py`, `episodic_sql.py`, `procedural_sql.py` include the quarantine filter
- The canonical `_fetch_ranked_rows_by_ids` in `base_vector_sql_store` includes the quarantine filter
- `LatestWinsFactResolver` excludes quarantined facts from canonical-row selection (degenerate all-quarantined case falls back across the full set with a warning; the chosen row remains quarantined and is filtered at retrieval anyway)
- Working memory `get_context` filters quarantined messages by default; `include_quarantined=True` is required to see them

The management API (`uma.api.management`) exposes `list_quarantined`, `reinstate_quarantined`, and `purge_quarantined` to review, restore, or permanently delete quarantined records. See `.claude/skills/quarantine.md` for the full lifecycle.

### Integrity Verification

`uma.api.management.verify_integrity(memory, record_id, lane, ...)` recomputes the canonical content hash for any stored Fact, Episode, Skill, or Chunk and compares it to the stored `content_hash` (or `meta["text_hash"]` for chunks). On mismatch the record is quarantined through the same path (`quarantined_at` set, audit log entry appended) and an `IntegrityVerificationResult(status="failed", quarantined=True, ...)` is returned. A clean record returns `status="verified"` without mutation.

`lint_memory_drift` automatically routes typed lane artifacts through `verify_integrity` so batch integrity checks run without calling the function directly. Background scanning across the full dataset is an Enterprise capability and is not part of this SDK.

### Trust Scoring at Retrieval

Trust is applied during the canonical retrieval pipeline (see below):

```
final_score = (1 - trust_weight) * existing_score + trust_weight * trust_score
candidates with trust_score < min_trust_score are dropped before truncation
```

Default `min_trust_score: 0.5` is calibrated to filter every medium-severity injection survivor (trust × 0.5 → 0.25) at retrieval, even when quarantine is disabled. Default `trust_weight: 0.15`.

For maintenance operations, Lite also allows post-write trust adjustment on a
single fact via `semantic_core.update_trust(fact_id, new_score, reason=..., ctx=RuntimeContext(...))`.
The update remains tenant- and owner-scoped, changes only `trust_score`, and
appends a per-update audit entry to `meta["trust_updates"]`.

### Retrieval Audit Log

Every retrieval call (`retrieve_context`, `retrieve_memory`) is recorded by default in `.uma/db/retrieval_audit.db`:

- A **SHA-256-hashed truncated preview** of `query_text` — never the raw text
- The pre-LLM scan severity (`scan_severity`)
- Whether downstream LLM hops were skipped (`llm_hops_skipped`)
- Scope: `tenant_id`, `user_id`, `session_id`

Disable via `security.retrieval_audit_enabled: false` in the runtime config. Query via `uma.api.management.list_retrieval_audit(memory, ...)`.

---

## Vector Isolation Contract

The vector adapter contract is defined in `uma/adapters/vector/base.py` and enforced identically across all three bundled backends (LanceDB, FAISS, InMemory). This is the architectural mechanism by which **LLM08 Vector and Embedding Weaknesses** is closed for UMA.

### The Contract

```python
class VectorIndex(ABC):
    def upsert(
        self, ids, vectors,
        *,
        tenant_ids: List[str],      # required parallel list; refuses empty strings
        owner_types: List[str],     # required parallel list
        owner_ids: List[str],       # required parallel list
        extra_metadata: Optional[List[Dict]] = None,  # everything non-isolation
    ): ...

    def query(
        self, vector,
        *,
        tenant_id: str,             # required; refuses empty string
        owner_type: str,            # required
        owner_id: str,              # required
        k: int = 10,
        extra_filters: Optional[Dict] = None,
    ): ...
```

**Key design points:**

- Isolation parameters are **explicit parallel lists**, never buried in a metadata dict.
- `extra_metadata` MUST NOT contain `tenant_id`, `owner_type`, or `owner_id` — adapters raise `ValueError` to refuse reserved-key injection.
- `extra_filters` is a query-time predicate map for non-isolation fields (`doc_id`, `kind`, `kb_lane`, etc.).
- Legacy call shapes (`metadata=...` on upsert, `filters=...` on query) raise `TypeError`.

### Implementation by Backend

| Backend | Isolation enforcement | Filter strategy |
|---|---|---|
| **LanceDB** (recommended) | `tenant_id` / `owner_type` / `owner_id` promoted to top-level indexed columns | DuckDB `WHERE` push-down before the candidate cap |
| **FAISS** | Stored in parallel scope dict | Oversample (`k × 4`) then post-filter in Python |
| **InMemory** | Stored in parallel `_scopes` dict | Isolation check before similarity computation |

**LanceDB push-down** is the architectural reason owner isolation holds under load. The query path:

```python
where = (
    f"tenant_id = '{_sql_escape(tenant_id)}' "
    f"AND owner_type = '{_sql_escape(owner_type)}' "
    f"AND owner_id = '{_sql_escape(owner_id)}'"
)
table.search(vec).where(where).limit(limit).to_list()
```

The cap (`limit`) is applied **after** scope narrowing. Hostile values (e.g. `t'malicious;DROP TABLE foo;--`) are SQL-escaped via standard single-quote doubling and round-trip safely.

**FAISS** does not support pushed-down predicates. The adapter oversamples by a factor of 4 and post-filters; under heavy multi-owner load FAISS can suffer recall loss. The docstring documents this; LanceDB is recommended for deployments with many agents or users.

### Score Normalization

LanceDB returns metric-dependent raw `_distance`. UMA normalizes via `score = math.exp(-max(0.0, distance))`, mapping `[0, ∞)` to `(0, 1]` — coherent with `trust_score ∈ [0, 1]` for the weighted blend.

### Atomicity

All three adapters validate every row in a batch before mutating any state. A bad row in position N causes the entire upsert to raise `ValueError`; no rows from the batch are committed. Partial state would surface as cross-scope leaks or silent retrieval misses on subsequent queries.

See `.claude/skills/vector-contract.md` for the full contract reference and the path to writing a custom backend.

---

## Canonical Retrieval Pipeline

Production retrieval runs in two phases: an **iterative coverage loop** (RLM) that determines what to retrieve, followed by a **single-pass ranking and rendering pipeline** that shapes the result.

### Phase 1 — RLM iterative retrieval loop

The RLM controller (`uma/retrieve/rlm/controller.py`) drives a bounded multi-step loop. Each step evaluates coverage of what has already been collected, then deterministically decides what to retrieve next, until coverage thresholds are met or hard budgets are reached.

```
1. Baseline retrieval
   Initial vector + lexical search across all active lanes for the query.
   All results are owner-scoped via the C1 isolation contract.

2. Coverage assessment  [repeats each step]
   Evaluate what has been found: semantic fact count, salience distribution,
   cluster summaries, graph nodes, novelty trend across recent steps.
   Produces a CoverageReport with an `enough` flag and novelty signals.

3. Stop / continue decision  [deterministic]
   should_stop() checks coverage confidence against thresholds,
   hard limits (max_steps, max_env_calls, token_budget), and recall intent.
   If satisfied → exit loop. If not → continue.

4. Navigation decision  [deterministic]
   deterministic_decision() inspects the current pack state and produces
   the next action: expand a specific predicate (fetch_more_facts),
   broaden semantic search, fetch episodic clusters, expand the graph,
   or fall back to raw chunks if fact retrieval yields nothing.
   No LLM is involved in this decision.

5. Execute action
   The chosen action runs against the appropriate store. New results
   are merged into the ContextPack. Go to step 2.

6. Post-loop LLM fact pruning
   After the loop exits, _prune_facts_with_llm() uses the LLM to discard
   facts not relevant to the query. This is the only step where the LLM
   participates in retrieval. Skipped when query scan severity is medium/high.
```

Hard limits: `max_steps` (default 4), `max_env_calls` (default 12), `timeout_s` (default 20s). All are configurable via `retrieval.rlm` in `uma.yaml`.

### Phase 2 — single-pass ranking and rendering pipeline

Once the RLM loop has collected the candidate set, all production retrieval follows this exact sequence:

```
1. Boundary scan on query_text
   scan_content(query_text) → severity flows to controller
   medium/high severity disables downstream LLM hops (snippet refiner, fact pruner)
   to prevent payload amplification

2. Candidate discovery
   dense vector search (top_k_dense) + optional lexical search (top_k_sparse)
   both owner-scoped via the C1 isolation contract
   quarantined records excluded at the store layer (AND quarantined_at IS NULL)

3. Fusion
   merge dense + lexical candidates via RRF or boost-on-overlap
   produces a single ranked candidate pool

4. Optional rerank
   reorders within the pool only — never expands it

5. Trust adjustment
   final_score = (1 - trust_weight) * existing_score + trust_weight * trust_score
   candidates with trust_score < min_trust_score are dropped before truncation

6. Selection
   deterministic truncation to max_chunks / max_facts

7. Snippet rendering
   merge adjacency, bound length, preserve source traceability
   LLM-driven refinement skips chunks with medium/high injection severity
```

Ranking logic lives only in `uma/retrieve/ranking.py`. It is never duplicated in stores, controllers, or rendering layers. Security primitives set `trust_score` and `quarantined_at` at write time; the retrieval pipeline consumes them at read time.

Every retrieval call is recorded in the audit log.

---

## Canonical Ingestion Pipeline

```
UMAMemory.ingest_document(file_path, owner_type, owner_id, tenant_id, ...)
  → MIME consistency check (raises MimeRejection)
  → file size check (raises FileSizeRejection if > max_file_bytes; default 50 MB)
  → parse + PDF page count cap (raises if > pdf_max_pages; default 5000)
  → ingest_service.capture_source
    → manifest gate (source_hash + owner; skip if unchanged)
      · same tenant/owner/source_path with a different content hash creates a new manifest version
      · the new manifest records `supersedes=<prior_doc_id>`
      · the prior manifest records `superseded_by=<new_doc_id>` and `superseded_at=<utc timestamp>`
    → chunk document; HTML/Markdown sanitization at chunk boundary
    → per-chunk injection scan; high-severity chunks marked quarantined
    → embed chunks (strict=True; raises on any batch failure)
    → persist chunks to SQL + vector store (C1 isolation contract)
    → write manifest (only after embed succeeds)
  → derive_memory_artifacts
    → extract semantic facts (LLM-driven; salience-gated)
    → quarantined chunks dropped before fact extraction (injected content cannot seed the semantic lane)
    → update graph if enabled
  → curate_compiled_memory
```

**The manifest is always written after embedding succeeds.** If embedding raises, no manifest is recorded and re-ingest is safe.

`tenant_id` is a parameter on every public method and defaults to `DEFAULT_TENANT_ID` (`"default"`) when the caller omits it. UMA Lite runs a single tenant, but the value is carried explicitly alongside `agent_id` and `user_id` and written on every durable row — it is never inferred at the storage boundary. Ownership stays the caller's concern — empty `owner_type` / `owner_id` raises `ValueError`.

HTML and Markdown chunks pass through `_sanitize_html` in `uma/ingest/parser.py`, stripping `<script>` / `<iframe>` tags, inline event handlers, `javascript:` / `data:` URLs, conditional comments, and inline SVG. Per-category removal counts are recorded in `meta["security"]["sanitization"]` on the document manifest.

---

## Ingestion Path: `process_turn`

Conversation turns flow through `MemoryPipeline.process_turn`:

```
UMAMemory.process_turn(user_id, user_msg, assistant_reply, session_id, tenant_id)
  → rate-limit hook (if set)
  → injection scan on user_msg (raises InjectionDetectedError on high severity)
  → append to working memory
      · each message scanned at write time
      · quarantined messages retained in buffer but filtered from get_context
  → index as episodic event
      · summarizes current turn (user_msg + assistant_reply) only
      · prior working memory is background context in the prompt, not summarized
      · assistant_reply scanned at write time in EpisodicCore.store_episode
      · episode trust_score = 0.8 (UMA-synthesised turn summary; between user 0.9 and assistant 0.7)
      · session_id stored as provenance; episodes retrievable cross-session
  → store raw turn chunks (each chunk scanned at write time)
  → extract semantic facts (session-local)
      · user_msg  → facts with trust_score = 0.9 (turn_user)
      · assistant_reply → facts with trust_score = 0.7 (turn_assistant)
      · both sets deduplicated before storage
  → optional: promote facts to durable memory (explicit policy)
```

`session_id` is required and must be a non-empty string. Pass `skip_scan=True` to bypass the Layer-2 entry scan (the downstream per-artifact scans still run).

---

## Operational Hardening

### Rate-Limit Hook

UMA exposes a single optional hook for SDK-level throttling. The hook fires at the top of `retrieve_context`, `retrieve_memory`, `process_turn`, and `ingest_document`. The hook raises to refuse; returning normally allows the call. Both sync and async hooks are supported.

```python
def hook(operation, ctx):
    if too_many_calls(ctx):
        raise RuntimeError(f"rate limit hit on {operation}")

memory.set_rate_limit_hook(hook)
```

UMA does not ship a default rate limiter — operators integrate with their existing throttling stack. `ctx` is a `RuntimeContext` for the three user-scoped operations; `ctx=None` for `ingest_document` (the API takes `owner_type`/`owner_id`, not a user scope).

### Resource Caps at Ingest

| Setting | Default | Effect |
|---|---|---|
| `IngestConfig.max_file_bytes` | 52428800 (50 MB) | Files larger raise `FileSizeRejection` before being opened |
| `IngestConfig.pdf_max_pages` | 5000 | PDFs declaring more pages raise before parsing |
| `IngestConfig.allow_empty_pages` | false | Reject documents with zero extractable pages |

### Score Normalization

LanceDB raw distances are mapped to scores in `(0, 1]` via `exp(-distance)` so that the trust-weighted blend `(1 - trust_weight) * similarity + trust_weight * trust` operates on coherent scales.

---

## Intent Routing in `retrieve_context`

The retrieval planner classifies the query and selects lanes:

| Intent | Lanes activated |
|--------|----------------|
| `TOPICAL` | `raw` + `semantic` |
| `PERSONAL` | `profile` + `procedural` + `semantic` + `episodic` |
| `MIXED` | `raw` + `semantic` + `profile` + `procedural` + `episodic` |
| History markers | `raw` + `episodic` + `semantic` |

PERSONAL intent requires both a first-person marker (`I`/`me`/`my`) and a personal-state cue (`prefer`, `like`, `own`, etc.). Instructional queries ("how do I...") are TOPICAL. History markers (e.g. "last time", "yesterday", "what happened") activate the episodic lane regardless of intent class.

---

## Public API Surface

```python
from uma import UMAMemory, InjectionDetectedError
from uma.api.management import (
    explain_result,
    lint_memory_drift,
    verify_integrity,
    list_quarantined,
    reinstate_quarantined,
    purge_quarantined,
    list_retrieval_audit,
    consolidate,
)

# Initialize
# Replace with the path to your uma.yaml
memory = UMAMemory.from_yaml("/path/to/your/uma.yaml")
# One instance serves every agent and every user. agent_id and user_id are
# required on each call; tenant_id defaults to "default" when omitted.

# Optional: pre-LLM injection gate (never raises)
scan = memory.scan_user_input(user_msg)
if scan["severity"] == "high":
    return safe_rejection_response()

# Retrieve
context = await memory.retrieve_context(query_text=..., agent_id=..., user_id=..., session_id=..., tenant_id=...)
result  = await memory.retrieve_memory(query_text=..., agent_id=..., user_id=..., session_id=..., tenant_id=...)
# result keys: compiled_memory, facts, evidence, provenance_valid
# facts: [{text, confidence, salience, source_chunk_ids}] — text is full "subject predicate object" triple
# evidence: [{id, text, source, source_document_id}]

# Ingest — raises InjectionDetectedError on high-severity user_msg
await memory.process_turn(agent_id=..., user_id=..., user_msg=..., assistant_reply=..., session_id=..., tenant_id=...)
await memory.process_turn(..., skip_scan=True)  # bypass Layer-2 entry scan
report = await memory.ingest_document(file_path, owner_type="user", owner_id=..., tenant_id=...)

# Opt one agent into profile-gated, copy-based promotion
await memory.set_agent_profile(agent_id=..., description=..., focus_areas=[...], tenant_id=...)
profile = await memory.get_agent_profile(agent_id=..., tenant_id=...)

# Bootstrap
await memory.load_memory_bootstrap("MEMORY.md", agent_id=..., user_id=..., tenant_id=...)
await memory.load_daily_diary_bootstrap("diary.md", agent_id=..., user_id=..., tenant_id=...)

# Operational
memory.set_rate_limit_hook(my_hook)
health = memory.health_check()
await memory.rebuild_vector_indexes(tenant_id="default", ...)

# Inspection and provenance
explanation = await explain_result(memory, result, user_id=...)
drift       = await lint_memory_drift(memory, artifact, user_id=...)

# Integrity and quarantine management
ver  = await verify_integrity(memory, record_id="fact-abc", lane="semantic", ...)
recs = await list_quarantined(memory, lane="semantic", owner_type="user", owner_id=..., tenant_id=...)
await reinstate_quarantined(memory, lane="semantic", record_id="fact-abc", ...)
await purge_quarantined(memory, lane="semantic", record_id="fact-abc", ...)

# Retrieval audit
audit = await list_retrieval_audit(memory, tenant_id="default", user_id=..., limit=100)
```

---

## Runtime Profile

UMA Lite uses a single embedded profile: SQLite (authoritative) + LanceDB (vector index). No external database services are required.

| Config | Use |
|--------|-----|
| `config/uma.yaml` | Convention — any accessible path works |

LLM and embedding values in `uma.yaml` are user-customizable baselines — set provider, model, and host to match your environment. All initialization goes through the same path: `UMAMemory.from_yaml(path)`. See [`configure.md`](.claude/skills/configure.md) for the install extras needed per provider.

---

## Status

UMA is in **beta**. Schema and API may change. No backward-compatibility guarantees are made; the codebase deliberately removes obsolete paths rather than preserving them.

For developer-facing documentation, see the nine Agent Skills under `.claude/skills/`:

- `overview.md` — orientation, philosophy, DAT invariants
- `api.md` — full public API reference
- `lanes.md` — six public filter lanes plus the profile and optional graph planner/plugin views
- `configure.md` — YAML configuration reference
- `security.md` — full security model
- `agent-loop.md` — end-to-end integration pattern
- `promotion.md` — public profile-gated promotion contract
- `vector-contract.md` — the C1 vector contract
- `quarantine.md` — quarantine lifecycle
