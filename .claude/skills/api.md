---
name: uma-api
description: Full public API reference for UMA — every UMAMemory method, immutable agent scoping, promotion profiles, management functions, scope fields, security entry points, and rate-limit hook registration. Use this skill when answering questions about method signatures, return shapes, required arguments, promotion, or any management or quarantine API.
---

# UMA — Public API Reference

## Initialization

```python
from uma import UMAMemory

# Pass any path — absolute or relative to where your application runs.
# "config/uma.yaml" is a convention, not a requirement.
runtime = UMAMemory.from_yaml("/path/to/your/uma.yaml")
```

`from_yaml` accepts any path — absolute or relative to the working directory of the running process. `"config/uma.yaml"` is the convention used throughout the examples, but the file can live anywhere accessible to your application at runtime (e.g. `"./uma.yaml"`, `"/etc/myapp/uma.yaml"`, or any other location you choose). There is one initialization path — no `init_lite()` or `init_cont()` variants.

### Identity is per call, never per instance

UMA is **single-tenant, multi-agent and multi-user**. One `UMAMemory` instance
serves every agent and every user in the process; identity travels with each
call:

| Field | Required | Default |
| --- | --- | --- |
| `agent_id` | yes | none — omitting it raises `ValueError` |
| `user_id` | yes on user-scoped APIs | none — omitting it raises `ValueError` |
| `tenant_id` | no | `"default"` (`DEFAULT_TENANT_ID`) |

```python
context_a = await memory.retrieve_context(
    query_text="...", agent_id="agent-a", user_id="user-1",
)
context_b = await memory.retrieve_context(
    query_text="...", agent_id="agent-b", user_id="user-1",
)
```

There is no `set_context`, no bound `agent_id` attribute, and no ambient
scope. Two agents calling the same instance concurrently cannot see each
other's agent-owned rows, because each call builds its own `RuntimeContext`
from its own arguments.

`tenant_id` is a real parameter on every public method even though Lite runs
single-tenant: it is carried explicitly to storage next to `agent_id` and
`user_id`, and only *defaults* when the caller omits it — it is never inferred
at the storage boundary.

---

## Promotion Profile APIs

### `set_agent_profile` — Enable profile-gated promotion

```python
profile = await memory.set_agent_profile(
    agent_id="agent-default",
    description="An infrastructure assistant focused on Kubernetes operations",
    focus_areas=["kubernetes", "containers", "incident response"],
)
```

The profile opts this agent into built-in promotion. Facts extracted by
`process_turn` remain session-local unless they pass quarantine, confidence,
salience, source-safety, and profile-scope gates. Promotion is copy-based and
preserves provenance; the source fact is never widened in place.

### `get_agent_profile` — Read the configured profile

```python
profile = await memory.get_agent_profile(agent_id="agent-default")
if profile is None:
    # No profile means automatic promotion is a no-op.
    ...
```

Promotion runs in a bounded background task after turn extraction. It never
blocks the reply path, and a promotion failure does not fail `process_turn`.
See `promotion.md` for the complete gate and ownership contract.

---

## Core Retrieval APIs

### `retrieve_context` — RAG-style context retrieval

```python
context = await memory.retrieve_context(
    query_text="the user's current question or task",
    agent_id="agent-default",      # REQUIRED
    user_id="user-123",            # REQUIRED
    tenant_id="default",           # optional; defaults to "default"
    request_id="req-1",            # optional; auto-generated if omitted
    session_id="session-1",        # optional
    workspace_id=None,             # optional
    lane_filter=["raw", "semantic"],  # optional
)
```

**Contract:**

- Intended for LLM context assembly, not durable memory projection
- Returns an evidence-oriented context bundle (chunks, facts, skills, working memory)
- `lane_filter` narrows retrieval to specific lanes; valid names: `raw`, `semantic`, `episodic`, `procedural`, `wiki`, `working_memory`
- The boundary scans `query_text` for injection patterns and propagates the severity to downstream LLM hops (snippet refiner, fact pruner) which skip amplification on medium/high. **High-severity queries are NOT blocked** — the scan is advisory; the caller still gets retrieval results, just without LLM-refined snippets.
- Every retrieval call is recorded in the retrieval audit log (toggle via `security.retrieval_audit_enabled`).

**Returns:** `Dict[str, Any]` — context pack with `snippets`, `facts`, `working_memory`, `meta`, and `query_scan_severity` ("none" | "low" | "medium" | "high").

---

### `retrieve_memory` — Compiled memory retrieval

```python
result = await memory.retrieve_memory(
    query_text="user's question",
    user_id="user-123",
    tenant_id="default",
    request_id="req-1",
    session_id="session-1",
    memory_intent="continuity",    # default
    include_debug=False,
)
```

**Contract:**

- `compiled_memory` is the primary field: `{status, summary, memory_intent, provenance_valid}`. `status="evidence_only"` when no wiki artifact exists.
- `facts` are full subject-predicate-object triples serialized as `text="subject predicate object"`. The predicate is never dropped.
- `evidence` is mandatory and attached to every result path.
- Does NOT degrade silently into plain chunk retrieval — `fallback` in `debug` signals degradation.

**Returns:** `Dict[str, Any]` with keys `compiled_memory`, `facts`, `evidence`, `provenance_valid`.

---

## Core Ingestion APIs

### `scan_user_input` — Pre-LLM injection gate

```python
scan = memory.scan_user_input(user_msg)
# {"severity": "none"|"low"|"medium"|"high", "matched_rules": [...], "score": float}
```

Call this at the **top of every agent turn**, before `retrieve_context` and before any LLM call. It is synchronous and never raises — the caller decides what to do with the result.

**Recommended use:**

```python
scan = memory.scan_user_input(user_msg)
if scan["severity"] == "high":
    return "I can't process that request."   # do not call retrieve_context / LLM
```

---

### `process_turn` — Persist a conversation turn

```python
from uma import InjectionDetectedError

try:
    await memory.process_turn(
        agent_id="agent-default",      # REQUIRED
        user_id="user-123",            # REQUIRED
        user_msg="what the user said",
        assistant_reply="what the assistant replied",
        session_id="session-1",        # REQUIRED; raises ValueError if missing
        tenant_id="default",           # optional; defaults to "default"
        workspace_id=None,
        extra_meta={"custom": "data"}, # optional
        skip_scan=False,               # see below
    )
except InjectionDetectedError as e:
    # e.severity == "high", e.matched_rules, e.score
    handle_security_event(e)
```

**Contract:**

- `session_id` is required — raises `ValueError` if missing or empty
- Re-scans `user_msg` at the storage boundary (Layer 2 of UMA's defense-in-depth model for memory poisoning)
- On `severity == "high"`, raises `InjectionDetectedError` and **nothing is stored** — no working memory, no episode, no chunks, no facts
- On `low` / `medium`, logged + trust-reduced (by 20% / 50%) and storage proceeds
- Persists working memory, extracts an episode from the current turn, extracts semantic facts (session-local by default until promoted)
- `skip_scan=True` bypasses the gate when the caller has independently validated the input

| Severity | Behaviour | Artifact trust |
|---|---|---|
| `none` | Proceed normally | Unchanged |
| `low` | Logged, proceeds | Reduced by 20% |
| `medium` | Logged, proceeds | Reduced by 50% |
| `high` | Raises `InjectionDetectedError`; turn dropped | Not stored |

---

### `ingest_document` — Ingest a document file

```python
report = await memory.ingest_document(
    file_path="/path/to/document.pdf",
    agent_id="agent-default",   # REQUIRED — the agent performing the ingest
    owner_type="user",          # required — durable owner, not the agent
    owner_id="user-123",        # required
    tenant_id="default",        # optional; defaults to "default"
    workspace_id=None,
    config=None,                # optional IngestConfig override
)

Documents are owner-scoped, so there is no `user_id` here. `agent_id` records
which agent ingested the document; it never substitutes for the owner tuple.
```

Chunks, embeds, and indexes the document through the canonical pipeline.

**Gating:**

- File size limit: `IngestConfig.max_file_bytes` (default 50 MB) — raises `FileSizeRejection` if exceeded
- MIME consistency: rejects executables, extension/content mismatches — raises `MimeRejection`
- PDF page count limit: `IngestConfig.pdf_max_pages` (default 5000)
- HTML/Markdown chunks are sanitized (`<script>`, `<iframe>`, inline event handlers, `javascript:` / `data:` URLs, conditional comments stripped)
- Each chunk is injection-scanned at write time; high-severity chunks are quarantined and excluded from fact extraction

---

## Bootstrap APIs

Seed a store from markdown before the first turn. Both require an explicit
`agent_id` and `user_id`; `tenant_id` defaults.

```python
await memory.load_memory_bootstrap(
    "MEMORY.md",
    agent_id="agent-default",
    user_id="user-123",
    tenant_id="default",
    session_id="session-1",
)

await memory.load_daily_diary_bootstrap(
    "diary.md",
    agent_id="agent-default",
    user_id="user-123",
    session_id="session-1",
)
```

---

## Rate-Limit Hook (M6)

UMA exposes a single optional hook that fires at the top of each expensive public method:

```python
def my_hook(operation: str, ctx):
    # operation ∈ {"retrieve_context", "retrieve_memory", "process_turn", "ingest_document"}
    # ctx is a RuntimeContext (or None for ingest_document)
    if too_many_calls(ctx):
        raise RateLimitExceeded(f"rate limit hit on {operation}")

# Both sync and async hooks supported — UMA detects coroutine returns
async def my_async_hook(operation, ctx):
    await my_redis_check(ctx.tenant_id, operation)

memory.set_rate_limit_hook(my_hook)        # returns self for chaining
memory.set_rate_limit_hook(None)           # clear
```

The hook **raises to refuse**. Returning normally allows the call. UMA does not ship a default rate limiter — operators integrate with their existing throttling stack (Redis, Envoy, in-process counter, etc.).

---

## Health and Maintenance

```python
status = memory.health_check()  # sync; {"status": "ok"|"error", "checks": {...}}

# Rebuild vector indexes from authoritative SQL data
# Maintenance is scoped by tenant plus the durable owner tuple. An agent's own
# rows are addressed as owner_type="agent", owner_id=<agent_id>.
await memory.rebuild_vector_indexes(
    tenant_id="default",
    owner_type="user",
    owner_id="user-123",
    include_episodic=True,
    include_semantic=True,
    include_procedural=True,
    batch_size=32,
)

memory.shutdown()  # release backend connections
```

---

## Management APIs (`uma.api.management`)

These keep inspection, curation, drift checks, and quarantine management off the main `UMAMemory` surface.

```python
from uma.api.management import (
    explain_result,
    lint_memory_drift,
    verify_integrity,
    list_quarantined,
    reinstate_quarantined,
    purge_quarantined,
    list_retrieval_audit,
)
```

### `explain_result` — Inspect provenance of a retrieval result

```python
explanation = await explain_result(
    memory, result,
    user_id="user-123",
    tenant_id="default",
)
# Returns: artifact_id, provenance, evidence, chunk_ids, lineage, conflicts, wiki_page
```

### `lint_memory_drift` — Check artifacts for provenance drift + integrity

```python
result = await lint_memory_drift(
    memory, artifacts,
    user_id="user-123",
    tenant_id="default",
    stale_after_seconds=86400,
)
# Returns: status, artifacts_scanned, findings, drift_statuses
```

Typed lane artifacts (Fact, Episode, Skill, Chunk) are automatically routed through `verify_integrity`. Integrity failures appear as findings with `category="integrity_failure"` and the tampered record is quarantined.

### `verify_integrity` — On-demand content hash verification

```python
from uma.api.management import verify_integrity

result = await verify_integrity(
    memory,
    record_id="fact-abc",
    lane="semantic",           # one of: semantic, episodic, procedural, raw
    owner_type="user",
    owner_id="user-123",
    tenant_id="default",
)
# result.status == "verified" | "failed"
# result.quarantined == True if mismatch detected and record was quarantined
```

Recomputes the canonical hash and compares to stored `content_hash` (or `meta["text_hash"]` for chunks). On mismatch, quarantines the record and appends an `integrity_failure` entry to the artifact's `meta.security.audit_log`.

### Quarantine management

```python
# List all quarantined records for a scope
records = await list_quarantined(
    memory,
    lane="semantic",            # or "episodic", "procedural", "raw"
    owner_type="user",
    owner_id="user-123",
    tenant_id="default",
    limit=100,
)

# Reinstate a quarantined record (clear quarantined_at)
await reinstate_quarantined(
    memory,
    lane="semantic",
    record_id="fact-abc",
    owner_type="user",
    owner_id="user-123",
    tenant_id="default",
)

# Permanently delete a quarantined record
await purge_quarantined(
    memory,
    lane="semantic",
    record_id="fact-abc",
    owner_type="user",
    owner_id="user-123",
    tenant_id="default",
)
```

Quarantined records stay in the database but are excluded from every retrieval query. See `quarantine.md` for the full lifecycle.

### `list_retrieval_audit` — Inspect retrieval audit log

```python
rows = await list_retrieval_audit(
    memory,
    tenant_id="default",
    user_id="user-123",
    limit=100,
    since_iso="2026-01-01T00:00:00Z",
)
# Each row: timestamp, operation, tenant_id, user_id, session_id,
#           query_hash, query_preview, scan_severity, result_count, llm_hops_skipped
```

Each retrieval call (`retrieve_context`, `retrieve_memory`) records a SHA-256 query digest (`query_hash`, for correlating one query across log lines), the first 80 characters of the query (`query_preview`, for human auditing), and metadata. The full query is never stored. The audit store is enabled by default; disable via `security.retrieval_audit_enabled: false` in your YAML.

---

## Scope Fields Reference

All retrieval and ingestion APIs require these fields:

| Field | Required | Description |
|---|---|---|
| `user_id` | Yes | Non-empty string; raises `ValueError` if missing |
| `tenant_id` | Yes | Defaults to `"default"` |
| `request_id` | No | Auto-generated as `request:{user_id}` if omitted |
| `session_id` | Context-dependent | Required for `process_turn`; optional for retrieval |
| `workspace_id` | No | Optional workspace scope |

Runtime scope is **never inferred from prior calls** — all fields must be passed explicitly at each call site.

---

## Errors

| Exception | When |
|---|---|
| `ValueError` | Missing required scope fields, empty isolation values, malformed inputs |
| `InjectionDetectedError` | `process_turn` detected high-severity injection in `user_msg` |
| `FileNotFoundError` | `ingest_document` path does not exist |
| `MimeRejection` | `ingest_document` rejected file by MIME consistency check |
| `FileSizeRejection` | `ingest_document` rejected file exceeding `max_file_bytes` |
| `TypeError` | Legacy adapter call shape (old `metadata=` / `filters=` kwargs on vector index) |
