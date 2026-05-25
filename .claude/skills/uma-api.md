---
name: uma-api
description: Full public API reference for UMA — every UMAMemory method, every management function, scope fields, security entry points, and rate-limit hook registration. Use this skill when answering questions about method signatures, return shapes, what arguments a UMA call requires, how to use `process_turn`/`retrieve_context`/`retrieve_memory`/`ingest_document`, how to register a rate-limit hook, or how to call any management or quarantine API.
---

# UMA — Public API Reference

## Initialization

```python
from uma import UMAMemory

memory = UMAMemory.from_yaml("config/uma.yaml")
```

`from_yaml` initializes retrieval synchronously (retrieval is instant after return) and schedules ingestion subsystems for background warmup. There is one initialization path — no `init_lite()` or `init_cont()` variants.

### Bind agent identity (optional, once per instance)

```python
memory = memory.set_context(agent_id="agent-default")
```

`set_context` binds the fixed agent identity to the instance. It is not per-request state — call it once at startup. Returns `self` for chaining.

---

## Core Retrieval APIs

### `retrieve_context` — RAG-style context retrieval

```python
context = await memory.retrieve_context(
    query_text="the user's current question or task",
    user_id="user-123",
    tenant_id="default",           # required; defaults to "default"
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
        user_id="user-123",
        user_msg="what the user said",
        assistant_reply="what the assistant replied",
        session_id="session-1",        # REQUIRED; raises ValueError if missing
        tenant_id="default",
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
- Re-scans `user_msg` at the storage boundary (defense in depth)
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
    owner_type="user",          # required
    owner_id="user-123",        # required
    tenant_id="default",        # defaults to "default"
    workspace_id=None,
    config=None,                # optional IngestConfig override
)
```

Chunks, embeds, and indexes the document through the canonical pipeline.

**Gating:**

- File size limit: `IngestConfig.max_file_bytes` (default 50 MB) — raises `FileSizeRejection` if exceeded
- MIME consistency: rejects executables, extension/content mismatches — raises `MimeRejection`
- PDF page count limit: `IngestConfig.pdf_max_pages` (default 5000)
- HTML/Markdown chunks are sanitized (`<script>`, `<iframe>`, inline event handlers, `javascript:` / `data:` URLs, conditional comments stripped)
- Each chunk is injection-scanned at write time; high-severity chunks are quarantined and excluded from fact extraction

---

## Animus Bootstrap APIs (Animus integration)

```python
memory.load_userprofile("USER.md")        # loads user profile into in-memory cache
memory.load_agentprofile("SOUL.md")       # loads agent profile (soul) into cache

await memory.load_memory_bootstrap(
    "MEMORY.md",
    user_id="user-123",
    tenant_id="default",
    session_id="session-1",
)

await memory.load_daily_diary_bootstrap(
    "diary.md",
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

Quarantined records stay in the database but are excluded from every retrieval query. See `uma-quarantine.md` for the full lifecycle.

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

Each retrieval call (`retrieve_context`, `retrieve_memory`) records a hashed query preview and metadata. The audit store is enabled by default; disable via `security.retrieval_audit_enabled: false` in your YAML.

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
