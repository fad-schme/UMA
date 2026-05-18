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

`set_context` binds the fixed agent identity to the instance. It is not per-request state — use it once at startup. Returns `self` for chaining.

---

## Core Retrieval APIs

### `retrieve_context` — RAG-style context retrieval

```python
context = await memory.retrieve_context(
    query_text="the user's current question or task",
    user_id="user-123",
    tenant_id="default",          # required; defaults to "default"
    request_id="req-1",           # optional; auto-generated if omitted
    session_id="session-1",       # optional
    workspace_id=None,            # optional
    lane_filter=["raw", "semantic"],  # optional; narrows which lanes are queried
)
```

**Contract:**
- Intended for LLM context assembly, not durable memory projection
- Returns an evidence-oriented context bundle (chunks, facts, skills, working memory)
- `lane_filter` narrows retrieval to specific lanes without requiring wiki state
- All results are source-traceable through `doc_id`, `chunk_ids`, `page_range`

**Returns:** `Dict[str, Any]` — context pack with `snippets`, `facts`, `working_memory`, `meta`

**Valid lane names:** `raw`, `semantic`, `episodic`, `procedural`, `wiki`, `working_memory`

---

### `retrieve_memory` — Compiled memory retrieval

```python
result = await memory.retrieve_memory(
    query_text="user's question",
    user_id="user-123",
    tenant_id="default",
    request_id="req-1",
    session_id="session-1",
    memory_intent="continuity",   # default; controls retrieval strategy
)
```

**Contract:**
- `memories` is the primary compiled-memory field in the result
- `evidence` is mandatory and attached to every result path
- Does NOT degrade silently into plain chunk retrieval — `fallback` field signals degradation
- Supporting facts/skills are secondary evidence, not the product identity

**Returns:** `Dict[str, Any]` — compiled memory result with `memories`, `evidence`, `product`, `fallback`

---

## Core Ingestion APIs

### `process_turn` — Persist a conversation turn

```python
await memory.process_turn(
    user_id="user-123",
    user_msg="what the user said",
    assistant_reply="what the assistant replied",
    session_id="session-1",       # required; must be non-empty
    tenant_id="default",
    workspace_id=None,
    extra_meta={"custom": "data"},  # optional
)
```

**Contract:**
- `session_id` is required — raises `ValueError` if missing or empty
- Persists working memory, extracts episodic events, extracts semantic facts
- Session-local facts remain session-scoped until explicitly promoted
- Triggers the canonical `MemoryPipeline` internally (lazy-initialized on first call)

---

### `ingest_document` — Ingest a document file

```python
result = await memory.ingest_document(
    file_path="/path/to/document.pdf",
    owner_type="user",      # optional; one of: agent, user, workspace, system
    owner_id="user-123",    # optional
    config=None,            # optional IngestConfig override
)
```

Chunks, embeds, and indexes the document through the canonical UMA ingest pipeline.

---

## Animus Bootstrap APIs (Animus integration)

```python
memory.load_userprofile("USER.md")      # loads user profile into in-memory cache
memory.load_agentprofile("SOUL.md")     # loads agent profile (soul) into cache

result = await memory.load_memory_bootstrap(
    "MEMORY.md",
    user_id="user-123",
    tenant_id="default",
    session_id="session-1",
)

result = await memory.load_daily_diary_bootstrap(
    "diary.md",
    user_id="user-123",
    session_id="session-1",
)
```

---

## Health and Maintenance

```python
status = memory.health_check()  # sync; returns {"status": "ok"|"error", "checks": {...}}

# Rebuild vector indexes from authoritative SQL data
result = await memory.rebuild_vector_indexes(
    tenant_id="default",
    owner_type="user",
    owner_id="user-123",
    include_episodic=True,
    include_semantic=True,
    include_procedural=True,
    batch_size=32,
)

# Rebuild both vector and graph indexes
result = await memory.rebuild_derived_indexes(...)

memory.shutdown()  # release backend connections
```

---

## Management APIs (`uma.api.management`)

These keep inspection, curation, and drift checks off the main `UMAMemory` surface.

```python
from uma.api.management import explain_result, lint_memory_drift
```

### `explain_result` — Inspect provenance of a retrieval result

```python
explanation = await explain_result(
    memory,
    result,          # output from retrieve_memory or retrieve_context
    user_id="user-123",
    tenant_id="default",
)
# Returns: artifact_id, provenance, evidence, chunk_ids, lineage, conflicts, wiki_page
```

### `lint_memory_drift` — Check artifacts for provenance drift

```python
result = await lint_memory_drift(
    memory,
    artifacts,          # single artifact or list
    user_id="user-123",
    tenant_id="default",
    stale_after_seconds=86400,
)
# Returns: status, artifacts_scanned, findings, drift_statuses
```

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
