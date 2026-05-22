# ARCHITECTURE.md — UMA-RLM

UMA-RLM is a memory and context runtime SDK for AI agents. It ingests data, stores it across typed memory lanes, and exposes two thin retrieval products. It does not generate replies, perform reasoning, or manage tool use — that belongs to the calling application.

---

## Core Products

| API | Purpose |
|-----|---------|
| `retrieve_context(...)` | Evidence-oriented RAG context for LLM prompting |
| `retrieve_memory(...)` | Compiled, evidence-backed memory for continuity |

Both products are owned by `UMAMemory`, initialized from a config file, and operated with explicit per-call request scope.

---

## Memory Lanes

UMA stores and retrieves across six typed lanes. Lane selection is explicit — ownership scope alone does not determine which lane to query.

| Lane | Store | Role | Scope default |
|------|-------|------|---------------|
| Working Memory | In-memory buffer | Recent message continuity within a session | Session-local |
| Raw Chunks (`raw`) | SQLite + vector index | Immutable source evidence from ingested documents | Durable |
| Semantic Facts (`semantic`) | SQLite + vector index | Structured statements extracted from chunks/turns | Session-local; promotable |
| Episodic (`episodic`) | SQLite + vector index | Time-ordered interaction history | Session-local; promotable |
| Procedural / Skills (`procedural`) | SQLite + vector index | Named skills and how-to knowledge | Durable |
| Compiled Wiki (`wiki`) | SQLite (document store) | Mutable, evidence-backed synthesis artifacts | Durable |
| Graph (optional) | Plugin backend | Relationship routing and entity expansion | — |

Graph is disabled in all public profiles. It is a supporting lane for relationship traversal, not the primary truth.

Each artifact (fact, episode, skill, chunk) additionally carries `trust_score` (float, classifier-derived via `uma.common.trust.score_source`) and `content_hash` (SHA-256 hex, where applicable) as security primitives (OWASP ASI06 baseline). At every write boundary, `uma.common.injection_scan.scan_content` checks incoming text against the YAML pattern catalog; high-severity hits set `trust_score` to 0.0 and record the scan result in `meta["security"]["injection_scan"]`.

Each artifact also carries a nullable `quarantined_at` timestamp. When a high-severity scan hit is detected and `SecurityConfig.quarantine_enabled=True`, the artifact is stored with `quarantined_at` set to the current UTC time. Quarantined artifacts are excluded from all normal retrieval queries (`AND quarantined_at IS NULL`) but remain in the database. The management API (`uma.api.management`) exposes `list_quarantined`, `reinstate_quarantined`, and `purge_quarantined` to review, restore, or permanently delete quarantined records.

File ingestion validates caller inputs at `UMAMemory.ingest_document` (non-empty path, file must exist and be a regular file), checks byte-level MIME consistency via `uma/ingest/mime_check.py` before parsing (raising `MimeRejection` for executable types or extension/content mismatches), and sanitizes HTML and Markdown through `_sanitize_html` in `uma/ingest/parser.py` (stripping scripts, iframes, inline event handlers, javascript: and data: URLs, conditional comments, and inline SVG); per-category removal counts are stored in `meta["security"]["sanitization"]` on the document manifest.

On-demand integrity verification is available through `uma.api.management.verify_integrity(memory, record_id, lane, ...)`. It recomputes the canonical content hash for any stored Fact, Episode, Skill, or Chunk and compares it to the stored `content_hash` (or `meta["text_hash"]` for chunks). On mismatch, the record is quarantined through the same PR4 path (`quarantined_at` set, audit log entry appended) and an `IntegrityVerificationResult(status="failed", ...)` is returned. A clean record returns `status="verified"` without mutation. `lint_memory_drift` automatically routes typed lane artifacts through `verify_integrity` so batch integrity checks can be run without calling the function directly. Background scanning across the full dataset is an Enterprise capability and is not part of this SDK.

---

## Storage Model

```
.uma/db/
  chunks.db       — authoritative chunk text and metadata
  documents.db    — ingested document manifests
  episodic.db     — episodic event history
  semantic.db     — extracted facts
  procedural.db   — skills and routines

.uma/vectors/
  vectors_chunks
  vectors_episodic
  vectors_semantic
  vectors_procedural
```

**Invariant:** SQL is always authoritative. Vector stores hold only ids and filterable metadata — never full chunk text. They are rebuildable from SQL at any time:

```python
await memory.rebuild_vector_indexes(tenant_id="default")
```

---

## Ownership and Tenancy (DAT Invariants)

Every stored and retrieved artifact must carry explicit ownership. These invariants are non-negotiable.

| Field | Requirement |
|-------|-------------|
| `owner_type` | One of: `agent`, `user`, `workspace`, `system` |
| `owner_id` | Required; consistent with the lane |
| `tenant_id` | Required for durable artifacts; preserved end-to-end |
| `session_id` + `agent_id` | Required for session-local artifacts |

Cross-tenant access is impossible by construction. Cross-agent sharing is denied by default unless the artifact owner is intentionally broader than agent scope.

**Practical rule:** If you cannot answer "which user/agent/project is allowed to see this row?" the design is wrong.

---

## Runtime Scope Invariants

- No shared mutable object stores current request scope. Patterns like `memory.user_id` or `controller.current_scope` are forbidden.
- Every API entry point operates from an explicit, immutable `RuntimeContext` built at the call boundary.
- Working memory and episodic turns are session-local by default. Semantic facts extracted from turns are also session-local by default and must be explicitly promoted to become durable.

---

## Canonical Retrieval Pipeline

All production retrieval follows this exact sequence:

```
1. Candidate discovery
   dense vector search (top_k_dense) + optional lexical search (top_k_sparse)
   both owner-scoped; quarantined records excluded at the store layer (PR4)

2. Fusion
   merge dense + lexical candidates via RRF or boost-on-overlap
   produces a single ranked candidate pool

3. Optional rerank
   reorders within the pool only — never expands it

4. Trust adjustment (PR5)
   final_score = (1 - trust_weight) * existing_score + trust_weight * trust_score
   candidates with trust_score < min_trust_score are dropped before truncation

5. Selection
   deterministic truncation to max_chunks / max_facts

6. Snippet rendering
   merge adjacency, bound length, preserve source traceability
```

Ranking logic lives only in `uma/retrieve/ranking.py`. It is never duplicated in stores, controllers, or rendering layers. Trust-aware ranking is a retrieval concern; security primitives (PR1–PR4) set `trust_score` at write time, and the ranking module consumes it at read time.

---

## Canonical Ingestion Pipeline

```
UMAMemory.ingest_document
  → ingest_service.capture_source
    → manifest gate (source_hash + owner; skip if unchanged)
    → chunk document
    → embed chunks (strict=True; raises on any batch failure)
    → persist chunks to SQL + vector store
    → write manifest (only after embed succeeds)
  → derive_memory_artifacts
    → extract semantic facts (LLM-driven; salience-gated)
    → update graph if enabled
  → curate_compiled_memory
```

The manifest is always written after embedding succeeds. If embedding raises, no manifest is recorded and re-ingest is safe.

---

## Ingestion Path: `process_turn`

Conversation turns flow through `MemoryPipeline.process_turn`:

```
process_turn(user_id, user_msg, assistant_reply, session_id)
  → append to working memory
  → index as episodic event
  → extract semantic facts (session-local)
  → optional: promote facts to durable memory (explicit policy)
```

---

## Intent Routing in `retrieve_context`

The retrieval planner classifies the query and selects lanes:

| Intent | Lanes activated |
|--------|----------------|
| `TOPICAL` | `raw` + `semantic` |
| `PERSONAL` | `profile` + `procedural` |
| `MIXED` | All four lanes |
| History markers | `raw` + `episodic` + `semantic` |

PERSONAL intent requires both a first-person marker (`I`/`me`/`my`) and a personal-state cue (`prefer`, `like`, `own`, etc.). Instructional queries ("how do I...") are TOPICAL.

---

## Public API Surface

```python
from uma import UMAMemory
from uma.api.management import explain_result, lint_memory_drift

# Initialize
memory = UMAMemory.from_yaml("config/uma.yaml").set_context(agent_id="agent-default")

# Retrieve
context = await memory.retrieve_context(query_text=..., user_id=..., session_id=..., tenant_id=...)
result  = await memory.retrieve_memory(query_text=..., user_id=..., session_id=..., tenant_id=...)

# Ingest
await memory.process_turn(user_id=..., user_msg=..., assistant_reply=..., session_id=...)
report = await memory.ingest_document(file_path, owner_type="agent", owner_id=...)

# Bootstrap
await memory.load_memory_bootstrap("MEMORY.md", user_id=...)
await memory.load_daily_diary_bootstrap("diary.md", user_id=...)

# Management
explanation = await explain_result(memory, result, user_id=...)
drift       = await lint_memory_drift(memory, artifact, user_id=...)
health      = memory.health_check()
```

---

## Runtime Profile

UMA uses a single embedded profile: SQLite (authoritative) + LanceDB (vector index). No external database services are required.

| Config | Use |
|--------|-----|
| `config/uma.yaml` | Default runnable config |
| `config/uma_lite.yaml` | Reference embedded profile (same storage settings) |

LLM and embedding values in both files are user-customizable baselines — set provider, model, and host to match your environment. All initialization goes through the same path: `UMAMemory.from_yaml(path)`.

---

## MCP Integration

UMA ships an MCP server at `mcp/server.py` that exposes the core retrieval and ingestion operations as MCP tools. See `mcp/README.md` for installation and Claude Desktop configuration.
