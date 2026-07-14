---
name: uma-lanes
description: Explains UMA's six memory lanes (working memory, semantic facts, raw chunks, episodic, procedural, wiki) — what each one stores, its retrieval role, its storage contract, session-vs-durable scope, quarantine semantics, and the canonical retrieval pipeline that all production retrieval follows. Use this skill when answering questions about which lane to use, how lanes differ, how to filter by lane, what `lane_filter` accepts, how candidates flow through fusion/rerank/trust-adjustment/selection, or any question about chunk metadata and the SQL-vs-vector storage split.
---

# UMA — Memory Lanes

## Overview

UMA stores and retrieves memory across eight typed lanes. Each lane has a distinct responsibility, storage contract, and retrieval role. Lane selection is explicit — ownership scope alone does not distinguish which lane to query.

`profile` and `semantic` share the same SQLite store and vector index. `profile` is a distinct retrieval lane at query time (the planner selects it independently) but physically both read from `semantic_sql` filtered by `kind`. `trace` (`decision_trace`) exists as a kind in `KB_LANES` but is debug metadata only — the planner unconditionally excludes it from retrieval.

---

## The Six Lanes

### 1. Working Memory (`working_memory`)

**What it is:** Short-term conversational continuity buffer.

**Role:** Holds recent messages within a session for immediate context injection.

**Storage:** In-memory buffer managed by `WorkingMemoryCore`. Backed by session-scoped state; not persisted across sessions.

**Security:** Each appended message is injection-scanned before persistence. High-severity messages are kept in the buffer with `quarantined_at` set, but `get_context` filters them out by default (pass `include_quarantined=True` to include).

**Config (in `uma.yaml`):**
```yaml
working_memory:
  max_tokens: 4096
  keep_recent_messages: 4
  keep_recent_token_fraction: 0.1
```

**When used:** Always included in `retrieve_context` by default (`include_working_memory: true`). Session-scoped; tied to `session_id`.

---

### 2. Semantic Facts (`semantic`)

**What it is:** Structured statements extracted from chunks and conversation turns.

**Role:** Preferred "truth layer." Facts are the distilled, structured knowledge extracted from raw evidence. Chunks are the evidence; facts are the conclusions.

**Storage:** `SemanticStore` (SQLite via `semantic_sql.py`) + vector index for retrieval.

**Extraction:** LLM-driven extraction during ingest; salience threshold gates what is stored.
```yaml
semantic:
  salience_threshold: 0.45  # facts below this are dropped
```

**Trust scoring:**
- Facts from `user_msg` → `trust_score=0.9` (user said it directly)
- Facts from `assistant_reply` → `trust_score=0.7` (assistant may synthesize or hallucinate)
- Reduced by 20% (low) / 50% (medium) injection severity hit at write time

**Conflict resolution:** When upserting a fact that already exists by `(subject, predicate)`, `LatestWinsFactResolver` chooses the canonical row by `latest non-quarantined updated_at`. Quarantined facts are excluded from consideration; the all-quarantined case falls back across the full set with a warning and the chosen row remains quarantined.

**Scope behavior:** Facts extracted from turns are **session-local by default**. Must be explicitly promoted to become durable `user`, `workspace`, or `agent` memory.

**Retrieval:**
```python
retrieve_context(..., lane_filter=["semantic"])
```
Up to `max_facts: 5` per retrieval by default. Filtered by `quarantined_at IS NULL`.

---

### 3. Raw Evidence Chunks (`raw`)

**What it is:** Authoritative source text from ingested documents and conversations.

**Role:** Immutable evidence. The original source from which facts are extracted and wiki pages are compiled. All other artifacts trace back to chunks.

**Storage:** `ChunkStore` (SQLite via `chunk_sql.py`) is **authoritative** for chunk text and metadata. Vector store is a rebuildable accelerator that holds only ids, vectors, isolation columns, and filterable metadata — not full text.

**Required chunk metadata:**
- `id`, `doc_id`, `text`, `position`, `page_range`
- `tenant_id`, `owner_type`, `owner_id` (refused if empty)
- `source_uri` / hash (if available)
- `trust_score`, `content_hash`, `quarantined_at`

**Chunking rules:**
- Never cut mid-sentence
- Prefer paragraph-level chunks
- Minimum ~80 characters per chunk
- Overlap must align to sentence boundaries

**Security at ingest:**
- Every chunk text is injection-scanned at write time
- High-severity chunks are stored with `quarantined_at` set and excluded from retrieval
- Quarantined chunks are also dropped before fact extraction so injected content never seeds the semantic lane

**Retrieval:** Candidate discovery via dense vector search + optional lexical search (hybrid fusion). Final ranking via `uma/retrieve/ranking.py` — the single canonical ranking module.

---

### 4. Episodic Memory (`episodic`)

**What it is:** Time-ordered memory of interactions and ingest events.

**Role:** Captures *when* things happened and what the context was. Enables temporal reasoning over agent/user history.

**Storage:** `EpisodicStore` (SQLite via `episodic_sql.py`) + vector index via `EpisodeIndexer`.

**Episode shape:** Built from the current turn only (`user_msg` + `assistant_reply`). Prior working memory is available to the LLM as background context for coherent summarization, but is not re-summarized on every turn.

**Security:** `EpisodicCore.store_episode` scans `assistant_reply` at write time (the assistant_reply trust starts at 0.7). High-severity hits quarantine the episode.

**Scope behavior:** Episodes are retrievable **across sessions** — `session_id` is stored as provenance metadata, not as a retrieval gate.

**Retrieval:**
```python
retrieve_context(..., lane_filter=["episodic"])
```
Up to `max_episodic: 2` per context retrieval by default; up to `max_episodes: 3` in memory retrieval.

---

### 5. Procedural Memory / Skills (`procedural`)

**What it is:** Named skills, routines, and how-to knowledge.

**Role:** Stores procedures that an agent can recall and apply. Distinct from facts (what is true) — procedural memory encodes how to do things.

**Storage:** `ProceduralStore` (SQLite via `procedural_sql.py`) + vector index via `SkillIndexer`.

**Feature loading:** Procedural memory is loaded as an optional feature:
```yaml
features:
  load:
    - name: procedural
      enabled: true
      provider: "uma.memory.procedural.feature:ProceduralFeature"
```

**Validation:** `Skill.owner_type` and `Skill.owner_id` must be non-empty strings (`_validate_skill` refuses missing values). This matches the C1 vector contract; the SQL write and vector upsert cannot disagree.

**Retrieval:**
```python
retrieve_context(..., lane_filter=["procedural"])
```
Up to `max_procedural: 2` per context retrieval. Up to `max_skills: 3` in memory retrieval.

---

### 6. Compiled Wiki Pages (`wiki`)

**What it is:** Mutable, evidence-backed compiled knowledge artifacts.

**Role:** The synthesis layer. Wiki pages are compiled from chunks, facts, and episodes into durable continuity artifacts. Unlike raw chunks (immutable evidence), wiki pages are updated as knowledge evolves.

**Storage:** Stored as normal documents in UMA with `kind="wiki_page"`, `kb_lane="wiki"`, `page_slug`, `page_title`, `category`, `status` metadata.

**Key distinction:** `wiki/*.md` files on disk are a **projection only** — human-readable, git-friendly, Obsidian-compatible. The in-database record is the canonical source of truth.

**Management:**
```python
from uma.api.management import lint_memory_drift

# Check for provenance drift + integrity
await lint_memory_drift(memory, artifact, user_id=..., stale_after_seconds=86400)
```

---

## Storage Model Summary

| Lane | Authoritative Store | Accelerator | Write-time scan | Read-time quarantine filter |
|---|---|---|---|---|
| Working Memory | In-memory buffer (session-scoped) | — | ✅ | ✅ |
| Semantic | SQLite (`semantic_sql.py`) | Vector index | ✅ | ✅ |
| Profile | SQLite (`semantic_sql.py`, `kind=profile_fact`) | Vector index (shared with semantic) | ✅ | ✅ |
| Raw Chunks | SQLite (`chunk_sql.py`) | Vector index | ✅ | ✅ |
| Episodic | SQLite (`episodic_sql.py`) | Vector index | ✅ | ✅ |
| Procedural | SQLite (`procedural_sql.py`) | Vector index | ✅ | ✅ |
| Wiki | SQLite (document store) | Vector index | — | n/a |
| Graph (optional) | Graph backend (plugin) | — | — | — |
| Trace | SQLite (debug metadata only) | — | — | — (never retrieved) |

**Security primitives:** Each artifact (fact, episode, skill, chunk) carries `trust_score` (float, default 0.5) and `content_hash` (SHA-256 hex, where applicable). Write-time scanning implements the **Pre-Write Sanitization** layer of UMA's defense-in-depth model for memory poisoning (ASI06): every storage boundary scans content via `scan_artifact_text` before persisting. Ingested files pass MIME consistency (`mime_check.enforce_mime_consistency`) before parsing; HTML/Markdown is sanitized via `_sanitize_html` with per-category counts recorded in `meta["security"]["sanitization"]`. The `trust_score` and `source_chunk_ids` fields implement the **Provenance Tracking** layer. The `tenant_id`/`owner_type`/`owner_id` enforcement implements the **Memory Isolation** layer.

**Invariant:** SQL is always authoritative. Vector stores are rebuildable from SQL at any time:
```python
await memory.rebuild_vector_indexes(tenant_id="default", ...)
```

---

## Canonical Retrieval Pipeline

All production retrieval follows this exact sequence:

1. **Candidate discovery** — dense vector search (`top_k_dense`) + optional lexical search (`top_k_sparse`), both owner-scoped; quarantined records excluded at the store layer
2. **Fusion** — merge dense + lexical candidates (RRF or boost-on-overlap) into a single candidate pool
3. **Optional rerank** — reorder within the candidate pool only; never expands the pool
4. **Trust adjustment** — `final_score = (1 - trust_weight) * existing + trust_weight * trust_score`; candidates below `min_trust_score` (default 0.5) are dropped before truncation
5. **Selection** — deterministic truncation to `max_chunks` / `max_facts`
6. **Snippet rendering** — presentation layer: merge adjacency, bound length, preserve traceability. Skips LLM refinement on chunks with medium/high injection severity.

**Policy:** Ranking logic lives only in `uma/retrieve/ranking.py`. Never in stores, snippet rendering, or controller layers.

**Score normalization (LanceDB):** Vector distances are mapped to scores via `exp(-distance)`, producing values in `(0, 1]` that are coherent with `trust_score` in `[0, 1]` for the weighted blend.

---

## Lane Ownership Scoping

Every lane query is owner-scoped. The scope fields required per call:

```python
tenant_id="default"      # required; isolates by tenant
user_id="user-123"       # required for user-scoped lanes
agent_id="agent-default" # bound via set_context(); required
session_id="session-1"   # required for session-local lanes
```

Cross-tenant access is impossible by construction:

- The vector index (LanceDB) promotes `tenant_id` / `owner_type` / `owner_id` to first-class columns and pushes them into every query's `WHERE` clause before the candidate cap is applied.
- SQL stores filter by `tenant_id AND owner_type AND owner_id` in every read path.

Cross-agent sharing requires explicit scope widening.

---

## Lane Filter Usage

`retrieve_context` accepts `lane_filter` to query specific lanes only:

```python
# RAG over raw evidence only
context = await memory.retrieve_context(
    query_text=..., user_id=...,
    lane_filter=["raw", "semantic"]
)

# Episodic-only recall
context = await memory.retrieve_context(
    query_text=..., user_id=...,
    lane_filter=["episodic"]
)
```

Valid lane names: `raw`, `semantic`, `episodic`, `procedural`, `wiki`, `working_memory`.
