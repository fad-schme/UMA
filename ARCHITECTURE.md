# UMA Architecture Index

Stable top-level package boundaries:

- `uma.api`
  Public entry points and caller-facing runtime surfaces.
  Keep `UMAMemory`, `UMARuntime`, request-bound handles, and explicit developer/admin management surfaces here.
  Product-facing memory behavior belongs on `UMAMemory`.
  Developer/debug and admin/internal memory management operations belong in `uma.api.management`.

- `uma.ingest`
  Source parsing, normalization, chunking, and ingest orchestration.
  Ingest is organized as three explicit stages: capture, derive, and curate.
  Capture produces normalized source records and raw chunks.
  Derive produces provenance-bearing memory artifacts from chunks.
  Curate produces compiled wiki/memory artifacts through the memory wiki and compiled-memory subsystems.
  Do not put retrieval ranking, store implementations, public facade glue, or wiki-domain lifecycle rules here.

- `uma.retrieve`
  Context retrieval, memory retrieval, ranking, fusion, snippet refinement, and retrieval planning.
  Do not put SQL/vector persistence or ambient runtime state here.

- `uma.memory`
  Memory-domain behavior for working memory, episodic, semantic, procedural, graph, consolidation, compiled memory, and wiki artifacts.
  `uma.memory.wiki` owns canonical wiki page behavior: identity, slugging, lifecycle/status, evidence links, deterministic updates, markdown projection, drift checks, and regeneration from evidence.
  Do not put public API wrappers, runtime orchestration, adapter behavior, or store backend wiring here.

- `uma.stores`
  Canonical durable stores and storage metadata.
  Do not encode retrieval policy or snippet rendering here.

- `uma.adapters`
  External integration boundaries: DBs, vectors, graphs, LLMs, observability.
  Do not put product policy here.

- `uma.common`
  Cross-domain primitives only: config, ownership, provenance, compiled-memory record primitives, shared types, identity, registries, storage metadata, and narrow helpers.
  Do not move domain logic here just to avoid picking an owner.

Canonical entry points:

- Ingest document: `uma.ingest.ingest_document(...)`
- Ingest capture stage: `uma.ingest.capture_source(...)`
- Ingest derive stage: `uma.ingest.derive_memory_artifacts(...)`
- Ingest curate stage: `uma.ingest.curate_compiled_memory(...)`
- Retrieve context: `uma.api.UMAMemory.retrieve_context(...)`
- Retrieve memory: `uma.api.UMAMemory.retrieve_memory(...)`
- Developer/admin memory management: `uma.api.management`

API surface split:

- `uma.api.UMAMemory`
  Product-facing memory API plus required Animus support APIs.
  Keep retrieval, ingest, turn processing, health/rebuild/shutdown behavior, and Animus profile/bootstrap support here.
  Do not expose developer/admin wiki management helpers as `UMAMemory` methods.

- `uma.api.management`
  Developer/debug and admin/internal memory management operations.
  Owns management entry points such as `explain_result(...)`, `update_wiki_page(...)`,
  `export_wiki_projection(...)`, and `lint_memory_drift(...)`.
  These functions delegate to runtime, provenance, evidence expansion, and `uma.memory.wiki`.
  They must not become a second runtime, second provenance model, or second wiki implementation.

Retrieval product split:

- `retrieve_context(...)`
  The canonical evidence-oriented context path for LLM context assembly.
  Chunks/documents are primary, provenance stays attached, and wiki participation is not required by default.
- `retrieve_memory(...)`
  The canonical compiled/evidence-backed memory path for continuity-oriented retrieval.
  `memories` is the primary contract field. If compiled memory is unavailable, the result must surface an explicit fallback instead of silently returning chunk-style context output.
- `uma.retrieve.planner.build_retrieval_plan(...)`
  The small canonical lane planner for both product paths.
  It selects participating lanes, marks excluded lanes with reasons, and keeps lane policy separate from backend execution mechanics.

Compiled memory and wiki model:

- Raw chunks are terminal evidence.
- Provenance is the trust path back to raw chunks.
- Compiled memory artifacts are synthesized views over evidence, not terminal truth.
- Wiki pages are canonical UMA records managed by `uma.memory.wiki`.
- Markdown is a rebuildable projection of canonical wiki records.
- Compiled-memory index entries are navigation metadata, not proof objects.
- Compiled-memory log events are append-only audit history, not source evidence.
- Evidence expansion is the canonical operation for opening memory answers, compiled artifacts,
  and wiki pages back to direct and transitive raw chunk evidence.

Canonical storage taxonomy:

- Shared storage vocabulary lives in `uma.common.storage_metadata`.
- Persisted records use one kind taxonomy:
  `raw_source`, `wiki_page`, `semantic_fact`, `episodic_event`,
  `procedural_rule`, `profile_fact`, `decision_trace`, `query_artifact`.
- Persisted and retrieved artifacts expose one lane taxonomy through `kb_lane`:
  `raw`, `wiki`, `semantic`, `episodic`, `procedural`, `profile`, `trace`.
- `wiki/*.md` is projection/export only. Canonical wiki state belongs in UMA records,
  not markdown files.
- Wiki records use `kind=wiki_page` and `kb_lane=wiki`.
- Raw source chunks use `kind=raw_source` and `kb_lane=raw`.
- Derived semantic, episodic, procedural, profile, and trace records must preserve provenance
  to raw chunks or be explicitly invalid/manual-audited according to the provenance model.

