# UMA Architecture Index

Stable top-level package boundaries:

- `uma.api`
  Public entry points and caller-facing runtime surfaces.
  Keep `UMAMemory`, `UMARuntime`, and request-bound handles here.
  Do not put retrieval policy, store code, or chunk/fact domain logic here.

- `uma.ingest`
  Source parsing, normalization, chunking, and ingest orchestration.
  Do not put retrieval ranking, store implementations, or public facade glue here.

- `uma.retrieve`
  Context retrieval, memory retrieval, ranking, fusion, snippet refinement, and retrieval planning.
  Do not put SQL/vector persistence or ambient runtime state here.

- `uma.memory`
  Memory-domain behavior for working memory, episodic, semantic, procedural, graph, and consolidation artifacts.
  Do not put public API wrappers or store backend wiring here.

- `uma.stores`
  Canonical durable stores and storage metadata.
  Do not encode retrieval policy or snippet rendering here.

- `uma.adapters`
  External integration boundaries: DBs, vectors, graphs, LLMs, observability.
  Do not put product policy here.

- `uma.common`
  Cross-domain primitives only: config, ownership, shared types, identity, registries, and narrow helpers.
  Do not move domain logic here just to avoid picking an owner.

Canonical entry points:

- Ingest document: `uma.ingest.ingest_document(...)`
- Retrieve context: `uma.api.UMAMemory.retrieve_context(...)`
- Retrieve memory: `uma.api.UMAMemory.retrieve_memory(...)`

Canonical storage taxonomy:

- Shared storage vocabulary lives in `uma.common.storage_metadata`.
- Persisted records use one kind taxonomy:
  `raw_source`, `wiki_page`, `semantic_fact`, `episodic_event`,
  `procedural_rule`, `profile_fact`, `decision_trace`, `query_artifact`.
- Persisted and retrieved artifacts expose one lane taxonomy through `kb_lane`:
  `raw`, `wiki`, `semantic`, `episodic`, `procedural`, `profile`, `trace`.
- `wiki/*.md` is projection/export only. Canonical wiki state belongs in UMA records,
  not markdown files.
