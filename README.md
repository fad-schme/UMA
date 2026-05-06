
  _   _ __  __   _      ___ _    __  __ 
 | | | |  \/  | /_\ ___| _ \ |  |  \/  |
 | |_| | |\/| |/ _ \___|   / |__| |\/| |
  \___/|_|  |_/_/ \_\  |_|_\____|_|  |_|
                                        

# UMA-RLM

Universal Memory Architecture

UMA-RLM is a production-first memory runtime for developers building AI agents. It combines raw evidence, semantic facts, episodic memory, procedural skills, graph links, profiles, and compiled wiki memory into a single SDK.

UMA-RLM implements the concept of RLM, or Recursive Language Model: an inference-time strategy that lets an LLM handle inputs far beyond its context window by treating long context as an external environment. Instead of stuffing the entire prompt into tokens, the model loads material into UMA and programmatically inspects, decomposes, and retrieves relevant evidence.

UMA manages memory only. It enforces explicit ownership boundaries, retrieves context, compiles evidence-backed memory, tracks provenance, and maintains rebuildable wiki projections. The developer owns all agent behavior, reasoning, tools, and final responses.

## Why UMA-RLM

- RLM retrieval you can ship: bounded, read-only, deterministic recursion with strict JSON decisions and time/call budgets.
- Memory as environment: the model peeks into memory via safe, snippet-first APIs instead of dumping long context into prompts.
- Explicit ownership contracts: write-facing and promotion-facing paths use explicit primitive ownership fields instead of ambiguous owner objects.
- Provenance as runtime invariant: memory answers, compiled artifacts, and wiki pages remain traceable back to raw chunks or are explicitly invalid/manual-audited.
- Compiled wiki memory: UMA maintains canonical wiki records as synthesized views over evidence, with markdown as a rebuildable projection.
- Predicate-scoped graph navigation: expand memory through fact edges to keep recall precise and controllable.
- Episodic clusters as chapters: precomputed cluster summaries give quick orientation before diving into raw episodes.
- Salience-aware facts: fact memory acts as a truth layer with conflict visibility and confidence/support signals.
- Pluggable backends: SQLite/Postgres, FAISS/Pinecone/Weaviate, Neo4j/Memgraph, OpenAI/Ollama, and other providers via extensions.
- SDK-first: UMA manages memory only; your agent owns reasoning, tools, and final responses.

### Logical Separation of Agent and User Memories

UMA-RLM enforces a first-class logical separation between an agent’s global knowledge and user-specific memory.

- Agent Memory, or Agent KB, contains durable cross-user knowledge such as domain facts, policies, procedures, and learned generalizations.
- User Memory contains private, user-scoped information such as conversations, preferences, and uploaded project data.
- Project Memory further subdivides user memory into isolated sub-contexts, ensuring that information from one project never leaks into another unless explicitly promoted.

This separation is not an afterthought or a naming convention. It is enforced through explicit ownership metadata across SQL storage, vector embeddings, graph nodes and edges, retrieval filters, write paths, and promotion paths.

As a result, UMA-RLM can safely support long-lived agents that learn over time without contaminating user privacy or cross-project boundaries.

### Hierarchical Knowledge Base Segmentation

UMA-RLM organizes memory into a hierarchical knowledge structure instead of a flat vector store.

- Agent-level knowledge sits at the top of the hierarchy and is shared across all users of the agent.
- User-level knowledge is scoped to an individual user.
- Project-level knowledge is scoped to a specific project within a user.

Each memory item is tagged with explicit ownership metadata, allowing retrieval to search the appropriate scope first, merge results deterministically across scopes, and apply promotion or demotion policies when knowledge should move between layers.

This hierarchy enables UMA-RLM to scale from single-user assistants to enterprise, multi-user, and multi-agent systems while preserving correctness, performance, and data governance.

## Core Features

### 1) Recursive Retrieval Controller, or RLM

`RLMController` iteratively queries memory with bounded recursion:

- starts with baseline retrieval
- uses structured decisions to refine what to fetch next
- stops deterministically with budgets for steps, actions, environment calls, and timeout

### 2) Evidence-First Memory Environment

- Structured access to raw chunks, semantic facts, episodic memory, procedural skills, graph links, profiles, and compiled wiki memory
- Small snippets and summaries by default
- Explicit evidence expansion when raw chunks are needed
- Provenance metadata preserved across retrieved, derived, and compiled artifacts

### 3) Temporal Graph Memory

- Predicate-scoped edges for precise traversal
- Episodic and fact nodes stay connected over time
- Safe graph neighbor queries with depth and limit controls

### 4) Consolidation and Salience

- Consolidation cycles compress episodic data into durable facts
- Salience scoring prioritizes what matters in retrieval
- Cluster summaries are precomputed for fast chapter recall

### 5) Provenance and Evidence Expansion

- Raw chunks are terminal evidence
- Derived artifacts and compiled memory carry provenance back to supporting chunks
- Memory answers expose supporting evidence, retrieval path metadata, and support/conflict signals
- Evidence expansion opens memory answers, compiled artifacts, and wiki pages back to raw chunks

### 6) Compiled Wiki Memory

- Wiki pages are canonical UMA records, not markdown files
- Page identity, slugging, status, evidence links, drift checks, and regeneration are managed by `uma.memory.wiki`
- Markdown export is projection-only and can be deleted and rebuilt
- Compiled-memory index entries are navigation metadata; compiled-memory log events are audit history

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is the repo's thin development convenience layer and resolves to the package metadata in `pyproject.toml` plus the `dev` extra.

For other install surfaces:

- Core package only: `pip install -e .`
- Development/test environment: `pip install -r requirements.txt`
- Vector backends: `pip install -e '.[vector]'`
- Graph backend: `pip install -e '.[graph]'`
- Ollama provider: `pip install -e '.[ollama]'`
- Parser extras: `pip install -e '.[parsers]'`
- Postgres backend: `pip install -e '.[postgres]'`

You can combine extras when needed, for example:

```bash
pip install -e '.[dev,vector,graph,ollama,parsers]'
```

### Config baseline

`config/uma.yaml` is the committed production-shaped example configuration. It is safe to commit, shows the intended UMA wiring, and avoids committed secrets. It may still require local or remote services, optional extras, and deployment-specific endpoints or credentials before it will run in your environment.

For local development:

1. Copy `config/uma.yaml` to `config/uma.local.yaml`
2. Add your real provider settings, endpoints, and secrets there
3. Run UMA with `--config config/uma.local.yaml` or `UMAMemory.from_yaml("config/uma.local.yaml")`

Keep secrets out of committed YAML configs. Use environment variables or a secret manager. Operators should copy or adapt `config/uma.yaml` for their deployment rather than assuming it runs unchanged.

`requirements.txt` is the development baseline only. Selected backends may require optional extras or provider packages, for example:

- Qdrant or other vector backends: install the matching vector/provider packages
- Neo4j or other graph backends: install the graph extra and run a reachable graph service
- Ollama/OpenAI-compatible providers: run or configure the selected LLM and embedding providers

### Storage paths

`storage.db_root` supports `~` and environment variables. For relative paths, set `storage.db_root_base` to control resolution: `auto`, `cwd`, or `config`.

### Canonical storage metadata

UMA uses one shared storage vocabulary across ingest and retrieval.

- `kind`: `raw_source`, `wiki_page`, `semantic_fact`, `episodic_event`, `procedural_rule`, `profile_fact`, `decision_trace`, `query_artifact`
- `kb_lane`: `raw`, `wiki`, `semantic`, `episodic`, `procedural`, `profile`, `trace`
- shared metadata fields on persisted artifacts: `kind`, `kb_lane`, `owner_type`, `owner_id`, `scope`, `source_id`, `source_type`, `created_at`, `updated_at`, `provenance`, `status`

`wiki/*.md` is projection-only output. Canonical wiki state belongs in UMA records with `kind="wiki_page"` and `kb_lane="wiki"`.

### Architecture status

The current architecture reflects the completed PR 1-8 cleanup sequence:

- explicit ownership and scope contracts for write-facing and promotion-facing paths
- provenance as a runtime invariant
- three-stage ingest: capture, derive, curate
- product-facing memory APIs separated from developer/admin management APIs
- compiled wiki pages as canonical UMA records with markdown as projection only

For package boundaries and invariants, see `ARCHITECTURE.md`.

## Typical Usage

```python
from uma import UMAMemory
from uma.api.management import explain_result, export_wiki_projection

memory = UMAMemory.from_yaml("config/uma.local.yaml").set_context(
    user_id="user-123",
    agent_id="agent-default",
    tenant_id="default",
    request_id="req-1",
    session_id="session-1",
)

# Get curated evidence-oriented context (RAG) for the current user message:
context = await memory.retrieve_context(
    query_text=user_message,
    lane_filter=["raw", "semantic"],  # optional
)

# Retrieve compiled/evidence-backed memory results for continuity-oriented use:
memory_result = await memory.retrieve_memory(
    query_text=user_message,
    memory_intent="continuity",
)

# Optional developer/debug inspection:
explanation = await explain_result(memory, memory_result)

# Optional wiki projection export when compiled wiki state exists:
# await export_wiki_projection(memory, memory_result["compiled_answer"], output_path="wiki/example.md")

# Your agent controls the system prompt and the LLM call:
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "system", "content": str(context)},
]
agent_reply = await agent_llm_generate(messages)

# Persist the turn into UMA memory:
await memory.process_turn(
    user_id="user-123",
    user_msg=user_message,
    assistant_reply=agent_reply,
    extra_meta={"session_id": "session-1"},
)
```

### Structured retrieval vs rendering

UMA retrieval returns structured data products, not final prompts.

- `retrieve_context(...)` returns evidence-oriented context for RAG-style use.
- `retrieve_memory(...)` returns compiled/evidence-backed memory for continuity-oriented use.
- Rendering snippets or prompts is a presentation concern controlled by the developer.

Keeping retrieval separate from rendering lets developers inspect evidence, route results, build custom prompts, and debug memory behavior without parsing strings.

### Retrieval products

UMA exposes two distinct retrieval products on `UMAMemory`.

- `retrieve_context(...)`
  Curated context retrieval for the LLM. This is the evidence-oriented RAG path. Raw chunks and documents are primary, provenance is attached, and wiki state is not required by default.
- `retrieve_memory(...)`
  Compiled/evidence-backed memory retrieval for continuity-oriented use. Results include compiled answer data where available, supporting evidence, provenance, retrieval path metadata, and direct/transitive evidence expansion support.

Both product paths run through one small lane-aware planner in `uma.retrieve.planner`. It decides which canonical lanes participate for the current product call, surfaces excluded lanes and reasons in retrieval trace data, and leaves backend mechanics such as hybrid or lexical retrieval below that boundary.

- Context retrieval defaults toward evidence lanes: usually `raw` first, then `semantic` when available.
- Memory retrieval defaults toward compiled-memory intent: `wiki` first in policy, then raw evidence expansion, with optional `semantic` and `episodic` support.
- `profile` is its own lane. UMA does not treat user-owned KB and user profile as the same retrieval target.

Callers should treat returned evidence and provenance as part of the contract. If no compiled artifact is available, UMA surfaces the evidence-backed fallback explicitly instead of silently pretending chunk retrieval is compiled memory.

### Ingest stages

Document ingest is split into three explicit internal stages:

1. Capture: parse and normalize raw input into source records and raw chunks.
2. Derive: derive facts, graph edges, temporal/salience markers, and other provenance-bearing memory artifacts from chunks.
3. Curate: create or refresh compiled wiki/memory artifacts from evidence and derived artifacts.

This allows UMA to ingest raw evidence without forcing derivation, rerun derivation without reparsing documents, and rebuild wiki state from evidence plus derived artifacts.

### Compiled wiki subsystem

UMA treats wiki pages as managed memory records.

- Canonical wiki state lives in UMA records with `kind="wiki_page"` and `kb_lane="wiki"`.
- `uma.memory.wiki` owns page identity, slugging, lifecycle/status, evidence links, deterministic updates, markdown projection, drift checks, and page regeneration.
- Markdown under `wiki/*.md` is projection/export only.
- Wiki pages remain synthesized views over evidence; they are not terminal truth.

### Developer/admin management APIs

Developer/debug and admin/internal operations live in `uma.api.management`, not on `UMAMemory`.

- `explain_result(...)` explains evidence, provenance, retrieval path, conflicts, drift, and direct/transitive chunk support for a result.
- `update_wiki_page(...)` performs controlled wiki curation through the canonical wiki/compiled-memory path.
- `export_wiki_projection(...)` exports rebuildable markdown or other projections from canonical wiki state.
- `lint_memory_drift(...)` reports stale, unsupported, conflicted, or invalid memory/wiki state without silently rewriting it.

These APIs inspect, curate, project, or lint. They do not replace the core ingest, retrieval, provenance, or wiki subsystems.

### Observability: Telemetry and Timing

UMA ships lightweight helpers for logging and timing critical paths. Use them around retrieval, embeddings, consolidation, and storage operations to surface latency and errors.

```python
from uma.adapters.observability.telemetry import log_call
from uma.adapters.observability.timing import time_block, async_time_block, timed

@log_call("embed_query")
def embed_query(text: str):
    ...

@timed
async def retrieve(user_id: str, query: str):
    ...

with time_block("vector_search"):
    index.search(vec)

async with async_time_block("consolidation_run"):
    await memory.consolidation_run(user_id)
```

### Health checks

UMA exposes a lightweight readiness report for dependency checks: SQL, vector, graph, LLM, and embedder.

```python
status = memory.health_check()
if status["status"] != "ok":
    print("Health issues:", status)
```

### Retries and error boundaries

External dependencies such as LLMs, graph backends, and vector services can be transiently unavailable. UMA applies conservative retries around these calls and keeps read paths resilient. If you need different retry behavior, wrap your adapters or supply custom providers.

### Data consistency and recovery

If a vector index drifts from SQL state, you can rebuild it from stored data.

```python
result = await memory.rebuild_vector_indexes(owner_type="user", owner_id="user:user-123")
print(result)
```

### Retrieval performance harness

Use the built-in script to measure end-to-end retrieval latency and emit metrics snapshots.

```bash
python3 scripts/perf_retrieval.py --iterations 100 --concurrency 20
```

Snapshot keys to watch:

- `uma.retrieve_context.latency` or the configured retrieval timing key for the active runtime path
- `uma.get_user_context.calls|path=rlm|` / `path=classic` / `path=wm_only`

### Security and config hygiene

Keep secrets out of YAML configs. Use environment variables or a secret manager. The config loader emits warnings when it detects likely secrets in config files.

### Logging configuration

UMA logs to both stdout/stderr and a file by default. Configure with:

- `UMA_LOG_PATH`, for example `stdout`, `stderr`, or a file path
- `UMA_LOG_TO_FILE`, set to `0` to disable file logging

### Custom LLM / Embedding Providers

You can configure Agent-LLM and UMA-LLM separately. If you only set a single `llm` section, UMA will use it for both.

```yaml
llms:
  agent:
    provider: "my_pkg.llm:MyLLM"
    model: "my-model"
    config:
      timeout: 20.0
  uma:
    provider: "my_pkg.llm:MySmallLLM"
    model: "my-small-model"
    config:
      timeout: 20.0

embedding:
  provider: "my_pkg.embed:embed"
  dimension: 1536
  config:
    preflight: true
```

### Extensions: Custom Adapters

The published `uma` package contains UMA core only. Adapters remain external to the package, and users continue to load config explicitly with `UMAMemory.from_yaml(config_path)`.

UMA resolves external adapter modules in two ways:

- Explicit adapter roots from `UMA_ADAPTER_ROOTS`. Multiple roots may be separated by `os.pathsep`; earlier entries win.
- Backward-compatible project-local `extensions/` or `plugins/` directories alongside your config root.

Folder layout:

```text
project_root/
  config/uma.yaml
  extensions/
    vector/
      my_qdrant.py
    db/
      my_sql.py
```

Config example, vector:

```yaml
storage:
  vector_backend: "vector.my_qdrant:make_index"
  vector_config:
    url: "http://localhost:6333"
    collection: "uma_vectors"
```

Notes:

- `vector_backend` accepts a plugin spec `module:callable`.
- The callable must accept `dim` as the first argument and return a `VectorIndex`.
- For installed use, make the directory that contains `vector/`, `graph/`, `db/`, or `llm/` import packages available via `UMA_ADAPTER_ROOTS` if it is not in Python's import path already.

#### Consolidation feature usage

Consolidation is an optional feature that runs an asynchronous sleep cycle for a user. It:

1. Fetches recent episodic memories
2. Clusters similar episodes
3. Summarizes clusters with an LLM
4. Extracts salient facts with an LLM
5. Upserts facts into fact memory
6. Prunes low-value episodes

This does not run automatically. You enable the feature in config, then call it from your own scheduler, batch job, or pipeline hook.

```yaml
features:
  load:
    - name: consolidation
      enabled: true
      provider: "uma.memory.consolidation.feature:ConsolidationFeature"
```

```python
result = await memory.consolidation_run(user_id="user-123")
if result.ok:
    print("facts:", result.data["facts"])
else:
    print("consolidation failed:", result.errors)
```

#### Procedural feature usage

Procedural memory is an optional feature that lets you store and retrieve skills using vector search plus rule-based matching. It exposes async methods that return `FeatureResult`. All procedural reads are owner-scoped and require explicit `user_id` at call time.

```yaml
features:
  load:
    - name: procedural
      enabled: true
      provider: "uma.memory.procedural.feature:ProceduralFeature"
      config:
        max_k: 50
```

```python
result = await memory.procedural_add_skill(skill, embedding)
if not result.ok:
    print("add failed:", result.errors)

result = await memory.procedural_find_skills("book a flight", user_id="user-123", k=5)
if result.ok:
    print("skills:", result.data)
else:
    print("find failed:", result.errors)
```

## License

MIT. See `LICENSE`.
