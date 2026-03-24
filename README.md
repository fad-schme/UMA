
  _   _ __  __   _      ___ _    __  __ 
 | | | |  \/  | /_\ ___| _ \ |  |  \/  |
 | |_| | |\/| |/ _ \___|   / |__| |\/| |
  \___/|_|  |_/_/ \_\  |_|_\____|_|  |_|
                                        

# UMA-RLM

Universal Memory Architecture
UMA-RLM is a production-first memory runtime for developers building AI agents.
It combines working, episodic, facts, skills, and temporal graph memory into a single SDK, and implements the concept of RLM (Recursive Language Model): an inference-time strategy that lets an LLM handle inputs far beyond its context window by treating the long prompt as an external environment. Instead of stuffing the entire prompt into tokens, the model “loads” it into an environment and then programmatically inspects, decomposes, and recursively calls itself on relevant snippets.
UMA-RLM only retrieves context from its stores; the developer owns all agent behavior, reasoning, and response generation.

## Why UMA-RLM

- RLM retrieval you can ship: bounded, read-only, deterministic recursion with strict JSON decisions and time/call budgets.
- Memory as environment: the model "peeks" into memory via safe, snippet-first APIs instead of dumping long context into prompts.
- Predicate-scoped graph navigation: expand memory through fact edges to keep recall precise and controllable.
- Episodic clusters as chapters: precomputed cluster summaries give quick orientation before diving into raw episodes.
- Salience-aware facts: fact memory acts as a truth layer with conflict resolution and confidence scores.
- Pluggable backends: SQLite/Postgres (via extensions), FAISS/Pinecone/Weaviate (via extensions), Neo4j/Memgraph (via extensions), OpenAI/Ollama (via extensions).
- SDK-first: UMA manages memory only; your agent owns reasoning, tools, and final responses.

### Logical Separation of Agent and User Memories

UMA-RLM enforces a **first-class logical separation** between an agent’s global knowledge and user-specific memory.

- **Agent Memory (Agent KB)** contains durable, cross-user knowledge such as domain facts, policies, procedures, and learned generalizations.
- **User Memory** contains private, user-scoped information such as conversations, preferences, and uploaded project data.
- **Project Memory** further subdivides user memory into isolated sub-contexts, ensuring that information from one project never leaks into another unless explicitly promoted.

This separation is not an afterthought or a naming convention. It is enforced at the data-model level across:
- SQL storage
- vector embeddings
- graph nodes and edges
- retrieval filters

As a result, UMA-RLM can safely support long-lived agents that learn over time **without contaminating user privacy or cross-project boundaries**.

### Hierarchical Knowledge Base Segmentation

UMA-RLM organizes memory into a **hierarchical knowledge structure** instead of a flat vector store:

- **Agent-level knowledge** sits at the top of the hierarchy and is shared across all users of the agent.
- **User-level knowledge** is scoped to an individual user.
- **Project-level knowledge** is scoped to a specific project within a user.

Each memory item is tagged with explicit ownership metadata (agent / user / project), allowing retrieval to:
- search the appropriate scope first,
- merge results deterministically across scopes,
- and apply promotion or demotion policies when knowledge should move between layers.

This hierarchy enables UMA-RLM to scale from single-user assistants to enterprise, multi-user and multi-agent systems while preserving correctness, performance, and data governance.

## Core Features

### 1) Recursive Retrieval Controller (RLM)
RLMController iteratively queries memory with bounded recursion:
- Starts with baseline retrieval
- Uses structured decisions to refine what to fetch next
- Stops deterministically with budgets (steps, actions, env calls, timeout)

### 2) Snippet-First Memory Environment
- Read-only access to facts, episodic, skills, and graph stores
- Small snippets by default (summaries and facts)
- Explicit expansion when raw transcripts are needed

### 3) Temporal Graph Memory
- Predicate-scoped edges for precise traversal
- Episodic and fact nodes stay connected over time
- Safe graph neighbor queries with depth and limit controls

### 4) Consolidation and Salience
- Consolidation cycles compress episodic data into durable facts
- Salience scoring prioritizes what matters in retrieval
- Cluster summaries are precomputed for fast "chapter" recall


## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Storage paths
`storage.db_root` supports `~` and environment variables. For relative paths, set
`storage.db_root_base` to control resolution (`auto`, `cwd`, or `config`).

## Typical Usage

```python
from uma import UMARuntime
from uma.core.uma_memory import UMAMemory
from uma.types import RuntimeContext

memory = UMAMemory.from_yaml("config/uma.yaml")
memory._agent_id = "agent-default"  # temporary write-side bridge for process_turn(...)
runtime = UMARuntime.from_memory(memory)

handle = runtime.bind(
    RuntimeContext(
        tenant_id="default",
        agent_id="agent-default",
        request_id="req-1",
        user_id="user-123",
        session_id="session-1",
    )
)

# Get UMA-RLM context (RLM retrieval) for the current user message:
context = await handle.retrieve_structured_context(user_message)

# Your agent controls the system prompt and the LLM call:
messages = [{"role": "system", "content": system_prompt}, {"role": "system", "content": str(context)}]
agent_reply = await agent_llm_generate(messages)

# Persist the turn into UMA memory:
await memory.process_turn(
    user_id="user-123",
    user_msg=user_message,
    assistant_reply=agent_reply,
    extra_meta={"session_id": "session-1"},
)
```

### Context pack vs snippet (important)
UMA retrieval returns a **structured data product** (facts, episodes, chunks, graph).
Snippet rendering is a **presentation layer** that formats that data into a string.
Keeping them separate lets developers:
- explicitly control rendering (no hidden wrappers)
- feed structured context to their own ranking/routing logic
- render different prompt styles
- log/debug retrieval results without string parsing

If you want UMA to render a string snippet using its configured context settings,
bind a runtime context and call `UMARequestHandle.retrieve_rendered_context(query_text)`.



### Observability (Telemetry + Timing)
UMA ships lightweight helpers for logging and timing critical paths. Use them around
retrieval, embeddings, consolidation, and storage operations to surface latency and errors.

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
UMA exposes a lightweight readiness report for dependency checks (SQL, vector, graph, LLM, embedder).

```python
status = memory.health_check()
if status["status"] != "ok":
    print("Health issues:", status)
```

### Retries and error boundaries
External dependencies (LLMs, graph backends, vector services) can be transiently unavailable.
UMA applies conservative retries around these calls and keeps read paths resilient.
If you need different retry behavior, wrap your adapters or supply custom providers.

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
- `uma.get_user_context.latency` (end-to-end retrieval latency)
- `uma.get_user_context.calls|path=rlm|` / `path=classic` / `path=wm_only`

### Security and config hygiene
Keep secrets out of YAML configs. Use environment variables or a secret manager.
The config loader emits warnings when it detects likely secrets in config files.

### Logging configuration
UMA logs to both stdout/stderr and a file by default. Configure with:
- `UMA_LOG_PATH` (e.g., `stdout`, `stderr`, or a file path)
- `UMA_LOG_TO_FILE` (set to `0` to disable file logging)

### Custom LLM / Embedding Providers
You can configure **Agent‑LLM** and **UMA‑LLM** separately. If you only
set a single `llm` section, UMA will use it for both.

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

### Extensions (custom adapters)
The published `uma` package contains UMA core only. Adapters remain external to
the package, and users continue to load config explicitly with
`UMAMemory.from_yaml(config_path)`.

UMA resolves external adapter modules in two ways:
- Explicit adapter roots from `UMA_ADAPTER_ROOTS` (multiple roots may be separated by `os.pathsep`; earlier entries win).
- Backward-compatible project-local `extensions/` or `plugins/` directories alongside your config root.

Folder layout:
```
project_root/
  config/uma.yaml
  extensions/
    vector/
      my_qdrant.py
    db/
      my_sql.py
```

Config example (vector):
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
- For installed use, make the directory that contains `vector/`, `graph/`, `db/`, or `llm/`
  import packages available via `UMA_ADAPTER_ROOTS` if it is not in Python's import path already.

#### Consolidation feature usage
Consolidation is an optional feature that runs an asynchronous "sleep cycle" for a user. It:
1) Fetches recent episodic memories
2) Clusters similar episodes
3) Summarizes clusters (LLM)
4) Extracts salient facts (LLM)
5) Upserts facts into fact memory
6) Prunes low-value episodes

This does not run automatically. You enable the feature in config, then call it from your own
scheduler, batch job, or pipeline hook.

```yaml
features:
  load:
    - name: consolidation
      enabled: true
      provider: "uma.features.consolidation.feature:ConsolidationFeature"
```

```python
# Run from your own scheduler / batch job
result = await memory.consolidation_run(user_id="user-123")
# result.data schema: {"facts": List[Fact], "fact_count": int}
if result.ok:
    print("facts:", result.data["fact_count"])
else:
    print("consolidation failed:", result.errors)
```

#### Procedural feature usage
Procedural memory is an optional feature that lets you store and retrieve skills using
vector search plus rule-based matching. It exposes async methods that return FeatureResult.
All procedural reads are owner-scoped and require explicit `user_id` at call time.

```yaml
features:
  load:
    - name: procedural
      enabled: true
      provider: "uma.features.procedural.feature:ProceduralFeature"
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
