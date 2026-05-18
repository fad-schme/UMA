# UMA — Configuration

## Runtime Profiles

UMA has one embedded profile: SQLite (authoritative) + LanceDB (vector index). No external services required. Initialize through `UMAMemory.from_yaml(path)`.

| Config file | Use |
|---|---|
| `config/uma.yaml` | Default runnable config |
| `config/uma_lite.yaml` | Reference embedded profile (same storage settings) |

LLM and embedding values in both files are user-customizable baselines — set provider, model, and host to match your environment.

---

## UMA Lite — Embedded Profile

No external services required. SQLite and LanceDB run in-process.

```bash
pip install -e .
python -c "
from uma import UMAMemory
memory = UMAMemory.from_yaml('config/uma.yaml')
print(memory.health_check())
"
```

Data is stored under `.uma/` in the working directory:
- `.uma/db/` — SQLite database files
- `.uma/vectors/` — LanceDB vector files

---

## Full YAML Structure

### Storage

```yaml
storage:
  db_root: ".uma/db"          # base path for SQLite databases
  db_root_base: "cwd"         # resolve db_root relative to current working directory
  sql_backend: "sqlite"

  # Vector backend options:
  # Embedded LanceDB (Lite profile):
  vector_backend: "uma.adapters.vector.lancedb:LanceDBIndex"
  vector_config:
    path: ".uma/vectors"

  # FAISS (in-process alternative, requires pip install -e '.[vector]'):
  # vector_backend: "uma.adapters.vector.faiss_adapter:FaissIndex"

  # Graph (disabled in public profiles):
  graph_backend: "disabled"
  graph_config:
    enabled: false
```

### Working Memory

```yaml
working_memory:
  max_tokens: 4096              # token budget for working memory buffer
  warning_ratio: 0.7            # log warning at 70% of budget
  hard_limit_ratio: 0.95        # hard truncation at 95%
  chunk_size: 20                # messages per chunk when summarizing
  keep_recent_messages: 4       # always retain N most recent messages
  keep_recent_token_fraction: 0.1  # fraction of budget reserved for recents
```

### Embedding Provider

```yaml
embedding:
  provider: "ollama"            # options: ollama, openai
  model: "nomic-embed-text"
  dimension: 768
  config:
    host: "http://localhost:11434"
    batch_size: 32
    timeout: 20.0
```

For OpenAI embeddings:
```yaml
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  dimension: 1536
  config:
    api_key: "${OPENAI_API_KEY}"
```

### LLM Provider (UMA internal use)

UMA uses an LLM internally for memory extraction, summarization, and fact extraction — not for generating agent replies.

```yaml
llms:
  uma:
    provider: "ollama"          # options: ollama, openai, anthropic
    model: "llama3"
    config:
      host: "http://localhost:11434"
      timeout: 20.0
```

For OpenAI:
```yaml
llms:
  uma:
    provider: "openai"
    model: "gpt-4o-mini"
    config:
      api_key: "${OPENAI_API_KEY}"
```

For Anthropic/Claude (install `pip install -e '.[llm]'` first):
```yaml
llms:
  uma:
    provider: "anthropic"
    model: "claude-haiku-4-5-20251001"
    config:
      api_key: "${ANTHROPIC_API_KEY}"
```

Supported providers: `ollama`, `openai`, `anthropic`. Anthropic is LLM-only (not supported as an embedding provider).

### Retrieval Parameters

```yaml
retrieval:
  max_episodes: 3               # max episodic items in memory retrieval
  max_facts: 5                  # max semantic facts in memory retrieval
  max_skills: 3                 # max procedural skills in memory retrieval
  max_graph_items: 3            # max graph items (when graph is enabled)
  max_evidence_chunks: 10       # max raw chunks per retrieval
  strict: true                  # enforce ownership scoping strictly
  debug_scores: false           # emit per-candidate score cards in results

  context:                      # parameters for retrieve_context
    max_working_messages: 6
    max_episodic: 2
    max_semantic: 4
    max_chunks: 2
    snippet_max_chars: 600      # max characters per rendered snippet
    snippet_refiner_enabled: true
    snippet_refiner_top_k: 6
    max_procedural: 2
    include_working_memory: true
    include_episodic: true
    include_graph: false        # set true only when graph is enabled
    include_procedural: true
```

### Semantic Salience

```yaml
semantic:
  salience_threshold: 0.45      # facts below this score are not stored
```

Lower = more facts stored; higher = stricter filtering. Default `0.45` is a reasonable baseline.

### Feature Loading

Optional features attach to UMA at startup. Procedural memory is the primary pluggable feature:

```yaml
features:
  load:
    - name: procedural
      enabled: true
      provider: "uma.memory.procedural.feature:ProceduralFeature"
      config:
        max_k: 50               # max skills to index per query
  policy:
    on_attach_error: "log_and_skip"   # or "raise"
    allow_method_override: false
```

To disable procedural memory: set `enabled: false` or remove the entry.

### Pipeline

```yaml
pipeline:
  defer_post_turn: false        # if true, post-turn processing runs in background queue
  post_turn_queue_max: 200      # max queued post-turn jobs (when deferred)
```

---

## Alternate Vector Backends

The vector backend is a user configuration choice. Set `vector_backend` + `vector_config` in your YAML:

**LanceDB (default, embedded — no install required):**
```yaml
vector_backend: "uma.adapters.vector.lancedb:LanceDBIndex"
vector_config:
  path: ".uma/vectors"
```

**FAISS (in-process alternative):**
```bash
pip install -e '.[vector]'
```
```yaml
vector_backend: "uma.adapters.vector.faiss_adapter:FaissIndex"
vector_config:
  path: ".uma/vectors/faiss"
```

**Qdrant (external service):**
```bash
pip install -e '.[vector]'
```
```yaml
vector_backend: "uma.adapters.vector.qdrant:QdrantIndex"
vector_config:
  url: "http://localhost:6333"
```

Vector indexes can always be rebuilt from authoritative SQLite data:
```python
await memory.rebuild_vector_indexes(tenant_id="default")
```

---

## Install Surfaces

```bash
# Minimal (Lite profile — no external services)
pip install -e .

# With Qdrant or FAISS support
pip install -e '.[vector]'

# With Anthropic LLM support
pip install -e '.[llm]'

# Development + tests
pip install -r requirements.txt
```

---

## Running Tests

```bash
PYTHONPATH=. python -m pytest -q
```

Run only failing tests while iterating; run full suite before committing.
