---
name: uma-configure
description: Complete configuration reference for UMA — runtime profiles (Lite/embedded), full YAML structure, storage backends, working-memory parameters, LLM/embedding providers (Ollama, OpenAI, Anthropic), retrieval parameters, security configuration (injection scan custom patterns, file size limits, retrieval audit toggle), feature loading, and install surfaces. Use this skill when answering questions about uma.yaml structure, how to set the trust weight, how to disable retrieval audit, how to swap vector backends, how to configure a custom injection pattern catalog, or what install extras are available.
---

# UMA — Configuration

## Runtime Profile

UMA Lite uses a single embedded profile: SQLite (authoritative) + LanceDB (vector index). No external services required. Initialize through `UMAMemory.from_yaml(path)`.

| Config file | Use |
|---|---|
| `config/uma.yaml` | Default runnable config |

LLM and embedding values are user-customizable baselines — set provider, model, and host to match your environment.

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
  db_root: ".uma/db"
  db_root_base: "cwd"
  sql_backend: "sqlite"

  # Embedded LanceDB (default):
  vector_backend: "uma.adapters.vector.lancedb:LanceDBIndex"
  vector_config:
    path: ".uma/vectors"
    # Optional tuning for ANN search depth:
    search_k_multiplier: 4
    search_k_max: 512

  # FAISS (in-process alternative, requires pip install -e '.[vector]'):
  # vector_backend: "uma.adapters.vector.faiss_adapter:FaissIndex"

  # InMemory (testing / CI):
  # vector_backend: "uma.adapters.vector.inmemory:InMemoryVectorIndex"

  # Graph (disabled in public profiles):
  graph_backend: "disabled"
  graph_config:
    enabled: false
```

### Working Memory

```yaml
working_memory:
  max_tokens: 4096
  warning_ratio: 0.7
  hard_limit_ratio: 0.95
  chunk_size: 20
  keep_recent_messages: 4
  keep_recent_token_fraction: 0.1
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

UMA uses an LLM internally for memory extraction, summarization, fact extraction, and snippet refinement — not for generating agent replies.

```yaml
llms:
  uma:
    provider: "ollama"          # options: ollama, openai, anthropic
    model: "llama3"
    config:
      host: "http://localhost:11434"
      timeout: 20.0

# Optional: used only by storage adapters that need credentials.
secrets:
  provider: "uma.adapters.secrets.EnvVarProvider"
  options:
    prefix: "UMA"
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

Supported LLM providers: `ollama`, `openai`, `anthropic`. Anthropic is LLM-only (not supported as an embedding provider).

### Retrieval Parameters

```yaml
retrieval:
  max_episodes: 3
  max_facts: 5
  max_skills: 3
  max_graph_items: 3
  max_evidence_chunks: 10

  # Trust-aware ranking
  trust_weight: 0.15            # final = (1 - tw) * similarity + tw * trust
  min_trust_score: 0.5          # candidates below this are dropped before truncation

  strict: true
  debug_scores: false

  context:                      # parameters for retrieve_context
    max_working_messages: 6
    max_episodic: 2
    max_semantic: 4
    max_chunks: 2
    snippet_max_chars: 600
    snippet_refiner_enabled: true
    snippet_refiner_top_k: 6
    max_procedural: 2
    include_working_memory: true
    include_episodic: true
    include_graph: false
    include_procedural: true
```

**`min_trust_score: 0.5`** filters every medium-severity injection survivor: trust starts at default 0.5, drops to 0.4 on medium hits (50% reduction), so anything flagged medium is dropped from retrieval. High hits already cause quarantine.

### Semantic Salience

```yaml
semantic:
  salience_threshold: 0.45      # facts below this score are not stored
```

Lower = more facts stored; higher = stricter filtering. Default `0.45` is a reasonable baseline.

### Security

These settings govern UMA's defense-in-depth model for memory poisoning (ASI06). Layers 1 (pre-write sanitization) and 4 (memory isolation) are always on. The settings below tune layer 1 behavior and the provenance audit trail (layer 2).

```yaml
security:
  # Path to a YAML file with additional injection patterns. UMA loads
  # `uma/common/injection_patterns.yaml` by default; `custom_patterns_path`
  # extends it. Set this to add organization-specific rules without
  # modifying the bundled catalog.
  custom_patterns_path: null

  # If true (default), high-severity write-time scans store the artifact
  # with quarantined_at set; retrieval skips quarantined records. Set
  # false only if you want flagged artifacts dropped instead of retained.
  quarantine_enabled: true

  # Retrieval audit log. Default on. Records hashed query previews,
  # severity, scope, and result counts for every retrieve_* call.
  retrieval_audit_enabled: true
  retrieval_audit_db_path: null   # defaults to `.uma/db/retrieval_audit.db`
```

### Ingest Limits

These are properties on `IngestConfig`, configurable per-call or by changing the default:

```yaml
ingest:
  max_file_bytes: 52428800       # 50 MB; files over this raise FileSizeRejection
  pdf_max_pages: 5000            # PDFs over this raise an error before parsing
  allow_empty_pages: false       # raise on documents with zero extractable pages
```

### Feature Loading

Optional features attach to UMA at startup. Procedural memory is the primary pluggable feature:

```yaml
features:
  load:
    - name: procedural
      enabled: true
      provider: "uma.memory.procedural.feature:ProceduralFeature"
      config:
        max_k: 50
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

The vector backend is a user configuration choice. All backends implement the same C1-contract `VectorIndex` interface — see `uma-vector-contract.md` for details.

**LanceDB (default, embedded — no extra install):**

```yaml
vector_backend: "uma.adapters.vector.lancedb:LanceDBIndex"
vector_config:
  path: ".uma/vectors"
```

LanceDB pushes tenant/owner filters into the database before the candidate cap. Recommended for multi-tenant deployments.

**FAISS (in-process alternative):**

```bash
pip install -e '.[vector]'
```
```yaml
vector_backend: "uma.adapters.vector.faiss_adapter:FaissIndex"
vector_config:
  path: ".uma/vectors/faiss"
```

FAISS does not support pushed-down predicates. The adapter oversamples (k × 4) and post-filters in Python — fine for single-tenant or smaller deployments.

**InMemory (testing / CI):**

```yaml
vector_backend: "uma.adapters.vector.inmemory:InMemoryVectorIndex"
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

# With FAISS support
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
