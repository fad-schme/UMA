# UMA-RLM

Universal Memory Architecture
UMA-RLM is a production-first memory runtime for developers building AI agents.
It combines working, episodic, semantic, procedural, and temporal graph memory into a single SDK, and implements the concept of RLM (Recursive Language Model): an inference-time strategy that lets an LLM handle inputs far beyond its context window by treating the long prompt as an external environment. Instead of stuffing the entire prompt into tokens, the model “loads” it into an environment and then programmatically inspects, decomposes, and recursively calls itself on relevant snippets. 

## Why UMA-RLM

- RLM retrieval you can ship: bounded, read-only, deterministic recursion with strict JSON decisions and time/call budgets.
- Memory as environment: the model "peeks" into memory via safe, snippet-first APIs instead of dumping long context into prompts.
- Predicate-scoped graph navigation: expand memory through semantic edges to keep recall precise and controllable.
- Episodic clusters as chapters: precomputed cluster summaries give quick orientation before diving into raw episodes.
- Salience-aware facts: semantic memory acts as a truth layer with conflict resolution and confidence scores.
- Pluggable backends: SQLite/Postgres, FAISS/Pinecone/Weaviate, Neo4j/Memgraph, OpenAI/Ollama.
- SDK-first: UMA manages memory only; your agent owns reasoning, tools, and final responses.

## Core Features

### 1) Recursive Retrieval Controller (RLM)
RLMController iteratively queries memory with bounded recursion:
- Starts with baseline retrieval
- Uses structured decisions to refine what to fetch next
- Stops deterministically with budgets (steps, actions, env calls, timeout)

### 2) Snippet-First Memory Environment
- Read-only access to semantic, episodic, procedural, and graph stores
- Small snippets by default (summaries and facts)
- Explicit expansion when raw transcripts are needed

### 3) Temporal Graph Memory
- Predicate-scoped edges for precise traversal
- Episodic and semantic nodes stay connected over time
- Safe graph neighbor queries with depth and limit controls

### 4) Consolidation and Salience
- Consolidation cycles compress episodic data into durable facts
- Salience scoring prioritizes what matters in retrieval
- Cluster summaries are precomputed for fast "chapter" recall

## Typical Usage

```python
from uma3.core.uma3_memory import UMAMemory
from uma3.core.pipeline import MemoryPipeline

memory = UMAMemory.from_yaml("config/uma3.yaml")
memory.initialize()

pipeline = MemoryPipeline(memory_client=memory, hooks=memory.hooks)

# After your agent generates a reply:
await pipeline.process_turn(
    user_id="user-123",
    user_msg="Hey, I love cold brew.",
    assistant_reply="Noted — I will remember that.",
)

# When building your next prompt:
ctx = await memory.get_user_context(
    user_id="user-123",
    query_text="What should I recommend for coffee?"
)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## License
MIT. See `LICENSE`.
