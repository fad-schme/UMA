# UMA Example Chatbot

Simple interactive chatbot demonstrating UMA memory usage.

Usage:

1. Run the example as a module from the repo root.
2. Install UMA dependencies in the current environment.
   Minimal package install: `pip install -e .`
   Development/test convenience install: `pip install -r requirements.txt`
3. Copy the safe baseline config to a local file, for example:

```bash
cp config/uma.yaml config/uma.local.yaml
```

4. Update `config/uma.local.yaml` for the backends you actually want to use.
5. Install any optional extras required by that local config.
   Examples:
   `pip install '.[vector]'` for Qdrant/FAISS-related vector backends
   `pip install '.[graph]'` for Neo4j
   `pip install '.[ollama]'` for Ollama-based providers
6. (Optional) install parser extras for document ingestion, e.g. `pip install '.[parsers]'`
7. Run:

```bash
python -m examples.chatbot_app.main --config config/uma.local.yaml --user user:local --agent agent-default
```

Supported execution mode:

- Run the example as a module from the repo root.
- Do not rely on direct script execution such as `python examples/chatbot_app/main.py`; that path can fail on import resolution depending on how Python is launched.

Startup expectations:

- If imports are correct but the configured backends or optional dependencies are missing, startup fails fast with an actionable message.
- The committed `config/uma.yaml` is a safe baseline, not a personal ready-to-run environment file.
- Your local `config/uma.local.yaml` may still reference optional infrastructure such as Ollama or custom vector/graph backends. If those are not installed or reachable, update the config or install the matching extras before running the example.

Commands inside the REPL:
- `/load` — load documents from `/material` into UMA (document ingestion pipeline).
- `/setprompt` — set the system prompt used for the assistant.
- `/quit` — exit.
