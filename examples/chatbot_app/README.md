# UMA-RLM Example Chatbot

Simple interactive chatbot demonstrating UMA-RLM memory usage.

Usage:

1. Run the example as a module from the repo root.
2. Install UMA dependencies in the current environment, for example `pip install -e .`.
3. Ensure `config/uma.yaml` is configured for the backends you actually want to use.
4. Install any optional extras required by that config.
   Examples:
   `pip install '.[vector]'` for Qdrant/FAISS-related vector backends
   `pip install '.[graph]'` for Neo4j
   `pip install '.[ollama]'` for Ollama-based providers
5. (Optional) install parser extras for document ingestion, e.g. `pip install '.[parsers]'`
6. Run:

```bash
python -m examples.chatbot_app.main --config config/uma.yaml --user user:local --agent agent-default
```

Supported execution mode:

- Run the example as a module from the repo root.
- Do not rely on direct script execution such as `python examples/chatbot_app/main.py`; that path can fail on import resolution depending on how Python is launched.

Startup expectations:

- If imports are correct but the configured backends or optional dependencies are missing, startup fails fast with an actionable message.
- The bundled `config/uma.yaml` currently references optional infrastructure such as vector, graph, and Ollama backends. If those are not installed or reachable, update the config or install the matching extras before running the example.

Commands inside the REPL:
- `/load` — load documents from `/material` into UMA (document ingestion pipeline).
- `/setprompt` — set the system prompt used for the assistant.
- `/quit` — exit.
