# Example Startup Fix Notes

## What Was Broken

The documented example startup path was inaccurate:

- `examples/chatbot_app/README.md` told users to run `python examples/chatbot_app/main.py ...`
- from a clean checkout, that direct script path failed with `ModuleNotFoundError: No module named 'uma'`
- after switching to a working import path manually, startup then failed on optional backend dependencies and config assumptions, but the failure surface was not clearly framed for the example user

There was also an example-specific runtime issue after startup:

- the chatbot loop called turn ingestion without the explicit `session_id` required by the current supported turn-processing contract

## Supported Execution Mode

Chosen canonical invocation:

```bash
python -m examples.chatbot_app.main --config config/uma.yaml --user user:local --agent agent-default
```

This is the supported example path because:

- it works from the repo root without hidden `PYTHONPATH` hacks
- it uses normal package/module resolution
- it matches the actual source layout of the repo

## What Changed

1. Updated `examples/chatbot_app/README.md` to document module execution from the repo root as the only supported invocation.
2. Added explicit notes that direct script execution is not supported because it can fail on import resolution.
3. Added a startup error formatter in `examples/chatbot_app/main.py` so immediate failures now explain whether the issue is:
   - import/execution mode
   - missing optional dependency
   - missing/unsupported backend setup
4. Wrapped example startup in `SystemExit` with the actionable startup message instead of a raw traceback for the initial failure path.
5. Updated the example chat session setup to bind an explicit `session_id`.
6. Updated the example turn-ingest call to pass `extra_meta={"session_id": ...}` so the example remains compatible with the current supported turn-processing contract.

## Remaining Limitations

- The bundled `config/uma.yaml` still references optional vector, graph, and Ollama backends.
- In the current environment, startup still fails because `qdrant-client` is not installed.
- That limitation is now honest and actionable rather than being masked by a broken import path.

This phase did not redesign config defaults or eager initialization behavior.
