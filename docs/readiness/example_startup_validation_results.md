# Example Startup Validation Results

## Documented Command Used

```bash
python -m examples.chatbot_app.main --config config/uma.yaml --user user:local --agent agent-default
```

## Import-Path Result

- Module execution from repo root: `PASS`
- Direct script execution (`python examples/chatbot_app/main.py ...`): `FAIL as expected and now explicitly unsupported`

Observed direct-script failure:

```text
ModuleNotFoundError: No module named 'uma'
```

## Startup Result

- Command: `python -m examples.chatbot_app.main --config config/uma.yaml --user user:local --agent agent-default`
- Result: `FAIL on real dependency/config issue, not import-path issue`

Observed startup message:

```text
Failed to start the UMA example with config 'config/uma.yaml'.
Cause: qdrant-client is not installed. Install it with `pip install qdrant-client`.
Install vector dependencies with `pip install '.[vector]'` or switch `storage.vector_backend` to a backend available in your environment.
The supported invocation is `python -m examples.chatbot_app.main --config config/uma.yaml --user user:local --agent agent-default`.
```

## Docs Alignment

- `examples/chatbot_app/README.md`: `PASS`
- `examples/chatbot_app/main.py` startup message: `PASS`

## Remaining Dependency / Infrastructure Limitations

- The bundled `config/uma.yaml` still assumes optional vector/graph/LLM infrastructure.
- In this environment, the first immediate blocker is missing `qdrant-client`.
- Those limitations remain, but they are now surfaced honestly and actionably through the supported example path.
