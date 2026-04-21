# Public API Validation Results

## Supported Import Path

- Command:

```bash
python3 - <<'PY'
from uma import UMAMemory, UMARuntime
print(UMAMemory.__module__, UMAMemory.__name__)
print(UMARuntime.__module__, UMARuntime.__name__)
PY
```

- Result: `PASS`

Observed:

```text
UMAMemory uma.core.uma_memory UMAMemory
UMARuntime uma.core.runtime.runtime UMARuntime
```

## Docs / Examples Alignment

- `README.md` import example aligned to package root: `PASS`
- `examples/chatbot_app/main.py` import aligned to package root: `PASS`
- README no longer names a non-exported `UMARequestHandle` symbol directly in usage prose: `PASS`

## Pytest Collection

- Command: `PYTHONPATH=. python3 -m pytest --collect-only -q`
- Result: `PASS`

Observed:

- `241 tests collected`
- collection completed without the prior `ImportError: cannot import name 'UMARuntime' from 'uma'`

## Full Pytest Run After Change

- Command: `PYTHONPATH=. python3 -m pytest -q`
- Result: `FAIL beyond import layer`

Observed summary:

- `196 passed`
- `45 failed`

Main remaining failure classes after import-surface fix:

- `AttributeError: 'UMAMemory' object has no attribute 'agent_id'`
- `AttributeError: 'UMARuntime' object has no attribute 'bind'`
- `AttributeError: 'UMAMemory' object has no attribute 'process_turn'`
- one brittle source-text assertion in `tests/test_ownership_only_retrieval.py`

## Conclusion

The public API/export mismatch is fixed.

The suite now moves past the import stage and exposes the next real blockers in the runtime contract, with `agent_id` consistency as the most prominent one.
