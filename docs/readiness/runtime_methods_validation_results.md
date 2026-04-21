# Runtime Methods Validation Results

## Affected Methods

- `UMARuntime.bind(...)`
- `UMARequestHandle`
- `UMAMemory.process_turn(...)`

## Export Surface Check

- Command:

```bash
python3 - <<'PY'
from uma import UMAMemory, UMARuntime, UMARequestHandle
print('exports_ok', UMAMemory.__name__, UMARuntime.__name__, UMARequestHandle.__name__)
PY
```

- Result: `PASS`

## Affected Tests Run

### Runtime binding and bound-context retrieval

- Command:

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_runtime_binding.py \
  tests/test_bound_context_retrieval.py \
  tests/test_retrieval_service_recall_scope.py
```

- Result: `FAIL beyond missing-method layer`
- Summary: `11 passed, 1 failed`

Remaining failure:

- `tests/test_bound_context_retrieval.py::test_bound_context_workspace_id_does_not_broaden_retrieval_owner_support`
- Current symptom: retrieval returned no chunks (`owner_types == set()`)
- This is no longer a missing-method failure.

### Turn-processing and working-memory path

- Command:

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_turn_session_local_defaults.py \
  tests/test_working_memory_session_scope.py \
  tests/test_turn_ingest_idempotent.py \
  tests/test_pipeline_graph.py
```

- Result: `PASS`
- Summary: `11 passed`

## Conclusion

- `bind(...)` gap resolved: `PASS`
- `process_turn(...)` gap resolved: `PASS`
- affected tests move past old missing-method failures: `PASS`

Remaining failures in the validated runtime slice are now beyond the runtime-surface gap and appear to be behavioral retrieval issues rather than missing contract methods.
