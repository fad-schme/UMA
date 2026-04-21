# UMA Baseline Command Log

Chronological baseline validation log for Phase 1.

## 1. Repository surface discovery

- Command: `pwd`
- Result: `PASS`
- Notes: Confirmed repo root.

- Command: `rg --files -g 'pyproject.toml' -g 'setup.py' -g 'setup.cfg' -g 'requirements*.txt' -g 'AGENTS.md' -g 'README*' -g 'Makefile' -g 'Dockerfile*' -g '.github/**' -g 'tests/**' -g 'examples/**' -g 'uma/**'`
- Result: `PASS`
- Notes: Confirmed package, example, and test surface.

- Command: `git status --short`
- Result: `PASS`
- Notes: Worktree status checked before assessment work.

## 2. Baseline file inspection

- Commands: `sed -n ...` / `nl -ba ...` across the files listed in `baseline_assessment.md`
- Result: `PASS`
- Notes: Used to inspect package metadata, README claims, runtime initialization path, config contents, and failing tests/helpers.

## 3. Config and startup checks

- Command:
  ```bash
  python3 - <<'PY'
  from uma.core.uma_memory import UMAMemory
  try:
      UMAMemory.from_yaml('config/uma.yaml')
      print('STARTUP_OK')
  except Exception as exc:
      print(type(exc).__name__, str(exc))
  PY
  ```
- Result: `FAIL`
- Key result:
  - Config warning about sensitive value in `storage.graph_config.password`
  - `RuntimeError: qdrant-client is not installed. Install it with pip install qdrant-client.`

- Command:
  ```bash
  python3 examples/chatbot_app/main.py --config config/uma.yaml --user user:local --agent agent-default
  ```
- Result: `FAIL`
- Key result:
  - `ModuleNotFoundError: No module named 'uma'`

- Command:
  ```bash
  PYTHONPATH=. python3 -m examples.chatbot_app.main --config config/uma.yaml --user user:local --agent agent-default
  ```
- Result: `FAIL`
- Key result:
  - Import path issue resolved
  - Startup then failed during `UMAMemory.from_yaml(...)`
  - `RuntimeError: qdrant-client is not installed`

## 4. Full test-suite baseline

- Command:
  ```bash
  /bin/zsh -lc 'PYTHONPATH=. python3 -m pytest -q'
  ```
- Result: `FAIL`
- Key result:
  - Test collection aborted with 5 import errors
  - Cause: `ImportError: cannot import name 'UMARuntime' from 'uma'`

## 5. Targeted critical-path tests

- Command:
  ```bash
  PYTHONPATH=. python3 -m pytest -q \
    tests/test_config_load.py \
    tests/test_config_types.py \
    tests/test_environment_api.py \
    tests/test_tenant_scoped_durable_boundaries.py \
    tests/test_public_scope_surfaces.py \
    tests/test_external_adapter_roots.py \
    tests/test_runtime_concurrency.py \
    tests/test_ownership_only_retrieval.py \
    tests/test_isolation_matrix.py
  ```
- Result: `FAIL`
- Key result:
  - `29 passed, 11 failed`
  - Main failure patterns:
    - `AttributeError: 'UMAMemory' object has no attribute 'agent_id'`
    - `NameError: name 'UMARuntime' is not defined`
    - `AttributeError: 'UMAMemory' object has no attribute '_build_retrieval_request'`
    - brittle `_agent_id` substring assertion failure

- Command:
  ```bash
  PYTHONPATH=. python3 -m pytest -q \
    tests/test_feature_loading.py \
    tests/test_rebuild_indexes.py \
    tests/test_chunk_retrieval_returns_objects.py \
    tests/test_vector_scores_plumbed.py \
    tests/test_ranking_score_cards.py \
    tests/test_semantic_search_subject_optional.py \
    tests/test_chunk_and_procedural_search_no_subject.py
  ```
- Result: `FAIL`
- Key result:
  - `10 passed, 4 failed`
  - Main failure pattern:
    - `AttributeError: 'UMAMemory' object has no attribute 'agent_id'`

## 6. Packaging and artifact validation

- Command:
  ```bash
  python3 -m pip wheel . --no-deps -w /tmp/uma_dist
  ```
- Result: `FAIL`
- Key result:
  - Isolated build attempted to fetch `setuptools>=64`
  - Failed due to restricted network / missing remote index access

- Command:
  ```bash
  python3 -m pip wheel . --no-deps --no-build-isolation -w /tmp/uma_dist
  ```
- Result: `FAIL`
- Key result:
  - Wheel built, but as `UNKNOWN-0.0.0-py3-none-any.whl`
  - This is not a valid expected package identity for UMA

- Command: inspect wheel contents with `zipfile`
- Result: `FAIL`
- Key result:
  - Wheel contained only `.dist-info`
  - No `uma` package modules present

- Commands:
  ```bash
  python3 -m venv /tmp/uma_empty_wheel_env
  /tmp/uma_empty_wheel_env/bin/pip install --no-deps /tmp/uma_dist/UNKNOWN-0.0.0-py3-none-any.whl
  /tmp/uma_empty_wheel_env/bin/python - <<'PY'
  import importlib.util
  print(importlib.util.find_spec('uma'))
  PY
  ```
- Result: `FAIL`
- Key result:
  - Installing succeeded as `UNKNOWN-0.0.0`
  - Importing `uma` from outside the repo root failed: `None` / `ModuleNotFoundError`

## 7. Supporting environment checks

- Command:
  ```bash
  python3 - <<'PY'
  import setuptools
  print(setuptools.__version__)
  PY
  ```
- Result: `PASS`
- Notes: Host had `setuptools 58.0.4`, which does not satisfy the declared isolated build requirement.

## Phase 1 Outcome

- Baseline status: `COMPLETED`
- Overall result: `FAIL`
- Summary:
  - Full suite does not collect
  - Public API surface does not match docs/tests
  - Example startup is not runnable as documented
  - Default config is unsafe and environment-specific
  - Built artifact is not a valid installable UMA package

No fixes were implemented in Phase 1.
