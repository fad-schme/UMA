# UMA Baseline Assessment

## Environment Used

- Date: 2026-04-21
- Working directory: `/Users/<user>/Library/CloudStorage/OneDrive-<organization>/PersProjects/dev/UMA-RLM`
- Shell: `zsh`
- Python: `3.9`
- Pip: `25.3`
- Local setuptools on host: `58.0.4`
- Network: restricted in this environment, which affected isolated build dependency resolution

## Files Inspected

- `pyproject.toml`
- `requirements.txt`
- `README.md`
- `uma/__init__.py`
- `config/uma.yaml`
- `examples/chatbot_app/README.md`
- `examples/chatbot_app/main.py`
- `uma/core/initializers/runtime.py`
- `uma/core/initializers/stores.py`
- `uma/core/uma_memory.py`
- `tests/helpers/runtime.py`
- `tests/test_ownership_only_retrieval.py`
- `tests/test_environment_api.py`
- `tests/test_isolation_matrix.py`
- tests importing `UMARuntime` from `uma`

## Main Observed Failures

### 1. Public API export surface is inconsistent

- `README.md` documents `from uma import UMARuntime`.
- Multiple tests import `UMARuntime` from `uma`.
- `uma/__init__.py` exports only `UMAMemory`.
- Result: full `pytest` collection fails before execution on those imports.

Likely root cause:
- The public package export surface was changed or partially migrated without updating the package initializer, docs, and tests together.

### 2. Built artifact is not a valid installable release artifact

- `pip wheel . --no-deps --no-build-isolation -w /tmp/uma_dist` produced `UNKNOWN-0.0.0-py3-none-any.whl`.
- Inspecting the wheel showed only `.dist-info` contents and no `uma` package files.
- Installing the wheel into a clean venv and importing from outside the repo root failed with `ModuleNotFoundError: No module named 'uma'`.

Likely root cause:
- Packaging metadata and/or setuptools package discovery are not being resolved correctly in the real build path.
- The current release artifact path is not validated by the repo’s baseline checks.

### 3. Normal documented example startup path is not runnable

- `examples/chatbot_app/README.md` instructs `python examples/chatbot_app/main.py ...`.
- Running that from a clean checkout failed immediately with `ModuleNotFoundError: No module named 'uma'`.
- Running the module form with `PYTHONPATH=.` got past import resolution but then failed during `UMAMemory.from_yaml()` because the default config requires `qdrant-client`, which was not installed.

Likely root cause:
- Docs assume either an installed package or an adjusted import path, but the command shown does not provide either.
- The default config also assumes optional external infrastructure and optional dependencies during eager startup.

### 4. Default committed config is not safe as a repo baseline

- `config/uma.yaml` contains private LAN endpoints for Qdrant, Neo4j, and Ollama.
- It also contains a plaintext Neo4j password.
- This conflicts with the README guidance that secrets should be kept out of YAML configs.

Likely root cause:
- A developer-local operational config appears to have been committed as the default example config instead of a safe baseline config.

### 5. Runtime initialization is eager and fails early on optional infrastructure

- `UMAMemory.from_yaml()` calls `init_retrieval_ready()` immediately.
- `init_retrieval_ready()` wires stores, LLM, embedder, retrieval cores, graph, and RLM before returning.
- Store initialization instantiates the configured vector backend immediately.
- With the committed default config, failure occurs during vector adapter startup, before example startup can proceed.

Likely root cause:
- Retrieval-ready boot is intentionally eager, but the default release path points to optional plugin infrastructure and external services, so startup fails before the package can even be exercised locally.

### 6. `agent_id` contract is inconsistent

- `UMAMemory` stores `_agent_id` internally.
- `UMAMemory._build_runtime_context()` references `self.agent_id`.
- There is no visible `agent_id` property on `UMAMemory`.
- Many targeted tests fail with `AttributeError: 'UMAMemory' object has no attribute 'agent_id'`.
- Test helpers still directly mutate `memory._agent_id`.

Likely root cause:
- Runtime identity handling is mid-migration between ambient/internal `_agent_id` usage and an intended explicit runtime-context contract.

### 7. Critical-path test suite is not fully credible as release evidence yet

- Some tests use real `UMAMemory.from_yaml()` bootstrapping and real SQLite/InMemory storage paths.
- However, many test helpers still use fake providers and direct internal `_agent_id` mutation.
- Several failures are caused by API contract drift rather than exercising end-to-end behavior.
- `tests/test_ownership_only_retrieval.py` includes a brittle substring assertion on `_agent_id`, which also matches unrelated names like `request_agent_id`.

Likely root cause:
- The suite contains both useful real-path coverage and partially migrated tests that still depend on internal implementation details or text-level assertions.

## Recommended Fix Order

1. Fix package/build integrity.
   Release artifact validity blocks Beta and Production immediately.

2. Fix public API/export surface.
   Test collection and documented imports must align before deeper validation has meaning.

3. Fix the `agent_id` contract.
   This is blocking multiple retrieval, environment, and isolation tests.

4. Fix documented startup path and import assumptions.
   The example app must run from a clean checkout or the docs must stop claiming it does.

5. Replace the committed default config with a safe baseline config.
   Remove secrets and machine-specific endpoints from the repo default.

6. Reassess eager initialization behavior against the intended release surface.
   Optional infrastructure should not make the default baseline unusable unless that is an explicit product contract.

7. Clean up critical-path test credibility issues.
   After API and identity contracts are fixed, re-run the suite and tighten tests that rely on internals or brittle source-text assertions.

## Phase 1 Conclusion

The baseline was completed. The current codebase is not ready for Beta or Production. The dominant blockers are packaging/build integrity, public API drift, eager startup against environment-specific config, and inconsistent runtime identity handling.

No fixes were implemented in Phase 1.
