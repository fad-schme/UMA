# UMA-RLM Codebase Review

**Scope:** full static review of the shipped package (`uma-0713.zip`) — 142 Python files, ~37.8k LOC.
**Method:** AST analysis + call-site cross-referencing + manual reading, checked against the invariants in `AGENTS.md` and `ARCHITECTURE.md`.
**Caveat:** the environment had no network, so the test suite could not be run and no third-party linter was installable. Dead-code / unused-import findings are from static analysis plus call-site tracing (high confidence), but a few *could* be reachable via dynamic dispatch or entry-point plugins not visible here. Each such case is flagged.

Everything compiles (`py_compile` clean). The load-bearing security discipline is genuinely well built (see *What's solid*). The problems cluster in **half-wired features**, **dead scaffolding / parallel paths**, and **doc drift** — exactly the "conceptual clutter" and "one canonical path" concerns `AGENTS.md` is built around.

---

## Severity legend
- **H** — breaks first-run, silently loses data, or contradicts a stated core guarantee.
- **M** — dead code, duplication, or parallel paths that violate the "one canonical path / lean" rules.
- **L** — documentation drift and naming/consistency nits.

---

## H1 — The default runnable config is not shipped
The README quickstart and `ARCHITECTURE.md` both present `config/uma.yaml` (and `config/uma.yaml`) as *the* first-run path, and every example hardcodes it:

- `examples/github_chat_eval.py:32`, `examples/batch_test.py:60`, `examples/chatbot_app/main.py:127`, `examples/chatbot_app/sim.py:32`, `examples/memory_app/main.py:93` → `config_path = "config/uma.yaml"`

There is **no `config/` directory anywhere in the package**. As delivered, `UMAMemory.from_yaml("config/uma.yaml")` fails immediately, which breaks the "install in one path" product test in `AGENTS.md §0`.

Compounding it: `setup.py` uses `include_package_data=True` with **no `MANIFEST.in`**, so even if the YAMLs existed at repo root they would not reliably ship in an sdist, and they live *outside* the `uma/` package tree so `packages.find` won't capture them.

**Fix (low effort):** ship `config/uma.yaml` + `config/uma.yaml`; add a `MANIFEST.in` (or move them under `uma/` as package data). If this is only a zip-packaging omission, confirm the release artifact actually contains them.

---

## H2 — `defer_post_turn=True` silently drops semantic memory
`uma/ingest/pipeline.py`

- `process_turn` early-returns after enqueuing when defer is on: `pipeline.py:359-377`
- the queue drainer `process_post_turn_queue` (`pipeline.py:179`) is **called only from tests** (`tests/test_process_turn_semantic_behavior.py`, `tests/test_runtime_concurrency.py`) and is **not on the public `UMAMemory` API**.

So an operator who sets the documented, config-validated flag `pipeline.defer_post_turn` (`config_types.py:479-486`, validated in `config.py:397-404`) gets an agent that stores episodes but **never extracts facts, never promotes, never updates the graph, never runs after-turn hooks** — unless they discover and call an undocumented internal coroutine. This violates `AGENTS.md`'s "failures degrade safely" and "no partially-migrated features."

**Fix:** either (a) delete the defer path entirely (`_DeferredPostTurnTask`, `_enqueue_post_turn`, `_post_turn_defer_enabled`, `_post_turn_queue_*`, `process_post_turn_queue`) if background draining isn't a real product requirement, or (b) finish it — expose a public drain entry point and document that enabling defer requires the caller to schedule draining. Given `AGENTS.md`'s "simplify before extending," (a) is the cleaner call for Lite.

---

## H3 — Consolidation auto-trigger is hardcoded off
`uma/ingest/ingest_service.py:1109-1113`

```python
await maybe_trigger_consolidation(
    memory=memory,
    user_id=...,
    enabled=False,     # <-- literal
)
```

`ConsolidationConfig.consolidation_enabled` is parsed, defaulted to `True`, and validated (`config_types.py:498`, `config.py:374`), but the **only** automatic trigger passes `enabled=False` as a constant, so the config flag has zero effect on the automatic path. Consolidation runs only if the caller manually invokes the feature method `memory.consolidation_run()`.

**Fix:** pass the resolved config value through (`enabled=self.consolidation_cfg.consolidation_enabled`), or delete `maybe_trigger_consolidation` + its call and document consolidation as manual-only. Right now the wrapper is dead scaffolding pretending to be a wired feature.

---

## M1 — The "RLM = LLM-controlled search" principle isn't implemented; dead LLM-decision scaffolding remains
`AGENTS.md §0` states the core product principle: *"RLM is always enabled. Retrieval is LLM-controlled search."* The controller docstring echoes it (`controller.py:64-76`).

In practice the navigation loop is **fully deterministic**: it calls `decisions.deterministic_decision(...)` (`controller.py:460`) and `decisions.next_predicate_scope(...)` (`controller.py:475`). The LLM is used only for optional *fact pruning* (`_prune_facts_with_llm`, `controller.py:993`) and snippet refinement — never to propose retrieval actions.

The machinery for an LLM-decision mode exists but is **dead in production**:
- `ControllerDecision.from_json` (`decisions.py:263`) — only its own log line + `tests/test_rlm_decisions.py`.
- `RetrievalAction.validate_action` (`decisions.py:98`) — only its own log line.

(`ControllerDecision` the class is *not* dead — `deterministic_decision` reuses it as a container — but the JSON-parse/validate path for LLM output is.)

**Fix:** either implement the LLM-controlled navigation the docs promise, or drop the claim and remove `from_json`/`validate_action`. Keeping unwired scaffolding for the product's headline differentiator is the most misleading form of clutter.

---

## M2 — Three overlapping graph-update paths; one is test-only
Graph is `disabled` by default in all public profiles (`config_types.py:146`), yet the write side is implemented three times:

1. `uma/ingest/graph_updater.py:75` `update_graph(...)` — used by document ingest (`ingest_service.py:24`).
2. `uma/ingest/pipeline.py:851` `MemoryPipeline._update_graph(...)` — used by the turn path, calls `GraphCore` methods directly (`add_episode`, `add_facts`, `link_episode_to_facts`, `link_temporal`).
3. `uma/memory/graph/updater.py:51` `GraphUpdater` class (~340 lines, methods `add_episode_node`, `add_fact`, `link_episode_to_facts`, `link_temporal`) — **referenced only by `tests/test_graph_core.py`**; no production caller.

`GraphUpdater` is a parallel abstraction that the live code bypasses. This is precisely the "two modules express the same concept with slightly different names/shapes → converge them" rule in `AGENTS.md §3`.

**Fix:** delete `GraphUpdater` (and its test, or repoint the test at the real path), and converge (1) and (2) onto one graph-write helper.

---

## M3 — `CoTMemoryBuilder` is orphaned public surface
`uma/retrieve/cot_memory_builder.py:31`, exported in `uma/retrieve/__init__.py` `__all__`.

Never instantiated in production — the canonical retrieval path uses `ContextPackBuilder` (`api/runtime.py:1303-1319`). Only `tests/test_context_builders.py:125` touches it, and its docstring references a `UMARequestHandle.retrieve_context` shape that isn't the live entry point. It enlarges the public API surface `AGENTS.md` wants "small and sharp" for no runtime benefit.

**Fix:** remove it (and its test), or document why it's a supported public helper and wire it into a real path.

---

## M4 — `UMARequestHandle` / `UMARuntime.bind()` is a test-only second entry shape
`uma/api/runtime.py:52` (class), `:335` (`bind`).

`UMARequestHandle.retrieve_context/retrieve_memory` are thin forwarders to `UMARuntime.retrieve_context(self.context, ...)`. The public `UMAMemory.retrieve_context` (`api/memory.py:435`) calls the runtime directly and never goes through `bind()`. The handle path is exercised only by tests (`test_retrieval_scoped_requests.py`, `test_isolation_matrix.py`).

Not a duplicate *implementation* (it delegates), but it is a second public-ish entry shape for the same behavior. `AGENTS.md §7` wants one obvious path.

**Fix:** either make `bind()` the canonical internal call path (and have `UMAMemory` use it) or drop it and have the tests call the runtime the way production does.

---

## M5 — Duplicated store methods that belong in the base class
`BaseVectorSQLStore` (`stores/base_vector_sql_store.py:38`) already abstracts `_table_name()` / `_id_column()`, yet each of the four vector stores re-implements identical CRUD-security methods:

- `quarantine_record` — near byte-for-byte identical in `chunk_sql.py:790`, `semantic_sql.py:1019`, `episodic_sql.py:1008`, `procedural_sql.py:649` (differ only in table name + log strings).
- `reinstate_quarantined_record` — same story across all four (`chunk_sql.py:716`, `semantic_sql.py:1064`, `episodic_sql.py:960`, `procedural_sql.py:601`).

That's ~8 near-identical method bodies. `AGENTS.md §3`: "converge duplicate concepts into one canonical path."

Related naming inconsistency: the scope-clause helper is `_scope_where` in `chunk_sql.py:178` and `semantic_sql.py:262` but `_require_scope` in `episodic_sql.py:220` — three names, one concept.

**Fix:** lift `quarantine_record` / `reinstate_quarantined_record` into `BaseVectorSQLStore` using `self._table_name()`; standardize on one scope-helper name.

---

## M6 — Confirmed dead functions (defined, never called in production)
Traced across `uma/`, `tests/`, `examples/`, `mcp/`. These have no production call site (references are only their own definition, own log strings, or tests):

| Symbol | Location | Notes |
|---|---|---|
| `log_call` | `adapters/observability/telemetry.py:17` | zero references anywhere |
| `merge_scan_results` | `common/injection_scan.py:282` | zero references |
| `deduplicate_strings` | `memory/consolidation/utils.py:14` | zero references |
| `parse_scores_list` | `memory/semantic/query_pruner.py:56` | zero references |
| `build_skill_from_definition` | `memory/procedural/skill_indexer.py:56` | zero references |
| `promote_and_update_graph` | `memory/promotion.py:367` | zero references |
| `is_expired` | `adapters/secrets/secrets.py:110` | credential TTL check; nothing consumes it in Lite |
| `_executemany` | `stores/base_sql_store.py:182` | only its own log string |
| `list_facts_for_subject` | `stores/semantic_sql.py:665` | only its own error strings |
| `get_cluster_members` | `stores/episodic_sql.py:894` | only its own log strings |
| `async_time_block` | `adapters/observability/timing.py:48` | only its own log string |
| `_filter_time_range` | `retrieve/rlm/environment.py:107` | test-only |
| `legacy_session_scope_for_user` | `memory/working_memory/core.py:46` | test-only; name literally says "legacy" — `AGENTS.md` says remove legacy paths |

**Fix:** delete. If any is a deliberately-public helper (e.g. `is_expired` for a future pool), document that and add a real caller; otherwise it's clutter.

---

## M7 — 34 unused imports (incomplete-refactor signal)
Full list available on request. The notable cluster: `scan_content`, `apply_scan`, `quarantine_enabled` are imported-but-unused in `ingest/pipeline.py:45`, `ingest/episodic_writer.py:12`, `ingest/ingest_service.py:34`, and `memory/episodic/core.py:39`. These are leftovers from before write-time scanning was consolidated into `scan_artifact_text(...)` (which *is* correctly wired — see *What's solid*). Harmless at runtime, but a clear sign the scan refactor left dead imports behind (`AGENTS.md §9`: "removal of obsolete code paths").

Others include `Tuple` (`chunk_sql.py:11`), `UMAMemory` (`api/management.py:24`, `common/initializers/runtime.py:27`), `quarantine_enabled` (`api/memory.py:86`), and several unused type-only imports in `consolidation/feature.py` and `procedural/feature.py`.

**Fix:** strip them; a linter in CI (ruff/pyflakes) would prevent recurrence.

---

## M8 — Pervasive silent broad exception handlers in core paths
78 broad, trivial-body handlers (`except Exception: pass|continue|return ...` with no logging), concentrated in retrieval: `retrieve/ranking.py` (`:89`, `:354`), `retrieve/rlm/controller.py` (~15 instances incl. `:562-574`, `:766`, `:798`, `:1078`), `retrieve/rlm/coverage.py`, `retrieve/rlm/domain.py`, `retrieve/rlm/environment.py`.

Many are legitimately best-effort (e.g. attaching optional debug-scoring metadata, `ranking.py:89` — commented as optional). But the volume and their concentration in core retrieval run against `AGENTS.md §3`: "Never swallow exceptions silently in core flows … Do not add defensive catch-all logic that hides broken invariants." At least some of these almost certainly mask real defects rather than tolerate optional data.

**Fix:** audit each; downgrade genuinely-optional ones to `logger.debug(..., exc_info=True)`; let real errors surface. Don't blanket-`pass`.

---

## L1 — Intent-routing table in `ARCHITECTURE.md` is stale vs `planner.py`
- Doc: `PERSONAL → profile + procedural`. Code: `(profile, procedural, semantic, episodic)` (`planner.py:152`).
- Doc: `MIXED → all four lanes`. Code returns **five** lanes (`planner.py:154`).

**Fix:** update the doc table to match the planner.

---

## L2 — Lane vocabulary mismatch across docs and code
`KB_LANES` (`common/storage_metadata.py:26`) defines **seven** lanes incl. `profile` and `trace`. `ARCHITECTURE.md`'s table lists six and omits both. Meanwhile the planner itself treats `trace` as "not a retrieval lane" (`planner.py:208`), and `profile` is really semantic facts with `kind=profile_fact` sharing the semantic store (`storage_metadata.py:56,74`; advertised alongside `semantic` in `runtime.py:_available_retrieval_lanes`). So `profile` and `semantic` can double-count the same store.

**Fix:** reconcile the lane list in one place; either promote `profile`/`trace` to first-class in the docs or drop them from `KB_LANES`. Clarify the profile/semantic store-sharing so retrieval doesn't double-count.

---

## L3 — Ambient agent-scope default (`set_context`)
`set_context(agent_id=...)` stores `self._agent_id` on the shared instance (`api/memory.py:390-401`), and `_resolve_runtime_context` falls back to it then to `"agent-default"` (`api/memory.py:301`). This is the `memory.agent_id` shared-mutable-request-scope pattern `AGENTS.md §1` explicitly forbids and that your own enterprise design notes flag. Today it's only a per-call *default* (each call can still pass its own scope), so it's not yet a leak — but it's the wrong default for the multi-agent direction the design docs describe.

**Fix:** keep `agent_id` strictly per-call in `RuntimeContext`; if a convenience default is wanted, make `set_context` store an immutable default that callers must still opt into, and add the cross-agent isolation test from the enterprise conversation to CI.

---

## Oversized unit worth noting (not a defect, a maintainability risk)
`RLMController.retrieve_context` is a single method spanning `controller.py:151-736` (~585 lines with no intervening method). `AGENTS.md §7` values "easy to trace end-to-end / one obvious path." A method this large is hard to trace and hard to test in isolation.

**Fix (larger effort):** extract the per-step navigation body and the stop/coverage evaluation into named helpers.

---

## What's solid (keep)
- **Store-layer ownership scoping is disciplined.** Every SELECT I traced builds its `WHERE` from `tenant_id/owner_type/owner_id` via `_require_scope`/`_scope_where`, and quarantine exclusion (`AND quarantined_at IS NULL`) is applied at the store boundary as documented. The DAT "impossible by construction" intent holds at the query layer.
- **Write-time injection scanning is real and wired** via `scan_artifact_text(...)` on document chunks (`ingest_service.py:755,1299`), turn chunks (`pipeline.py:742`), episodes (`memory/episodic/core.py:121`), and bootstrap/diary writers (`ingest/episodic_writer.py:80,200`).
- **Manifest-after-embed invariant is correctly implemented** (`ingest_service.py:686-720`) with explicit comments — embedding is attempted first, and the manifest is written only on success, so re-ingest stays safe.
- **Ranking logic is centralized** in `retrieve/ranking.py` (consumed by the controller via `Ranker`), matching the "one ranking module" rule.
- **Store class hierarchy is clean:** `BaseSQLStore → BaseVectorSQLStore → {Chunk, Episodic, Procedural, Semantic}`, `DocumentSQLStore → BaseSQLStore`. (The duplication in M5 is in leaf methods, not the hierarchy.)
- **Secrets-provider resolution is carefully validated** (`api/memory.py:245-286`): import-path parsing, class check, `SecretsProvider` subclass check, and construction all raise with clear messages.

---

## Suggested order of work (effort × impact)
1. **H1** ship the config files + `MANIFEST.in` (minutes; unblocks first run).
2. **H2 / H3 / M1** decide-and-delete the three half-wired features (defer queue, consolidation auto-trigger, LLM-decision scaffolding). Mostly deletions; each removes a correctness trap or a misleading claim.
3. **M6 / M7** delete dead functions and unused imports; add ruff/pyflakes to CI to hold the line.
4. **M5** lift the quarantine methods into the base store; unify the scope-helper name.
5. **M2 / M3 / M4** collapse the parallel graph/context/handle paths onto one canonical route.
6. **M8** audit the silent excepts in retrieval.
7. **L1–L3** doc/vocab reconciliation and the ambient-scope default.
8. Refactor the 585-line `retrieve_context` when touching the controller next.
