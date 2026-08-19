# UMA Roadmap — Verified Gaps

Derived from a code-verified assessment of a competitor (gbrain) feature review,
2026-08-10. Every item below was checked against the codebase; claims from the
review that did not survive verification are recorded in
[Not gaps](#not-gaps--closed-with-evidence) so they do not resurface.

Ordered by impact, not by effort.

---

## G1 — No fact-level contradiction detection

**Impact: high · Effort: 5–8 days · Status: open**

Competing `(subject, predicate)` values silently coexist as separate rows. UMA has
no notion of one fact superseding another.

The conflict resolver does not do this job and never did.
`SemanticSQLStore._resolve_conflict` (`uma/stores/semantic_sql.py:417`) pins
`object=?` in its WHERE clause:

```sql
WHERE tenant_id=? AND owner_type=? AND owner_id=? AND subject=? AND predicate=? AND object=?
```

It therefore only ever matches *identical* facts, so `_archived`
(`uma/stores/semantic_sql.py:446`) is empty for competing values. Discarding true
duplicates is correct behavior — nothing is being wrongly dropped. The gap is that
contradictions are never detected in the first place.

The only contradiction logic in the codebase is `detect_contradictions`
(`uma/retrieve/rlm/coverage.py:280`), a coarse verb-polarity keyword heuristic over
episode *summaries*, explicitly documented as "NOT a semantic truth checker." It
does not touch facts.

**Prerequisite — predicate cardinality.** This cannot be built as a storage change
alone. UMA has no model of whether a predicate is single-valued or multi-valued, so
there is no basis to decide whether `prefers_language: Python → Go` is a
contradiction or an evolution. Define cardinality first; supersession follows from it.

**Reuse, do not invent.** The `superseded_by` / `superseded_at` pattern already
exists on document manifests (see `tests/test_memory_management.py:1026`). Extend
that canonical path rather than introducing a second supersession concept.

Unlocks G4's third signal and any future conflict-flagging in consolidation.

---

## G2 — RLM retrieval loop is opaque

**Impact: medium–high (operability) · Effort: ~3 days · Status: open**

When the RLM loop returns something surprising there is no way to see which
iteration surfaced which artifact, which decision the controller took, or why the
loop terminated. Per-artifact scoring is covered (see G3); per-*stage* attribution
is not.

Attaches to `uma/retrieve/rlm/controller.py` and `uma/retrieve/rlm/decisions.py`.

Purely operational. Prioritized above cheaper items because it makes every
subsequent retrieval change debuggable.

---

## G3 — `score_card` is computed but never reaches callers

**Impact: medium (operability) · Effort: ~0.5 day · Status: DONE**

Wired `include_debug` through to score-card emission:
`UMAMemory.retrieve_context` / `retrieve_memory` → `UMARuntime` →
`RetrievalRequest.debug` → `ContextPack.debug` → `Ranker.rank_*(debug=...)` →
`_emit_scorecards`. Mirrors the existing `query_scan_severity` threading exactly.

The flag is passed **per call** rather than set on the Ranker: a single Ranker is
constructed once at runtime init (`uma/common/initializers/runtime.py:178`) and
shared across concurrent requests, so mutating instance state per request would
race. The global `retrieval.debug_scores` config default is preserved — either
source enables emission.

`retrieve_memory` previously did not forward `include_debug` to its internal
`retrieve_context` call; it does now, so the memory path gets cards too.

`ScoreCard` (`uma/retrieve/ranking.py:360`) carries `vector_score`,
`lexical_score`, `rerank_score`, `route`, `method`, `final_score`, `trust_score`,
and `final_score_with_trust` per candidate; `_emit_scorecards`
(`uma/retrieve/ranking.py:569`) writes it onto each object's `meta`, which
`_chunk_payload` already serializes out.

Still open in this area: nothing *aggregates* the emitted cards into a single
ranked view. That is closer to G2's per-stage attribution and is tracked there.

---

## G4 — Gap analysis not surfaced in `retrieve_memory`

**Impact: medium · Effort: 1–2 days · Status: DONE (reduced scope, as planned)**

New `uma/retrieve/gaps.py` — a pure, total function, no I/O — surfaced as
`MemoryResult.gaps`. Two signals, as scoped:

- `stale_support` — the **newest** chunk supporting a fact is older than
  `retrieval.gap_max_support_age_days` (default 180). Keying on the freshest
  support means one recent corroboration clears the flag.
- `weak_support` — the fact rests on a single chunk whose `trust_score` is below
  `retrieval.gap_min_support_trust` (default 0.6). That bar sits **above**
  `min_trust_score` deliberately: chunks below the retrieval floor are already
  filtered out, so reusing that value would flag nothing.

A fact can appear under both reasons; they are independent and acted on
differently.

**Reporting only.** Nothing here filters, reranks, or alters trust or
provenance — a flagged fact is still returned in `facts`, pinned by test.

Computed from the **raw** `context.facts` / `context.chunks`, not the
serialized projections: `_chunk_payload` drops `created_at` and `trust_score`,
which are precisely the two signals needed.

Facts with no resolvable supporting chunk are deliberately **not** reported —
that is already covered by provenance invalidation, and duplicating it would
give operators two places to look for one problem.

The third signal from the original review — "pairs where the resolver dropped a
competing value" — remains **vacuous** until G1 lands, because the resolver never
drops competing values (see G1). It arrives free with G1.

---

## G5 — Promotion duplicate guard is scoped only by `turn_id`

**Impact: medium · Effort: 1–2 days · Status: DONE**

Added `SemanticSQLStore.durable_fact_exists(fact, *, tenant_id, owner_type,
owner_id)`, delegated through `SemanticCore`, and gated the promotion path on it
in `MemoryPipeline._maybe_promote_facts` — **before** `policy.promote()` mints
the durable Fact, so a duplicate costs one indexed lookup and nothing else.

Binary exists/novel, no similarity threshold, as scoped.

**Matched on `(subject, predicate, object)` rather than `content_hash`.** The
two are equivalent by construction — `content_hash` *is* the SHA-256 of that
tuple — but the column is nullable and NULL on rows written before it existed,
so hash matching would silently pass duplicates through on older data. The tuple
is also the comparison `_resolve_conflict` already uses, so this adds no new
notion of fact identity. Backed by `idx_facts_owner_sub_pred`.

Fails **open**: a store or guard error logs and treats the fact as novel, since
promotion is best-effort and a guard fault must not block legitimate promotions.

Also fixed while here: `await_pending_background` now discards drained tasks
explicitly. `add_done_callback` fires via `call_soon`, so the set was not
reliably empty when the drain returned — latent before, and the extra await in
the promotion path made it reproducible.

Original diagnosis, for the record: `_find_idempotent_duplicate`
(`uma/stores/semantic_sql.py:369`) matches on `(subject, predicate, object)`
**plus `meta.turn_id`**, so the same fact arriving from two different turns
passed the guard and was promoted twice.

---

## G6 — `trust_score` per source at write time

**Impact: n/a · Effort: none · Status: NOT A GAP — already implemented**

This entry was wrong. Verified 2026-08-11: `trust_score` **is** assigned per
source at write time, and has been.

`uma/common/trust.py` owns the policy — `score_source(SourceDescriptor(...))`
maps a provenance kind to a score: `turn_user` 0.9 (authenticated session),
`turn_assistant` 0.7, `document` 0.7, bootstrap 0.8 manual / 0.6 default,
`tool_output` 0.5, `promotion` inherits the parent, unknown 0.5.

It is wired into every persisted write path:

| Path | Site | Kind |
|---|---|---|
| Turn chunks | `uma/ingest/pipeline.py:731` | `turn_user` / `turn_assistant` |
| Turn facts | `uma/memory/semantic/core.py:231` | turn `source_kind` |
| Document chunks | `uma/ingest/ingest_service.py:743` | `document` |
| Document facts | `uma/ingest/ingest_service.py:907` | `document` |
| Bootstrap memory | `uma/ingest/ingest_service.py:1258` | `bootstrap_memory` |
| Document episodes | `uma/ingest/episodic_writer.py:71` | `document` |
| Diary episodes | `uma/ingest/episodic_writer.py:185` | `bootstrap_diary` |
| Promotion | `uma/memory/promotion.py:529` | `promotion` (inherits) |

End-to-end coverage already exists: `tests/test_process_turn.py:45-46` pins
user-derived facts at 0.9 and assistant-derived at 0.7;
`tests/test_ingest_pipeline.py:144,298` pin document facts and chunks at 0.7;
`tests/test_security_trust.py` covers the policy across every kind.

Two things that look like gaps and are not:

- `working_memory/core.py:147` passes `trust_score=1.0` into `scan_artifact_text`.
  That is a transient scan baseline for the live buffer, not persisted artifact
  trust — working memory has no trust-aware ranking layer to consume it.
- `tool_output` is defined and tested but has no production call site, because
  UMA has no tool-output ingestion surface. Adding one would mean inventing that
  surface, not fixing trust assignment.

**Open product question, not a defect.** `document` (0.7) ties `turn_assistant`
(0.7) and loses to `turn_user` (0.9). The original competitor review wanted
curated documents to outrank conversational content; the shipped policy does the
opposite, which matches the reasoning in the original assessment — a user's
directly stated preference should outrank a stale document. Changing these
numbers is a decision about what UMA should trust, not a bug fix, and needs an
explicit call before anyone edits `score_source`.

**Still correct and still rejected:** adding an `owner_type` / source-tier term
to the ranker. That would create a second parallel weighting axis for a concern
`trust_score` already owns.

---

## G7 — Subject normalization has no casefolding or aliasing

**Impact: low standalone · Effort: medium · Status: deferred — depends on G1**

`_normalize_subject` (`uma/memory/semantic/extractor_utils.py:228`) only normalizes
whitespace and truncates to 12 words. No casefolding, no alias resolution — "K8s"
and "Kubernetes" are separate rows, and indexes key on raw `(subject, predicate)`.

Deferred deliberately:
- Retrieval is vector-first with lexical fusion, so embeddings already bridge these
  variants at *retrieval* time. An alias table improves *merge/resolution*
  precision, which has no consumer until G1 exists.
- An alias table is a standing maintenance liability: LLM-populated makes it a new
  trust surface; hand-curated rots.

Revisit only after G1.

---

## G8 — Evidence attribution is artifact-level, not claim-level

**Impact: low relative to cost · Effort: high · Status: deferred**

The compiled-truth / evidence split is **already largely built** in
`uma/common/compiled_memory.py`:

- rewritable compiled body (`text` / `summary`)
- evidence set (`direct_source_chunk_ids` + transitive collection)
- append-only timeline (`compiled_memory_log`, appended not replaced)
- `provenance_valid` + `_invalidate_provenance` on unreachable raw chunks
- `conflicts`, `has_conflicts`, `conflict_count` on the index entry

What remains is **per-claim** evidence IDs — evidence attaches to the artifact, not
to individual sentences. That residual is the expensive part: it needs claim
segmentation plus per-claim attribution, making it LLM-dependent and an evaluation
problem rather than a schema problem.

Migrating the JSON-in-`meta` representation to FK'd rows is separable and worth
doing only against a concrete query requirement.

---

## Suggested sequence

| Order | Item | Effort | Rationale |
|-------|------|--------|-----------|
| ~~1~~ | ~~G3~~ | ~~0.5d~~ | **Done** — cheapest; makes everything below observable |
| ~~2~~ | ~~G5~~ | ~~1–2d~~ | **Done** — self-contained duplicate guard |
| ~~3~~ | ~~G4~~ | ~~1–2d~~ | **Done** — two signals only |
| ~~4~~ | ~~G6~~ | — | **Closed** — verified already implemented, no work needed |
| 5 | G2 | 3d | Operability before the largest change |
| 6 | G1 | 5–8d | The prize; retroactively unlocks G4.3 and G7 |

### Test-suite health (resolved 2026-08-11)

The suite is fully green: **775 passed, 2 skipped, 0 failed**. Four separate
issues were masking each other:

- **Missing parser deps.** `beautifulsoup4` / `markdown` / `PyPDF2` / `pandas`
  were not installed, so 18 HTML/markdown **sanitization** tests failed at
  import — a security-relevant lane silently untested. The declaration in
  `[dev]` (via `uma-mem[parsers]`) was already correct; only the environment
  was stale.
- **Store-format mismatch.** `test_episodic_fetch_summaries_owner_scoped` used a
  fixed path outside the test tree, so a DB written by an older build survived
  between runs and tripped the `uma_store_meta` format check with a stale
  `'uma-rlm'`. `'uma'` is and remains the correct value — nothing in the source
  ever produced `'uma-rlm'`. Now uses `tmp_path`.
- **Entry-point discovery.** `uma_entry_point()` looked only in the interpreter
  scheme's script directory, missing `--user` installs (pip's fallback when the
  base install is not writable). Now checks the user scheme and PATH too.
- **Deadlock on any barrier-party failure.** See below.

`pytest-timeout` is now a `[dev]` dependency with `timeout = 60` and
`timeout_method = "thread"` (signal-based timeouts are POSIX-only). The six
`threading.Barrier` waits in `test_concurrency.py` / `test_isolation_and_tenancy.py`
are bounded by `_BARRIER_TIMEOUT_S`. Previously, one party raising before it
reached the barrier left the other blocked on a non-daemon `asyncio.to_thread`
pool thread; the test reported `F` and then the interpreter hung forever joining
it at shutdown. That is now a `BrokenBarrierError` and an honest failure.

G7 and G8 stay deferred pending G1 and a concrete query need respectively.

---

## Not gaps — closed with evidence

Recorded so these do not resurface as proposals.

### Consolidation scheduling — by design, not missing

`Consolidator` (`uma/memory/consolidation/consolidator.py`) is fully implemented:
`run_once`, clustering, summarizing, `_persist_facts`, `_prune`. It is exposed as a
public feature method, `await memory_client.consolidation_run(user_id)`
(`uma/memory/consolidation/feature.py`), config-gated by `consolidation_enabled`.

Nothing invokes it automatically **on purpose**. In UMA core, consolidation is
caller-invoked; users who want a scheduler wire one themselves against that public
method. Automation and full orchestration are **enterprise-tier**. `CHANGELOG.md`
records that the earlier `consolidation_trigger.py` auto-trigger scaffolding was
intentionally removed, not left unfinished.

Do not propose adding schedulers, cron hooks, or fire-and-forget auto-triggers for
consolidation to core. The fire-and-forget promotion pass in
`uma/ingest/pipeline.py:278` sits on the reply path and is a different case — it is
not a template for scheduling core maintenance jobs.

### Deterministic entity auto-linking — already exists

`GraphUpdater` (`uma/memory/graph/updater.py`) already creates deterministic,
zero-LLM typed edges from extracted facts: `add_facts_batch`, `add_fact`, and
`link_episode_to_facts` (`:305`) produce `(Episode)-[:MENTIONS]->(Fact)` and
`(Episode)-[:<PREDICATE>]->(Entity)`, with ownership stamped on every edge and
`_sanitize_predicate` applied to the predicate.

The requested guardrail — an edge must never widen trust, bypass quarantine, or
promote a fact — is already structurally true: edges are written by the updater and
never touch the gated promotion path.

Graph *adapter* maturity (`uma/adapters/graph/base.py`, 81 lines) is a separate
concern and not tracked here.

### Trust weighting in ranking — already exists

See G6. Ranking already consumes `trust_score`. Only the write-time assignment is
open, and it belongs upstream.
