---
name: uma-quarantine
description: Complete quarantine lifecycle in UMA — when a record gets quarantined (write-time injection scan, integrity verification failure), how quarantined records behave at retrieval (excluded by SQL WHERE clauses), the management API to list, reinstate, or purge quarantined records, the audit-log entries quarantining writes to artifact meta, and how this composes with trust scoring and min_trust_score. Use this skill when answering questions about what happens to a flagged record, how to recover from a false-positive quarantine, what's in `quarantined_at`, how to view all quarantined facts, how integrity failures get handled, or any question about the quarantine semantics across the typed lanes.
---

# UMA — Quarantine Lifecycle

Quarantine is UMA's primary mechanism for tolerating untrusted content without losing it. A quarantined record stays in the database but is excluded from every retrieval query. The record can be inspected, reinstated, or purged through the management API.

Quarantine is part of UMA's **Pre-Write Sanitization** layer (layer 1 of the defense-in-depth model for ASI06 memory poisoning): high-severity injection hits are retained as forensic evidence rather than silently dropped, so operators can review what was attempted. The **Provenance Tracking** layer (layer 2) extends this with SHA-256 integrity verification — `verify_integrity` can quarantine records post-write if their content hash drifts.

This skill covers the full lifecycle: how a record gets quarantined, what happens at retrieval, and how to manage quarantined records.

---

## What Triggers Quarantine

Three paths set `quarantined_at`:

### 1. Write-time injection scan (high severity)

Every storage write boundary scans the artifact text against the injection pattern catalog. High-severity hits trigger quarantine:

```python
# uma/common/injection_scan.py::scan_artifact_text
def scan_artifact_text(text, trust, meta, *, log_context, now=None):
    result = scan_content(text)
    if result.severity == "high" and quarantine_enabled():
        trust = 0.0
        meta["security"]["injection_scan"] = {...}
        return trust, meta, datetime.utcnow()   # quarantined_at set
    elif result.severity == "medium":
        trust *= 0.5                            # not quarantined
    elif result.severity == "low":
        trust *= 0.8                            # not quarantined
    return trust, meta, None
```

This fires on:

- Document chunks during `ingest_document`
- Turn chunks during `process_turn` (via `_store_turn_chunks`)
- `assistant_reply` during `EpisodicCore.store_episode`
- Each appended message in `WorkingMemoryCore.append`

### 2. `process_turn` user_msg high severity

A high-severity `user_msg` is special — it raises `InjectionDetectedError` and **nothing is stored**. The turn is dropped entirely; there is no quarantined record because the write never happened.

### 3. Integrity verification failure

`verify_integrity` re-derives the canonical content hash and compares to the stored `content_hash`. On mismatch:

```python
result = await verify_integrity(memory, record_id="fact-abc", lane="semantic", ...)
# result.status == "failed"
# result.quarantined == True   (the record was just quarantined)
```

The record is quarantined through the same `quarantined_at`-set path, and an `integrity_failure` entry is appended to `meta.security.audit_log`.

`lint_memory_drift` automatically routes typed lane artifacts through `verify_integrity`, so quarantine can happen as a side effect of a drift check.

---

## What Quarantine Means

| Property | Value |
|---|---|
| `quarantined_at` | UTC datetime when the record was quarantined |
| `trust_score` | Reduced to `0.0` for write-time scan hits; preserved on integrity failures |
| `meta.security.injection_scan` | Rule names, severity, score (for scan hits) |
| `meta.security.audit_log` | Append-only entries for integrity failures |

The record is **retained in the database**, **not deleted**. It can be inspected, reinstated, or purged.

---

## Retrieval Behavior

Every retrieval query at every typed lane filters out quarantined records:

```sql
-- Embedded into every search query at every store
... AND quarantined_at IS NULL
```

This applies at:

- `ChunkSQLStore.search` and `_fetch_ranked_rows_by_ids`
- `SemanticSQLStore.search` and `fetch_by_ids`
- `EpisodicSQLStore.search` and `fetch_by_ids`
- `ProceduralSQLStore.search`
- The canonical `_fetch_ranked_rows_by_ids` in `base_vector_sql_store`

Working memory quarantine is filtered in Python:

```python
# uma/memory/working_memory/buffer.py
def get_context(self, *, include_quarantined: bool = False):
    if include_quarantined:
        return self._messages
    return [m for m in self._messages if m.quarantined_at is None]
```

Pass `include_quarantined=True` only when you specifically want to see flagged messages — never in normal retrieval.

---

## Conflict Resolution Awareness

`LatestWinsFactResolver` (canonical fact upsert) excludes quarantined facts:

```python
eligible = [f for f in facts if f.quarantined_at is None]
if eligible:
    canonical = max(eligible, key=updated_at)
else:
    # All candidates quarantined — fall back across the full set
    # but warn; the resulting canonical remains quarantined and
    # will be filtered at retrieval anyway.
    canonical = max(facts, key=updated_at)
    logger.warning("all candidates quarantined; canonical remains quarantined")
```

This means a quarantined fact cannot become the canonical row for a `(subject, predicate)` slot while a non-quarantined alternative exists. If all alternatives are quarantined, the slot has a quarantined canonical row and retrieval returns nothing for that slot until something legitimate arrives.

---

## Management API

```python
from uma.api.management import (
    list_quarantined,
    reinstate_quarantined,
    purge_quarantined,
)
```

### `list_quarantined`

```python
records = await list_quarantined(
    memory,
    lane="semantic",            # one of: semantic, episodic, procedural, raw
    owner_type="user",
    owner_id="user-123",
    tenant_id="default",
    limit=100,
)
```

Returns a list of records currently quarantined for the given scope and lane. Each record carries its original fields plus `quarantined_at`, `trust_score`, and `meta["security"]`.

Use this to:

- Triage false positives — recent quarantines may be legitimate user input that hit an aggressive pattern
- Audit injection attempts — review what was attempted against your deployment
- Verify the scanner is calibrated correctly for your domain

### `reinstate_quarantined`

```python
await reinstate_quarantined(
    memory,
    lane="semantic",
    record_id="fact-abc",
    owner_type="user", owner_id="user-123",
    tenant_id="default",
)
```

Clears `quarantined_at` (sets to NULL). The record is now retrievable again. `trust_score` is **not** automatically restored — the caller decides whether to update it separately if needed.

Use this for false positives. **Reinstate carefully.** If a pattern flagged a record, the pattern matched something. Confirm the content is benign before reinstating.

### `purge_quarantined`

```python
await purge_quarantined(
    memory,
    lane="semantic",
    record_id="fact-abc",
    owner_type="user", owner_id="user-123",
    tenant_id="default",
)
```

Permanently deletes the record. Irreversible. Use for confirmed-malicious content where keeping the row for forensics adds no value.

The associated vector-index entry is also removed.

---

## What Lives in `meta.security`

Each quarantined record carries a `meta.security` blob describing why it was flagged:

```json
{
  "injection_scan": {
    "severity": "high",
    "matched_rules": ["prompt_reset", "role_impersonation"],
    "score": 0.93,
    "scanned_at": "2026-05-25T10:42:11Z"
  },
  "audit_log": [
    {
      "event": "integrity_failure",
      "at": "2026-05-25T11:13:08Z",
      "expected_hash": "...",
      "actual_hash": "...",
      "verified_by": "verify_integrity"
    }
  ]
}
```

`audit_log` is append-only. New events are added on subsequent integrity checks, re-quarantines after reinstatement, etc.

---

## Composition With Trust Scoring

Quarantine and trust scoring are two layers of the same defense:

| Severity | Quarantined? | Trust | Visible at retrieval? |
|---|---|---|---|
| `none` | No | Unchanged (default 0.5) | Yes |
| `low` | No | × 0.8 → 0.4 | **No** (below `min_trust_score: 0.5`) |
| `medium` | No | × 0.5 → 0.25 | **No** (below `min_trust_score: 0.5`) |
| `high` | **Yes** | 0.0 | No (SQL filter) |

`min_trust_score: 0.5` is calibrated so that low and medium hits are filtered by the trust gate even though they're not quarantined. If you raise `min_trust_score` above 0.5, lower-severity hits stay visible. If you lower it below 0.4, low hits become visible again.

**Set `security.quarantine_enabled: false`** to drop flagged artifacts instead of retaining them:

```yaml
security:
  quarantine_enabled: false
```

This means a high-severity hit becomes a hard drop, not a quarantine. The artifact is never written. Forensics are lost; retrieval is unchanged. Useful for high-throughput pipelines where retaining flagged data is a compliance liability.

---

## Inspecting Quarantine in `lint_memory_drift`

`lint_memory_drift` reports findings that include quarantine status:

```python
result = await lint_memory_drift(memory, artifacts, user_id="...")

for finding in result["findings"]:
    if finding["category"] == "integrity_failure":
        # The record was just quarantined by verify_integrity
        print(f"Quarantined {finding['lane']}/{finding['record_id']}")
    elif finding["category"] == "stale_provenance":
        # Older than stale_after_seconds; not quarantined
        ...
```

`integrity_failure` findings are the side effect of quarantine triggered during the drift check. Run drift checks periodically; treat new `integrity_failure` findings as alert-worthy.

---

## Common Patterns

### Daily quarantine triage

```python
from uma.api.management import list_quarantined

for lane in ("semantic", "episodic", "procedural", "raw"):
    records = await list_quarantined(
        memory, lane=lane,
        tenant_id=TENANT,
        owner_type="user", owner_id=USER,
        limit=500,
    )
    for r in records:
        if r.quarantined_at < cutoff_ago_24h:
            continue  # skip old
        review(r)
```

### Bulk-purge after pattern catalog tightening

If you tightened a pattern and want to purge records flagged by old (false-positive) versions of the same pattern, list them, filter by rule name in `meta.security.injection_scan.matched_rules`, and `reinstate` or `purge` selectively. UMA does not ship a bulk-reinstate API by design — every reinstatement is a deliberate decision.

### Integrity sweep

```python
from uma.api.management import lint_memory_drift

drift = await lint_memory_drift(memory, all_my_artifacts, user_id=USER, stale_after_seconds=86400)
new_failures = [f for f in drift["findings"] if f["category"] == "integrity_failure"]
if new_failures:
    alert(new_failures)
```

Run this nightly. The typed lane artifacts are auto-routed through `verify_integrity`; mismatches surface as new quarantines plus findings.

---

## What Quarantine Is Not

- **Not a moderation system.** UMA's pattern catalog detects prompt-injection-shaped content. It does not detect toxicity, PII, copyright issues, etc. Pair with a separate moderation layer if you need those.
- **Not deletion.** Quarantined records stay in the database. Use `purge_quarantined` for actual deletion.
- **Not a security guarantee against motivated adversaries.** Pattern matching is a baseline defense, not a complete one. For high-risk deployments, layer a model-based classifier and allow-list known-good sources.
- **Not free.** Each quarantined record still consumes SQL row space and a vector-index entry. Purge periodically if your scanner is aggressive.
