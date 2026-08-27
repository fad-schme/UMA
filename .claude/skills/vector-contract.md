---
name: uma-vector-contract
description: The C1 vector-index contract — how UMA's three vector backends (LanceDB, FAISS, InMemory) implement explicit tenant/owner isolation, the upsert and query signatures, the extra_filters/extra_metadata split, atomicity guarantees, score normalization via exp(-distance), SQL escape for predicate values, and what you need to know to write a custom backend. Use this skill when answering questions about how to add a new vector backend, why LanceDB has top-level columns for tenant/owner, how filter push-down works, what extra_filters versus extra_metadata is for, why the upsert signature uses parallel lists, how to interpret query scores, or any question about the C1 patch series and the vector adapter interface.
---

# UMA — Vector Index Contract

UMA's vector adapters (LanceDB, FAISS, InMemory) implement a single contract defined in `uma/adapters/vector/base.py`. The contract enforces tenant/owner isolation at the storage layer — not as an application-layer post-filter that runs after a possibly-truncated retrieval.

This contract is the **Memory Isolation** layer of UMA's defense-in-depth model for ASI06 memory poisoning: strict per-user isolation enforced at the storage layer so a malicious interaction with one user cannot poison the knowledge base for others.

This skill covers the contract itself, the three bundled implementations, and what a custom backend must provide.

---

## The Contract

```python
from typing import Any, Dict, List, Optional, Tuple

class VectorIndex(ABC):
    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        *,
        tenant_ids: List[str],
        owner_types: List[str],
        owner_ids: List[str],
        extra_metadata: Optional[List[Dict]] = None,
    ) -> None: ...

    @abstractmethod
    def query(
        self,
        vector: List[float],
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]: ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None: ...
```

The signatures are non-negotiable. Old-shape calls (`metadata=...` on upsert, `filters=...` on query, missing isolation kwargs) raise `TypeError`.

---

## Why This Shape

Before C1, UMA stored tenant/owner inside a serialized `metadata_json` blob. Filtering happened in Python after the backend returned its top-k. Under multi-tenant load, the top-k could be dominated by one tenant; other tenants saw silent empty results after the Python filter dropped everything.

The new shape enforces isolation at the type level rather than by convention:

- `tenant_ids` / `owner_types` / `owner_ids` are **explicit parallel-list parameters**, not buried inside a metadata dict. Each list must be the same length as `ids`.
- `query` requires `tenant_id`, `owner_type`, `owner_id` as keyword-only parameters. Empty values raise `ValueError`.
- Adapters push these into the backend's native predicate (where possible) **before** the candidate cap is applied.
- `extra_metadata` / `extra_filters` carry the non-isolation keys — `doc_id`, `kind`, `kb_lane`, etc.

`extra_metadata` MUST NOT contain `tenant_id`, `owner_type`, or `owner_id`. Reserved-key violations raise `ValueError` at upsert time. This prevents callers from accidentally double-storing isolation values and ensures the source of truth is the explicit parameter.

---

## LanceDB Implementation

LanceDB is the recommended backend for multi-tenant deployments. The table schema:

```
id              TEXT PRIMARY KEY
vector          FLOAT[dim]
tenant_id       TEXT       -- promoted from metadata_json
owner_type      TEXT       -- promoted
owner_id        TEXT       -- promoted
metadata_json   TEXT       -- everything else (kind, kb_lane, doc_id, etc.)
```

`query` builds a `WHERE` clause and pushes it into LanceDB's DuckDB engine:

```python
where = (
    f"tenant_id = '{_sql_escape(tenant_id)}' "
    f"AND owner_type = '{_sql_escape(owner_type)}' "
    f"AND owner_id = '{_sql_escape(owner_id)}'"
)
table.search(vec).where(where).limit(limit).to_list()
```

The cap (`limit`) is applied **after** scope narrowing. Cross-tenant rows cannot leak past the storage layer.

### SQL Escape

Values are escaped via standard SQL single-quote doubling (`'` → `''`), the same pattern LanceDB's `_delete_from_table` already used for id-list construction, before being interpolated into the `WHERE` clause DuckDB executes.

### Score Normalization

LanceDB returns metric-dependent raw `_distance`. UMA normalizes via:

```python
d = max(0.0, float(distance))
score = math.exp(-d)
```

This maps `[0, ∞)` to `(0, 1]`, monotonically, so it's coherent with `trust_score ∈ [0, 1]` in the trust-weight blend. Without it, the blend mixes incompatible scales and produces incoherent rankings.

---

## InMemory Implementation

Storage uses three parallel dicts keyed by id:

- `_vectors: Dict[str, List[float]]`
- `_scopes: Dict[str, Tuple[str, str, str]]` — `(tenant_id, owner_type, owner_id)`
- `_extra: Dict[str, Dict[str, Any]]`

`query` iterates `_vectors`, applies the isolation check first (skip if `_scopes.get(id) != scope_key`), then applies `extra_filters`, then computes cosine similarity. Isolation runs before similarity — no cross-scope row reaches the cosine computation.

**Intended use:** development, CI, tests. Not for production data volumes (O(N) scan).

---

## FAISS Implementation

FAISS does not support pushed-down metadata predicates — the library only knows about vectors and integer IDs. The adapter compensates with **oversampling**:

```python
oversample_k = min(int(self.index.ntotal), max(k * 4, k))
scores, idxs = self.index.search(arr, oversample_k)
```

After FAISS returns the top `oversample_k` candidates, the adapter applies the isolation filter in Python. Then `extra_filters`. Then truncates to `k`.

This is a **heuristic, not a guarantee**. Under heavy cross-tenant load FAISS can still suffer recall loss — the docstring documents this. **For multi-tenant deployments, use LanceDB.** FAISS is fine for single-tenant or smaller deployments.

The oversample multiplier is a class attribute (`_oversample_multiplier = 4`). Increase it for higher recall at the cost of more in-memory filtering work per query.

---

## Atomicity Under Bad Input

All three adapters validate every row in a batch **before** mutating any state. A bad row in position N causes the entire upsert to raise `ValueError`; no rows from the batch are committed.

Implementation pattern:

```python
# Pass 1: validate all rows into local lists
prepared: List[tuple] = []
for sid, vec, tid, ot, oid, extra in zip(...):
    # validate isolation, dimensions, reserved keys
    prepared.append((sid, vec, (tid, ot, oid), dict(extra)))

# Pass 2: mutate state only after all validation passed
for sid, vec, scope, extra in prepared:
    self._vectors[sid] = vec
    self._scopes[sid] = scope
    self._extra[sid] = extra
```

This is the **atomicity invariant**: the index either commits all rows in a batch or none. Partial state would surface as cross-scope leaks (a vector present without a scope entry, looking like it belongs to nobody) or silent retrieval misses on subsequent queries.

LanceDB inherits this from its build-rows-then-`table.add(rows)` pattern. FAISS implements it explicitly via `prepared_scopes` / `prepared_extras` local lists.

---

## Reserved Keys

`extra_metadata` MUST NOT contain `tenant_id`, `owner_type`, or `owner_id`. Each adapter raises `ValueError` if it does:

```python
for reserved in ("tenant_id", "owner_type", "owner_id"):
    if reserved in extra:
        raise ValueError(
            f"extra_metadata must not contain reserved isolation key "
            f"{reserved!r}; pass via the explicit parallel-list parameter instead"
        )
```

This is a contract guard, not theater. Without it, callers could pass `extra_metadata={"tenant_id": "wrong-tenant"}` and silently store a row whose JSON blob disagrees with its column. Refusing at the boundary keeps the source of truth singular.

---

## Delete

`delete(ids: List[str])` is unchanged from the pre-C1 signature. UMA generates SQL-unique ids across all callers, so a cross-tenant id collision would require an id generator outside UMA's control. Scoping deletes by tenant would be ceremonial.

---

## Writing a Custom Backend

To implement a new backend (e.g. Weaviate, Pinecone, Qdrant):

1. Subclass `VectorIndex`.
2. Implement `upsert`, `query`, `delete` with the exact signatures above.
3. **Isolation is non-optional.** The contract requires you to respect `tenant_id`/`owner_type`/`owner_id` on every read. If your backend supports server-side filters (most cloud vector DBs do), push isolation into the filter. If not, accept the oversample-and-post-filter pattern and document the recall caveat.
4. Validate-all-then-commit on upsert. Partial state is a bug.
5. Refuse reserved keys in `extra_metadata` at upsert.
6. Refuse empty isolation values at upsert and query.
7. Normalize scores to `(0, 1]` or document the range and adjust `min_trust_score` / `trust_weight` to compensate.

Register your backend via `vector_backend` in YAML:

```yaml
storage:
  vector_backend: "my_package.my_module:MyVectorIndex"
  vector_config:
    # whatever your backend's __init__ takes
```

The factory passes `vector_config` as keyword arguments to your class's `__init__`. You must accept a `dim: int` argument (UMA's embedding dimension).

---

## Health-Check Pattern

`uma/common/health.py::_check_vector_index` probes connectivity by calling `query` with placeholder isolation values:

```python
index.query(
    [0.0] * dim,
    tenant_id="__health__",
    owner_type="system",
    owner_id="__health__",
    k=1,
)
```

We expect this to return `[]` (no real row matches `__health__`). What we're testing is that the query call goes through without raising. If your backend's `query` requires a live network connection, this is the touch point that surfaces a connectivity failure as a health-check error.

---

## Why Filter-Before-Cap Matters

Filtering by tenant *after* the k-nearest cap (the pre-C1 approach) lets a large tenant's rows fill the entire candidate pool before a small tenant's rows are ever considered, silently truncating that tenant's results to empty even though its documents are in the index. Filtering before the cap — what LanceDB's push-down and FAISS's oversample-and-post-filter both do — is the only way to guarantee a tenant's query sees its own top-k regardless of how much data other tenants have indexed. Push-down is preferred where a backend supports it; oversample-and-post-filter is the documented fallback.
