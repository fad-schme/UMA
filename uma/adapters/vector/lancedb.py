from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any, Dict, List, Optional, Tuple

from .base import VectorIndex

logger = logging.getLogger(__name__)

try:
    import lancedb  # type: ignore
except Exception as exc:  # pragma: no cover
    lancedb = None  # type: ignore
    logger.error("Failed to import lancedb: %s", exc)


def _sql_escape(value: str) -> str:
    """Escape a string for use as a single-quoted SQL literal.

    LanceDB uses DuckDB SQL under the hood and accepts standard
    single-quote-doubled escaping (`'` -> `''`). Matches the existing
    pattern used in `_delete_from_table` for id-list construction.

    The values passed here (tenant_id, owner_type, owner_id) come from
    validated internal call chains (DAT invariant: every artifact write
    populates them as non-empty strings), but we escape defensively as
    a belt-and-suspenders against any future caller path that doesn't
    enforce the invariant.
    """
    return value.replace("'", "''")


class LanceDBIndex(VectorIndex):
    """Persistent LanceDB-backed vector index for UMA's embedded lite profile."""

    def __init__(
        self,
        dim: int,
        *,
        path: str,
        table_name: str = "uma_vectors",
        search_k_multiplier: int = 8,
        search_k_max: int = 512,
        **_: Any,
    ) -> None:
        if lancedb is None:
            raise RuntimeError("lancedb is not installed. Install it with `pip install lancedb`.")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("LanceDBIndex: dim must be a positive integer.")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("LanceDBIndex: path must be a non-empty string.")
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError("LanceDBIndex: table_name must be a non-empty string.")

        self.dim = dim
        self.dimension = dim
        self.index = self
        self.path = path
        self.table_name = table_name.strip()
        self._search_k_multiplier = max(1, int(search_k_multiplier))
        self._search_k_max = max(1, int(search_k_max))
        self._lock = threading.RLock()
        self._db = lancedb.connect(path)

        logger.info(
            "Initialized LanceDBIndex path=%s table=%s dim=%d",
            path,
            self.table_name,
            dim,
        )

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        *,
        tenant_ids: List[str],
        owner_types: List[str],
        owner_ids: List[str],
        extra_metadata: Optional[List[Dict]] = None,
    ) -> None:
        self._validate_upsert_inputs(ids, vectors)
        if not vectors:
            logger.debug("LanceDBIndex.upsert called with empty vectors; no-op.")
            return

        # C1: validate parallel isolation lists. Length-mismatch is a caller
        # bug — refuse loudly rather than silently writing partial state.
        n = len(ids)
        if len(tenant_ids) != n:
            raise ValueError(
                f"LanceDBIndex.upsert: tenant_ids length ({len(tenant_ids)}) "
                f"does not match ids length ({n})."
            )
        if len(owner_types) != n:
            raise ValueError(
                f"LanceDBIndex.upsert: owner_types length ({len(owner_types)}) "
                f"does not match ids length ({n})."
            )
        if len(owner_ids) != n:
            raise ValueError(
                f"LanceDBIndex.upsert: owner_ids length ({len(owner_ids)}) "
                f"does not match ids length ({n})."
            )

        extra_list = extra_metadata or [{} for _ in ids]
        if len(extra_list) != n:
            raise ValueError(
                f"LanceDBIndex.upsert: extra_metadata length ({len(extra_list)}) "
                f"does not match ids length ({n})."
            )

        rows = []
        for sid, vector, tid, ot, oid, extra in zip(
            ids, vectors, tenant_ids, owner_types, owner_ids, extra_list,
        ):
            # Validate per-row isolation fields. Empty strings are explicitly
            # rejected — the architecture invariant requires them to be
            # populated.
            if not isinstance(tid, str) or not tid.strip():
                raise ValueError(
                    f"LanceDBIndex.upsert: tenant_id must be a non-empty string (id={sid!r})."
                )
            if not isinstance(ot, str) or not ot.strip():
                raise ValueError(
                    f"LanceDBIndex.upsert: owner_type must be a non-empty string (id={sid!r})."
                )
            if not isinstance(oid, str) or not oid.strip():
                raise ValueError(
                    f"LanceDBIndex.upsert: owner_id must be a non-empty string (id={sid!r})."
                )

            extra = extra or {}
            if not isinstance(extra, dict):
                raise ValueError("LanceDBIndex.upsert: extra_metadata items must be dicts.")
            # C1: refuse to silently double-store the isolation keys. If a
            # caller accidentally also puts them in extra_metadata, that's a
            # bug — the explicit parameters are the source of truth.
            for reserved in ("tenant_id", "owner_type", "owner_id"):
                if reserved in extra:
                    raise ValueError(
                        f"LanceDBIndex.upsert: extra_metadata must not contain "
                        f"reserved isolation key {reserved!r}; pass via the "
                        f"explicit parallel-list parameter instead (id={sid!r})."
                    )

            rows.append(
                {
                    "id": sid,
                    "vector": [float(value) for value in vector],
                    "tenant_id": tid.strip(),
                    "owner_type": ot.strip(),
                    "owner_id": oid.strip(),
                    "metadata_json": json.dumps(extra, sort_keys=True),
                }
            )

        with self._lock:
            table = self._get_or_create_table(seed_rows=rows)
            self._delete_from_table(table, ids)
            table.add(rows)

    def query(
        self,
        vector: List[float],
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        if len(vector) != self.dim:
            raise ValueError(
                f"LanceDBIndex.query: expected query vector dim={self.dim}, got={len(vector)}"
            )
        if not isinstance(k, int) or k <= 0:
            raise ValueError("LanceDBIndex.query: k must be a positive integer.")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("LanceDBIndex.query: tenant_id must be a non-empty string.")
        if not isinstance(owner_type, str) or not owner_type.strip():
            raise ValueError("LanceDBIndex.query: owner_type must be a non-empty string.")
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("LanceDBIndex.query: owner_id must be a non-empty string.")
        if extra_filters is not None and not isinstance(extra_filters, dict):
            raise ValueError("LanceDBIndex.query: extra_filters must be a dict or None.")

        table = self._open_table()
        if table is None:
            logger.debug("LanceDBIndex.query: table missing; returning [].")
            return []

        limit = min(max(k * self._search_k_multiplier, k), self._search_k_max)

        # C1: push isolation filter down into LanceDB so the cap applies
        # AFTER tenant/owner narrowing. Without this, heavy users in one
        # tenant can occupy the top-k globally and starve other tenants.
        #
        # SQL-quote escape for string-literal column values matches the
        # existing pattern in _delete_from_table. tenant/owner ids come
        # from validated internal call chains (DAT invariant) but are
        # quoted defensively.
        where = (
            f"tenant_id = '{_sql_escape(tenant_id.strip())}' "
            f"AND owner_type = '{_sql_escape(owner_type.strip())}' "
            f"AND owner_id = '{_sql_escape(owner_id.strip())}'"
        )
        try:
            rows = (
                table.search([float(value) for value in vector])
                .where(where)
                .limit(limit)
                .to_list()
            )
        except Exception:
            logger.exception("LanceDBIndex.query failed table=%s", self.table_name)
            raise

        results: List[Tuple[str, float]] = []
        for row in rows:
            sid = row.get("id")
            if not isinstance(sid, str) or not sid:
                continue

            # extra_filters: apply non-isolation predicates in Python.
            # Isolation already happened in the WHERE clause above.
            if extra_filters:
                meta = self._parse_metadata(row.get("metadata_json"))
                if any(meta.get(key) != value for key, value in extra_filters.items()):
                    continue

            # M3: normalize LanceDB's raw `_distance` (default L2) via
            # exp(-distance). Monotonic, maps [0, inf) to (0, 1], stable
            # across queries, coherent with trust_score's [0, 1] for
            # the trust-weight blend in retrieve/ranking.py.
            distance = row.get("_distance")
            if isinstance(distance, (float, int)):
                # Guard against negative distances from exotic metrics;
                # treat as 0 (perfect match) for safety.
                d = max(0.0, float(distance))
                score = math.exp(-d)
            else:
                score = 0.0
            results.append((sid, score))
            if len(results) >= k:
                break

        # NOTE: the M4 silent-truncation warning has been removed. With
        # C1's pushed-down isolation filter, the cap is applied AFTER
        # tenant/owner narrowing, so cross-tenant load can no longer
        # silently truncate a requesting tenant's recall. Any case where
        # the post-result is under k now reflects genuinely sparse data
        # within the requesting tenant's scope, not a multi-tenant bug.

        return results

    def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        table = self._open_table()
        if table is None:
            return
        with self._lock:
            self._delete_from_table(table, ids)

    def _open_table(self):
        try:
            return self._db.open_table(self.table_name)
        except Exception:
            return None

    def _get_or_create_table(self, seed_rows: List[Dict[str, Any]]):
        table = self._open_table()
        if table is not None:
            return table
        return self._db.create_table(self.table_name, data=seed_rows)

    def _delete_from_table(self, table: Any, ids: List[str]) -> None:
        escaped = [sid.replace("'", "''") for sid in ids if isinstance(sid, str) and sid]
        if not escaped:
            return
        predicate = "id IN ({})".format(", ".join(f"'{sid}'" for sid in escaped))
        table.delete(predicate)

    @staticmethod
    def _parse_metadata(raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, str) or not raw.strip():
            return {}
        try:
            value = json.loads(raw)
        except Exception:
            logger.exception("LanceDBIndex: failed decoding metadata_json")
            return {}
        return value if isinstance(value, dict) else {}

    def _validate_upsert_inputs(self, ids: List[str], vectors: List[List[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("LanceDBIndex.upsert: ids and vectors length mismatch.")
        for sid in ids:
            if not isinstance(sid, str) or not sid:
                raise ValueError("LanceDBIndex.upsert: all ids must be non-empty strings.")
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self.dim:
                raise ValueError(
                    f"LanceDBIndex.upsert: expected vector dim={self.dim}, got={len(vector) if isinstance(vector, list) else 'invalid'}."
                )
            for value in vector:
                if not isinstance(value, (float, int)):
                    raise ValueError("LanceDBIndex.upsert: vectors must contain numeric values.")