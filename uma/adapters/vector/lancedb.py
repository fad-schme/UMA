from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from .base import VectorIndex

logger = logging.getLogger(__name__)

try:
    import lance_namespace  # type: ignore

    # Narrow import-time compatibility for the currently published lancedb wheel
    # set in this environment. Keep this local to the LanceDB adapter boundary.
    if (
        not hasattr(lance_namespace, "CreateEmptyTableRequest")
        and hasattr(lance_namespace, "CreateTableRequest")
    ):
        lance_namespace.CreateEmptyTableRequest = lance_namespace.CreateTableRequest

    import lancedb  # type: ignore
except Exception as exc:  # pragma: no cover
    lancedb = None  # type: ignore
    logger.error("Failed to import lancedb: %s", exc)


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
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        self._validate_upsert_inputs(ids, vectors)
        if not vectors:
            logger.debug("LanceDBIndex.upsert called with empty vectors; no-op.")
            return

        metadata_list = metadata or [{} for _ in ids]
        if len(metadata_list) != len(ids):
            raise ValueError("LanceDBIndex.upsert: metadata length mismatch with ids.")

        rows = []
        for sid, vector, meta in zip(ids, vectors, metadata_list):
            meta = meta or {}
            if not isinstance(meta, dict):
                raise ValueError("LanceDBIndex.upsert: metadata items must be dicts.")
            rows.append(
                {
                    "id": sid,
                    "vector": [float(value) for value in vector],
                    "metadata_json": json.dumps(meta, sort_keys=True),
                }
            )

        with self._lock:
            table = self._get_or_create_table(seed_rows=rows)
            self._delete_from_table(table, ids)
            table.add(rows)

    def query(
        self,
        vector: List[float],
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        if len(vector) != self.dim:
            raise ValueError(
                f"LanceDBIndex.query: expected query vector dim={self.dim}, got={len(vector)}"
            )
        if not isinstance(k, int) or k <= 0:
            raise ValueError("LanceDBIndex.query: k must be a positive integer.")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("LanceDBIndex.query: filters must be a dict or None.")

        table = self._open_table()
        if table is None:
            logger.debug("LanceDBIndex.query: table missing; returning [].")
            return []

        limit = min(max(k * self._search_k_multiplier, k), self._search_k_max)
        try:
            rows = table.search([float(value) for value in vector]).limit(limit).to_list()
        except Exception:
            logger.exception("LanceDBIndex.query failed table=%s", self.table_name)
            raise

        results: List[Tuple[str, float]] = []
        for row in rows:
            sid = row.get("id")
            if not isinstance(sid, str) or not sid:
                continue

            meta = self._parse_metadata(row.get("metadata_json"))
            if filters and any(meta.get(key) != value for key, value in filters.items()):
                continue

            distance = row.get("_distance")
            score = -float(distance) if isinstance(distance, (float, int)) else 0.0
            results.append((sid, score))
            if len(results) >= k:
                break

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
