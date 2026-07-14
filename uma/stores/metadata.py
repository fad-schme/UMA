"""
store_metadata.py
=================

Helpers for tracking UMA storage format metadata inside SQL stores.
"""

from __future__ import annotations

import logging
from typing import Dict

from uma.version import __version__ as UMA_RLM_VERSION

logger = logging.getLogger(__name__)

META_TABLE = "uma_store_meta"
FORMAT_NAME = "uma-rlm"
# B608: META_TABLE is a module-level constant, never derived from input.
# The assert below is the machine-checkable proof: it fires immediately at
# import time if the value is ever changed to something outside the known set.
assert META_TABLE in {"uma_store_meta"}, (
    f"META_TABLE {META_TABLE!r} is not a known UMA schema table; "
    "update this assertion if adding a new metadata table."
)


def ensure_store_metadata(store: object, conn: object, store_name: str) -> Dict[str, str]:
    """
    Ensure UMA metadata exists in the store and validate format.

    This writes a lightweight key/value table into each SQL DB to record:
    - format_name: "uma-rlm"
    - uma_rlm_version: package version that created the store
    - store_name: semantic | episodic | procedural
    """
    store_name = (store_name or "").strip()
    if not store_name:
        raise ValueError("ensure_store_metadata: store_name must be non-empty")

    store._execute(
        conn,
        f"""
        CREATE TABLE IF NOT EXISTS {META_TABLE} (
            meta_key TEXT PRIMARY KEY,
            meta_value TEXT NOT NULL
        );
        """,
        log_context="meta_schema",
    )

    rows = store._query_all(
        conn,
        f"SELECT meta_key, meta_value FROM {META_TABLE}",  # nosec B608 — META_TABLE is a module constant asserted at import time
        log_context="meta_read",
    )
    meta = {r["meta_key"]: r["meta_value"] for r in rows}

    existing_format = meta.get("format_name")
    if existing_format and existing_format != FORMAT_NAME:
        raise ValueError(
            f"UMA store format mismatch: expected {FORMAT_NAME!r}, got {existing_format!r}"
        )

    existing_version = meta.get("uma_rlm_version")
    if existing_version and existing_version != UMA_RLM_VERSION:
        logger.warning(
            "UMA store '%s' created with UMA %s; running %s.",
            store_name,
            existing_version,
            UMA_RLM_VERSION,
        )

    desired = {
        "format_name": FORMAT_NAME,
        "uma_rlm_version": UMA_RLM_VERSION,
        "store_name": store_name,
    }
    for key, value in desired.items():
        if meta.get(key):
            continue
        store._execute(
            conn,
            f"""
            INSERT INTO {META_TABLE} (meta_key, meta_value)
            VALUES (?, ?)
            ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value
            """,
            params=[key, value],
            log_context="meta_upsert",
        )

    return desired
