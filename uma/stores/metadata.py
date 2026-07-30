"""
store_metadata.py
=================

Helpers for tracking UMA storage format metadata inside SQL stores.
"""

from __future__ import annotations

import logging

from uma.version import __version__ as UMA_VERSION

logger = logging.getLogger(__name__)

FORMAT_NAME = "uma"


def ensure_store_metadata(store: object, conn: object, store_name: str) -> dict[str, str]:
    """
    Ensure UMA metadata exists in the store and validate format.

    This writes a lightweight key/value table into each SQL DB to record:
    - format_name: UMA store format name
    - uma_version: package version that created the store
    - store_name: semantic | episodic | procedural
    """
    store_name = (store_name or "").strip()
    if not store_name:
        raise ValueError("ensure_store_metadata: store_name must be non-empty")

    store._execute(
        conn,
        """
        CREATE TABLE IF NOT EXISTS uma_store_meta (
            meta_key TEXT PRIMARY KEY,
            meta_value TEXT NOT NULL
        );
        """,
        log_context="meta_schema",
    )

    rows = store._query_all(
        conn,
        "SELECT meta_key, meta_value FROM uma_store_meta",
        log_context="meta_read",
    )
    meta = {r["meta_key"]: r["meta_value"] for r in rows}

    existing_format = meta.get("format_name")
    if existing_format and existing_format != FORMAT_NAME:
        raise ValueError(
            f"UMA store format mismatch: expected {FORMAT_NAME!r}, got {existing_format!r}"
        )

    existing_version = meta.get("uma_version")
    if existing_version and existing_version != UMA_VERSION:
        logger.warning(
            "UMA store '%s' created with UMA %s; running %s.",
            store_name,
            existing_version,
            UMA_VERSION,
        )

    desired = {
        "format_name": FORMAT_NAME,
        "uma_version": UMA_VERSION,
        "store_name": store_name,
    }
    for key, value in desired.items():
        if meta.get(key):
            continue
        store._execute(
            conn,
            """
            INSERT INTO uma_store_meta (meta_key, meta_value)
            VALUES (?, ?)
            ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value
            """,
            params=[key, value],
            log_context="meta_upsert",
        )

    return desired
