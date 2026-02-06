"""
Store initialization helpers split out of `uma_memory` for readability.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..utils.config_types import parse_plugin_spec
from ...adapters.db.sqlite_adapter import SQLiteAdapter
from ...stores.episodic_sql import EpisodicSQLStore
from ...stores.semantic_sql import SemanticSQLStore
from ...stores.procedural_sql import ProceduralSQLStore
from ...stores.chunk_sql import ChunkSQLStore
from ...stores.document_sql import DocumentSQLStore

logger = logging.getLogger(__name__)


def initialize_stores(memory: "Any") -> dict:
    """
    Wire SQL + vector stores based on UMA's unified storage config.
    """
    dim = memory.embedding_cfg.dimension
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"Invalid embedding.dimension={dim!r}; must be > 0 integer.")

    storage_cfg = memory.raw_config.storage

    # --------------------------------------------------------------
    # Validate and compute DB paths
    # --------------------------------------------------------------
    db_root = os.path.expandvars(os.path.expanduser(storage_cfg.db_root)).rstrip("/") + "/"
    if not os.path.isabs(db_root):
        cwd_root = os.path.abspath(db_root)
        cfg_root = (
            os.path.abspath(os.path.join(memory._config_dir, db_root))
            if memory._config_dir
            else None
        )
        db_root_base = (
            (storage_cfg.get("db_root_base") or "auto")
            if isinstance(storage_cfg, dict)
            else "auto"
        )
        db_root_base = str(db_root_base).strip().lower() or "auto"
        db_files = ("episodic.db", "semantic.db", "procedural.db")

        def _has_db_files(root: str) -> bool:
            return any(os.path.exists(os.path.join(root, name)) for name in db_files)

        if db_root_base in {"config", "config_dir"}:
            db_root = cfg_root or cwd_root
        elif db_root_base in {"cwd", "workdir"}:
            db_root = cwd_root
        elif db_root_base == "auto":
            if os.path.exists(cwd_root) and (
                _has_db_files(cwd_root) or not (cfg_root and os.path.exists(cfg_root))
            ):
                db_root = cwd_root
            elif cfg_root and os.path.exists(cfg_root):
                db_root = cfg_root
            else:
                db_root = cwd_root
        else:
            logger.warning(
                "Unknown storage.db_root_base=%r; falling back to auto resolution.",
                db_root_base,
            )
            db_root = cwd_root
    db_root = os.path.abspath(db_root.rstrip("/"))

    episodic_db_path = os.path.join(db_root, "episodic.db")
    semantic_db_path = os.path.join(db_root, "semantic.db")
    procedural_db_path = os.path.join(db_root, "procedural.db")
    chunks_db_path = os.path.join(db_root, "chunks.db")
    documents_db_path = os.path.join(db_root, "documents.db")

    # --------------------------------------------------------------
    # SQL BACKEND SELECTION
    # --------------------------------------------------------------
    sql_backend = storage_cfg.sql_backend
    sql_backend_str = str(sql_backend or "")

    if sql_backend == "sqlite":
        sql_adapter_cls = SQLiteAdapter

    else:
        if ":" not in sql_backend_str:
            raise ValueError(f"Unsupported storage.sql_backend={sql_backend!r}")
        sql_plugin = parse_plugin_spec(sql_backend_str)
        if not callable(sql_plugin):
            raise TypeError("storage.sql_backend plugin must be a callable 'module:attr'")
        sql_adapter_cls = sql_plugin

    # Instantiate DB adapters
    epi_db = sql_adapter_cls(episodic_db_path)
    sem_db = sql_adapter_cls(semantic_db_path)
    pro_db = sql_adapter_cls(procedural_db_path)
    chunk_db = sql_adapter_cls(chunks_db_path)
    doc_db = sql_adapter_cls(documents_db_path)

    # --------------------------------------------------------------
    # VECTOR BACKEND SELECTION
    # --------------------------------------------------------------
    vector_backend = storage_cfg.vector_backend
    vector_cfg = storage_cfg.get("vector_config", {}) if isinstance(storage_cfg, dict) else {}
    vector_backend_str = str(vector_backend or "")

    if vector_backend == "faiss":
        from uma.adapters.vector.inmemory import InMemoryVectorIndex
        from uma.adapters.vector.faiss_adapter import FaissIndex

        def vector_init(d: int):
            try:
                return FaissIndex(d)
            except Exception:
                logger.exception(
                    "Failed to initialize FaissIndex; falling back to InMemoryVectorIndex."
                )
                return InMemoryVectorIndex.fallback_if_faiss_unavailable(d)

    elif vector_backend == "inmemory":
        from uma.adapters.vector.inmemory import InMemoryVectorIndex

        vector_init = lambda d: InMemoryVectorIndex(d)

    else:
        if ":" not in vector_backend_str:
            raise ValueError(f"Unsupported storage.vector_backend={vector_backend!r}")
        plugin = parse_plugin_spec(vector_backend_str)
        if not callable(plugin):
            raise TypeError("storage.vector_backend plugin must be a callable 'module:attr'")
        if not isinstance(vector_cfg, dict):
            raise ValueError("storage.vector_config must be a mapping for plugin vector backend")

        def vector_init(d: int):
            return plugin(d, **vector_cfg)

    epi_idx = vector_init(dim)
    sem_idx = vector_init(dim)
    pro_idx = vector_init(dim)
    chunk_idx = vector_init(dim)

    try:
        stores = {
            "episodic": EpisodicSQLStore(epi_db, epi_idx),
            "semantic": SemanticSQLStore(sem_db, sem_idx),
            "procedural": ProceduralSQLStore(pro_db, pro_idx),
            "chunk": ChunkSQLStore(chunk_db, chunk_idx),
        }
        memory.document_store = DocumentSQLStore(doc_db)
    except Exception:
        logger.exception("Failed to initialize one or more SQL/vector stores.")
        raise

    logger.info(
        "Stores initialized (sql_backend=%s, vector_backend=%s, db_root=%s, dim=%d)",
        sql_backend,
        vector_backend,
        db_root,
        dim,
    )
    return stores
