"""
Store initialization helpers split out of `uma_memory` for readability.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ...adapters.db.sqlite_adapter import SQLiteAdapter
from ...adapters.vector.faiss_adapter import FaissIndex
from ...stores.episodic_sql import EpisodicSQLStore
from ...stores.semantic_sql import SemanticSQLStore
from ...stores.procedural_sql import ProceduralSQLStore

logger = logging.getLogger(__name__)


def initialize_stores(memory: "Any") -> None:
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

    # --------------------------------------------------------------
    # SQL BACKEND SELECTION
    # --------------------------------------------------------------
    sql_backend = storage_cfg.sql_backend

    if sql_backend == "sqlite":
        sql_adapter_cls = SQLiteAdapter

    elif sql_backend == "postgres":
        try:
            from uma.adapters.db.postgres_adapter import PostgresAdapter
        except ImportError as exc:
            logger.exception("PostgresAdapter import failed.")
            raise RuntimeError(
                "storage.sql_backend='postgres' but PostgresAdapter is not available. "
                "Install the required dependency or switch to 'sqlite'."
            ) from exc
        sql_adapter_cls = PostgresAdapter

    else:
        raise ValueError(f"Unsupported storage.sql_backend={sql_backend!r}")

    # Instantiate DB adapters
    epi_db = sql_adapter_cls(episodic_db_path)
    sem_db = sql_adapter_cls(semantic_db_path)
    pro_db = sql_adapter_cls(procedural_db_path)

    # --------------------------------------------------------------
    # VECTOR BACKEND SELECTION
    # --------------------------------------------------------------
    vector_backend = storage_cfg.vector_backend
    vector_cfg = storage_cfg.get("vector_config", {}) if isinstance(storage_cfg, dict) else {}

    if vector_backend == "faiss":
        from uma.adapters.vector.inmemory import InMemoryVectorIndex

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

    elif vector_backend == "pinecone":
        from uma.adapters.vector.pinecone_adapter import PineconeIndex

        index_name = vector_cfg.get("index_name", "")
        if not index_name:
            raise ValueError("Pinecone vector backend requires storage.vector_config.index_name")
        vector_init = lambda d: PineconeIndex(index_name=index_name, dim=d)

    elif vector_backend == "weaviate":
        from uma.adapters.vector.weaviate_adapter import WeaviateIndex

        url = vector_cfg.get("url", "")
        api_key = vector_cfg.get("api_key", "")
        class_name = vector_cfg.get("class_name", "")
        if not (url and api_key and class_name):
            raise ValueError(
                "Weaviate vector backend requires storage.vector_config.url, "
                "storage.vector_config.api_key, and storage.vector_config.class_name"
            )
        vector_init = lambda d: WeaviateIndex(
            url=url,
            api_key=api_key,
            class_name=class_name,
            dim=d,
        )

    else:
        raise ValueError(f"Unsupported storage.vector_backend={vector_backend!r}")

    epi_idx = vector_init(dim)
    sem_idx = vector_init(dim)
    pro_idx = vector_init(dim)

    try:
        memory.episodic_store = EpisodicSQLStore(epi_db, epi_idx)
        memory.semantic_store = SemanticSQLStore(sem_db, sem_idx)
        memory.procedural_store = ProceduralSQLStore(pro_db, pro_idx)
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
