"""
UMA Unified Configuration Loader (Simplified Edition)

This configuration format uses:
• ONE db_root for all SQL stores
• Declarative backend selection for SQL, Vector, and Graph
• Strict validation of all required fields
"""

from __future__ import annotations
import yaml
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class UMAConfig(dict):
    """Strict config loader with dot-access."""

    @classmethod
    def load_yaml(cls, path: str) -> "UMAConfig":
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except Exception:
            logger.exception("Failed to load UMA config YAML")
            raise RuntimeError(f"Invalid config YAML at: {path}")

        if not isinstance(data, dict):
            raise ValueError("UMA config must be a dict at top-level")

        cfg = cls(data)
        cfg._validate()
        return cfg

    # Dot-access
    def __getattr__(self, item: str):
        if item not in self:
            raise AttributeError(f"Missing config section '{item}'")
        val = self[item]
        return UMAConfig(val) if isinstance(val, dict) else val

    # ----- Small helpers -----
    def _require(self, section, key):
        if section not in self or key not in self[section]:
            raise ValueError(f"Missing required config key: '{section}.{key}'")

    def _require_nonempty_str(self, section, key):
        self._require(section, key)
        val = self[section][key]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"'{section}.{key}' must be a non-empty string")

    def _require_positive_int(self, section, key):
        self._require(section, key)
        val = self[section][key]
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"'{section}.{key}' must be a positive integer")

    def _require_ratio(self, section, key):
        self._require(section, key)
        val = self[section][key]
        if not (0 < val < 1):
            raise ValueError(f"'{section}.{key}' must be between 0 and 1")

    def _warn_on_secrets(self) -> None:
        sensitive_keys = ("api_key", "apikey", "secret", "token", "password", "passwd", "pwd")
        placeholders = ("YOUR_", "CHANGEME", "REPLACE", "INSERT", "<", "...")

        def _is_placeholder(value: str) -> bool:
            upper = value.strip().upper()
            if not upper:
                return True
            if upper.startswith("YOUR_") or upper.startswith("CHANGEME"):
                return True
            if upper.startswith("REPLACE") or upper.startswith("INSERT"):
                return True
            if value.strip() in {"...", "<redacted>", "<REDACTED>"}:
                return True
            return False

        def _walk(node: Any, path: str = "") -> None:
            if isinstance(node, dict):
                for key, val in node.items():
                    key_str = str(key)
                    new_path = f"{path}.{key_str}" if path else key_str
                    if any(s in key_str.lower() for s in sensitive_keys):
                        if isinstance(val, str) and not _is_placeholder(val):
                            logger.warning(
                                "Sensitive value detected in config at '%s'. "
                                "Prefer environment variables or a secret manager.",
                                new_path,
                            )
                    _walk(val, new_path)
            elif isinstance(node, list):
                for idx, item in enumerate(node):
                    _walk(item, f"{path}[{idx}]")

        _walk(self)

    # ----- MAIN VALIDATION -----
    def _validate(self):
        logger.info("Validating UMA config...")

        # -----------------------
        # STORAGE
        # -----------------------
        self._require_nonempty_str("storage", "db_root")

        sql_backend = self.storage.get("sql_backend")
        if sql_backend not in ("sqlite", "postgres"):
            raise ValueError("'storage.sql_backend' must be 'sqlite' or 'postgres'")

        vector_backend = self.storage.get("vector_backend")
        if vector_backend not in ("faiss", "pinecone", "weaviate", "inmemory"):
            raise ValueError(
                "'storage.vector_backend' must be one of: faiss, pinecone, weaviate, inmemory"
            )
        vector_cfg = self.storage.get("vector_config") or {}
        if vector_backend == "pinecone":
            if not isinstance(vector_cfg, dict) or not vector_cfg.get("index_name"):
                raise ValueError(
                    "'storage.vector_config.index_name' is required for pinecone backend"
                )
        if vector_backend == "weaviate":
            if not isinstance(vector_cfg, dict):
                raise ValueError("'storage.vector_config' must be a mapping for weaviate backend")
            for key in ("url", "api_key", "class_name"):
                if not vector_cfg.get(key):
                    raise ValueError(f"'storage.vector_config.{key}' is required for weaviate backend")

        graph_backend = self.storage.get("graph_backend")
        if graph_backend not in ("neo4j", "memgraph", "disabled"):
            raise ValueError(
                "'storage.graph_backend' must be 'neo4j', 'memgraph', or 'disabled'"
            )

        # -----------------------
        # WORKING MEMORY
        # -----------------------
        self._require_positive_int("working_memory", "max_tokens")
        self._require_ratio("working_memory", "warning_ratio")
        self._require_ratio("working_memory", "hard_limit_ratio")
        self._require_positive_int("working_memory", "chunk_size")
        # optional tunables (present under the top-level `working_memory` section)
        wm_section = self.get("working_memory", {}) if isinstance(self, dict) else {}
        if "keep_recent_messages" in wm_section:
            self._require_positive_int("working_memory", "keep_recent_messages")
        if "keep_recent_token_fraction" in wm_section:
            self._require_ratio("working_memory", "keep_recent_token_fraction")

        # -----------------------
        # SECURITY
        # -----------------------
        # Deprecated: config-embedded code execution has been removed.
        if "security" in self:
            sec = self.security
            if not isinstance(sec, dict):
                raise ValueError("'security' section must be a mapping")
            if "allow_config_code" in sec:
                logger.warning("'security.allow_config_code' is deprecated and ignored.")
        
        # -----------------------
        # EMBEDDING
        # -----------------------
        self._require("embedding", "provider")
        provider = self.embedding.provider
        if provider in ("openai", "ollama"):
            self._require_nonempty_str("embedding", "model")
        elif not isinstance(provider, str) or not provider.strip():
            raise ValueError("embedding.provider must be a non-empty string")

        self._require_positive_int("embedding", "dimension")
        if "config" in self.embedding and not isinstance(self.embedding["config"], dict):
            raise ValueError("'embedding.config' must be a mapping")

        # -----------------------
        # LLM
        # -----------------------
        self._require("llm", "provider")
        prov = self.llm.provider
        if not isinstance(prov, str) or not prov.strip():
            raise ValueError("llm.provider must be a non-empty string")

        if prov == "openai":
            self._require_nonempty_str("llm", "model")
        elif prov == "ollama":
            if not (self.llm.get("model") or self.llm.get("ollama_model")):
                raise ValueError(
                    "llm.provider='ollama' requires either 'model' or 'ollama_model'"
                )
        if "config" in self.llm and not isinstance(self.llm["config"], dict):
            raise ValueError("'llm.config' must be a mapping")

        # -----------------------
        # RETRIEVAL
        # -----------------------
        for key in ("max_episodes", "max_facts", "max_skills", "max_graph_items"):
            self._require_positive_int("retrieval", key)

        # -----------------------
        # RETRIEVAL.RLM (optional)
        # -----------------------
        rlm = self.retrieval.get("rlm")
        if rlm is not None:
            if not isinstance(rlm, dict):
                raise ValueError("'retrieval.rlm' must be a mapping")

            if "enabled" in rlm and not isinstance(rlm["enabled"], bool):
                raise ValueError("'retrieval.rlm.enabled' must be boolean")

            def _pos_int(key):
                if key in rlm and (not isinstance(rlm[key], int) or rlm[key] <= 0):
                    raise ValueError(f"'retrieval.rlm.{key}' must be a positive integer")

            def _pos_float(key):
                if key in rlm and (not isinstance(rlm[key], (int, float)) or rlm[key] <= 0):
                    raise ValueError(f"'retrieval.rlm.{key}' must be a positive number")

            _pos_int("max_steps")
            _pos_int("max_actions_per_step")
            _pos_int("max_items_per_type")
            _pos_int("llm_max_tokens")
            _pos_int("max_env_calls")
            _pos_int("max_return_chars")
            _pos_float("timeout_s")

        # -----------------------
        # CONSOLIDATION
        # -----------------------
        self._require("consolidation", "enabled")
        self._require_positive_int("consolidation", "max_episodes_per_cycle")

        cs = self.consolidation.get("cluster_similarity")
        if cs is None or not (0 < cs <= 1):
            raise ValueError("'consolidation.cluster_similarity' must be 0 < x <= 1")

        p = self.consolidation.get("prune_min_fact_salience")
        if p is None or not (0 <= p <= 1):
            raise ValueError("'consolidation.prune_min_fact_salience' must be between 0 and 1")

        # -----------------------
        # FEATURES
        # -----------------------
        features = self.get("features")
        if not isinstance(features, dict):
            raise ValueError("'features' must be a mapping")

        if "load" in features:
            load_cfg = features.get("load")
            if not isinstance(load_cfg, list):
                raise ValueError("'features.load' must be a list")
            for item in load_cfg:
                if not isinstance(item, dict):
                    raise ValueError("Each entry in 'features.load' must be a mapping")
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("'features.load[].name' must be a non-empty string")
                if "enabled" in item and not isinstance(item["enabled"], bool):
                    raise ValueError("'features.load[].enabled' must be boolean")
                if "provider" in item and not isinstance(item["provider"], str):
                    raise ValueError("'features.load[].provider' must be a string")
                if "config" in item and not isinstance(item["config"], dict):
                    raise ValueError("'features.load[].config' must be a mapping")
        else:
            self._require("features", "procedural_enabled")
            self._require("features", "consolidation_enabled")

        policy = features.get("policy")
        if policy is not None:
            if not isinstance(policy, dict):
                raise ValueError("'features.policy' must be a mapping")
            if "on_attach_error" in policy and policy["on_attach_error"] not in (
                "log_and_skip",
                "raise",
            ):
                raise ValueError("'features.policy.on_attach_error' must be 'log_and_skip' or 'raise'")
            if "allow_method_override" in policy and not isinstance(
                policy["allow_method_override"], bool
            ):
                raise ValueError("'features.policy.allow_method_override' must be boolean")

        self._warn_on_secrets()
        logger.info("UMA configuration validated successfully.")
