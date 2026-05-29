"""
UMA Unified Configuration Loader (Simplified Edition)

This configuration format uses:
• ONE db_root for all SQL stores
• Declarative backend selection for SQL, Vector, and Graph
• Strict validation of all required fields
"""

from __future__ import annotations
import logging
import os
from typing import Any

import yaml
from uma.common.initializers.runtime import init_runtime_env

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

        data.setdefault("profile", "lite")
        cfg = cls(data)
        try:
            cfg._source_path = os.path.abspath(path)
            cfg._source_dir = os.path.dirname(cfg._source_path)
        except Exception:
            # Keep non-fatal: path bookkeeping should not block config loading.
            cfg._source_path = None
            cfg._source_dir = None

        # Initialize lightweight runtime environment (plugin roots, process defaults).
        # MUST remain cheap: no DB/LLM/embedder initialization here.
        init_runtime_env(cfg)

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
        known_sql = ("sqlite",)
        is_sql_plugin = isinstance(sql_backend, str) and ":" in sql_backend
        if sql_backend not in known_sql and not is_sql_plugin:
            raise ValueError(
                "'storage.sql_backend' must be 'sqlite' or a plugin spec 'module:callable'"
            )

        vector_backend = self.storage.get("vector_backend")
        if not isinstance(vector_backend, str) or not vector_backend.strip():
            raise ValueError("'storage.vector_backend' must be a non-empty string")
        vector_cfg = self.storage.get("vector_config") or {}
        if not isinstance(vector_cfg, dict):
            raise ValueError("'storage.vector_config' must be a mapping for plugin vector backends")

        graph_backend = self.storage.get("graph_backend")
        is_graph_plugin = isinstance(graph_backend, str) and ":" in graph_backend
        if graph_backend not in ("disabled",) and not is_graph_plugin:
            raise ValueError(
                "'storage.graph_backend' must be 'disabled' or a plugin spec 'module:callable'"
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
        # EMBEDDING
        # -----------------------
        self._require("embedding", "provider")
        provider = self.embedding.provider
        if provider == "ollama":
            self._require_nonempty_str("embedding", "model")
        elif not isinstance(provider, str) or not provider.strip():
            raise ValueError("embedding.provider must be a non-empty string")

        self._require_positive_int("embedding", "dimension")
        if "config" in self.embedding and not isinstance(self.embedding["config"], dict):
            raise ValueError("'embedding.config' must be a mapping")

        # -----------------------
        # SECRETS
        # -----------------------
        secrets_cfg = self.get("secrets")
        if secrets_cfg is not None:
            if not isinstance(secrets_cfg, dict):
                raise ValueError("'secrets' must be a mapping")
            provider = secrets_cfg.get("provider")
            if not isinstance(provider, str) or not provider.strip():
                raise ValueError("'secrets.provider' must be a non-empty string")
            options = secrets_cfg.get("options")
            if options is not None and not isinstance(options, dict):
                raise ValueError("'secrets.options' must be a mapping")

        # -----------------------
        # LLM / LLMS
        # -----------------------
        if "llms" in self and isinstance(self.llms, dict):
            if "uma" not in self.llms:
                raise ValueError("'llms.uma' section is required")
            for key in ("uma", "agent"):
                if key not in self.llms:
                    continue
                section = self.llms.get(key)
                if not isinstance(section, dict):
                    raise ValueError(f"'llms.{key}' must be a mapping")
                if "provider" not in section or not isinstance(section.get("provider"), str) or not section.get("provider").strip():
                    raise ValueError(f"llms.{key}.provider must be a non-empty string")
                prov = section.get("provider")
                if prov == "ollama" and not (section.get("model") or section.get("ollama_model")):
                    raise ValueError(
                        f"llms.{key}.provider='ollama' requires either 'model' or 'ollama_model'"
                    )
                if "config" in section and not isinstance(section.get("config"), dict):
                    raise ValueError(f"llms.{key}.config must be a mapping")
        else:
            self._require("llm", "provider")
            prov = self.llm.provider
            if not isinstance(prov, str) or not prov.strip():
                raise ValueError("llm.provider must be a non-empty string")

            if prov == "ollama":
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
        if "strict" in self.retrieval and not isinstance(self.retrieval.get("strict"), bool):
            raise ValueError("'retrieval.strict' must be boolean")
        if "debug_scores" in self.retrieval and not isinstance(self.retrieval.get("debug_scores"), bool):
            raise ValueError("'retrieval.debug_scores' must be boolean")
        if "max_evidence_chunks" in self.retrieval:
            val = self.retrieval.get("max_evidence_chunks")
            if not isinstance(val, int) or val < 0:
                raise ValueError("'retrieval.max_evidence_chunks' must be a non-negative integer")
        if "neighbor_window" in self.retrieval:
            val = self.retrieval.get("neighbor_window")
            if not isinstance(val, int) or val < 0:
                raise ValueError("'retrieval.neighbor_window' must be a non-negative integer")
        if "max_expanded_chunks" in self.retrieval:
            val = self.retrieval.get("max_expanded_chunks")
            if not isinstance(val, int) or val < 0:
                raise ValueError("'retrieval.max_expanded_chunks' must be a non-negative integer")
        if "chunk_shortlist_k" in self.retrieval:
            val = self.retrieval.get("chunk_shortlist_k")
            if not isinstance(val, int) or val < 0:
                raise ValueError("'retrieval.chunk_shortlist_k' must be a non-negative integer")
        if "chunk_shortlist_max_per_doc" in self.retrieval:
            val = self.retrieval.get("chunk_shortlist_max_per_doc")
            if not isinstance(val, int) or val < 0:
                raise ValueError("'retrieval.chunk_shortlist_max_per_doc' must be a non-negative integer")
        if "hybrid" in self.retrieval:
            hybrid = self.retrieval.get("hybrid")
            if not isinstance(hybrid, dict):
                raise ValueError("'retrieval.hybrid' must be a mapping")
            if "enabled" in hybrid and not isinstance(hybrid.get("enabled"), bool):
                raise ValueError("'retrieval.hybrid.enabled' must be boolean")
            for key in ("top_k_dense", "top_k_sparse"):
                if key in hybrid and (not isinstance(hybrid.get(key), int) or int(hybrid.get(key)) < 0):
                    raise ValueError(f"'retrieval.hybrid.{key}' must be an integer >= 0")
            if "fusion_strategy" in hybrid:
                strat = hybrid.get("fusion_strategy")
                if not isinstance(strat, str) or not strat.strip():
                    raise ValueError("'retrieval.hybrid.fusion_strategy' must be a non-empty string")
                if strat.strip().lower() not in ("rrf", "overlap_boost"):
                    raise ValueError("'retrieval.hybrid.fusion_strategy' must be one of: rrf, overlap_boost")
        context_cfg = self.retrieval.get("context")
        if context_cfg is not None:
            if not isinstance(context_cfg, dict):
                raise ValueError("'retrieval.context' must be a mapping")
            for key in ("max_working_messages", "max_episodic", "max_semantic", "max_procedural", "max_graph"):
                if key in context_cfg and (not isinstance(context_cfg[key], int) or context_cfg[key] < 0):
                    raise ValueError(f"'retrieval.context.{key}' must be a non-negative integer")

        # -----------------------
        # RETRIEVAL.RLM (optional)
        # -----------------------
        rlm = self.retrieval.get("rlm")
        if rlm is not None:
            if not isinstance(rlm, dict):
                raise ValueError("'retrieval.rlm' must be a mapping")

            if "test_mode" in rlm and not isinstance(rlm["test_mode"], bool):
                raise ValueError("'retrieval.rlm.test_mode' must be boolean")

            def _pos_int(key):
                if key in rlm and (not isinstance(rlm[key], int) or rlm[key] <= 0):
                    raise ValueError(f"'retrieval.rlm.{key}' must be a positive integer")

            def _pos_float(key):
                if key in rlm and (not isinstance(rlm[key], (int, float)) or rlm[key] <= 0):
                    raise ValueError(f"'retrieval.rlm.{key}' must be a positive number")

            _pos_int("max_steps")
            _pos_int("max_actions_per_step")
            _pos_int("max_items_per_type")
            _pos_int("max_env_calls")
            _pos_float("timeout_s")
            if "chunk_fallback_enabled" in rlm and not isinstance(rlm["chunk_fallback_enabled"], bool):
                raise ValueError("'retrieval.rlm.chunk_fallback_enabled' must be boolean")
            if "chunk_fallback_k_multiplier" in rlm:
                val = rlm.get("chunk_fallback_k_multiplier")
                if not isinstance(val, int) or val <= 0:
                    raise ValueError("'retrieval.rlm.chunk_fallback_k_multiplier' must be a positive integer")

            if "predicate_allowlist" in rlm:
                pal = rlm.get("predicate_allowlist")
                if not isinstance(pal, dict):
                    raise ValueError("'retrieval.rlm.predicate_allowlist' must be a mapping")
                for dom, preds in pal.items():
                    if not isinstance(dom, str) or not dom.strip():
                        raise ValueError("'retrieval.rlm.predicate_allowlist' keys must be non-empty strings")
                    if not isinstance(preds, list) or not all(isinstance(p, str) and p.strip() for p in preds):
                        raise ValueError(
                            f"'retrieval.rlm.predicate_allowlist.{dom}' must be a list of non-empty strings"
                        )

        # -----------------------
        # CONSOLIDATION
        # -----------------------
        if "consolidation" in self:
            self._require("consolidation", "enabled")
            self._require_positive_int("consolidation", "max_episodes_per_cycle")

            cs = self.consolidation.get("cluster_similarity")
            if cs is None or not (0 < cs <= 1):
                raise ValueError("'consolidation.cluster_similarity' must be 0 < x <= 1")

            p = self.consolidation.get("prune_min_fact_salience")
            if p is None or not (0 <= p <= 1):
                raise ValueError("'consolidation.prune_min_fact_salience' must be between 0 and 1")

        # -----------------------
        # SEMANTIC (optional overrides)
        # -----------------------
        if "semantic" in self:
            semantic_cfg = self.semantic
            if not isinstance(semantic_cfg, dict):
                raise ValueError("'semantic' section must be a mapping")
            if "salience_threshold" in semantic_cfg:
                val = semantic_cfg["salience_threshold"]
                if not isinstance(val, (int, float)):
                    raise ValueError("'semantic.salience_threshold' must be a number")
                if not (0 <= val <= 1):
                    raise ValueError("'semantic.salience_threshold' must be between 0 and 1")

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

        # -----------------------
        # PIPELINE (optional)
        # -----------------------
        pipeline_cfg = self.get("pipeline")
        if pipeline_cfg is not None:
            if not isinstance(pipeline_cfg, dict):
                raise ValueError("'pipeline' must be a mapping")
            if "defer_post_turn" in pipeline_cfg and not isinstance(
                pipeline_cfg["defer_post_turn"], bool
            ):
                raise ValueError("'pipeline.defer_post_turn' must be boolean")
            if "post_turn_queue_max" in pipeline_cfg:
                val = pipeline_cfg["post_turn_queue_max"]
                if not isinstance(val, int) or val <= 0:
                    raise ValueError("'pipeline.post_turn_queue_max' must be a positive integer")

        self._warn_on_secrets()
        logger.info("UMA configuration validated successfully.")
