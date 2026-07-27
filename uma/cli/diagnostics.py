"""Offline diagnostics for the UMA CLI."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from uma.common.config_types import RuntimeConfig


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def doctor_offline(
    config: dict[str, Any],
    config_path: Path,
) -> tuple[dict[str, Any], str, str, int]:
    runtime = RuntimeConfig.from_uma_config(config)
    checks: list[dict[str, str]] = [
        {
            "name": "config",
            "status": "ok",
            "detail": str(config_path),
        },
        {
            "name": "python",
            "status": "ok" if sys.version_info >= (3, 9) else "error",
            "detail": sys.version.split()[0],
        },
    ]

    backends = (
        ("sql", runtime.storage.sql_backend, None),
        (
            "vector",
            runtime.storage.vector_backend,
            {
                "faiss": "faiss",
                "uma.adapters.vector.faiss_adapter:FaissIndex": "faiss",
                "uma.adapters.vector.lancedb:LanceDBIndex": "lancedb",
            }.get(runtime.storage.vector_backend),
        ),
        ("graph", runtime.storage.graph_backend, None),
    )
    for name, backend, dependency in backends:
        if backend in {"sqlite", "inmemory", "disabled"}:
            available = True
        else:
            module_name = dependency or backend.split(":", 1)[0]
            available = _module_available(module_name)
        checks.append(
            {
                "name": f"storage:{name}",
                "status": "ok" if available else "error",
                "detail": str(backend),
            }
        )

    providers = [
        ("llm:uma", runtime.llm),
        ("embedding", runtime.embedding),
    ]
    if isinstance(config.get("llms"), dict) and "agent" in config["llms"]:
        providers.append(("llm:agent", runtime.agent_llm))

    provider_dependencies = {
        "ollama": "openai",
        "openai": "openai",
        "anthropic": "anthropic",
    }
    default_credentials = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    for name, provider_config in providers:
        provider = provider_config.provider
        dependency = provider_dependencies.get(provider)
        available = dependency is not None and _module_available(dependency)
        checks.append(
            {
                "name": name,
                "status": "ok" if available else "error",
                "detail": f"{provider}:{provider_config.model}",
            }
        )

        if provider not in default_credentials:
            continue
        inline_key = provider_config.config.get("api_key")
        env_name = provider_config.config.get("api_key_env") or default_credentials[provider]
        credential_available = bool(inline_key) or bool(os.environ.get(str(env_name)))
        checks.append(
            {
                "name": f"{name}:credential",
                "status": "ok" if credential_available else "error",
                "detail": "configured directly" if inline_key else str(env_name),
            }
        )

    custom_patterns_path = runtime.security.custom_patterns_path
    if custom_patterns_path:
        pattern_path = Path(custom_patterns_path).resolve()
        checks.append(
            {
                "name": "security:custom_patterns",
                "status": "ok" if pattern_path.is_file() else "error",
                "detail": str(pattern_path),
            }
        )
    else:
        checks.append(
            {
                "name": "security:custom_patterns",
                "status": "ok",
                "detail": "bundled catalogs",
            }
        )

    failed = any(check["status"] == "error" for check in checks)
    status = "error" if failed else "ok"
    lines = ["UMA doctor (offline)"]
    lines.extend(
        f"[{check['status']}] {check['name']}: {check['detail']}"
        for check in checks
    )
    lines.append(f"Overall: {status}")
    return (
        {
            "mode": "offline",
            "config_path": str(config_path),
            "checks": checks,
        },
        "\n".join(lines),
        status,
        1 if failed else 0,
    )
