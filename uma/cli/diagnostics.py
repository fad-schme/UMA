"""Offline and runtime diagnostics for the UMA CLI."""

from __future__ import annotations

import importlib.util
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from uma.api.memory import UMAMemory
from uma.common.config_types import RuntimeConfig


_EXIT_CODES = {
    "ok": 0,
    "degraded": 4,
    "error": 1,
}


class RuntimeHealthTimeout(TimeoutError):
    """Raised when runtime initialization or health checks exceed a deadline."""


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
                "uma.adapters.vector.qdrant:QdrantIndex": "qdrant_client",
            }.get(runtime.storage.vector_backend),
        ),
        ("graph", runtime.storage.graph_backend, None),
    )
    for name, backend, dependency in backends:
        module_name: str | None = None
        if backend in {"sqlite", "inmemory", "disabled"}:
            available = True
        else:
            module_name = dependency or backend.split(":", 1)[0]
            available = _module_available(module_name)
        detail = str(backend)
        if not available and module_name:
            detail = f"{backend} (missing dependency: {module_name})"
        checks.append(
            {
                "name": f"storage:{name}",
                "status": "ok" if available else "error",
                "detail": detail,
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
        detail = f"{provider}:{provider_config.model}"
        if not available:
            detail = (
                f"{detail} (missing dependency: "
                f"{dependency or 'unsupported provider'})"
            )
        checks.append(
            {
                "name": name,
                "status": "ok" if available else "error",
                "detail": detail,
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


@contextmanager
def _runtime_deadline(timeout_seconds: float | None) -> Iterator[None]:
    if timeout_seconds is None:
        yield
        return
    if timeout_seconds <= 0:
        raise ValueError("timeout must be greater than zero")

    can_interrupt = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
    )
    if not can_interrupt:
        started = time.monotonic()
        yield
        if time.monotonic() - started > timeout_seconds:
            raise RuntimeHealthTimeout(
                f"runtime health check exceeded {timeout_seconds:g} seconds"
            )
        return

    def _handle_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise RuntimeHealthTimeout(
            f"runtime health check exceeded {timeout_seconds:g} seconds"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            elapsed = time.monotonic() - started
            remaining = max(previous_timer[0] - elapsed, 0.000001)
            signal.setitimer(
                signal.ITIMER_REAL,
                remaining,
                previous_timer[1],
            )


def _error_check(name: str, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "error",
        "detail": detail,
        "latency_ms": None,
    }


def _health_data(health: Any) -> tuple[str, dict[str, Any]]:
    if hasattr(health, "model_dump"):
        serialized = health.model_dump(mode="json")
    elif isinstance(health, dict):
        serialized = dict(health)
    elif hasattr(health, "status") and hasattr(health, "checks"):
        serialized = {
            "status": health.status,
            "checks": health.checks,
        }
    else:
        raise TypeError("health_check() returned an unsupported result")

    status = str(serialized.get("status", "error"))
    if status not in _EXIT_CODES:
        raise ValueError(f"health_check() returned unknown status {status!r}")
    checks = serialized.get("checks")
    if not isinstance(checks, dict):
        raise TypeError("health_check() result must contain a checks mapping")

    normalized_checks: dict[str, Any] = {}
    for name, check in checks.items():
        if hasattr(check, "model_dump"):
            normalized = check.model_dump(mode="json")
        elif hasattr(check, "to_dict"):
            normalized = check.to_dict()
        elif isinstance(check, dict):
            normalized = dict(check)
        else:
            normalized = {
                "name": getattr(check, "name", str(name)),
                "status": getattr(check, "status", "error"),
                "detail": getattr(check, "detail", ""),
                "latency_ms": getattr(check, "latency_ms", None),
            }
        normalized_checks[str(name)] = normalized
    return status, normalized_checks


def _runtime_text(data: dict[str, Any], status: str) -> str:
    lines = [
        "UMA health",
        f"Config: {data['config_path']}",
    ]
    if data.get("agent_id"):
        lines.append(f"Agent: {data['agent_id']}")
    for name, check in data["checks"].items():
        check_status = check.get("status", "error")
        detail = check.get("detail", "")
        latency_ms = check.get("latency_ms")
        latency = (
            f" ({latency_ms:.1f} ms)"
            if isinstance(latency_ms, (int, float))
            else ""
        )
        lines.append(f"[{check_status}] {name}: {detail}{latency}")
    lines.extend(
        (
            "LLM probe: initialization only; no provider generation request "
            "was performed.",
            f"Overall: {status}",
        )
    )
    return "\n".join(lines)


def runtime_health(
    config_path: Path,
    *,
    agent_id: str | None,
    timeout_seconds: float | None,
) -> tuple[dict[str, Any], str, str, int]:
    """Initialize UMA, run its lightweight health probe, and always shut down."""

    memory: UMAMemory | None = None
    checks: dict[str, Any] = {}
    status = "error"
    phase = "initialization"
    started = time.monotonic()
    try:
        with _runtime_deadline(timeout_seconds):
            memory = UMAMemory.from_yaml(str(config_path))
            health_target: Any = memory
            if agent_id is not None:
                phase = "agent context"
                health_target = memory.set_context(agent_id=agent_id)
            phase = "health check"
            health = health_target.health_check()
            status, checks = _health_data(health)
    except RuntimeHealthTimeout as exc:
        checks["timeout"] = _error_check("timeout", str(exc))
        status = "error"
    except Exception as exc:
        checks[phase.replace(" ", "_")] = _error_check(
            phase.replace(" ", "_"),
            f"{phase} failed: {exc}",
        )
        status = "error"
    finally:
        if memory is not None:
            try:
                memory.shutdown()
            except Exception as exc:
                checks["shutdown"] = _error_check(
                    "shutdown",
                    f"shutdown failed: {exc}",
                )
                status = "error"

    data = {
        "config_path": str(config_path),
        "agent_id": agent_id,
        "timeout_seconds": timeout_seconds,
        "duration_ms": round((time.monotonic() - started) * 1000),
        "llm_probe": "initialization_only",
        "checks": checks,
    }
    return (
        data,
        _runtime_text(data, status),
        status,
        _EXIT_CODES[status],
    )


def doctor_runtime(
    config: dict[str, Any],
    config_path: Path,
    *,
    agent_id: str | None,
    timeout_seconds: float | None,
) -> tuple[dict[str, Any], str, str, int]:
    """Combine offline readiness diagnostics with live runtime health."""

    offline_data, offline_text, offline_status, _ = doctor_offline(
        config,
        config_path,
    )
    runtime_data, runtime_text, runtime_status, _ = runtime_health(
        config_path,
        agent_id=agent_id,
        timeout_seconds=timeout_seconds,
    )
    if "error" in {offline_status, runtime_status}:
        status = "error"
    elif "degraded" in {offline_status, runtime_status}:
        status = "degraded"
    else:
        status = "ok"
    text = (
        f"{offline_text}\n\n{runtime_text}\n\n"
        f"UMA doctor overall: {status}"
    )
    return (
        {
            "mode": "runtime",
            "config_path": str(config_path),
            "offline": offline_data,
            "runtime": runtime_data,
        },
        text,
        status,
        _EXIT_CODES[status],
    )
