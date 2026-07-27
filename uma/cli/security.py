"""Security commands for the UMA CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from uma.api.memory import UMAMemory


def scan_input(
    config: dict[str, Any],
    config_path: Path,
    text_to_scan: str,
    fail_on: Optional[str],
) -> tuple[dict[str, Any], str, str, int]:
    memory = UMAMemory(config, config_path=str(config_path))
    try:
        result = memory.scan_user_input(text_to_scan)
        threshold = fail_on or memory.cfg.security.scan_severity_threshold
    finally:
        memory.shutdown()

    severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    if threshold not in severity_order:
        raise ValueError(f"invalid security scan severity threshold: {threshold!r}")

    threshold_reached = severity_order[result["severity"]] >= severity_order[threshold]
    matched_rules = result["matched_rules"]
    lines = [
        f"Severity: {result['severity']}",
        f"Score: {result['score']}",
        f"Matched rules: {', '.join(matched_rules) if matched_rules else 'none'}",
        f"Fail threshold: {threshold}",
    ]
    return (
        {
            **result,
            "fail_on": threshold,
            "threshold_reached": threshold_reached,
        },
        "\n".join(lines),
        "findings" if threshold_reached else "ok",
        1 if threshold_reached else 0,
    )
