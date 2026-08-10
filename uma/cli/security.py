"""Security commands for the UMA CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from uma.api.memory import UMAMemory


logger = logging.getLogger(__name__)


def scan_input(
    config: dict[str, Any],
    config_path: Path,
    text_to_scan: str,
    fail_on: Optional[str],
) -> tuple[dict[str, Any], str, str, int]:
    memory: UMAMemory | None = None
    data: dict[str, Any] = {}
    text = "UMA security scan"
    status = "error"
    exit_code = 1
    try:
        memory = UMAMemory(config, config_path=str(config_path))
        result = memory.scan_user_input(text_to_scan)
        threshold = fail_on or memory.cfg.security.scan_severity_threshold
        severity_order = {
            "none": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
        }
        if threshold not in severity_order:
            raise ValueError(
                f"invalid security scan severity threshold: {threshold!r}"
            )

        severity = result.get("severity")
        if severity not in severity_order:
            raise RuntimeError(
                f"security scanner returned invalid severity: {severity!r}"
            )
        matched_rules = result.get("matched_rules")
        if not isinstance(matched_rules, list):
            raise RuntimeError(
                "security scanner returned invalid matched_rules"
            )

        threshold_reached = (
            severity_order[severity] >= severity_order[threshold]
        )
        data = {
            **result,
            "fail_on": threshold,
            "threshold_reached": threshold_reached,
        }
        text = "\n".join(
            [
                f"Severity: {severity}",
                f"Score: {result.get('score')}",
                (
                    "Matched rules: "
                    f"{', '.join(matched_rules) if matched_rules else 'none'}"
                ),
                f"Fail threshold: {threshold}",
            ]
        )
        status = "findings" if threshold_reached else "ok"
        exit_code = 1 if threshold_reached else 0
    except Exception as exc:
        logger.debug("CLI security scan failed", exc_info=True)
        data = {"error": str(exc)}
        text = f"UMA security scan\n[error] {exc}"
    finally:
        if memory is not None:
            try:
                memory.shutdown()
            except Exception as exc:
                logger.debug(
                    "CLI security scan shutdown failed",
                    exc_info=True,
                )
                data = {
                    **data,
                    "shutdown_error": str(exc),
                }
                text = f"{text}\n[error] shutdown failed: {exc}"
                status = "error"
                exit_code = 1

    return data, text, status, exit_code
