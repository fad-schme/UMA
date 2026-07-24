"""Command-line companion for UMA developers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from uma.version import __version__


_SECRET_KEYS = ("api_key", "apikey", "secret", "token", "password", "passwd", "pwd", "credential")


def _resolve_config_path(explicit_path: Optional[str]) -> Path:
    configured_path = explicit_path or os.environ.get("UMA_CONFIG")
    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"UMA config file not found: {path}")
        return path

    candidates = (Path.cwd() / "uma.yaml", Path.cwd() / "config" / "uma.yaml")
    for path in candidates:
        if path.is_file():
            return path.resolve()

    searched = ", ".join(str(path.resolve()) for path in candidates)
    raise ValueError(
        "UMA config file not found. Use --config or UMA_CONFIG. "
        f"Searched: {searched}"
    )


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            sensitive = any(marker in str(key).lower() for marker in _SECRET_KEYS)
            redacted[key] = (
                "<redacted>"
                if sensitive and not isinstance(item, (dict, list))
                else _redact(item)
            )
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _emit(command: str, data: dict[str, Any], output_format: str, text: str) -> None:
    if output_format == "json":
        print(
            json.dumps(
                {
                    "schema_version": "1",
                    "command": command,
                    "status": "ok",
                    "data": data,
                },
                sort_keys=True,
            )
        )
        return
    print(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uma",
        description="Developer companion for the UMA memory runtime.",
    )
    parser.add_argument("--config", help="Path to uma.yaml (or set UMA_CONFIG).")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="Show the installed UMA version.")

    config_parser = commands.add_parser("config", help="Validate or inspect UMA configuration.")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    config_commands.add_parser("validate", help="Validate the resolved UMA configuration.")
    config_commands.add_parser("show", help="Show the resolved configuration with secrets redacted.")

    args = parser.parse_args(argv)

    try:
        if args.command == "version":
            _emit(
                "version",
                {"version": __version__},
                args.output_format,
                f"uma {__version__}",
            )
            return 0

        from uma.common.config import UMAConfig

        config_path = _resolve_config_path(args.config)
        config = UMAConfig.load_yaml(str(config_path))

        if args.config_command == "validate":
            _emit(
                "config.validate",
                {"path": str(config_path), "profile": config.get("profile")},
                args.output_format,
                f"Valid UMA configuration: {config_path}",
            )
            return 0

        config_data = _redact(dict(config))
        _emit(
            "config.show",
            {"path": str(config_path), "config": config_data},
            args.output_format,
            f"# Config: {config_path}\n"
            f"{yaml.safe_dump(config_data, sort_keys=False).rstrip()}",
        )
        return 0
    except Exception as exc:
        if args.output_format == "json":
            print(
                json.dumps(
                    {
                        "schema_version": "1",
                        "command": (
                            f"config.{args.config_command}"
                            if args.command == "config"
                            else args.command
                        ),
                        "status": "error",
                        "data": None,
                        "errors": [{"message": str(exc)}],
                    },
                    sort_keys=True,
                )
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
