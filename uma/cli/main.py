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


def _emit(
    command: str,
    data: dict[str, Any],
    output_format: str,
    text: str,
    status: str = "ok",
) -> None:
    if output_format == "json":
        print(
            json.dumps(
                {
                    "schema_version": "1",
                    "command": command,
                    "status": status,
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

    doctor_parser = commands.add_parser("doctor", help="Run UMA diagnostic checks.")
    doctor_parser.add_argument(
        "--offline",
        action="store_true",
        required=True,
        help="Check configuration and local dependencies without initializing UMA.",
    )

    security_parser = commands.add_parser("security", help="Run UMA security tools.")
    security_commands = security_parser.add_subparsers(dest="security_command", required=True)
    scan_parser = security_commands.add_parser("scan", help="Scan text for prompt injection patterns.")
    scan_parser.add_argument("text", nargs="?", help="Text to scan.")
    scan_parser.add_argument("--file", dest="input_file", type=Path, help="Read text from a UTF-8 file.")
    scan_parser.add_argument("--stdin", action="store_true", help="Read text from standard input.")
    scan_parser.add_argument(
        "--fail-on",
        choices=("low", "medium", "high"),
        help="Exit with status 1 at or above this severity.",
    )

    dev_parser = commands.add_parser("dev", help="Run UMA development tools.")
    dev_commands = dev_parser.add_subparsers(dest="dev_command", required=True)
    check_parser = dev_commands.add_parser("check", help="Run predefined development checks.")
    check_parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help="Check profile to run (default: quick).",
    )
    check_parser.add_argument("--only", help="Run only these comma-separated checks.")
    check_parser.add_argument("--skip", help="Skip these comma-separated checks.")
    check_parser.add_argument("--list", action="store_true", help="List available checks.")
    check_parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure.")

    args = parser.parse_args(argv)
    command_name = args.command
    if args.command == "config":
        command_name = f"config.{args.config_command}"
    elif args.command == "security":
        command_name = f"security.{args.security_command}"
    elif args.command == "dev":
        command_name = f"dev.{args.dev_command}"

    try:
        if args.command == "version":
            _emit(
                "version",
                {"version": __version__},
                args.output_format,
                f"uma {__version__}",
            )
            return 0

        if args.command == "dev":
            from .development import run_checks

            data, text, status, exit_code = run_checks(
                profile=args.profile,
                only=args.only,
                skip=args.skip,
                list_only=args.list,
                fail_fast=args.fail_fast,
            )
            _emit("dev.check", data, args.output_format, text, status)
            return exit_code

        from uma.common.config import UMAConfig

        config_path = _resolve_config_path(args.config)
        config = UMAConfig.load_yaml(str(config_path))

        if args.command == "config" and args.config_command == "validate":
            _emit(
                "config.validate",
                {"path": str(config_path), "profile": config.get("profile")},
                args.output_format,
                f"Valid UMA configuration: {config_path}",
            )
            return 0

        if args.command == "config":
            config_data = _redact(dict(config))
            _emit(
                "config.show",
                {"path": str(config_path), "config": config_data},
                args.output_format,
                f"# Config: {config_path}\n"
                f"{yaml.safe_dump(config_data, sort_keys=False).rstrip()}",
            )
            return 0

        if args.command == "doctor":
            from .diagnostics import doctor_offline

            data, text, status, exit_code = doctor_offline(config, config_path)
            _emit("doctor", data, args.output_format, text, status)
            return exit_code

        sources = sum(
            (
                args.text is not None,
                args.input_file is not None,
                args.stdin,
            )
        )
        if sources != 1:
            raise ValueError("security scan requires exactly one of TEXT, --file, or --stdin")

        if args.text is not None:
            text_to_scan = args.text
        elif args.input_file is not None:
            text_to_scan = args.input_file.read_text(encoding="utf-8")
        else:
            text_to_scan = sys.stdin.read()

        from .security import scan_input

        data, text, status, exit_code = scan_input(
            config,
            config_path,
            text_to_scan,
            args.fail_on,
        )
        _emit("security.scan", data, args.output_format, text, status)
        return exit_code
    except Exception as exc:
        if args.output_format == "json":
            print(
                json.dumps(
                    {
                        "schema_version": "1",
                        "command": command_name,
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
