"""Command-line companion for UMA developers."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from uma.cli.scopes import (
    add_audit_scope_arguments,
    add_owner_scope_arguments,
    add_record_scope_arguments,
    add_request_scope_arguments,
    non_empty_value,
)
from uma.version import __version__


logger = logging.getLogger(__name__)

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "credential",
    "credentials",
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
}
_SECRET_PREFIXES = (
    "credential_",
    "credentials_",
    "password_",
    "secret_",
    "token_",
)
_SECRET_SUFFIXES = (
    "_api_key",
    "_apikey",
    "_credential",
    "_credentials",
    "_password",
    "_passwd",
    "_pwd",
    "_secret",
    "_token",
)


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return seconds


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timeout",
        dest="timeout_seconds",
        type=_positive_seconds,
        default=30.0,
        metavar="SECONDS",
        help="Runtime initialization and health timeout (default: 30).",
    )


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _add_yes_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the exact resolved target without an interactive prompt.",
    )


def _add_index_target_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_graph: bool,
) -> None:
    add_owner_scope_arguments(parser)
    lanes = ["episodic", "semantic", "procedural"]
    if include_graph:
        lanes.append("graph")
    parser.add_argument("--lane", choices=tuple(lanes))
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    _add_yes_argument(parser)


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


def _is_secret_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return (
        normalized in _SECRET_KEYS
        or normalized.startswith(_SECRET_PREFIXES)
        or normalized.endswith(_SECRET_SUFFIXES)
    )


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            redacted[key] = "<redacted>" if _is_secret_key(key) else _redact(item)
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


def _emit_error(
    command: str,
    message: str,
    output_format: str,
) -> None:
    if output_format == "json":
        print(
            json.dumps(
                {
                    "schema_version": "1",
                    "command": command,
                    "status": "error",
                    "data": None,
                    "errors": [{"message": message}],
                },
                sort_keys=True,
            )
        )
        return
    print(f"error: {message}", file=sys.stderr)


def _read_security_input(args: argparse.Namespace) -> str:
    sources = sum(
        (
            args.text is not None,
            args.input_file is not None,
            args.stdin,
        )
    )
    if sources != 1:
        raise ValueError(
            "security scan requires exactly one of TEXT, --file, or --stdin"
        )
    if args.text is not None:
        return args.text
    if args.input_file is not None:
        if not args.input_file.is_file():
            raise ValueError(
                f"security scan input file not found: {args.input_file}"
            )
        return args.input_file.read_text(encoding="utf-8")
    return sys.stdin.read()


def _build_parser() -> argparse.ArgumentParser:
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
        help="Check configuration and local dependencies without initializing UMA.",
    )
    _add_runtime_options(doctor_parser)

    health_parser = commands.add_parser(
        "health",
        help="Initialize UMA and check runtime dependency health.",
    )
    _add_runtime_options(health_parser)

    retrieve_parser = commands.add_parser(
        "retrieve",
        help="Run scoped UMA retrieval.",
    )
    retrieve_commands = retrieve_parser.add_subparsers(
        dest="operation",
        required=True,
    )
    for name, help_text in (
        ("context", "Retrieve curated context."),
        ("memory", "Retrieve compiled memory."),
    ):
        operation_parser = retrieve_commands.add_parser(name, help=help_text)
        operation_parser.add_argument("query", help="Retrieval query text.")
        add_request_scope_arguments(operation_parser)

    ingest_parser = commands.add_parser(
        "ingest",
        help="Ingest scoped content into UMA.",
    )
    ingest_commands = ingest_parser.add_subparsers(
        dest="operation",
        required=True,
    )
    document_parser = ingest_commands.add_parser(
        "document",
        help="Ingest one document for an explicit durable owner.",
    )
    document_parser.add_argument("file", type=Path)
    # Documents are owner-scoped, not user- or agent-scoped: the owner tuple
    # alone decides who reads the document back, so there is no --agent here.
    add_owner_scope_arguments(document_parser)

    turn_parser = ingest_commands.add_parser(
        "turn",
        help="Ingest one conversation turn.",
    )
    turn_parser.add_argument(
        "--user-message",
        "--user-msg",
        required=True,
        dest="user_message",
    )
    turn_parser.add_argument("--assistant-reply", required=True)
    add_request_scope_arguments(turn_parser)

    for name, help_text in (
        ("memory-bootstrap", "Ingest a MEMORY.md bootstrap."),
        ("diary-bootstrap", "Ingest a daily diary bootstrap."),
    ):
        bootstrap_parser = ingest_commands.add_parser(name, help=help_text)
        bootstrap_parser.add_argument("file", type=Path)
        add_request_scope_arguments(bootstrap_parser)

    audit_parser = commands.add_parser(
        "audit",
        help="Inspect tenant-scoped retrieval audit records.",
    )
    audit_commands = audit_parser.add_subparsers(
        dest="operation",
        required=True,
    )
    audit_list_parser = audit_commands.add_parser(
        "list",
        help="List retrieval audit records for one tenant.",
    )
    add_audit_scope_arguments(audit_list_parser)
    audit_list_parser.add_argument(
        "--severity-min",
        choices=("none", "low", "medium", "high"),
    )
    audit_list_parser.add_argument("--limit", type=_positive_int, default=100)

    auth_parser = commands.add_parser(
        "auth",
        help="Manage bearer tokens for `uma-mcp --http`.",
    )
    auth_commands = auth_parser.add_subparsers(
        dest="operation",
        required=True,
    )
    auth_create_parser = auth_commands.add_parser(
        "create",
        help="Issue a new bearer token for a (tenant, user) pair.",
    )
    auth_create_parser.add_argument(
        "label",
        help="Human-readable label (e.g. 'perplexity', 'claude-desktop').",
    )
    auth_create_parser.add_argument("--tenant", dest="tenant_id")
    auth_create_parser.add_argument(
        "--user",
        dest="user_id",
        required=True,
    )
    auth_create_parser.add_argument("--tokens-db", type=Path, default=None)

    auth_list_parser = auth_commands.add_parser(
        "list",
        help="List issued tokens.",
    )
    auth_list_parser.add_argument("--tenant", dest="tenant_id")
    auth_list_parser.add_argument(
        "--include-revoked",
        action="store_true",
        help="Include revoked tokens in the output.",
    )
    auth_list_parser.add_argument("--tokens-db", type=Path, default=None)

    auth_revoke_parser = auth_commands.add_parser(
        "revoke",
        help="Revoke a bearer token by its token_id.",
    )
    auth_revoke_parser.add_argument("token_id")
    auth_revoke_parser.add_argument("--tokens-db", type=Path, default=None)

    quarantine_parser = commands.add_parser(
        "quarantine",
        help="Inspect owner-scoped quarantined records.",
    )
    quarantine_commands = quarantine_parser.add_subparsers(
        dest="operation",
        required=True,
    )
    quarantine_list_parser = quarantine_commands.add_parser(
        "list",
        help="List quarantined records for one durable owner.",
    )
    add_owner_scope_arguments(quarantine_list_parser)
    quarantine_list_parser.add_argument(
        "--lane",
        choices=("semantic", "episodic", "procedural", "raw"),
    )
    quarantine_list_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=100,
    )
    for name, help_text in (
        ("reinstate", "Reinstate one exact quarantined record."),
        ("purge", "Permanently purge one exact quarantined record."),
    ):
        mutation_parser = quarantine_commands.add_parser(name, help=help_text)
        add_record_scope_arguments(mutation_parser)
        mutation_parser.add_argument(
            "--reason",
            type=non_empty_value,
            default=("CLI reinstatement" if name == "reinstate" else None),
        )
        _add_yes_argument(mutation_parser)

    index_parser = commands.add_parser(
        "index",
        help="Rebuild one exact owner/lane index scope.",
    )
    index_commands = index_parser.add_subparsers(
        dest="operation",
        required=True,
    )
    rebuild_vectors_parser = index_commands.add_parser(
        "rebuild-vectors",
        help="Rebuild one vector lane for one exact owner.",
    )
    _add_index_target_arguments(
        rebuild_vectors_parser,
        include_graph=False,
    )
    rebuild_derived_parser = index_commands.add_parser(
        "rebuild-derived",
        help="Rebuild one derived lane for one exact owner.",
    )
    _add_index_target_arguments(
        rebuild_derived_parser,
        include_graph=True,
    )

    integrity_parser = commands.add_parser(
        "integrity",
        help="Enforce integrity for one exact stored record.",
    )
    integrity_commands = integrity_parser.add_subparsers(
        dest="operation",
        required=True,
    )
    integrity_enforce_parser = integrity_commands.add_parser(
        "enforce",
        help="Verify and quarantine one record if its hash mismatches.",
    )
    add_record_scope_arguments(integrity_enforce_parser)
    _add_yes_argument(integrity_enforce_parser)

    maintenance_parser = commands.add_parser(
        "maintenance",
        help="Run manual maintenance jobs. Caller-invoked only; UMA core never schedules these.",
    )
    maintenance_commands = maintenance_parser.add_subparsers(
        dest="operation",
        required=True,
    )
    consolidate_parser = maintenance_commands.add_parser(
        "consolidate",
        help="Run one consolidation cycle for a user: cluster episodes, extract "
        "facts, then prune. Destructive — deletes episodes and facts.",
    )
    add_audit_scope_arguments(consolidate_parser)
    _add_yes_argument(consolidate_parser)

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

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command_name = args.command
    if args.command == "config":
        command_name = f"config.{args.config_command}"
    elif args.command == "security":
        command_name = f"security.{args.security_command}"
    elif args.command == "dev":
        command_name = f"dev.{args.dev_command}"
    elif args.command in {
        "retrieve",
        "ingest",
        "audit",
        "auth",
        "quarantine",
        "index",
        "integrity",
        "maintenance",
    }:
        command_name = f"{args.command}.{args.operation}"

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

        if args.command == "auth":
            # Auth ops don't need uma.yaml — the token store is standalone
            # so operators can issue tokens before configuring UMA.
            from . import auth as auth_cli

            handler = {
                "create": auth_cli.handle_create,
                "list": auth_cli.handle_list,
                "revoke": auth_cli.handle_revoke,
            }[args.operation]
            data, text, status, exit_code = handler(args)
            _emit(command_name, data, args.output_format, text, status)
            return exit_code

        security_input = (
            _read_security_input(args)
            if args.command == "security"
            else None
        )

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

        if args.command in {"doctor", "health"}:
            from .diagnostics import (
                doctor_offline,
                doctor_runtime,
                runtime_health,
            )

            if args.command == "health":
                data, text, status, exit_code = runtime_health(
                    config_path,
                    timeout_seconds=args.timeout_seconds,
                )
            elif args.offline:
                data, text, status, exit_code = doctor_offline(
                    config,
                    config_path,
                )
            else:
                data, text, status, exit_code = doctor_runtime(
                    config,
                    config_path,
                    timeout_seconds=args.timeout_seconds,
                )
            _emit(args.command, data, args.output_format, text, status)
            return exit_code

        if args.command in {
            "retrieve",
            "ingest",
            "audit",
            "quarantine",
            "index",
            "integrity",
            "maintenance",
        }:
            from .operations import run_scoped_operation

            data, text, status, exit_code = run_scoped_operation(
                config_path,
                args,
            )
            _emit(command_name, data, args.output_format, text, status)
            return exit_code

        from .security import scan_input

        if security_input is None:
            raise RuntimeError("security input was not resolved")
        data, text, status, exit_code = scan_input(
            config,
            config_path,
            security_input,
            args.fail_on,
        )
        _emit("security.scan", data, args.output_format, text, status)
        return exit_code
    except BrokenPipeError:
        return 1
    except KeyboardInterrupt:
        _emit_error(command_name, "interrupted", args.output_format)
        return 130
    except Exception as exc:
        logger.debug("CLI command %s failed", command_name, exc_info=True)
        _emit_error(command_name, str(exc), args.output_format)
        return 2
