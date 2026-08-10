"""Scoped retrieval, ingestion, and administrative CLI operations."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import logging
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

from uma.api import management as management_api
from uma.api.memory import UMAMemory
from uma.cli.confirmation import require_confirmation
from uma.common.results import DerivedRebuildReport, VectorRebuildReport


logger = logging.getLogger(__name__)

_RETRIEVAL_EFFECTS = ["retrieval_audit_write"]
_INGEST_EFFECTS = ["memory_write", "vector_index_write"]
_ARGUMENT_FLAGS = {
    "agent_id": "--agent",
    "owner_id": "--owner-id",
    "owner_type": "--owner-type",
    "lane": "--lane",
    "reason": "--reason",
    "record_id": "--record-id",
    "session_id": "--session",
    "user_id": "--user",
}


class OperationUsageError(ValueError):
    """A missing or incompatible CLI operation argument."""


def _require(args: Any, *names: str) -> None:
    missing = [
        _ARGUMENT_FLAGS.get(name, f"--{name.replace('_', '-')}")
        for name in names
        if not getattr(args, name, None)
    ]
    if missing:
        raise OperationUsageError(
            f"{args.command} {args.operation} requires "
            f"{', '.join(missing)}"
        )


def _require_input_file(args: Any) -> None:
    input_path = getattr(args, "file", None)
    if not isinstance(input_path, Path) or not input_path.is_file():
        raise OperationUsageError(
            f"{args.command} {args.operation} input file not found: "
            f"{input_path}"
        )


def _request_scope(args: Any) -> dict[str, Any]:
    return {
        "tenant_id": args.tenant_id,
        "agent_id": args.agent_id,
        "user_id": args.user_id,
        "session_id": args.session_id,
        "workspace_id": args.workspace_id,
        "request_id": args.request_id,
    }


def _owner_scope(args: Any) -> dict[str, Any]:
    return {
        "tenant_id": args.tenant_id,
        "owner_type": args.owner_type,
        "owner_id": args.owner_id,
    }


def _record_scope(args: Any) -> dict[str, Any]:
    return {
        **_owner_scope(args),
        "lane": args.lane,
        "record_id": args.record_id,
    }


def _validate_operation_args(args: Any) -> None:
    if args.command == "retrieve":
        _require(args, "agent_id", "user_id")
    elif args.command == "ingest" and args.operation == "document":
        _require(args, "owner_type", "owner_id")
        _require_input_file(args)
    elif args.command == "ingest" and args.operation == "turn":
        _require(args, "agent_id", "user_id", "session_id")
    elif args.command == "ingest":
        _require(args, "agent_id", "user_id")
        _require_input_file(args)
    elif args.command == "quarantine" and args.operation == "list":
        _require(args, "owner_type", "owner_id")
    elif args.command == "quarantine":
        _require(args, "owner_type", "owner_id", "lane", "record_id")
        if args.operation == "purge":
            _require(args, "reason")
    elif args.command == "index":
        _require(args, "owner_type", "owner_id", "lane")
    elif args.command == "integrity":
        _require(args, "owner_type", "owner_id", "lane", "record_id")


def _operation_effects(args: Any) -> list[str]:
    if args.command == "retrieve":
        return list(_RETRIEVAL_EFFECTS)
    if args.command == "ingest":
        return list(_INGEST_EFFECTS)
    if args.command == "quarantine" and args.operation == "reinstate":
        return ["quarantine_state_write", "security_audit_write"]
    if args.command == "quarantine" and args.operation == "purge":
        return ["memory_delete", "vector_index_delete"]
    if args.command == "index":
        return (
            ["graph_index_write"]
            if args.lane == "graph"
            else ["vector_index_write"]
        )
    if args.command == "integrity":
        return [
            "integrity_verification",
            "quarantine_state_write_on_mismatch",
            "security_audit_write_on_mismatch",
        ]
    return []


def _is_guarded_operation(args: Any) -> bool:
    return (
        args.command in {"index", "integrity"}
        or (
            args.command == "quarantine"
            and args.operation in {"reinstate", "purge"}
        )
    )


def _confirmation_target(args: Any) -> dict[str, Any]:
    target = {
        "tenant_id": args.tenant_id,
        "owner_type": args.owner_type,
        "owner_id": args.owner_id,
        "lane": args.lane,
    }
    if args.command == "index":
        target["record_scope"] = "all records in this exact owner/lane"
    else:
        target["record_id"] = args.record_id
    if args.command == "quarantine":
        target["reason"] = args.reason
    return target


def _confirmation_message(args: Any, target: dict[str, Any]) -> str:
    resolved_target = ", ".join(
        f"{name}={value!r}"
        for name, value in target.items()
    )
    return (
        f"Confirm {args.command} {args.operation}; "
        f"exact resolved target: {resolved_target}"
    )


def _index_flags(lane: str) -> dict[str, bool]:
    return {
        "include_episodic": lane == "episodic",
        "include_semantic": lane == "semantic",
        "include_procedural": lane == "procedural",
    }


def _result_outcome(args: Any, result: Any) -> tuple[str, int]:
    if args.command == "index" and isinstance(
        result, (VectorRebuildReport, DerivedRebuildReport)
    ):
        if isinstance(result, VectorRebuildReport):
            selected = result.report.get(args.lane)
        elif args.lane == "graph":
            selected = result.graph
        else:
            selected = result.vector.report.get(args.lane)
        selected_status = selected.status if selected is not None else None
        if selected_status == "error":
            return "error", 1
        if selected_status == "skipped":
            return "degraded", 4
        if selected_status == "ok":
            return "ok", 0
        return "error", 1
    if args.command == "integrity":
        result_status = getattr(result, "status", None)
        if result_status == "failed":
            return "findings", 1
    if (
        args.command == "quarantine"
        and args.operation in {"reinstate", "purge"}
        and result is not True
    ):
        return "error", 1
    return "ok", 0


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(mode="python"))
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _to_jsonable(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _text_result(command: str, data: dict[str, Any]) -> str:
    return (
        f"UMA {command}\n"
        f"{yaml.safe_dump(data, sort_keys=False).rstrip()}"
    )


async def _run_scoped_operation(
    config_path: Path,
    args: Any,
) -> tuple[dict[str, Any], str, str, int]:
    memory: UMAMemory | None = None
    command = f"{args.command}.{args.operation}"
    _validate_operation_args(args)
    effects = _operation_effects(args)
    data: dict[str, Any] = {"effects": effects}
    target: dict[str, Any] | None = None
    status = "error"
    exit_code = 1
    try:
        if _is_guarded_operation(args):
            target = _confirmation_target(args)
            require_confirmation(
                message=_confirmation_message(args, target),
                assume_yes=args.yes,
                stdin_is_tty=sys.stdin.isatty(),
            )
        memory = UMAMemory.from_yaml(str(config_path))

        if args.command == "retrieve":
            scoped = memory.set_context(agent_id=args.agent_id)
            kwargs = {
                "query_text": args.query,
                "user_id": args.user_id,
                "tenant_id": args.tenant_id,
                "request_id": args.request_id,
                "workspace_id": args.workspace_id,
                "session_id": args.session_id,
            }
            if args.operation == "context":
                result = await scoped.retrieve_context(**kwargs)
            else:
                result = await scoped.retrieve_memory(**kwargs)
            scope = _request_scope(args)

        elif args.command == "ingest" and args.operation == "document":
            result = await memory.ingest_document(
                str(args.file),
                owner_type=args.owner_type,
                owner_id=args.owner_id,
                tenant_id=args.tenant_id,
            )
            scope = _owner_scope(args)

        elif args.command == "ingest" and args.operation == "turn":
            scoped = memory.set_context(agent_id=args.agent_id)
            await scoped.process_turn(
                user_id=args.user_id,
                user_msg=args.user_message,
                assistant_reply=args.assistant_reply,
                session_id=args.session_id,
                tenant_id=args.tenant_id,
                workspace_id=args.workspace_id,
            )
            result = {"status": "ingested"}
            scope = _request_scope(args)

        elif args.command == "ingest":
            scoped = memory.set_context(agent_id=args.agent_id)
            kwargs = {
                "user_id": args.user_id,
                "tenant_id": args.tenant_id,
                "request_id": args.request_id,
                "workspace_id": args.workspace_id,
                "session_id": args.session_id,
            }
            if args.operation == "memory-bootstrap":
                result = await scoped.load_memory_bootstrap(
                    str(args.file),
                    **kwargs,
                )
            else:
                result = await scoped.load_daily_diary_bootstrap(
                    str(args.file),
                    **kwargs,
                )
            scope = _request_scope(args)

        elif args.command == "audit":
            result = await management_api.list_retrieval_audit(
                memory,
                tenant_id=args.tenant_id,
                user_id=args.user_id,
                severity_min=args.severity_min,
                limit=args.limit,
            )
            scope = {
                "tenant_id": args.tenant_id,
                "user_id": args.user_id,
            }

        elif args.command == "quarantine" and args.operation == "list":
            result = await management_api.list_quarantined(
                memory,
                tenant_id=args.tenant_id,
                owner_type=args.owner_type,
                owner_id=args.owner_id,
                lane=args.lane,
                limit=args.limit,
            )
            scope = _owner_scope(args)

        elif args.command == "quarantine":
            kwargs = {
                "record_id": args.record_id,
                "lane": args.lane,
                "owner_type": args.owner_type,
                "owner_id": args.owner_id,
                "tenant_id": args.tenant_id,
                "reason": args.reason,
            }
            if args.operation == "reinstate":
                result = await management_api.reinstate_quarantined(
                    memory,
                    **kwargs,
                )
            else:
                result = await management_api.purge_quarantined(
                    memory,
                    **kwargs,
                )
            scope = _record_scope(args)

        elif args.command == "index":
            kwargs = {
                "tenant_id": args.tenant_id,
                "owner_type": args.owner_type,
                "owner_id": args.owner_id,
                **_index_flags(args.lane),
                "batch_size": args.batch_size,
            }
            if args.operation == "rebuild-vectors":
                result = await memory.rebuild_vector_indexes(**kwargs)
            else:
                result = await memory.rebuild_derived_indexes(
                    **kwargs,
                    include_graph=args.lane == "graph",
                )
            scope = {
                **_owner_scope(args),
                "lane": args.lane,
                "record_scope": "all records in this exact owner/lane",
            }

        else:
            result = await management_api.verify_integrity(
                memory,
                record_id=args.record_id,
                lane=args.lane,
                owner_type=args.owner_type,
                owner_id=args.owner_id,
                tenant_id=args.tenant_id,
            )
            scope = _record_scope(args)

        data = {
            "scope": scope,
            "effects": effects,
            "result": _to_jsonable(result),
        }
        if target is not None:
            data["target"] = target
        status, exit_code = _result_outcome(args, result)
    except Exception as exc:
        logger.debug(
            "CLI operation %s failed",
            command,
            exc_info=True,
        )
        data = {
            "effects": effects,
            "error": str(exc),
        }
        if target is not None:
            data["target"] = target
    finally:
        if memory is not None:
            try:
                memory.shutdown()
            except Exception as exc:
                logger.debug(
                    "CLI operation %s shutdown failed",
                    command,
                    exc_info=True,
                )
                data = {
                    **data,
                    "effects": effects,
                    "shutdown_error": str(exc),
                }
                status = "error"
                exit_code = 1
    return data, _text_result(command, data), status, exit_code


def run_scoped_operation(
    config_path: Path,
    args: Any,
) -> tuple[dict[str, Any], str, str, int]:
    """Run one async public SDK operation from the synchronous CLI."""

    return asyncio.run(_run_scoped_operation(config_path, args))


__all__ = [
    "OperationUsageError",
    "run_scoped_operation",
]
