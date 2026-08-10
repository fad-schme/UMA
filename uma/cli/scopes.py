"""Reusable CLI argument registration for UMA scope types."""

from __future__ import annotations

import argparse
import os


_WILDCARD_CHARACTERS = frozenset("*?[]")


def non_empty_value(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("must be a non-empty string")
    return normalized


def exact_scope_value(value: str) -> str:
    normalized = non_empty_value(value)
    if any(character in normalized for character in _WILDCARD_CHARACTERS):
        raise argparse.ArgumentTypeError(
            "wildcards and glob syntax are not allowed"
        )
    return normalized


def _default_tenant() -> str:
    return (os.environ.get("UMA_TENANT_ID") or "default").strip() or "default"


def _add_tenant_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tenant",
        dest="tenant_id",
        type=exact_scope_value,
        default=_default_tenant(),
        help="Tenant ID (default: UMA_TENANT_ID, then 'default').",
    )


def add_request_scope_arguments(parser: argparse.ArgumentParser) -> None:
    """Register request-scope syntax without attaching command behavior."""

    _add_tenant_argument(parser)
    parser.add_argument("--agent", dest="agent_id", type=non_empty_value)
    parser.add_argument("--user", dest="user_id", type=non_empty_value)
    parser.add_argument("--session", dest="session_id", type=non_empty_value)
    parser.add_argument("--workspace", dest="workspace_id", type=non_empty_value)
    parser.add_argument("--request-id", dest="request_id", type=non_empty_value)


def add_owner_scope_arguments(parser: argparse.ArgumentParser) -> None:
    """Register durable owner-scope syntax without attaching command behavior."""

    _add_tenant_argument(parser)
    parser.add_argument(
        "--owner-type",
        choices=("agent", "user", "workspace", "system"),
    )
    parser.add_argument("--owner-id", type=exact_scope_value)


def add_audit_scope_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the tenant/user filters supported by the public audit API."""

    _add_tenant_argument(parser)
    parser.add_argument("--user", dest="user_id", type=non_empty_value)


def add_record_scope_arguments(parser: argparse.ArgumentParser) -> None:
    """Register exact owner, lane, and record syntax for one admin target."""

    add_owner_scope_arguments(parser)
    parser.add_argument(
        "--lane",
        choices=("semantic", "episodic", "procedural", "raw"),
    )
    parser.add_argument("--record-id", type=exact_scope_value)


__all__ = [
    "add_audit_scope_arguments",
    "add_owner_scope_arguments",
    "add_record_scope_arguments",
    "add_request_scope_arguments",
    "exact_scope_value",
    "non_empty_value",
]
