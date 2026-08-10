"""Central confirmation gate for guarded CLI administration."""

from __future__ import annotations

import sys


class ConfirmationRequired(RuntimeError):
    """A non-interactive mutation omitted the explicit --yes flag."""


class ConfirmationDeclined(RuntimeError):
    """An interactive operator declined a guarded mutation."""


def require_confirmation(
    *,
    message: str,
    assume_yes: bool,
    stdin_is_tty: bool,
) -> None:
    """Require explicit consent while keeping machine-readable stdout clean."""

    resolved_message = message.strip()
    if not resolved_message:
        raise ValueError("confirmation message must be non-empty")

    print(resolved_message, file=sys.stderr)
    if assume_yes:
        return
    if not stdin_is_tty:
        raise ConfirmationRequired(
            "non-interactive administration requires --yes"
        )

    print("Continue? [y/N] ", end="", file=sys.stderr, flush=True)
    response = sys.stdin.readline().strip().lower()
    if response not in {"y", "yes"}:
        raise ConfirmationDeclined("operation declined")


__all__ = [
    "ConfirmationDeclined",
    "ConfirmationRequired",
    "require_confirmation",
]
