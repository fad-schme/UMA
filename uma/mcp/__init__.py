"""UMA MCP server package.

Exposes UMA's memory API as MCP tools over stdio. The `uma-mcp` console
script (registered in ``pyproject.toml``) is the entry point:

    uma-mcp

Requires the ``mcp`` optional extra:

    pip install 'uma-mem[mcp]'

Configuration is via environment variables read by the server module:

    UMA_CONFIG_PATH   Absolute path to uma.yaml (required).

The server holds no agent, user, or tenant identity. Each tool call carries
its own ``agent_id`` (required) and ``user_id``, so a single uma-mcp process
serves every agent.

Client configuration lives in ``docs/mcp/STDIO_CLIENTS.md``.
"""

from typing import Any

__all__ = ["main"]


def __getattr__(name: str) -> Any:
    """Resolve ``main`` on first access rather than at package import.

    ``uma.mcp.server`` requires the ``mcp`` optional extra, but the sibling
    modules ``tokens`` and ``auth`` are stdlib-only. Importing the server
    eagerly here made ``import uma.mcp.tokens`` fail whenever the extra was
    absent. The ``uma-mcp`` console script targets ``uma.mcp.server:main``
    directly and never went through this re-export.
    """
    if name == "main":
        from uma.mcp.server import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
