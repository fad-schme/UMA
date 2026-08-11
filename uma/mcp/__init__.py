"""UMA MCP server package.

Exposes UMA's memory API as MCP tools over stdio. The `uma-mcp` console
script (registered in ``pyproject.toml``) is the entry point:

    uma-mcp

Requires the ``mcp`` optional extra:

    pip install 'uma-mem[mcp]'

Configuration is via environment variables read by the server module:

    UMA_CONFIG_PATH   Absolute path to uma.yaml (required).
    UMA_AGENT_ID      Agent identity bound to this server (default:
                      "agent-default"). Immutable per process — spawn
                      one uma-mcp per agent identity.

Client configuration lives in ``docs/mcp/STDIO_CLIENTS.md``.
"""

from uma.mcp.server import main

__all__ = ["main"]
