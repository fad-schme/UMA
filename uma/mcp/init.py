"""UMA MCP server package.

Exposes UMA's memory API as MCP tools. The `uma-mcp` console script
(registered in ``pyproject.toml``) is the entry point:

    uma-mcp                                  # stdio mode
    uma-mcp --http --port 3131               # HTTP + bearer tokens

Requires the ``mcp`` optional extra:

    pip install 'uma-mem[mcp]'

Client configuration:
    docs/mcp/STDIO_CLIENTS.md   (stdio: Claude Code, Codex, Cursor, ...)
    docs/mcp/DEPLOY.md          (HTTP: Claude Desktop remote, Cowork, Perplexity)

Submodule note: this package's ``__init__`` is intentionally import-free
so that ``uma.mcp.tokens`` (stdlib-only) can be used by the CLI without
requiring the ``mcp`` extra. Only ``uma.mcp.server`` and
``uma.mcp.auth`` need the MCP SDK.
"""
