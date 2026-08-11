"""Smoke tests for the uma-mcp server.

These verify wiring, not behaviour: the FastMCP server object exists, the
five tools are registered under their expected names, and main() is
callable. Actual tool invocations require a booted UMAMemory — those live
in the e2e suite.

Skipped if the `mcp` optional extra isn't installed.
"""

from __future__ import annotations

import importlib.util

import pytest

MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None
pytestmark = pytest.mark.skipif(
    not MCP_AVAILABLE,
    reason="mcp extra not installed (pip install 'uma-mem[mcp]')",
)


EXPECTED_TOOLS = frozenset(
    {
        "retrieve_context",
        "retrieve_memory",
        "process_turn",
        "ingest_document",
        "health_check",
    }
)


def test_module_imports():
    """The server module must import without a running UMAMemory."""
    from uma.mcp import server as srv  # noqa: F401


def test_main_is_callable():
    """main() must exist and be callable (we do not invoke it — it blocks)."""
    from uma.mcp.server import main

    assert callable(main)


def test_fastmcp_server_object_present():
    """The module must expose a FastMCP-compatible ``mcp`` object."""
    from uma.mcp import server as srv

    assert srv.mcp is not None
    assert srv.mcp.name == "uma"


@pytest.mark.asyncio
async def test_expected_tools_registered():
    """All five UMA tools must be registered under their canonical names."""
    from uma.mcp import server as srv

    tools = await srv.mcp.list_tools()
    registered = {t.name for t in tools}
    missing = EXPECTED_TOOLS - registered
    unexpected = registered - EXPECTED_TOOLS
    assert not missing, f"missing MCP tools: {sorted(missing)}"
    assert not unexpected, f"unexpected MCP tools: {sorted(unexpected)}"


def test_get_memory_errors_without_config_env(monkeypatch):
    """_get_memory must refuse to boot without UMA_CONFIG_PATH."""
    from uma.mcp import server as srv

    monkeypatch.delenv("UMA_CONFIG_PATH", raising=False)
    # Reset the lazy singleton so the guard runs even if a prior test set it.
    monkeypatch.setattr(srv, "_memory", None)

    with pytest.raises(RuntimeError, match="UMA_CONFIG_PATH"):
        srv._get_memory()
