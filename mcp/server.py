"""UMA MCP Server — exposes UMA memory operations as MCP tools.

Configuration via environment variables:
  UMA_CONFIG_PATH  Path to uma.yaml (required)
  UMA_AGENT_ID     Agent identity bound to this server (default: agent-default)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("uma.mcp")

# ---------------------------------------------------------------------------
# UMA lazy singleton
# ---------------------------------------------------------------------------
_memory = None


def _get_memory():
    global _memory
    if _memory is None:
        config_path = os.environ.get("UMA_CONFIG_PATH")
        if not config_path:
            raise RuntimeError("UMA_CONFIG_PATH environment variable is required.")
        agent_id = os.environ.get("UMA_AGENT_ID", "agent-default")

        # Ensure the uma package is importable when server.py runs standalone.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from uma.api.memory import UMAMemory

        logger.info("Initializing UMAMemory from %s", config_path)
        _memory = UMAMemory.from_yaml(config_path).set_context(agent_id=agent_id)
    return _memory


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP("uma")


@mcp.tool()
async def retrieve_context(
    query_text: str,
    user_id: str,
    session_id: str = "",
    tenant_id: str = "default",
) -> str:
    """Retrieve curated RAG context from UMA for the given query.

    Returns a JSON object with facts, chunks, and supporting evidence
    scoped to the user and session.
    """
    memory = _get_memory()
    result = await memory.retrieve_context(
        query_text=query_text,
        user_id=user_id,
        tenant_id=tenant_id,
        session_id=session_id or None,
    )
    return json.dumps(result, default=str)


@mcp.tool()
async def retrieve_memory(
    query_text: str,
    user_id: str,
    session_id: str = "",
    tenant_id: str = "default",
    memory_intent: str = "continuity",
) -> str:
    """Retrieve compiled, evidence-backed memory from UMA.

    Returns a JSON object with compiled memories and supporting evidence.
    Use memory_intent='continuity' for long-term recall or 'topical' for
    domain-specific knowledge.
    """
    memory = _get_memory()
    result = await memory.retrieve_memory(
        query_text=query_text,
        user_id=user_id,
        tenant_id=tenant_id,
        session_id=session_id or None,
        memory_intent=memory_intent,
    )
    return json.dumps(result, default=str)


@mcp.tool()
async def process_turn(
    user_id: str,
    session_id: str,
    user_msg: str,
    assistant_reply: str,
    tenant_id: str = "default",
) -> str:
    """Ingest a conversation turn into UMA memory.

    Stores the exchange as episodic memory and extracts semantic facts.
    Returns a JSON status object.
    """
    memory = _get_memory()
    await memory.process_turn(
        user_id=user_id,
        user_msg=user_msg,
        assistant_reply=assistant_reply,
        session_id=session_id,
        tenant_id=tenant_id,
    )
    return json.dumps({"status": "ok", "user_id": user_id, "session_id": session_id})


@mcp.tool()
async def ingest_document(
    file_path: str,
    owner_type: str = "agent",
    owner_id: str = "",
) -> str:
    """Ingest a document file into UMA's knowledge base.

    Chunks, embeds, and indexes the document. Subsequent retrieve_context
    calls will surface its content.
    Returns a JSON IngestReport.
    """
    memory = _get_memory()
    resolved_owner_id = owner_id or memory.agent_id or "agent-default"
    result = await memory.ingest_document(
        file_path,
        owner_type=owner_type,
        owner_id=resolved_owner_id,
    )
    return json.dumps(result.__dict__ if hasattr(result, "__dict__") else result, default=str)


@mcp.tool()
def health_check() -> str:
    """Return UMA runtime health status.

    Returns a JSON object with 'status' and per-component check results.
    """
    memory = _get_memory()
    return json.dumps(memory.health_check(), default=str)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
