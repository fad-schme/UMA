# UMA MCP Server

Exposes UMA memory operations as MCP tools so any MCP-compatible client
(Claude Desktop, Claude Code, etc.) can retrieve context, recall memory,
ingest documents, and store conversation turns.

## Tools

| Tool | Description |
|------|-------------|
| `retrieve_context` | RAG-style context retrieval for LLM prompting |
| `retrieve_memory` | Compiled, evidence-backed memory recall |
| `process_turn` | Store a conversation turn (episodic + semantic) |
| `ingest_document` | Ingest a document file into the knowledge base |
| `health_check` | Runtime health status |

## Prerequisites

- Python 3.9+
- UMA installed in the same environment (`pip install -e .` from the repo root)
- A running Ollama instance with `nomic-embed-text` and your LLM of choice
- A valid `uma.yaml` config (see `config/uma.yaml`)

## Install

```bash
# From the repo root — install UMA and the MCP dependency
pip install -e .
pip install "mcp[cli]>=1.0.0"
```

## Claude Desktop configuration

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) and add:

```json
{
  "mcpServers": {
    "uma": {
      "command": "python",
      "args": ["C:/path/to/uma/mcp/server.py"],
      "env": {
        "UMA_CONFIG_PATH": "C:/path/to/uma/config/uma.yaml",
        "UMA_AGENT_ID": "agent-default",
        "PYTHONPATH": "C:/path/to/uma"
      }
    }
  }
}
```

Replace `C:/path/to/uma` with the absolute path to this repository.

> **Tip — using `uv`:** If you manage the venv with `uv`, replace `"command": "python"` with
> `"command": "uv"` and set `"args": ["run", "--project", "C:/path/to/uma", "python", "mcp/server.py"]`.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `UMA_CONFIG_PATH` | Yes | — | Absolute path to `uma.yaml` |
| `UMA_AGENT_ID` | No | `agent-default` | Agent identity bound to this server |

## Quick test (CLI)

```bash
UMA_CONFIG_PATH=/path/to/uma/config/uma.yaml \
  python mcp/server.py
```

The server speaks JSON-RPC over stdio; Claude Desktop handles the protocol.
