# UMA MCP — stdio clients

UMA ships an MCP server, `uma-mcp`, that exposes its public API as MCP tools.
This page covers **stdio** transport, where the client launches the server as a
child process and talks to it over stdin/stdout. That is the right mode for
local coding agents and desktop clients running on the same machine as your
memory store.

For a server other machines connect to over HTTP, see [`DEPLOY.md`](DEPLOY.md)
(bearer tokens) and [`CHATGPT.md`](CHATGPT.md) (OAuth 2.1).

---

## Install

```bash
pip install 'uma-mem[mcp]'
```

This puts the `uma-mcp` executable on PATH. UMA requires Python 3.10+.

You also need a working `uma.yaml`. UMA never packages one — it is a file you
author and edit, and the path is yours to choose. See
[`configure.md`](../../.claude/skills/configure.md) for the full reference.

Confirm the runtime is healthy before wiring any client, because a client that
launches a broken server just fails silently:

```bash
uma --config /absolute/path/to/uma.yaml health
```

Exit code 0 means stores, vector indexes, and both providers are reachable.

---

## Configuration

The stdio server is configured through a single environment variable:

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `UMA_CONFIG_PATH` | **yes** | — | Absolute path to your `uma.yaml`. The server raises at first tool call if unset. |

The server holds no identity of its own. UMA is single-tenant, multi-agent and
multi-user, so **`agent_id` is a required argument on every tool call** —
`retrieve_context`, `retrieve_memory`, `process_turn`, and `ingest_document`
all reject a missing or empty value. `user_id` is required too in stdio mode
(in HTTP mode the bearer token supplies it), and `tenant_id` is optional and
defaults to `"default"`.

One server process therefore serves every agent: isolate assistant
configurations by passing distinct `agent_id` values, not by running a server
per agent.

Use **absolute paths**. Clients launch the server with an unpredictable working
directory, and a relative `db_root` in your `uma.yaml` resolves against either
the config file's directory or the process CWD depending on `db_root_base`. An
absolute `UMA_CONFIG_PATH` plus `db_root_base: "config"` is the combination that
behaves the same no matter who starts the process.

---

## Tools

Five tools, each a thin pass-through to the corresponding public UMA method.
Every one returns a JSON string.

| Tool | Returns |
| --- | --- |
| `retrieve_context` | `ContextBundle` — chunks, facts, episodic, working memory, provenance, `query_scan_severity` |
| `retrieve_memory` | `MemoryResult` — `compiled_memory`, facts as subject-predicate-object triples, evidence, `provenance_valid` |
| `process_turn` | `{"status": "ok", ...}`, or `{"status": "injection_blocked", "severity": ..., "matched_rules": [...]}` when the write-time scan rejected `user_msg` at high severity. On a block, nothing was stored. |
| `ingest_document` | `IngestReport` |
| `health_check` | `HealthStatus` |

**In stdio mode the assistant must pass `user_id` on every call.** There is no
token to derive identity from, so the tool arguments are authoritative and an
empty `user_id` is an error. `process_turn` additionally requires a non-empty
`session_id`.

`ingest_document` is scoped by `(tenant_id, owner_type, owner_id)` rather than
by a request scope. `owner_type` is `"agent"` (the default — the document joins
the calling agent's shared KB) or `"user"` (private to the ingesting user), and
an empty `owner_id` defaults to the caller in both cases. It may not name anyone
else: an agent-owned document is owned by the `agent_id` on the call, and a
user-owned one by the calling user, so `user_id` is required in stdio mode when
`owner_type` is `"user"` and unnecessary otherwise.

---

## Client setup

Use the absolute path to your `uma.yaml` in every example below, and restart the
client fully after editing its config.

### Claude Code

```bash
claude mcp add uma --env UMA_CONFIG_PATH=/absolute/path/to/uma.yaml -- uma-mcp
```

No agent flag is needed — the calling model passes `agent_id` in each tool
call.

### Claude Desktop

Edit the config file for your platform:

| Platform | Path |
| --- | --- |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "uma": {
      "command": "uma-mcp",
      "env": {
        "UMA_CONFIG_PATH": "/absolute/path/to/uma.yaml"
      }
    }
  }
}
```

If `uma-mcp` is not on the PATH the desktop app sees — common when UMA is
installed in a virtualenv — give the absolute path to the executable instead,
e.g. `/path/to/.venv/bin/uma-mcp` (or `...\.venv\Scripts\uma-mcp.exe` on
Windows).

### Cursor

`.cursor/mcp.json` in the project, or `~/.cursor/mcp.json` globally:

```json
{
  "mcpServers": {
    "uma": {
      "command": "uma-mcp",
      "env": {
        "UMA_CONFIG_PATH": "/absolute/path/to/uma.yaml"
      }
    }
  }
}
```

### Windsurf

`~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "uma": {
      "command": "uma-mcp",
      "env": {
        "UMA_CONFIG_PATH": "/absolute/path/to/uma.yaml"
      }
    }
  }
}
```

### Codex

`~/.codex/config.toml`:

```toml
[mcp_servers.uma]
command = "uma-mcp"
env = { UMA_CONFIG_PATH = "/absolute/path/to/uma.yaml" }
```

---

## Troubleshooting

**The client shows the server as failed, with no output.**
The server logs to stderr only — stdout is reserved for the MCP JSON-RPC frame,
and anything else written there corrupts it. Check the client's MCP log pane.
Run `uma-mcp` directly in a terminal to see startup errors: it will wait on
stdin, which is the healthy state.

**`UMA_CONFIG_PATH environment variable is required`.**
The variable did not reach the child process. Confirm it is inside the server's
own `env` block rather than exported in your shell — most clients do not inherit
the shell environment.

**`user_id is required when running in stdio mode`.**
The assistant omitted `user_id`. It is a required argument on every retrieval
and ingestion tool in stdio mode.

**First call is very slow, or times out.**
A local provider such as Ollama loads the model into memory on the first request
after boot; that can take over a minute, while later calls return in
milliseconds. Raise `embedding.config.timeout` and `llms.uma.config.timeout` in
your `uma.yaml`, or warm the model first with `uma --config <path> health`.

**Memory appears empty across restarts.**
Almost always a `db_root` resolving to a different directory than you expect.
Set `db_root_base: "config"` so the database travels with the config file rather
than following the process working directory.
