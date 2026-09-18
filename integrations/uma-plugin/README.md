# UMA plugin

Packages UMA's existing `uma-mcp` server as an installable plugin for Claude
Code, Codex, and Kimi Code. One implementation lives here; each client gets
a thin manifest under its own `.claude-plugin/`, `.codex-plugin/`, or
`.kimi-plugin/` folder (the root-level marketplace manifests at the repo root
point at this directory).

This plugin does not bundle UMA itself and does not ship a default
`uma.yaml` — `uma.yaml` is user-authored and never packaged (see
[`../../.claude/skills/configure.md`](../../.claude/skills/configure.md)).
Install and configure UMA first via
[`../../docs/mcp/STDIO_CLIENTS.md`](../../docs/mcp/STDIO_CLIENTS.md), then
install this plugin.

## What it adds over the raw MCP setup

- The five `uma-mcp` tools (`retrieve_context`, `retrieve_memory`,
  `process_turn`, `ingest_document`, `health_check`), wired up automatically
  instead of hand-edited into the client's MCP config.
- **SessionStart hook** — runs `uma health` against yoclaude /loginur config at the start
  of every session and warns (without blocking) if the store is unreachable.
- **Stop hook** — captures the session's last user/assistant exchange via
  `uma ingest turn`, so continuity does not depend on the agent remembering
  to call `process_turn` itself. Best-effort: it silently skips if any
  required identity variable is missing, the transcript is empty, or the
  `uma` CLI is not on PATH.

Everything else — retrieval, ranking, injection scanning, lane storage — is
unchanged UMA behavior reached through the same public API described in
[`../../docs/mcp/STDIO_CLIENTS.md`](../../docs/mcp/STDIO_CLIENTS.md).

## Required environment

UMA is single-tenant, multi-agent, multi-user: identity is required on every
call, not set once per session.

| Variable | Required by | Meaning |
| --- | --- | --- |
| `UMA_CONFIG_PATH` | MCP server, both hooks | Absolute path to your `uma.yaml`. |
| `UMA_AGENT_ID` | Stop hook | Agent identity passed to `uma ingest turn`. |
| `UMA_USER_ID` | Stop hook | User identity passed to `uma ingest turn`. |
| `UMA_TENANT_ID` | Stop hook (optional) | Defaults to `"default"` if unset. |

The MCP server itself only needs `UMA_CONFIG_PATH` — the calling model
supplies `agent_id`/`user_id` on each tool call per
[`STDIO_CLIENTS.md`](../../docs/mcp/STDIO_CLIENTS.md). The Stop hook has no
model in the loop, so it reads identity from the environment instead. If
`UMA_AGENT_ID` or `UMA_USER_ID` is unset, the Stop hook simply does not
capture — the MCP tools still work.

Set these in your shell profile or the client's own env passthrough before
installing the plugin; the plugin does not prompt for or scaffold them.

## Install

### Claude Code

```bash
claude plugin marketplace add fad-schme/UMA
claude plugin install uma
```

### Codex

Add the marketplace at the repo root (`.codex-plugin/marketplace.json`) per
your Codex client's plugin-marketplace instructions, then install `uma`.

### Kimi Code

Add the marketplace at the repo root (`.kimi-plugin/marketplace.json`) per
your Kimi Code client's plugin-marketplace instructions, then install `uma`.

## Troubleshooting

See [`../../docs/mcp/STDIO_CLIENTS.md`](../../docs/mcp/STDIO_CLIENTS.md) —
the plugin launches the same `uma-mcp` executable, so every failure mode
documented there (missing `UMA_CONFIG_PATH`, cold-start timeouts, `db_root`
resolving to the wrong directory) applies here too.
