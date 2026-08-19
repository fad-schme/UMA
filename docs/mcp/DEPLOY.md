# UMA MCP — HTTP deployment with bearer tokens

This page covers running `uma-mcp` over HTTP so remote MCP clients — Claude
Desktop's remote connector, Claude Cowork, Perplexity — can reach your memory
layer. Authentication is an opaque bearer token issued locally by UMA.

For ChatGPT, or any client that requires OAuth 2.1, see
[`CHATGPT.md`](CHATGPT.md). For local coding agents, stdio is simpler — see
[`STDIO_CLIENTS.md`](STDIO_CLIENTS.md).

---

## Install

```bash
pip install 'uma-mem[mcp]'
```

---

## Start the server

```bash
export UMA_CONFIG_PATH=/absolute/path/to/uma.yaml

uma-mcp --http --port 3131
```

That binds `127.0.0.1:3131` and serves the MCP endpoint at
`http://127.0.0.1:3131/mcp` using streamable-http.

`UMA_CONFIG_PATH` works exactly as in stdio mode, and `agent_id` is likewise
required on every tool call — see
[`STDIO_CLIENTS.md`](STDIO_CLIENTS.md#configuration).

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--http` | off | Enable HTTP (streamable-http). Stdio is the default transport. |
| `--port` | `3131` | Bind port. |
| `--bind` | `127.0.0.1` | Bind interface. Set to `0.0.0.0` when serving through a tunnel or reverse proxy. |
| `--public-url` | derived | Externally-reachable base URL. **Required whenever `--bind` is not loopback.** |
| `--tokens-db` | `.uma/db/mcp_tokens.db` | SQLite path for the bearer-token store. Mutually exclusive with `--oauth-issuer`. |

### `--public-url` is not optional off loopback

The server advertises issuer and resource identifiers derived from its own URL.
If it binds `0.0.0.0` but advertises `http://0.0.0.0:3131`, clients receive an
address they cannot connect to and fail with a connection error that looks like
a network fault rather than a configuration one. The server refuses to start in
that combination:

```
uma-mcp: --bind=0.0.0.0 requires --public-url so the issuer and resource
identifiers match the URL clients see.
```

Behind a tunnel or proxy:

```bash
uma-mcp --http --bind 0.0.0.0 --port 3131 \
        --public-url https://memory.example.com
```

The MCP endpoint clients should be pointed at is then
`https://memory.example.com/mcp`.

---

## Issue a token

Tokens are created out of band with the `uma auth` commands. Each token is bound
to exactly one `(tenant_id, user_id)` pair.

```bash
uma auth create claude-desktop --user alice
uma auth create perplexity     --user alice --tenant acme
```

```bash
uma auth list                 # issued tokens (add --include-revoked for all)
uma auth revoke <token_id>    # revoke one token
```

`--tokens-db PATH` selects a non-default store; pass the same path to
`uma-mcp --tokens-db`.

**The plaintext token is printed exactly once, at creation.** Only a SHA-256
hash is stored, so it cannot be recovered later — if it is lost, revoke it and
issue another. Server logs never contain the token; only its short `token_id`
handle appears.

---

## Connect a client

Point the client at `<public-url>/mcp` and give it the token as a bearer
credential. Most remote-MCP clients ask for the URL and an
`Authorization: Bearer <token>` header.

| Client | Endpoint | Credential |
| --- | --- | --- |
| Claude Desktop (remote connector) | `https://your-host/mcp` | Bearer token |
| Claude Cowork | `https://your-host/mcp` | Bearer token |
| Perplexity | `https://your-host/mcp` | Bearer token (or OAuth — see [`CHATGPT.md`](CHATGPT.md)) |

Verify by hand before wiring a client:

```bash
curl -sS -H "Authorization: Bearer $UMA_TOKEN" https://your-host/mcp
```

A `401` means the token is wrong, revoked, or the server is reading a different
`--tokens-db` than `uma auth create` wrote to.

---

## Scope and isolation

In HTTP mode **the token is authoritative for identity**, not the tool
arguments. Each token's `client_id` encodes `tenant_id:user_id`; the server
resolves that pair on every call and passes it to UMA, which enforces the same
`(tenant_id, owner_type, owner_id)` isolation at the SQL and vector layers that
every other UMA caller gets.

If a tool call also passes `user_id` or a non-default `tenant_id` and they do
not match the token, the call is rejected rather than silently overridden:

```
user_id='bob' does not match authenticated user (token identifies user='alice')
```

That is deliberate. A silent override would let a confused or manipulated client
believe it wrote to one scope while UMA wrote to another.

Practical consequences:

- One token per user. Sharing a token across people merges their memory.
- Revoking a token cuts access immediately; it does not delete anything already
  stored under that scope.
- `ingest_document` has no `user_id` — documents are owned by
  `(tenant_id, owner_type, owner_id)`. The token's tenant is still enforced.

---

## Hardening

The defaults are aimed at a local or tunnelled single-operator deployment. For
anything more exposed:

- **Terminate TLS in front of the server.** `uma-mcp` speaks plain HTTP; a
  bearer token on an unencrypted connection is readable in transit. Put it
  behind a reverse proxy or tunnel that provides HTTPS, and set `--public-url`
  to the `https://` address.
- **Keep the bind loopback** unless something in front of it is doing
  authentication and TLS.
- **Rate limiting is yours.** UMA ships no limiter. Register one with
  `set_rate_limit_hook`, or enforce it at the proxy.
- **Protect the tokens database** with filesystem permissions — it sits beside
  your memory stores under `.uma/db/` by default.
- **Rotate deliberately.** `uma auth create` a replacement, move the client
  over, then `uma auth revoke` the old `token_id`.

---

## Troubleshooting

**Server exits immediately with a `--public-url` message.**
You bound a non-loopback interface without telling the server its external URL.
See above.

**`401` on every request.**
The token is not in the store the server is reading. Confirm both `uma auth
create` and `uma-mcp` used the same `--tokens-db`, and that the token was not
revoked (`uma auth list --include-revoked`).

**Tool calls fail with a user/tenant mismatch.**
The client is sending an explicit `user_id` that disagrees with the token. Drop
it from the call — in HTTP mode the token supplies identity.

**`--oauth-issuer and --tokens-db are mutually exclusive`.**
Those are two different auth modes. Bearer validates opaque tokens from the
local store; OAuth validates JWTs from an external IdP. Pick one.
