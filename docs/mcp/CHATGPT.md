# UMA MCP — OAuth 2.1 (ChatGPT and other IdP-backed clients)

ChatGPT requires an MCP server to authenticate with OAuth 2.1 rather than a
static bearer token. In this mode UMA acts as a **pure OAuth resource server**:
it verifies JWTs issued by your identity provider and never mints tokens itself.

If your client accepts a static bearer token, [`DEPLOY.md`](DEPLOY.md) is
simpler. For local coding agents, use [`STDIO_CLIENTS.md`](STDIO_CLIENTS.md).

---

## Install

```bash
pip install 'uma-mem[mcp,oauth]'
```

The `oauth` extra adds JWT verification. Without it, OAuth mode cannot start.

---

## What you need from your IdP

Any RFC 8414-compliant provider works — Auth0, Microsoft Entra ID, Google,
Okta, Keycloak, Authentik. Register UMA as an API / resource server and collect:

| Value | Maps to | Notes |
| --- | --- | --- |
| Issuer URL | `--oauth-issuer` | Must equal the JWT `iss` claim exactly, trailing slash included. |
| Audience / API identifier | `--oauth-audience` | Must equal the JWT `aud` claim. |
| JWKS endpoint | `--oauth-jwks-uri` | Only if your IdP does not publish at `{issuer}/.well-known/jwks.json`. |
| Scopes (optional) | `--oauth-required-scope` | Repeat the flag per required scope. |

Signing must be **RS256 or ES256**. UMA rejects everything else, HS256
included — accepting a shared-secret algorithm alongside public-key
verification is the classic algorithm-confusion attack, so the allowlist is
fixed rather than configurable.

---

## Start the server

```bash
export UMA_CONFIG_PATH=/absolute/path/to/uma.yaml

uma-mcp --http --bind 0.0.0.0 --port 3131 \
        --public-url https://memory.example.com \
        --oauth-issuer https://your-tenant.auth0.com/ \
        --oauth-audience https://memory.example.com/mcp
```

Clients connect to `https://memory.example.com/mcp`.

### OAuth flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--oauth-issuer` | — | IdP issuer URL. Setting it switches from opaque bearer to JWT verification. |
| `--oauth-audience` | — | Expected `aud` claim. **Required** with `--oauth-issuer`. |
| `--oauth-jwks-uri` | `{issuer}/.well-known/jwks.json` | Override for non-standard JWKS paths. |
| `--oauth-tenant` | `default` | UMA tenant every authenticated user maps to. |
| `--oauth-required-scope` | none | Required scope; repeat for several. All must be present. If omitted, any valid JWT passes. |

Two rules the server enforces at startup rather than at first request:

- `--oauth-issuer` without `--oauth-audience` is refused. An unverified audience
  means a token minted for a different API would be accepted.
- `--oauth-issuer` and `--tokens-db` are mutually exclusive — they are different
  auth modes.

The issuer advertised in the RFC 9728 protected-resource metadata is your
**IdP**, not UMA. UMA is not an authorization server and does not pretend to be
one.

---

## Per-IdP flag sets

Replace hostnames and identifiers with your own.

**Auth0** — note the trailing slash on the issuer, which Auth0 includes in `iss`:

```bash
--oauth-issuer   https://your-tenant.auth0.com/ \
--oauth-audience https://memory.example.com/mcp
```

**Microsoft Entra ID** (v2.0 endpoint):

```bash
--oauth-issuer    https://login.microsoftonline.com/<tenant-guid>/v2.0 \
--oauth-audience  api://<application-id> \
--oauth-jwks-uri  https://login.microsoftonline.com/<tenant-guid>/discovery/v2.0/keys
```

**Google**:

```bash
--oauth-issuer   https://accounts.google.com \
--oauth-audience <your-client-id>.apps.googleusercontent.com \
--oauth-jwks-uri https://www.googleapis.com/oauth2/v3/certs
```

**Okta**:

```bash
--oauth-issuer   https://your-org.okta.com/oauth2/default \
--oauth-audience api://default
```

**Keycloak**:

```bash
--oauth-issuer   https://kc.example.com/realms/<realm> \
--oauth-audience uma-mcp \
--oauth-jwks-uri https://kc.example.com/realms/<realm>/protocol/openid-connect/certs
```

**Authentik**:

```bash
--oauth-issuer   https://auth.example.com/application/o/<slug>/ \
--oauth-audience uma-mcp
```

---

## Connect ChatGPT

1. Serve over **HTTPS** at a publicly reachable hostname. ChatGPT will not
   connect to plain HTTP or to a private address.
2. Add the connector, pointing it at `https://memory.example.com/mcp`.
3. Complete the OAuth flow against your IdP when prompted.

---

## Identity mapping

The JWT `sub` claim becomes the UMA `user_id`. The tenant comes from
`--oauth-tenant`, which is a **server-level setting**: one OAuth deployment
serves one UMA tenant. Multi-tenant claim mapping is not implemented — run a
separate server per tenant if you need several.

Beyond that, scope handling matches bearer mode: the token is authoritative, and
a tool call passing a conflicting `user_id` or `tenant_id` is rejected rather
than silently overridden. See
[`DEPLOY.md`](DEPLOY.md#scope-and-isolation).

---

## Troubleshooting

**`--oauth-issuer requires --oauth-audience`.**
Supply the audience. See above for why it is mandatory.

**Every request returns 401.**
Decode the token your client is sending and compare `iss` and `aud` against your
flags, character for character. A trailing-slash difference on the issuer is the
single most common cause. Confirm the algorithm is RS256 or ES256.

**401 with `missing_scopes` in the server log.**
The JWT lacks a scope you required with `--oauth-required-scope`. Either grant
it in the IdP or drop the requirement.

**JWKS fetch failures.**
The default JWKS path is `{issuer}/.well-known/jwks.json`. Providers that
publish elsewhere — Entra ID and Google among them — need an explicit
`--oauth-jwks-uri`.

**ChatGPT will not add the connector.**
Confirm the endpoint is HTTPS, publicly resolvable, and that `--public-url`
matches the address ChatGPT is given. A mismatch makes the advertised resource
metadata point somewhere the client cannot reach.
