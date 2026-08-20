"""Shared Ollama endpoint resolution for the opt-in e2e quality gates."""

from __future__ import annotations

import os


def resolve_ollama_host() -> str:
    """Return an OLLAMA_HOST value usable as an HTTP client base.

    `OLLAMA_HOST` follows Ollama's own server convention and is commonly set
    without a scheme and bound to `0.0.0.0` (e.g. `0.0.0.0:11434`). Neither
    form is usable by the OpenAI-compatible client: a missing scheme raises
    `UnsupportedProtocol`, and `0.0.0.0` is a bind address, not a routable
    destination.
    """
    host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip().rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    return host.replace("://0.0.0.0", "://127.0.0.1")
