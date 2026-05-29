"""
Secrets provider interface and reference implementation.

Location: this module lives at `uma/adapters/secrets.py` because it is
the seam between Lite (where the interface and the env-var reference
implementation live) and Enterprise (where Vault, CyberArk, AWS, Azure,
and GCP implementations live). It sits next to the storage adapters
because storage adapters are its primary consumer — UMA core never calls
into this module directly.

Lite itself does not call into this module — the embedded SQLite adapter
has no credentials to fetch. The interface lives here so that:

  1. Community storage adapters (Postgres, MySQL, etc.) written against
     Lite have a stable interface to depend on without taking a hard
     dependency on uma-enterprise.

  2. Enterprise providers (Vault, CyberArk, AWS Secrets Manager, Azure
     Key Vault, GCP Secret Manager) are interchangeable behind one
     contract.

Design rules (kept short on purpose):

  - The provider returns a `Secret` value object. TTL is optional and
    advisory: the env-var reference impl does not set one by default
    because env vars do not rotate without a process restart. Real
    secrets backends (Vault leases, AWS rotation schedules) populate
    `expires_at` from the backing store, and consumer caches honor it.
  - The provider never logs the secret value. Callers must not either.
  - The provider raises `SecretNotFound` for missing references and
    `SecretsProviderError` for transport / auth failures. Both are
    distinguishable so adapters can decide whether to retry.
  - The reference implementation reads from environment variables. It
    is suitable for development and for single-tenant deployments where
    the operator is comfortable managing rotation through process
    restarts.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SecretsProviderError(Exception):
    """
    Raised when a secrets provider cannot complete a request for reasons
    other than the secret being absent: transport failures, auth failures,
    misconfiguration, etc. Adapters typically treat this as retriable.
    """


class SecretNotFound(SecretsProviderError):
    """
    Raised when a requested secret reference does not exist in the backing
    store. Adapters typically treat this as a hard configuration error —
    retrying will not help.
    """


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Secret:
    """
    A resolved secret value plus the metadata a caller needs to cache and
    rotate it correctly.

    Attributes
    ----------
    value:
        The secret string. Never log this. The repr is overridden to make
        accidental logging visible without exposing the value.
    expires_at:
        UTC timestamp after which the value should be considered stale and
        re-resolved. `None` means "the provider gives no expiry guidance";
        the caller should pick a sensible default (typically a short TTL).
    version:
        Optional version identifier returned by the provider when the
        backing store supports versioning (Vault leases, AWS Secrets Manager
        version stages, etc.). Useful for audit correlation.
    """

    value: str
    expires_at: Optional[datetime] = None
    version: Optional[str] = field(default=None)

    def __repr__(self) -> str:
        # Hide the value in repr so it never leaks through debug logging
        # or test framework diff output. The length is included to make
        # "secret missing" vs "secret empty" distinguishable in diagnostics.
        return (
            f"Secret(value=<redacted len={len(self.value)}>, "
            f"expires_at={self.expires_at!r}, version={self.version!r})"
        )

    def is_expired(self, *, now: Optional[datetime] = None) -> bool:
        """
        Return True if the secret's advertised TTL has passed. A secret
        with no `expires_at` is treated as never expiring from the value
        object's perspective — the caller's cache decides its own ceiling.
        """
        if self.expires_at is None:
            return False
        current = now if now is not None else datetime.now(timezone.utc)
        return current >= self.expires_at


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class SecretsProvider(ABC):
    """
    Abstract interface for resolving secrets by reference.

    A "reference" is an opaque string the consumer agreed on with the
    operator who configured the provider — for example "postgres/main" or
    "qdrant/api_key". The provider maps the reference to a backing-store
    path according to its own rules; UMA core does not interpret the
    reference shape.

    Implementations must be safe to share across threads / async tasks.
    They should be cheap to construct — expensive work (auth handshake,
    client construction) belongs in `resolve`, behind whatever caching the
    implementation needs.
    """

    @abstractmethod
    def resolve(self, reference: str) -> Secret:
        """
        Resolve a secret reference to a `Secret` value object.

        Parameters
        ----------
        reference:
            Opaque identifier for the secret. Non-empty string.

        Returns
        -------
        Secret
            The resolved secret. Never `None`; on absence raise
            `SecretNotFound`.

        Raises
        ------
        SecretNotFound
            If the reference does not exist in the backing store.
        SecretsProviderError
            For transport, auth, or misconfiguration failures.
        ValueError
            If `reference` is empty or not a string.
        """


# ---------------------------------------------------------------------------
# Reference implementation: environment variables
# ---------------------------------------------------------------------------

# References are mapped to env var names by uppercasing and replacing any
# non-alphanumeric character with an underscore. So "postgres/main" becomes
# "POSTGRES_MAIN", "qdrant.api-key" becomes "QDRANT_API_KEY". This keeps the
# mapping predictable without forcing operators to learn a new convention.
_REFERENCE_TO_ENVVAR = re.compile(r"[^A-Za-z0-9]+")


class EnvVarProvider(SecretsProvider):
    """
    Reads secrets from environment variables.

    Suitable for development, CI, and single-tenant deployments where the
    operator manages credentials through the process environment. Not
    suitable for multi-tenant production — there is no per-tenant scoping
    and no rotation story beyond restarting the process.

    Parameters
    ----------
    prefix:
        Optional uppercase prefix prepended to every resolved env var name.
        For example, prefix="UMA" turns reference "postgres/main" into
        env var "UMA_POSTGRES_MAIN". Useful for namespacing in shared
        environments. Defaults to no prefix.
    default_ttl:
        Optional TTL to advertise on resolved secrets. Defaults to `None`,
        which means "no expiry guidance" — the consumer's cache decides
        its own ceiling. This is the right default for env vars, which
        do not rotate without a process restart: advertising a TTL would
        force unnecessary re-resolution mid-session without any security
        benefit. Operators who do remount secrets (e.g. Kubernetes secret
        rotation followed by a SIGHUP-driven reload) can pass an explicit
        TTL to opt in to periodic re-reading.
    """

    def __init__(
        self,
        *,
        prefix: Optional[str] = None,
        default_ttl: Optional[timedelta] = None,
    ) -> None:
        if prefix is not None and not prefix:
            # Explicit empty string is almost certainly a configuration
            # bug — the caller meant to pass None or a real prefix.
            raise ValueError("prefix must be None or a non-empty string")
        if default_ttl is not None and default_ttl <= timedelta(0):
            raise ValueError("default_ttl must be positive when provided")

        self._prefix = prefix.upper() if prefix else None
        self._default_ttl = default_ttl

        logger.debug(
            "EnvVarProvider initialized prefix=%s default_ttl_seconds=%s",
            self._prefix,
            int(self._default_ttl.total_seconds()) if self._default_ttl else None,
        )

    def resolve(self, reference: str) -> Secret:
        if not isinstance(reference, str):
            raise ValueError("reference must be a string")
        if not reference.strip():
            raise ValueError("reference must be a non-empty string")

        env_name = self._to_env_name(reference)

        # os.environ.get is intentional over os.getenv so we can distinguish
        # "unset" from "set to empty string" — empty is still a hard error
        # rather than a silent empty credential.
        raw = os.environ.get(env_name)
        if raw is None:
            logger.warning(
                "Secret not found reference=%s env_name=%s",
                reference,
                env_name,
            )
            raise SecretNotFound(
                f"No environment variable {env_name!r} for reference {reference!r}"
            )
        if raw == "":
            logger.warning(
                "Secret env var is empty reference=%s env_name=%s",
                reference,
                env_name,
            )
            raise SecretsProviderError(
                f"Environment variable {env_name!r} is set but empty "
                f"for reference {reference!r}"
            )

        if self._default_ttl is not None:
            expires_at: Optional[datetime] = (
                datetime.now(timezone.utc) + self._default_ttl
            )
        else:
            expires_at = None

        logger.debug(
            "Secret resolved reference=%s env_name=%s expires_at=%s",
            reference,
            env_name,
            expires_at.isoformat() if expires_at else None,
        )

        return Secret(value=raw, expires_at=expires_at, version=None)

    # -- helpers ------------------------------------------------------------

    def _to_env_name(self, reference: str) -> str:
        """
        Map a reference like "postgres/main" to "POSTGRES_MAIN", applying
        the optional prefix. Collapses runs of separators and strips
        leading/trailing underscores so references with surrounding
        punctuation produce clean names.
        """
        normalized = _REFERENCE_TO_ENVVAR.sub("_", reference).strip("_").upper()
        if not normalized:
            # Reference was all punctuation. Raise rather than silently
            # resolving "" to whatever happens to be in os.environ[""].
            raise ValueError(
                f"reference {reference!r} contains no alphanumeric characters"
            )
        if self._prefix:
            return f"{self._prefix}_{normalized}"
        return normalized


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "Secret",
    "SecretsProvider",
    "EnvVarProvider",
    "SecretsProviderError",
    "SecretNotFound",
]