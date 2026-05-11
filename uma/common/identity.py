"""
uma.common.identity
=======================

Identity and normalization helpers.

In UMA-RLM v1 we standardize:
    subject = "user:<id>"

This is used across:
- Semantic facts (Fact.subject)
- Graph user node keys
- Any subject-scoped memory representation

Rationale
---------
Using a typed prefix avoids collisions and enables safe multi-tenant joins
(e.g., "user:123" vs "org:123") in future expansions.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


USER_PREFIX = "user:"


def normalize_user_id(user_id_or_subject: str) -> str:
    """
    Normalize a user identifier into the canonical UMA subject format.

    Parameters
    ----------
    user_id_or_subject : str
        Either a raw user_id (e.g., "123") or a pre-prefixed subject (e.g., "user:123").

    Returns
    -------
    str
        Canonical subject string: "user:<id>"

    Raises
    ------
    ValueError
        If the input is empty or not a string.
    """
    if not isinstance(user_id_or_subject, str) or not user_id_or_subject.strip():
        raise ValueError("normalize_user_id: input must be a non-empty string.")

    s = user_id_or_subject.strip()
    if s.startswith(USER_PREFIX):
        return s

    normalized = f"{USER_PREFIX}{s}"
    logger.debug("Normalized subject %r -> %r", s, normalized)
    return normalized
