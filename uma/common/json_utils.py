"""
Shared JSON parsing helpers.

Centralizes best-effort JSON salvage to avoid drift across modules.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)
logger = logging.getLogger(__name__)


def try_parse_json_object(raw: str) -> Optional[dict[str, Any]]:
    """
    Best-effort JSON object parsing.

    Attempts strict parsing first, then salvages the first JSON object substring.
    Returns None when parsing fails or result is not an object.
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception as exc:
        logger.debug(
            "try_parse_json_object: strict parse failed; attempting salvage: %s",
            exc,
            exc_info=True,
        )

    m = _JSON_OBJECT_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception as exc:
        logger.debug("try_parse_json_object: salvage failed: %s", exc, exc_info=True)
        return None


def try_parse_json_list(raw: str) -> Optional[list[Any]]:
    """
    Best-effort JSON list parsing.

    Attempts strict parsing first, then salvages the first JSON list substring.
    Returns None when parsing fails or result is not a list.
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else None
    except Exception as exc:
        logger.debug(
            "try_parse_json_list: strict parse failed; attempting salvage: %s",
            exc,
            exc_info=True,
        )

    m = _JSON_LIST_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, list) else None
    except Exception as exc:
        logger.debug("try_parse_json_list: salvage failed: %s", exc, exc_info=True)
        return None
