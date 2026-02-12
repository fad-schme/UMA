"""
Shared JSON parsing helpers.

Centralizes best-effort JSON salvage to avoid drift across modules.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)


def try_parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
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
    except Exception:
        # Best-effort parse failure: log and raise for callers to handle explicitly.
        import logging
        logging.getLogger(__name__).exception("try_parse_json_object: strict json.loads failed")
        raise

    m = _JSON_OBJECT_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def try_parse_json_list(raw: str) -> Optional[List[Any]]:
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
    except Exception:
        import logging
        logging.getLogger(__name__).exception("try_parse_json_list: strict json.loads failed")
        raise

    m = _JSON_LIST_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, list) else None
    except Exception:
        return None
