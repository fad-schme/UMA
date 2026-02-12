from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_WARNED: set[str] = set()


def warn_once(key: str, message: str, *, exc: Optional[BaseException] = None) -> None:
    """
    Log a warning only once per key.
    """
    if key in _WARNED:
        return
    _WARNED.add(key)
    if exc:
        logger.warning("%s (%s)", message, exc)
    else:
        logger.warning("%s", message)


def optional_import(module_path: str, attr: Optional[str] = None) -> Any:
    """
    Import a module or attribute with a warn-once fallback.
    """
    try:
        module = __import__(module_path, fromlist=[attr] if attr else [])
        return getattr(module, attr) if attr else module
    except Exception as exc:
        key = f"optional_import:{module_path}:{attr or ''}"
        warn_once(key, f"Optional dependency '{module_path}' is unavailable.", exc=exc)
        return None
