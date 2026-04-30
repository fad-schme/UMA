"""
Ownership type definitions (UMA v1).

NOTE:
- Keep this module minimal to avoid circular imports.
- OwnerType defines the canonical durable ownership vocabulary.
"""

from __future__ import annotations

from typing import Literal

OwnerType = Literal["agent", "user", "workspace", "system"]
