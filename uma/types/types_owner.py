"""
Ownership type definitions (UMA v1).

NOTE:
- Keep this module minimal to avoid circular imports.
- OwnerType is intentionally simple: "agent" | "user".
"""

from __future__ import annotations

from typing import Literal

OwnerType = Literal["agent", "user"]
