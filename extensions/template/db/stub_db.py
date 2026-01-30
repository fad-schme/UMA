"""
Template SQL adapter.

Implement a DBAdapter-compatible class.
"""

from __future__ import annotations

from typing import Any

from uma.adapters.db.base import DBAdapter, DBConnection


class ExampleDBAdapter(DBAdapter):
    def __init__(self, db_path: str, **kwargs: Any) -> None:
        raise NotImplementedError("Implement DBAdapter for your backend")

    def connect(self) -> DBConnection:
        raise NotImplementedError
