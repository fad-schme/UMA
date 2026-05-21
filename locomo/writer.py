from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TextIO


class JsonlWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._handle: TextIO | None = None

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlWriter is not open.")
        self._handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        self._handle.write("\n")
        self._handle.flush()


def write_jsonl(path: str, records: Iterable[dict[str, Any]]) -> None:
    with JsonlWriter(path) as writer:
        for record in records:
            writer.write(record)
