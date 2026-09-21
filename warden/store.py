"""M1: in-memory versioned store.

Every write to a key is appended to that key's history; nothing is ever
overwritten in place. `version` is the 1-based position of a write within
its key's history.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class WriteRecord:
    value: Any
    version: int
    ts: float


class Store:
    def __init__(self) -> None:
        self._history: dict[str, list[WriteRecord]] = {}

    def write(self, key: str, value: Any) -> int:
        """Append a new version for key. Returns the new version number (starts at 1)."""
        records = self._history.setdefault(key, [])
        version = len(records) + 1
        records.append(WriteRecord(value=value, version=version, ts=time.time()))
        return version

    def read(self, key: str) -> Optional[WriteRecord]:
        """Latest WriteRecord for key, or None if the key was never written."""
        records = self._history.get(key)
        return records[-1] if records else None

    def history(self, key: str) -> list[WriteRecord]:
        """Full write history for key, oldest first. Empty list for an unknown key."""
        return list(self._history.get(key, []))

    def keys(self) -> list[str]:
        return list(self._history.keys())
