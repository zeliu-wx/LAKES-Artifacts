from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from toolrank.kb_extract.models import AssertionStatus, LedgerEntry


class AssertionLedger:
    """Append-only assertion store, backed by JSON file."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: list[LedgerEntry] = []

    def load(self) -> None:
        if not self.path.exists():
            self.entries = []
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries = [LedgerEntry.model_validate(item) for item in payload]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([item.model_dump() for item in self.entries], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def append(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)

    def query(self, tool_id: str, predicate: Optional[str] = None) -> list[LedgerEntry]:
        return [
            item
            for item in self.entries
            if item.tool_id == tool_id and (predicate is None or item.predicate == predicate)
        ]

    def accepted_for_tool(self, tool_id: str) -> list[LedgerEntry]:
        return [item for item in self.query(tool_id) if item.status == AssertionStatus.accepted]

    def conflicts_for_tool(self, tool_id: str) -> list[LedgerEntry]:
        return [item for item in self.query(tool_id) if item.status == AssertionStatus.conflict]
