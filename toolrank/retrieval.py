from __future__ import annotations

import json
from pathlib import Path
from typing import List

from toolrank.schemas import ToolCard


def load_toolcards(toolcards_dir: str | Path) -> List[ToolCard]:
    base = Path(toolcards_dir)
    cards: List[ToolCard] = []
    for path in sorted(base.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if "tool_id" not in payload or "tool_name" not in payload:
            # Ignore non-toolcard JSON files (e.g., dataset KB snapshots).
            continue
        cards.append(ToolCard.model_validate(payload))
    return cards
