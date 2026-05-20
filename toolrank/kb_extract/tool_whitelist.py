from __future__ import annotations

import json
from pathlib import Path


def load_toolcard_whitelist(path: str | Path) -> set[str]:
    resolved_path = Path(path)
    if not resolved_path.exists() and not resolved_path.is_absolute():
        resolved_path = Path(__file__).resolve().parent / "config" / resolved_path.name
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    return {str(item) for item in payload.get("tool_ids", []) if str(item).strip()}
