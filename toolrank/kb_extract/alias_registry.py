from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


def _slug(text: str) -> str:
    s = text.strip().lower().replace("+", "plus")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def resolve_tool(surface: str, registry: dict[str, str]) -> Optional[str]:
    key = _slug(surface)
    if not key:
        return None
    return registry.get(key, key if key in set(registry.values()) else None)
