from __future__ import annotations

import json
from typing import Any


DOSSIER_TIMEOUT_SEC = 120.0


def coerce_stream_content(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
        return "".join(parts)
    return ""


def parse_json_content(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


__all__ = [
    "DOSSIER_TIMEOUT_SEC",
    "coerce_stream_content",
    "parse_json_content",
]
