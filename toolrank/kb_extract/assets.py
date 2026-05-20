from __future__ import annotations

import base64
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FullAssetTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caption: str = ""
    html: str = ""
    img_path: str = ""


class FullAssetImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caption: str = ""
    context: str = ""
    img_path: str = ""


class FullAssetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    markdown_path: str
    full_markdown: str
    tables: list[FullAssetTable] = Field(default_factory=list)
    images: list[FullAssetImage] = Field(default_factory=list)


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"}:
            text = " ".join("".join(self._current_cell).split())
            self._current_row.append(text)
            self._current_cell = []
            self._in_cell = False
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)


def find_markdown_file(mineru_output_dir: Path) -> Path | None:
    matches = sorted(mineru_output_dir.glob("**/*.md"))
    return matches[0] if matches else None


def load_content_items(mineru_output_dir: Path) -> list[dict[str, Any]]:
    matches = sorted(mineru_output_dir.glob("**/*_content_list.json"))
    if not matches:
        return []
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("content_list", "items", "blocks"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def markdown_context(markdown_text: str, anchor: str, *, before: int = 500, after: int = 1000) -> str:
    if not anchor:
        return ""
    match = re.search(re.escape(anchor), markdown_text)
    if not match:
        fallback = anchor.split(":", 1)[0].strip()
        if fallback and fallback != anchor:
            match = re.search(re.escape(fallback), markdown_text)
    if not match:
        return ""
    start = max(0, match.start() - before)
    end = min(len(markdown_text), match.start() + after)
    return markdown_text[start:end]


def load_full_asset_input(mineru_output_dir: Path) -> FullAssetInput | None:
    markdown_path = find_markdown_file(mineru_output_dir)
    if markdown_path is None:
        return None
    full_markdown = markdown_path.read_text(encoding="utf-8", errors="ignore")
    entries = load_content_items(mineru_output_dir)
    if not entries:
        return None

    tables: list[FullAssetTable] = []
    images: list[FullAssetImage] = []
    for item in entries:
        entry_type = item.get("type")
        if entry_type == "table":
            tables.append(
                FullAssetTable(
                    caption=" ".join(item.get("table_caption") or []),
                    html=item.get("table_body") or "",
                    img_path=item.get("img_path") or "",
                )
            )
        elif entry_type == "image":
            caption = " ".join(item.get("image_caption") or [])
            images.append(
                FullAssetImage(
                    caption=caption,
                    context=markdown_context(full_markdown, caption),
                    img_path=item.get("img_path") or "",
                )
            )

    return FullAssetInput(
        markdown_path=str(markdown_path),
        full_markdown=full_markdown,
        tables=tables,
        images=images,
    )


def data_url(image_path: Path) -> str:
    mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_html_table_rows(html: str) -> list[list[str]]:
    parser = _HTMLTableParser()
    parser.feed(html)
    return parser.rows


__all__ = [
    "FullAssetImage",
    "FullAssetInput",
    "FullAssetTable",
    "data_url",
    "find_markdown_file",
    "load_content_items",
    "load_full_asset_input",
    "markdown_context",
    "parse_html_table_rows",
]
