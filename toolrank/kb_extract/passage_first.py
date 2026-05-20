from __future__ import annotations

from pathlib import Path
from typing import Any

from toolrank.kb_extract.relation_first import (
    CANONICAL_CATEGORIES,
    RELATION_FIRST_GATE_VERSION,
    RELATION_FIRST_PROMPT_VERSION,
    CritiqueEntry,
    RelationFirstLlmParseError,
    RelationFirstProjection,
    RelationFirstResponse,
    extract_relation_first_raw_response,
    project_relation_first_response,
)
from toolrank.openai_compat import OpenAICompatClient


PASSAGE_FIRST_PROMPT_VERSION = RELATION_FIRST_PROMPT_VERSION
PASSAGE_FIRST_GATE_VERSION = RELATION_FIRST_GATE_VERSION
PassageFirstLlmParseError = RelationFirstLlmParseError
PassageFirstProjection = RelationFirstProjection
PassageFirstResponse = RelationFirstResponse


def extract_passage_first_raw_response(
    *,
    client: OpenAICompatClient,
    model: str,
    mineru_output_dir: Path,
    doc_id: str,
    allowed_tool_ids: set[str],
    diagnostics_dir: Path | None = None,
) -> dict[str, Any]:
    return extract_relation_first_raw_response(
        client=client,
        model=model,
        mineru_output_dir=mineru_output_dir,
        doc_id=doc_id,
        allowed_tool_ids=allowed_tool_ids,
        diagnostics_dir=diagnostics_dir,
    )


def project_passage_first_response(
    raw_response: dict[str, Any],
    *,
    doc_id: str,
    allowed_tool_ids: set[str],
) -> RelationFirstProjection:
    return project_relation_first_response(
        raw_response,
        doc_id=doc_id,
        allowed_tool_ids=allowed_tool_ids,
    )


__all__ = [
    "CANONICAL_CATEGORIES",
    "CritiqueEntry",
    "PASSAGE_FIRST_GATE_VERSION",
    "PASSAGE_FIRST_PROMPT_VERSION",
    "PassageFirstLlmParseError",
    "PassageFirstProjection",
    "PassageFirstResponse",
    "extract_passage_first_raw_response",
    "project_passage_first_response",
]
