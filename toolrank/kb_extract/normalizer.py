from __future__ import annotations

from typing import Any, Optional

from toolrank.kb_extract.alias_registry import resolve_tool
from toolrank.kb_extract.models import CoverageSemantics, ObservationBasis
from toolrank.solc_range import VERSION_UNKNOWN_RANGE, normalize_solc_range

VALID_TRI = {"no", "partial", "yes"}
VALID_CUSTOM = {"none", "limited", "full"}
VALID_MODE = {"static", "symbolic", "fuzz", "ml", "llm"}
VALID_STRENGTH = {"weak", "medium", "strong"}


def canonicalize_tool_name(surface: str, alias_registry: dict[str, str]) -> Optional[str]:
    return resolve_tool(surface, alias_registry)


def normalize_solc_with_semantics(raw_text: str, observation_basis: ObservationBasis) -> Optional[dict[str, Any]]:
    normalized = normalize_solc_range(raw_text)
    if normalized == VERSION_UNKNOWN_RANGE:
        return None
    semantics = (
        CoverageSemantics.claimed_supported_range
        if observation_basis in {ObservationBasis.author_explicit_claim, ObservationBasis.table_explicit_entry}
        else CoverageSemantics.evaluated_on_range
    )
    start, end = normalized.split("-", 1)
    return {
        "raw_text": raw_text,
        "normalized": normalized,
        "intervals": [{"start": start, "end": end}],
        "coverage_semantics": semantics.value,
    }


def extract_normalized_value(value_type: str, value: dict[str, Any]) -> Any:
    if value_type == "solc_range":
        return value
    normalized = value.get("normalized")
    if normalized is not None:
        return normalized
    if value_type == "strength_label":
        for key in ("raw", "value", "label"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate in VALID_STRENGTH:
                return candidate
        return None
    if value_type == "bool":
        for v in value.values():
            if isinstance(v, bool):
                return v
        return None
    if value_type == "free_text":
        for v in value.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None
    if value_type == "float":
        for v in value.values():
            if isinstance(v, (int, float)):
                return v
        return None
    if value_type == "int":
        for v in value.values():
            if isinstance(v, int):
                return v
        return None
    return None


def validate_predicate_value(value_type: str, value: Any) -> bool:
    if value_type == "bool":
        return isinstance(value, bool)
    if value_type == "int":
        return isinstance(value, int) and value >= 0
    if value_type == "tri_state":
        return isinstance(value, str) and value in VALID_TRI
    if value_type == "custom_rule_level":
        return isinstance(value, str) and value in VALID_CUSTOM
    if value_type == "detection_mode":
        return isinstance(value, str) and value in VALID_MODE
    if value_type == "solc_range":
        if not isinstance(value, dict):
            return False
        normalized = value.get("normalized")
        return isinstance(normalized, str) and normalized != VERSION_UNKNOWN_RANGE
    if value_type == "free_text":
        return isinstance(value, str) and len(value.strip()) > 0
    if value_type == "float":
        return isinstance(value, (int, float)) and value >= 0
    if value_type == "strength_label":
        return isinstance(value, str) and value in VALID_STRENGTH
    if value_type == "tool_name_list":
        return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)
    if value_type in {"qualitative", "relation"}:
        return isinstance(value, dict)
    return False
