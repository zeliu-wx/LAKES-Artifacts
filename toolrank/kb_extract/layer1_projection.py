from __future__ import annotations

from collections import defaultdict
from typing import Any

from toolrank.kb_extract.models import PaperDossier
from toolrank.kb_extract.normalizer import extract_normalized_value, validate_predicate_value
from toolrank.schemas import DASP10_CATEGORIES


MAIN_TOOL_ROLES = {"study_subject", "proposed_tool", "baseline"}
STRENGTH_MAP = {
    "strong": "strong",
    "high": "strong",
    "best_in_paper": "strong",
    "medium": "medium",
    "moderate": "medium",
    "weak": "weak",
    "low": "weak",
}
_CANONICAL_SCENARIOS = {
    "majority voting": "majority_voting",
    "inclusive top-recall tool combination": "inclusive_top_recall_combination",
    "inclusive top-recall combination": "inclusive_top_recall_combination",
    "inclusive_top_recall_selection": "inclusive_top_recall_combination",
}
_PERCENT_METRICS = {"precision", "recall", "f1_score", "found_vulnerabilities_ratio"}


def _normalize_combination_hint(item: Any) -> dict[str, Any] | None:
    tools = [tool.lower() for tool in item.tools if isinstance(tool, str) and tool.strip()]
    if len(tools) < 2:
        return None
    scenario = item.scenario.strip()
    scenario = _CANONICAL_SCENARIOS.get(scenario, scenario.replace(" ", "_"))
    metric = item.metric.strip()
    value = float(item.value)
    if metric in _PERCENT_METRICS and value > 1.0:
        value = value / 100.0
    return {
        "tools": tools,
        "scenario": scenario,
        "metric": metric,
        "value": value,
        "avg_runtime_sec": item.avg_runtime_sec,
        "evidence_excerpt": item.evidence_excerpt,
    }


def _combination_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(sorted(entry["tools"])),
        entry["scenario"],
        entry["metric"],
        round(float(entry["value"]), 6),
        None if entry["avg_runtime_sec"] is None else round(float(entry["avg_runtime_sec"]), 6),
    )


def project_dossier_to_tool_updates(
    dossier: PaperDossier,
    schema_registry: dict[str, Any],
    allowed_tool_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    predicates = schema_registry.get("predicates", {})
    role_by_tool = {
        (tool.canonical_id or tool.surface_name.lower()): tool.role
        for tool in dossier.tools
    }

    def _include_tool(tool_id: str) -> bool:
        if role_by_tool.get(tool_id) not in MAIN_TOOL_ROLES:
            return False
        if allowed_tool_ids is None:
            return True
        return tool_id in allowed_tool_ids

    projected: dict[str, dict[str, Any]] = {}

    def _ensure(tool_id: str) -> dict[str, Any]:
        return projected.setdefault(
            tool_id,
            {
                "tool_id": tool_id,
                "role": role_by_tool.get(tool_id, "unknown"),
                "capability_hints": {},
                "d5_strength_labels": {},
                "category_ranking_knowledge": [],
                "combination_hints": [],
            },
        )

    for item in dossier.working_memory.capability_hints:
        tool_id = item.tool.lower()
        if not _include_tool(tool_id):
            continue
        spec = predicates.get(item.predicate)
        if spec is None:
            continue
        normalized = extract_normalized_value(spec["value_type"], item.value)
        if not validate_predicate_value(spec["value_type"], normalized):
            continue
        if spec["value_type"] == "solc_range":
            _ensure(tool_id)["capability_hints"][item.predicate] = normalized
        else:
            _ensure(tool_id)["capability_hints"][item.predicate] = {"normalized": normalized}

    for item in dossier.working_memory.global_strength_hints:
        tool_id = item.tool.lower()
        if not _include_tool(tool_id):
            continue
        normalized = STRENGTH_MAP.get(item.strength)
        if normalized is None:
            continue
        current = _ensure(tool_id)["d5_strength_labels"].get(item.category)
        rank = {"weak": 0, "medium": 1, "strong": 2}
        if current is None or rank[normalized] > rank[current]:
            _ensure(tool_id)["d5_strength_labels"][item.category] = normalized

    ranking_map: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for item in dossier.working_memory.global_ranking_hints:
        if item.category not in DASP10_CATEGORIES:
            continue
        tool_id = item.stronger_tool.lower()
        if not _include_tool(tool_id):
            continue
        for other in item.stronger_than:
            other_id = other.lower()
            if not _include_tool(other_id):
                continue
            if other_id == tool_id:
                continue
            ranking_map[tool_id][item.category].add(other_id)
    for tool_id, categories in ranking_map.items():
        bucket = _ensure(tool_id)["category_ranking_knowledge"]
        for category, stronger_than in sorted(categories.items()):
            if stronger_than:
                bucket.append({"category": category, "stronger_than": sorted(stronger_than)})

    for item in dossier.working_memory.combination_hints:
        payload = _normalize_combination_hint(item)
        if payload is None:
            continue
        tools = [tool for tool in payload["tools"] if _include_tool(tool)]
        if len(tools) < 2:
            continue
        payload = {**payload, "tools": tools}
        key = _combination_key(payload)
        for tool_id in tools:
            bucket = _ensure(tool_id)["combination_hints"]
            seen = {
                _combination_key(entry): idx
                for idx, entry in enumerate(bucket)
            }
            if key in seen:
                idx = seen[key]
                if len(payload["evidence_excerpt"]) > len(bucket[idx]["evidence_excerpt"]):
                    bucket[idx] = payload
            else:
                bucket.append(payload)

    return projected
