from __future__ import annotations

import json
from typing import Any

from toolrank.solc_range import VERSION_UNKNOWN_RANGE, normalize_solc_range

from toolrank.kb_extract.models import AssertionStatus, LedgerEntry
from toolrank.kb_extract.normalizer import extract_normalized_value
from toolrank.schemas import DASP10_CATEGORIES

TRI_RANK = {"no": 0, "partial": 1, "yes": 2}
CUSTOM_RANK = {"none": 0, "limited": 1, "full": 2}
_CANONICAL_SCENARIOS = {
    "majority voting": "majority_voting",
    "inclusive top-recall tool combination": "inclusive_top_recall_combination",
    "inclusive top-recall combination": "inclusive_top_recall_combination",
    "inclusive_top_recall_selection": "inclusive_top_recall_combination",
}
_PERCENT_METRICS = {"precision", "recall", "f1_score", "found_vulnerabilities_ratio"}


def _normalize_combination_hint(entry: dict[str, Any]) -> dict[str, Any] | None:
    tools = [str(item).lower() for item in entry.get("tools", []) if isinstance(item, str) and item.strip()]
    if len(tools) < 2:
        return None
    scenario = str(entry.get("scenario") or "").strip()
    scenario = _CANONICAL_SCENARIOS.get(scenario, scenario.replace(" ", "_"))
    metric = str(entry.get("metric") or "").strip()
    try:
        value = float(entry.get("value", 0.0))
    except (TypeError, ValueError):
        return None
    if metric in _PERCENT_METRICS and value > 1.0:
        value = value / 100.0
    avg_runtime = entry.get("avg_runtime_sec")
    return {
        "tools": tools,
        "scenario": scenario,
        "metric": metric,
        "value": value,
        "avg_runtime_sec": avg_runtime,
        "evidence_excerpt": entry.get("evidence_excerpt", ""),
    }


def _combination_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    normalized = _normalize_combination_hint(entry)
    if normalized is None:
        return (), "", "", 0.0, None
    tools = tuple(sorted(normalized["tools"]))
    scenario = normalized["scenario"]
    metric = normalized["metric"]
    value = round(float(normalized["value"]), 6)
    avg_runtime = normalized.get("avg_runtime_sec")
    avg_runtime_norm = None if avg_runtime is None else round(float(avg_runtime), 6)
    return tools, scenario, metric, value, avg_runtime_norm


def _dedupe_combination_hints(existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in existing:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_combination_hint(item)
        if normalized is None:
            continue
        key = _combination_key(normalized)
        current = deduped.get(key)
        if current is None or len(normalized.get("evidence_excerpt", "")) > len(current.get("evidence_excerpt", "")):
            deduped[key] = normalized
    return list(deduped.values())


def default_toolcard(name: str, tool_id: str) -> dict[str, Any]:
    return {
        "toolcard_schema_version": "1.1.0",
        "tool_id": tool_id,
        "tool_name": name,
        "aliases": [],
        "d7_input_support": {"sol": False, "bytecode": False, "runtime": False},
        "d8_mode": "static",
        "d2_solidity_versions": VERSION_UNKNOWN_RANGE,
        "d3_multifile_support": "partial",
        "d3_external_calls_support": "partial",
        "d3_multicontract_support": "partial",
        "d4_custom_rules": "limited",
        "d6_outputs": {
            "text": True,
            "json": False,
            "sarif": False,
            "pdf": False,
            "code_locate": False,
            "explanation": False,
        },
        "d9_activity": {"stars": 0, "last_update_days": 3650},
        "evidence": [],
        "d5_strength_labels": {},
        "category_ranking_knowledge": [],
        "combination_hints": [],
    }


def _union_solc(old: str, new_value: dict[str, Any]) -> tuple[str, bool]:
    normalized = normalize_solc_range(new_value.get("normalized"))
    old_norm = normalize_solc_range(old)
    if old_norm == VERSION_UNKNOWN_RANGE:
        return normalized, normalized != old_norm
    if normalized == VERSION_UNKNOWN_RANGE:
        return old_norm, False
    start_old, end_old = old_norm.split("-", 1)
    start_new, end_new = normalized.split("-", 1)
    merged = f"{min(start_old, start_new)}-{max(end_old, end_new)}"
    return merged, merged != old_norm


def materialize_toolcard(
    base_card: dict,
    accepted_entries: list[LedgerEntry],
    conflict_entries: list[LedgerEntry],
) -> tuple[dict, list[str]]:
    card = json.loads(json.dumps(base_card))
    changes: list[str] = []

    existing_combination_hints = card.get("combination_hints")
    if isinstance(existing_combination_hints, list):
        deduped_existing = _dedupe_combination_hints(existing_combination_hints)
        if deduped_existing != existing_combination_hints:
            card["combination_hints"] = deduped_existing
            changes.append("combination_hints")

    existing_ranking = card.get("category_ranking_knowledge")
    if isinstance(existing_ranking, list):
        filtered_ranking = [
            item for item in existing_ranking
            if isinstance(item, dict) and item.get("category") in DASP10_CATEGORIES
        ]
        if filtered_ranking != existing_ranking:
            card["category_ranking_knowledge"] = filtered_ranking
            changes.append("category_ranking_knowledge")

    for entry in accepted_entries:
        data = entry.candidate_data
        predicate = entry.predicate or data.get("predicate")
        if not predicate:
            continue
        if predicate == "d2.solidity_versions":
            merged, changed = _union_solc(card.get("d2_solidity_versions", VERSION_UNKNOWN_RANGE), data.get("value", {}))
            if changed:
                card["d2_solidity_versions"] = merged
                changes.append("d2_solidity_versions")
        elif predicate.startswith("d3."):
            mapping = {
                "d3.multifile_support": "d3_multifile_support",
                "d3.external_calls_support": "d3_external_calls_support",
                "d3.multicontract_support": "d3_multicontract_support",
            }
            field = mapping[predicate]
            new = data.get("value", {}).get("normalized")
            old = card.get(field)
            if new in TRI_RANK and old in TRI_RANK and TRI_RANK[new] > TRI_RANK[old]:
                card[field] = new
                changes.append(field)
        elif predicate == "d4.custom_rules":
            new = data.get("value", {}).get("normalized")
            old = card.get("d4_custom_rules")
            if new in CUSTOM_RANK and old in CUSTOM_RANK and CUSTOM_RANK[new] > CUSTOM_RANK[old]:
                card["d4_custom_rules"] = new
                changes.append("d4_custom_rules")
        elif predicate.startswith("d6.outputs."):
            key = predicate.split(".")[-1]
            new = data.get("value", {}).get("normalized")
            if new is True and card["d6_outputs"].get(key) is not True:
                card["d6_outputs"][key] = True
                changes.append(f"d6_outputs.{key}")
        elif predicate.startswith("d7.input."):
            key = predicate.split(".")[-1]
            new = data.get("value", {}).get("normalized")
            if new is True and card["d7_input_support"].get(key) is not True:
                card["d7_input_support"][key] = True
                changes.append(f"d7_input_support.{key}")
        elif predicate == "d8.mode":
            new = data.get("value", {}).get("normalized")
            old = card.get("d8_mode")
            if old in {None, "static"} and new and new != old:
                card["d8_mode"] = new
                changes.append("d8_mode")
        elif predicate == "d9.activity.stars":
            new = data.get("value", {}).get("normalized")
            old = card["d9_activity"].get("stars")
            if isinstance(new, int) and (old is None or new > old):
                card["d9_activity"]["stars"] = new
                changes.append("d9_activity.stars")
        elif predicate == "d9.activity.last_update_days":
            new = data.get("value", {}).get("normalized")
            old = card["d9_activity"].get("last_update_days")
            if isinstance(new, int) and (old is None or new < old):
                card["d9_activity"]["last_update_days"] = new
                changes.append("d9_activity.last_update_days")
        elif predicate.startswith("d5.strength."):
            category = predicate.split(".", 2)[2]
            normalized = extract_normalized_value("strength_label", data.get("value", {}))
            if normalized in {"weak", "medium", "strong"}:
                if card.setdefault("d5_strength_labels", {}).get(category) != normalized:
                    card["d5_strength_labels"][category] = normalized
                    changes.append(f"d5_strength_labels.{category}")
        elif predicate.startswith("category_ranking.") and predicate.endswith(".stronger_than"):
            category = predicate.split(".", 2)[1]
            if category not in DASP10_CATEGORIES:
                continue
            normalized = data.get("value", {}).get("normalized")
            if isinstance(normalized, list):
                existing = card.setdefault("category_ranking_knowledge", [])
                incoming = {
                    "category": category,
                    "stronger_than": [item for item in normalized if isinstance(item, str) and item.strip()],
                }
                if incoming["stronger_than"]:
                    replaced = False
                    for idx, item in enumerate(existing):
                        if item.get("category") == category:
                            if item != incoming:
                                existing[idx] = incoming
                                changes.append(f"category_ranking_knowledge.{category}")
                            replaced = True
                            break
                    if not replaced:
                        existing.append(incoming)
                        changes.append(f"category_ranking_knowledge.{category}")
        elif predicate == "combination.hint":
            normalized = data.get("value", {}).get("normalized")
            if isinstance(normalized, dict):
                incoming = _normalize_combination_hint(normalized)
                if incoming is None:
                    continue
                existing = card.setdefault("combination_hints", [])
                existing_index = {
                    _combination_key(item): idx
                    for idx, item in enumerate(existing)
                    if isinstance(item, dict)
                }
                key = _combination_key(incoming)
                if key in existing_index:
                    idx = existing_index[key]
                    if len(incoming["evidence_excerpt"]) > len(existing[idx].get("evidence_excerpt", "")):
                        existing[idx] = incoming
                        changes.append("combination_hints")
                else:
                    existing.append(incoming)
                    changes.append("combination_hints")

    card.pop("d1_metrics", None)
    card.pop("d5_category_capability", None)
    card.pop("d5_dasp10_coverage", None)
    card.pop("qualitative_evidence", None)
    card.pop("relation_evidence", None)
    card.pop("composition_gain_knowledge", None)
    return card, sorted(set(changes))


def check_blast_radius(
    materialized_change_count: int,
    paper_entries: list[LedgerEntry],
    max_materialized_changes: int = 50,
    max_new_entities: int = 15,
    max_unresolved_conflicts: int = 10,
) -> bool:
    new_entities = {item.tool_id for item in paper_entries if item.tool_id and item.status == AssertionStatus.manual_review}
    conflicts = [item for item in paper_entries if item.status == AssertionStatus.conflict]
    return (
        materialized_change_count > max_materialized_changes
        or len(new_entities) > max_new_entities
        or len(conflicts) > max_unresolved_conflicts
    )
