"""Build recall coverage matrices from performance knowledge bases."""

from __future__ import annotations

import re

from toolrank.schemas import PerformanceKnowledgeBase, PerformanceEntry
from toolrank.schemas_v2 import RecallCoverageEntry, RecallCoverageMatrix


def _tool_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _entry_weight(entry: PerformanceEntry) -> float:
    count = entry.dataset_profile.contract_count_total
    if count is not None and count > 0:
        return float(count)
    return 1.0


def _support_level(r_hat: float | None) -> str:
    if r_hat is None or r_hat == 0:
        return "unsupported"
    if r_hat >= 0.6:
        return "strong"
    if r_hat >= 0.3:
        return "medium"
    return "weak"


def build_recall_coverage(
    kb: PerformanceKnowledgeBase,
    tool_ids: list[str],
    matched_source_ids: set[str] | None = None,
) -> RecallCoverageMatrix:
    requested_by_key: dict[str, str] = {}
    for tool_id in tool_ids:
        requested_by_key.setdefault(_tool_key(tool_id), tool_id)

    selected_source_ids = set(matched_source_ids) if matched_source_ids is not None else None
    weighted_scores: dict[tuple[str, str], tuple[float, float]] = {}
    count_scores: dict[tuple[str, str], tuple[int, int]] = {}
    observed_categories: set[str] = set()

    for entry in kb.entries:
        if selected_source_ids is not None and entry.source_id not in selected_source_ids:
            continue

        weight = _entry_weight(entry)
        for observation in entry.tool_performance_data:
            tool = requested_by_key.get(_tool_key(observation.tool_name))
            if tool is None:
                continue

            scores = observation.vulnerability_scores or {}
            counts = observation.vulnerability_score_counts or {}
            for category in set(scores) | set(counts):
                observed_categories.add(category)
                count = counts.get(category)
                if count is not None:
                    detected, total = count_scores.get((tool, category), (0, 0))
                    count_scores[(tool, category)] = (
                        detected + count.detected,
                        total + count.total,
                    )
                    continue
                score = scores.get(category)
                if score is None:
                    continue
                total_score, total_weight = weighted_scores.get((tool, category), (0.0, 0.0))
                weighted_scores[(tool, category)] = (
                    total_score + (float(score) * weight),
                    total_weight + weight,
                )

    matrix: list[RecallCoverageEntry] = []
    weak_categories_by_tool: dict[str, list[str]] = {}
    for tool in tool_ids:
        weak_categories: list[str] = []
        for category in sorted(observed_categories):
            score_weight = weighted_scores.get((tool, category))
            score_count = count_scores.get((tool, category))
            r_hat = None
            detected = None
            total = None
            if score_count is not None:
                detected, total = score_count
                r_hat = detected / total if total > 0 else None
            elif score_weight is not None:
                total_score, total_weight = score_weight
                r_hat = total_score / total_weight if total_weight > 0 else None

            support_level = _support_level(r_hat)
            matrix.append(
                RecallCoverageEntry(
                    tool=tool,
                    category=category,
                    detected=detected,
                    total=total,
                    R_hat=r_hat,
                    support_level=support_level,
                )
            )
            if r_hat is None or r_hat < 0.3:
                weak_categories.append(category)
        weak_categories_by_tool[tool] = weak_categories

    return RecallCoverageMatrix(
        taxonomy_level="parent",
        matrix=matrix,
        weak_categories_by_tool=weak_categories_by_tool,
    )
