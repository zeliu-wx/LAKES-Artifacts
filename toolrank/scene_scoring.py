"""Compute scene-conditioned nominal tool scores."""

from __future__ import annotations

import re

from toolrank.schemas import D1Metric, PerformanceEntry, PerformanceKnowledgeBase
from toolrank.schemas_v2 import NominalToolScore, ScenePool


def _tool_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _raw_metric(metric: D1Metric) -> float | None:
    if metric.f1 is not None:
        return metric.f1
    if metric.recall is not None:
        return metric.recall
    if metric.precision is not None:
        return metric.precision
    return None


def _rank_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    if len(raw_scores) == 1:
        return {next(iter(raw_scores)): 0.0}
    ranked = sorted(raw_scores.items(), key=lambda item: (-item[1], item[0]))
    return {tool: (len(ranked) - index) / len(ranked) for index, (tool, _score) in enumerate(ranked)}


def _weighted_average(values: list[tuple[float, float]]) -> float | None:
    total_weight = sum(weight for weight, _value in values)
    if total_weight <= 0.0:
        return None
    return sum(weight * value for weight, value in values) / total_weight


def _evidence_level(slice_count: int, participated_weight: float, total_weight: float) -> str:
    if slice_count >= 3 and total_weight > 0.0 and participated_weight / total_weight >= 0.5:
        return "local_strong"
    if slice_count >= 2:
        return "local_moderate"
    if slice_count == 1:
        return "local_weak"
    return "unsupported"


def compute_scene_scores(
    scene_pool: ScenePool,
    kb: PerformanceKnowledgeBase,
    tool_ids: list[str],
) -> tuple[list[NominalToolScore], dict[str, dict[str, float]]]:
    entries_by_source: dict[str, PerformanceEntry] = {entry.source_id: entry for entry in kb.entries}
    requested_by_key = {_tool_key(tool): tool for tool in tool_ids}
    total_scene_weight = 1.0 if scene_pool.neighbors else 0.0

    tool_slice_scores: dict[str, dict[str, float]] = {tool: {} for tool in tool_ids}
    score_values: dict[str, list[tuple[float, float]]] = {tool: [] for tool in tool_ids}
    precision_values: dict[str, list[tuple[float, float]]] = {tool: [] for tool in tool_ids}
    recall_values: dict[str, list[tuple[float, float]]] = {tool: [] for tool in tool_ids}
    f1_values: dict[str, list[tuple[float, float]]] = {tool: [] for tool in tool_ids}
    participated_weights: dict[str, float] = {tool: 0.0 for tool in tool_ids}

    primary_neighbor = scene_pool.neighbors[0] if scene_pool.neighbors else None
    for neighbor in scene_pool.neighbors:
        if neighbor.paper_id is None:
            continue
        entry = entries_by_source.get(neighbor.paper_id)
        if entry is None:
            continue

        raw_scores: dict[str, float] = {}
        metrics_by_tool: dict[str, D1Metric] = {}
        for observation in entry.tool_performance_data:
            tool = requested_by_key.get(_tool_key(observation.tool_name))
            if tool is None:
                continue
            raw = _raw_metric(observation.metrics)
            if raw is None:
                continue
            raw_scores[tool] = raw
            metrics_by_tool[tool] = observation.metrics

        for tool, rank_score in _rank_scores(raw_scores).items():
            tool_slice_scores[tool][neighbor.slice_id] = rank_score
            if neighbor != primary_neighbor:
                continue
            score_values[tool].append((1.0, rank_score))
            participated_weights[tool] += 1.0
            metric = metrics_by_tool[tool]
            if metric.precision is not None:
                precision_values[tool].append((1.0, metric.precision))
            if metric.recall is not None:
                recall_values[tool].append((1.0, metric.recall))
            if metric.f1 is not None:
                f1_values[tool].append((1.0, metric.f1))

    nominal_scores: list[NominalToolScore] = []
    for tool in tool_ids:
        s_scene = _weighted_average(score_values[tool]) or 0.0
        nominal_scores.append(
            NominalToolScore(
                tool=tool,
                S_scene=s_scene,
                rank=1,
                P_scene=_weighted_average(precision_values[tool]),
                R_scene=_weighted_average(recall_values[tool]),
                F1_scene=_weighted_average(f1_values[tool]),
                evidence_level=_evidence_level(
                    len(score_values[tool]),
                    participated_weights[tool],
                    total_scene_weight,
                ),
            )
        )

    nominal_scores.sort(key=lambda item: (-item.S_scene, item.tool))
    ranked_scores = [
        item.model_copy(update={"rank": rank})
        for rank, item in enumerate(nominal_scores, start=1)
    ]
    return ranked_scores, tool_slice_scores
