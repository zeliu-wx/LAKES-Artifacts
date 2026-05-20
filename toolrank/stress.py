"""Stress ranking utilities."""

from __future__ import annotations

from toolrank.schemas_v2 import SceneNeighbor, ScenePool, StressRankings


def _supported_categories(neighbors: list[SceneNeighbor]) -> set[str]:
    return {
        category
        for neighbor in neighbors
        for category, value in neighbor.category_profile.items()
        if value > 0.0
    }


def _normalize_profile(profile: dict[str, float]) -> dict[str, float]:
    positive = {category: value for category, value in profile.items() if value > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {category: value / total for category, value in positive.items()}


def _category_margins(
    neighbors: list[SceneNeighbor],
    weights: list[float],
    categories: set[str],
) -> dict[str, float]:
    return {
        category: sum(
            weight * neighbor.category_profile.get(category, 0.0)
            for neighbor, weight in zip(neighbors, weights)
        )
        for category in categories
    }


def _rake_weights(
    neighbors: list[SceneNeighbor],
    base_weights: list[float],
    target_profile: dict[str, float],
) -> list[float]:
    weights = list(base_weights)
    for _ in range(20):
        margins = _category_margins(neighbors, weights, set(target_profile))
        for category, target in target_profile.items():
            current = margins.get(category, 0.0)
            if current <= 0.0:
                continue
            factor = target / current
            weights = [
                weight * (factor ** neighbor.category_profile.get(category, 0.0))
                for neighbor, weight in zip(neighbors, weights)
            ]
    return weights


def _rank_tools(
    neighbors: list[SceneNeighbor],
    weights: list[float],
    tool_scores: dict[str, dict[str, float]],
) -> list[str]:
    scores = {
        tool: sum(
            weight * slice_scores.get(neighbor.slice_id, 0.0)
            for neighbor, weight in zip(neighbors, weights)
        )
        for tool, slice_scores in tool_scores.items()
    }
    return sorted(scores, key=lambda tool: (-scores[tool], tool))


def run_stress_test(
    scene_pool: ScenePool,
    tool_scores: dict[str, dict[str, float]],
    global_category_profile: dict[str, float],
) -> StressRankings:
    neighbors = scene_pool.neighbors
    base_weights = [neighbor.weight for neighbor in neighbors]
    support = _supported_categories(neighbors)

    local = _rank_tools(neighbors, base_weights, tool_scores)

    support_failures: list[str] = []
    global_target = _normalize_profile(global_category_profile)
    if any(category not in support for category in global_target):
        support_failures.append("global")
    global_weights = _rake_weights(neighbors, base_weights, global_target)
    global_ranking = _rank_tools(neighbors, global_weights, tool_scores)

    uniform_target = (
        {category: 1.0 / len(support) for category in support} if support else {}
    )
    uniform_weights = _rake_weights(neighbors, base_weights, uniform_target)
    uniform_supported = _rank_tools(neighbors, uniform_weights, tool_scores)

    local_top = local[0] if local else None
    top1_flip = bool(
        local_top
        and (
            (global_ranking and global_ranking[0] != local_top)
            or (uniform_supported and uniform_supported[0] != local_top)
        )
    )

    return StressRankings(
        local=local,
        global_=global_ranking,
        uniform_supported=uniform_supported,
        top1_flip=top1_flip,
        support_failures=support_failures,
    )
