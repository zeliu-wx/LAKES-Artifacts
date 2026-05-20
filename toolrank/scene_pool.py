"""Build benchmark scene pools from performance KB dataset matches."""

from __future__ import annotations

from toolrank.dataset_kb import match_datasets
from toolrank.schemas import ContractFeatures, PerformanceEntry, PerformanceKnowledgeBase
from toolrank.schemas_v2 import SceneNeighbor, ScenePool


def build_scene_pool(
    features: ContractFeatures,
    kb: PerformanceKnowledgeBase,
    top_k: int = 5,
) -> ScenePool:
    if top_k <= 0 or not kb.entries:
        return ScenePool(neighbors=[])

    entries_by_key = {
        (entry.source_id, entry.dataset_profile.dataset_name): entry for entry in kb.entries
    }
    matches = match_datasets(kb, features, top_n=top_k)
    selected = [
        (match, entries_by_key.get((match.source_id, match.dataset_name)))
        for match in matches
    ]
    selected = [(match, entry) for match, entry in selected if entry is not None]
    if not selected:
        return ScenePool(neighbors=[])

    raw_weights = [match.support_ratio for match, _entry in selected]
    weight_total = sum(raw_weights)
    if weight_total <= 0:
        raw_weights = [1.0 for _match, _entry in selected]
        weight_total = sum(raw_weights)

    neighbors = [
        SceneNeighbor(
            slice_id=f"{entry.source_id}_{entry.dataset_profile.dataset_name}",
            benchmark_family=_benchmark_family(entry.dataset_profile.dataset_name),
            paper_id=entry.source_id,
            weight=raw_w / weight_total,
            distance=match.distance,
            category_profile=_category_profile(entry),
            provenance_refs=[entry.source_id],
        )
        for (match, entry), raw_w in zip(selected, raw_weights)
    ]
    return ScenePool(neighbors=neighbors)


def _benchmark_family(dataset_name: str) -> str:
    parts = dataset_name.split()
    return parts[0] if parts else dataset_name


def _category_profile(entry: PerformanceEntry) -> dict[str, float]:
    profile = entry.dataset_profile
    categories = getattr(profile, "vulnerability_categories", None)
    if not categories and profile.complexity_stats is not None:
        categories = profile.complexity_stats.vulnerability_categories
    if not categories:
        return {}

    unique_categories = list(dict.fromkeys(categories))
    weight = 1.0 / len(unique_categories)
    return {category: weight for category in unique_categories}
