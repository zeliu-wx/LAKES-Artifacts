from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from toolrank.schemas import (
    ContractFeatures,
    D1Metric,
    DatasetLocBin,
    DatasetMatch,
    PerformanceKnowledgeBase,
    ToolPerformanceObservation,
)


def load_performance_db(path: str | Path) -> PerformanceKnowledgeBase:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PerformanceKnowledgeBase.model_validate(payload)


def _solc_stratum(version: str | None) -> str | None:
    if not version:
        return None
    text = str(version).strip()
    if text.startswith("0."):
        parts = text.split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
    return None


def _bin_by_label(bins: List[DatasetLocBin]) -> dict[str, DatasetLocBin]:
    return {item.label: item for item in bins}


def _loc_bin_representative(item: DatasetLocBin) -> float:
    if item.max_exclusive is None:
        return float(item.min_inclusive)
    return (float(item.min_inclusive) + float(item.max_exclusive)) / 2.0


def _positive_count(raw_count) -> float | None:
    try:
        count = float(raw_count)
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    return count


def _entry_loc_samples(
    loc_profile: dict,
    bins_by_label: dict[str, DatasetLocBin],
) -> list[tuple[str, float, float]]:
    samples: list[tuple[str, float, float]] = []
    for bin_label, solc_counts in loc_profile.items():
        if not isinstance(solc_counts, dict):
            continue
        loc_bin = bins_by_label.get(str(bin_label))
        if loc_bin is None:
            continue
        loc_center = _loc_bin_representative(loc_bin)
        for solc_text, raw_count in solc_counts.items():
            count = _positive_count(raw_count)
            stratum = _solc_stratum(str(solc_text))
            if count is None or stratum is None:
                continue
            samples.append((stratum, loc_center, count))
    return samples


def _gower_solc_loc_distance(
    *,
    target_stratum: str,
    target_loc: int,
    sample_stratum: str,
    sample_loc: float,
    loc_range: float,
) -> float:
    solc_distance = 0.0 if sample_stratum == target_stratum else 1.0
    loc_distance = abs(sample_loc - float(target_loc)) / loc_range
    return 0.5 * solc_distance + 0.5 * min(loc_distance, 1.0)


def _loc_range(samples: list[tuple[str, float, float]]) -> float:
    loc_values = [loc for _stratum, loc, _count in samples]
    if not loc_values:
        return 1.0
    value = max(loc_values) - min(loc_values)
    return value if value > 0 else 1.0


def match_datasets(
    kb: PerformanceKnowledgeBase,
    features: ContractFeatures,
    top_n: int = 1,
    neighbor_k: int = 50,
) -> List[DatasetMatch]:
    raw_bins = kb.criteria.get("loc_bins") or []
    bins = [DatasetLocBin.model_validate(item) for item in raw_bins]
    bins_by_label = _bin_by_label(bins)
    target_stratum = _solc_stratum(features.primary_solidity_version)

    if top_n <= 0 or neighbor_k <= 0 or not target_stratum or features.loc_total <= 0 or not bins_by_label:
        return []

    all_samples: list[tuple[float, str, str, float, float]] = []
    dataset_totals: dict[tuple[str, str], float] = {}
    target_solc_totals: dict[tuple[str, str], float] = {}
    entry_samples: dict[tuple[str, str], list[tuple[str, float, float]]] = {}

    for entry in kb.entries:
        loc_profile = entry.dataset_profile.loc_profile.get("loc_bin_counts_by_solc") or {}
        if not isinstance(loc_profile, dict):
            continue
        samples = _entry_loc_samples(loc_profile, bins_by_label)
        if not samples:
            continue
        key = (entry.source_id, entry.dataset_profile.dataset_name)
        total = sum(count for _stratum, _loc, count in samples)
        if total <= 0:
            continue
        dataset_totals[key] = total
        target_solc_totals[key] = sum(
            count for stratum, _loc, count in samples if stratum == target_stratum
        )
        entry_samples[key] = samples

    flat_samples = [sample for samples in entry_samples.values() for sample in samples]
    loc_range = _loc_range(flat_samples)

    for key, samples in entry_samples.items():
        source_id, dataset_name = key
        for sample_stratum, sample_loc, count in samples:
            distance = _gower_solc_loc_distance(
                target_stratum=target_stratum,
                target_loc=features.loc_total,
                sample_stratum=sample_stratum,
                sample_loc=sample_loc,
                loc_range=loc_range,
            )
            all_samples.append((distance, source_id, dataset_name, sample_loc, count))

    if not all_samples:
        return []

    all_samples.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    cumulative = 0.0
    radius = all_samples[-1][0]
    for distance, _source_id, _dataset_name, _sample_loc, count in all_samples:
        cumulative += count
        radius = distance
        if cumulative >= neighbor_k:
            break

    aggregates: dict[tuple[str, str], dict[str, float]] = {}
    for distance, source_id, dataset_name, _sample_loc, count in all_samples:
        if distance > radius:
            continue
        key = (source_id, dataset_name)
        aggregate = aggregates.setdefault(key, {"neighbor_count": 0.0, "distance_sum": 0.0})
        aggregate["neighbor_count"] += count
        aggregate["distance_sum"] += distance * count

    ranked: list[tuple[float, float, float, str, DatasetMatch]] = []
    for key, aggregate in aggregates.items():
        source_id, dataset_name = key
        total = dataset_totals.get(key, 0.0)
        neighbor_count = aggregate["neighbor_count"]
        if total <= 0 or neighbor_count <= 0:
            continue
        support_ratio = neighbor_count / total
        mean_distance = aggregate["distance_sum"] / neighbor_count
        target_support = target_solc_totals.get(key, 0.0)
        target_support_ratio = target_support / total if total > 0 else 0.0
        scene_distance = max(0.0, 1.0 - support_ratio)
        match = DatasetMatch(
            source_id=source_id,
            dataset_name=dataset_name,
            distance=scene_distance,
            support_count=int(neighbor_count),
            support_ratio=support_ratio,
            solc_stratum=target_stratum,
            reasons=[
                f"solc_stratum={target_stratum}",
                f"knn_k={neighbor_k}",
                f"knn_radius={radius:.4f}",
                f"neighbor_count={int(neighbor_count)}",
                f"dataset_count={int(total)}",
                f"knn_coverage={support_ratio:.4f}",
                f"mean_gower_distance={mean_distance:.4f}",
                f"target_solc_support_count={int(target_support)}",
                f"target_solc_support_ratio={target_support_ratio:.4f}",
            ],
        )
        ranked.append((support_ratio, neighbor_count, mean_distance, source_id, match))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return [match for _score, _count, _mean_distance, _source_id, match in ranked[:top_n]]


def _tool_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def aggregate_tool_observation(
    kb: PerformanceKnowledgeBase,
    matches: List[DatasetMatch],
    tool_name: str,
    features: Optional[ContractFeatures] = None,
) -> Tuple[Optional[D1Metric], Dict[str, float]]:
    """Retrieve benchmark evidence for *tool_name* from the primary scene d* only."""
    if not matches:
        return None, {}

    primary = matches[0]
    wanted = _tool_key(tool_name)
    obs = _find_observation(kb, primary.source_id, wanted)
    if obs is None:
        return None, {}
    return _extract_metric_vuln(obs)


def aggregate_tool_observation_with_fallback(
    kb: PerformanceKnowledgeBase,
    matches: List[DatasetMatch],
    tool_name: str,
    features: Optional[ContractFeatures] = None,
) -> Tuple[Optional[D1Metric], Dict[str, float], Optional[str]]:
    """Retrieve benchmark evidence from the matched KNN scene order."""
    _ = features
    if not matches:
        return None, {}, None

    wanted = _tool_key(tool_name)
    primary = matches[0]
    primary_obs = _find_observation(kb, primary.source_id, wanted)
    if primary_obs is not None:
        metric, vuln = _extract_metric_vuln(primary_obs)
        return metric, vuln, None

    for match in matches[1:]:
        obs = _find_observation(kb, match.source_id, wanted)
        if obs is None:
            continue
        metric, vuln = _extract_metric_vuln(obs)
        return metric, vuln, match.source_id

    return None, {}, None


def _find_observation(
    kb: PerformanceKnowledgeBase, source_id: str, wanted_key: str
) -> Optional[ToolPerformanceObservation]:
    """Find a specific tool observation within a specific dataset entry."""
    for entry in kb.entries:
        if entry.source_id != source_id:
            continue
        for item in entry.tool_performance_data:
            if _tool_key(item.tool_name) == wanted_key:
                return item
    return None


def _extract_metric_vuln(
    obs: ToolPerformanceObservation,
) -> Tuple[Optional[D1Metric], Dict[str, float]]:
    """Extract D1Metric and vulnerability scores from an observation."""
    metrics = obs.metrics
    has_data = any(
        getattr(metrics, f) is not None
        for f in ("precision", "recall", "f1", "time_sec", "execution_time_avg")
    )
    metric = metrics if has_data else None
    vuln: Dict[str, float] = {}
    if obs.vulnerability_scores:
        vuln = {str(k): float(v) for k, v in obs.vulnerability_scores.items()}
    return metric, vuln


# ---------------------------------------------------------------------------
# Issue 4: Dataset / Benchmark KB refresh path
# Paper (§3.3): "The benchmark base is refreshed when new curated benchmark
# artifacts are incorporated."
# ---------------------------------------------------------------------------


def refresh_performance_db(
    existing_path: str | Path,
    new_artifact_paths: List[str | Path],
    output_path: Optional[str | Path] = None,
) -> PerformanceKnowledgeBase:
    """Ingest new benchmark performance artifacts and merge into existing DB.

    Each *new_artifact_path* must be a JSON file with the same schema as
    ``performance_db.json`` (i.e., ``PerformanceKnowledgeBase``).
    New entries are appended; if an entry with the same ``source_id`` already
    exists, the new entry *replaces* it (fresher evidence wins).
    """
    existing_path = Path(existing_path)
    if existing_path.exists():
        base = load_performance_db(existing_path)
    else:
        base = PerformanceKnowledgeBase(knowledge_base_type="performance", entries=[])

    existing_ids = {entry.source_id for entry in base.entries}

    for artifact_path in new_artifact_paths:
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        incoming = PerformanceKnowledgeBase.model_validate(payload)
        for entry in incoming.entries:
            if entry.source_id in existing_ids:
                # Replace existing entry with fresher evidence
                base.entries = [e for e in base.entries if e.source_id != entry.source_id]
            base.entries.append(entry)
            existing_ids.add(entry.source_id)

    out = Path(output_path) if output_path else existing_path
    out.write_text(
        json.dumps(base.model_dump(by_alias=True), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return base
