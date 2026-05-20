"""Precision-side gate for category ownership candidates."""

from __future__ import annotations

from statistics import mean

from toolrank.schemas_v2 import Step1EvidencePacket


MIN_MATCHED_PRECISION = 0.15
MATCHED_PRECISION_MEAN_FACTOR = 0.5


def matched_source_ids(packet: Step1EvidencePacket) -> set[str]:
    ids = {neighbor.paper_id for neighbor in packet.scene_pool.neighbors if neighbor.paper_id}
    for neighbor in packet.scene_pool.neighbors:
        ids.update(neighbor.provenance_refs)
    return ids


def top_scene_source_ids(packet: Step1EvidencePacket) -> list[str]:
    if not packet.scene_pool.neighbors:
        return []
    top = packet.scene_pool.neighbors[0]
    ids: list[str] = []
    if top.paper_id:
        ids.append(top.paper_id)
    ids.extend(top.provenance_refs)
    return list(dict.fromkeys(ids))


def _matched_precision_by_tool(packet: Step1EvidencePacket) -> dict[str, float]:
    top_sources = top_scene_source_ids(packet)
    if not top_sources:
        return {}
    source_rank = {source_id: index for index, source_id in enumerate(top_sources)}
    by_tool: dict[str, float] = {}
    by_rank: dict[str, int] = {}
    for row in packet.tool_overall_metrics:
        if row.source_id not in source_rank or row.precision is None:
            continue
        rank = source_rank[row.source_id]
        previous_rank = by_rank.get(row.tool)
        if previous_rank is None or rank < previous_rank:
            by_tool[row.tool] = row.precision
            by_rank[row.tool] = rank
    return by_tool


def matched_precision_threshold(packet: Step1EvidencePacket) -> float | None:
    values = list(_matched_precision_by_tool(packet).values())
    if not values:
        return None
    return max(MIN_MATCHED_PRECISION, mean(values) * MATCHED_PRECISION_MEAN_FACTOR)


def candidate_passes_precision_gate(packet: Step1EvidencePacket, tool: str) -> bool:
    by_tool = _matched_precision_by_tool(packet)
    precision = by_tool.get(tool)
    if precision is None:
        return True
    threshold = matched_precision_threshold(packet)
    return threshold is None or precision >= threshold
