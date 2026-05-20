"""Shared category candidate selection for composition planning and RAG retrieval."""

from __future__ import annotations

from toolrank.assignment_evidence import is_assignment_eligible
from toolrank.precision_gate import candidate_passes_precision_gate
from toolrank.schemas_v2 import Step1EvidencePacket


def _matched_source_ids(packet: Step1EvidencePacket) -> set[str]:
    ids = {neighbor.paper_id for neighbor in packet.scene_pool.neighbors if neighbor.paper_id}
    for neighbor in packet.scene_pool.neighbors:
        ids.update(neighbor.provenance_refs)
    return ids


def _rcov_by_tool_category(matrix) -> dict[tuple[str, str], object]:
    return {(row.tool, row.category): row for row in matrix.matrix}


def _external_rankings_by_tool(packet: Step1EvidencePacket, category: str) -> dict[str, list[dict]]:
    matched_sources = _matched_source_ids(packet)
    rows_by_dataset: dict[tuple[str, str], list] = {}
    for row in packet.performance_db_view:
        if row.category != category or row.source_id in matched_sources or row.R_hat is None:
            continue
        rows_by_dataset.setdefault((row.source_id, row.dataset_name), []).append(row)

    rankings_by_tool: dict[str, list[dict]] = {}
    for dataset_key, rows in rows_by_dataset.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                -(row.R_hat if row.R_hat is not None else -1.0),
                -(row.total or 0),
                row.tool,
                row.evidence_id,
            ),
        )
        for index, row in enumerate(ranked, start=1):
            next_rate = ranked[index].R_hat if index < len(ranked) else None
            margin = row.R_hat - next_rate if next_rate is not None else None
            rankings_by_tool.setdefault(row.tool, []).append(
                {
                    "row": row,
                    "rank": index,
                    "peer_count": len(ranked),
                    "margin": margin,
                    "dataset_key": dataset_key,
                }
            )
    return rankings_by_tool


def _best_external_ranking(rankings: list[dict]) -> dict | None:
    if not rankings:
        return None
    return min(
        rankings,
        key=lambda item: (
            item["rank"],
            0
            if is_assignment_eligible(
                item["row"].detected,
                item["row"].total,
                item["row"].R_hat,
            )
            else 1,
            -(item["row"].R_hat if item["row"].R_hat is not None else -1.0),
            -(item["row"].total or 0),
            item["row"].source_id,
            item["row"].dataset_name,
        ),
    )


def _eligible_external_rankings(rankings: list[dict]) -> list[dict]:
    return [
        item
        for item in rankings
        if is_assignment_eligible(item["row"].detected, item["row"].total, item["row"].R_hat)
    ]


def category_candidate_tools(
    packet: Step1EvidencePacket,
    category: str,
    limit: int = 3,
) -> list[str]:
    primary_tool = packet.primary_attention.primary_tool
    feasible_tools = sorted(
        entry.tool for entry in packet.tool_table if entry.feasible and entry.tool != primary_tool
    )
    local_by_key = _rcov_by_tool_category(packet.recall_coverage)
    external_by_tool = _external_rankings_by_tool(packet, category)

    def add_tool(items: list[str], tool: str) -> None:
        if tool not in items and len(items) < limit:
            items.append(tool)

    def local_sort_key(tool: str) -> tuple[float, int, str]:
        entry = local_by_key.get((tool, category))
        return (-(entry.R_hat if entry and entry.R_hat is not None else -1.0), -(entry.total or 0) if entry else 0, tool)

    def external_sort_key(tool: str) -> tuple[int, int, int, float, int, str]:
        eligible = _eligible_external_rankings(external_by_tool.get(tool, []))
        best = _best_external_ranking(eligible)
        if best is None:
            return (9999, 0, 0, 1.0, 0, tool)
        row = best["row"]
        eligible_datasets = {item["dataset_key"] for item in eligible}
        rank1_datasets = {item["dataset_key"] for item in eligible if item["rank"] == 1}
        return (
            best["rank"],
            -len(eligible_datasets),
            -len(rank1_datasets),
            -(row.R_hat if row.R_hat is not None else -1.0),
            -(row.total or 0),
            tool,
        )

    selected: list[str] = []
    external_tools = [
        tool
        for tool in feasible_tools
        if candidate_passes_precision_gate(packet, tool)
        and _eligible_external_rankings(external_by_tool.get(tool, []))
    ]
    external_tools.sort(key=external_sort_key)
    if external_tools:
        add_tool(selected, external_tools[0])

    local_tools = [
        tool
        for tool in feasible_tools
        if local_by_key.get((tool, category)) is not None
        and local_by_key[(tool, category)].R_hat is not None
        and candidate_passes_precision_gate(packet, tool)
    ]
    for tool in sorted(local_tools, key=local_sort_key):
        add_tool(selected, tool)

    for tool in external_tools:
        add_tool(selected, tool)

    return selected
