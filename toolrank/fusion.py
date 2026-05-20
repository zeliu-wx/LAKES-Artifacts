"""Finding-level fusion for multi-tool composition.

Implements category-ownership fusion semantics:
- the primary tool keeps only findings for categories it still owns
- each complement contributes only findings for categories assigned to it
- overlapping findings are preserved; later scoring/statistics layers may deduplicate if needed
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from toolrank.schemas import CompositionPlan, Finding, FusedReport


def _normalize_location(location: str) -> str:
    """Normalize a location string for deduplication comparison."""
    return location.strip().lower().replace("\\", "/")


def fuse_reports(
    anchor_findings: Sequence[Finding],
    complement_findings_by_tool: Dict[str, Sequence[Finding]],
    composition: CompositionPlan,
    weak_categories: Optional[Sequence[str]] = None,
    findings_source: str = "synthesized",
) -> FusedReport:
    """Produce a category-ownership fused report.

    Final ownership comes from ``composition.category_assignments``. Any category
    assigned to a complement is removed from the anchor and sourced only from the
    assigned complement. ``weak_categories`` is accepted for backward
    compatibility but no longer controls replacement behavior.
    """
    anchor_id = composition.anchor_tool_id or ""
    assignment_map = dict(composition.category_assignments)
    replacement_categories = sorted(
        category for category, tool_id in assignment_map.items() if tool_id != anchor_id
    )
    strong_categories = sorted(
        category for category, tool_id in assignment_map.items() if tool_id == anchor_id
    )

    retained: List[Finding] = []
    removed_count = 0
    for finding in anchor_findings:
        owner = assignment_map.get(finding.category, anchor_id)
        if owner == anchor_id:
            retained.append(finding)
        else:
            removed_count += 1

    inserted_count = 0
    for tool_id, findings in complement_findings_by_tool.items():
        for finding in findings:
            if assignment_map.get(finding.category) != tool_id:
                continue
            retained.append(finding)
            inserted_count += 1

    deduplicated_count = 0

    summary_parts = [
        f"anchor={anchor_id}",
        f"strong_categories={strong_categories}",
        f"weak_categories={replacement_categories}",
        f"removed_anchor={removed_count}",
        f"inserted_complement={inserted_count}",
        f"deduplicated={deduplicated_count}",
        f"total_findings={len(retained)}",
        f"source={findings_source}",
    ]

    return FusedReport(
        anchor_tool_id=anchor_id,
        weak_categories=replacement_categories,
        strong_categories=strong_categories,
        findings=retained,
        removed_anchor_findings_count=removed_count,
        inserted_complement_findings_count=inserted_count,
        deduplicated_count=deduplicated_count,
        findings_source=findings_source,
        summary="; ".join(summary_parts),
    )


def compact_fused_report_payload(fused_report: FusedReport) -> dict[str, list[dict[str, Any]]]:
    """Return the user-facing LAKES_out report payload.

    Internal fusion metadata stays on ``FusedReport``. The report file should be
    a concise list of findings and their tool source.
    """
    findings: list[dict[str, Any]] = []
    for finding in fused_report.findings:
        finding_payload: dict[str, Any] = {"category": finding.category}
        if finding.location:
            finding_payload["location"] = finding.location
        if finding.severity:
            finding_payload["severity"] = finding.severity
        if finding.explanation:
            finding_payload["explanation"] = finding.explanation

        findings.append(
            {
                "tool": finding.source_tool,
                "finding": finding_payload,
            }
        )
    return {"findings": findings}
