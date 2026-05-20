"""Certify Step 1 nominal winners or candidate sets."""

from __future__ import annotations

from toolrank.schemas_v2 import (
    CategoryDiagnostics,
    CertificationVerdict,
    NominalToolScore,
    RecallCoverageMatrix,
    ScorePanel,
    ToolTableEntry,
)


_SUFFICIENT_EVIDENCE = {"local_moderate", "local_strong"}


def _feasible_tools(tool_table: list[ToolTableEntry]) -> set[str]:
    return {entry.tool for entry in tool_table if entry.feasible}


def _candidate_set(top_tool: str) -> list[str]:
    return [top_tool]


def _ranked_nominal_scores(score_panel: ScorePanel) -> list[NominalToolScore]:
    return sorted(score_panel.nominal_scores, key=lambda item: (item.rank, -item.S_scene, item.tool))


def certify(
    score_panel: ScorePanel,
    diagnostics: CategoryDiagnostics,
    rcov: RecallCoverageMatrix,
    tool_table: list[ToolTableEntry],
) -> CertificationVerdict:
    if not score_panel.nominal_scores:
        return CertificationVerdict(
            status="no_feasible_tool",
            reason_codes=["NO_NOMINAL_SCORES"],
        )

    reason_codes: list[str] = []
    feasible = _feasible_tools(tool_table)
    top_score = None
    for score in _ranked_nominal_scores(score_panel):
        if score.tool in feasible:
            top_score = score
            reason_codes.append("FEASIBLE_TOOL")
            break
        reason_codes.append("TOP1_NOT_FEASIBLE")

    if top_score is None:
        return CertificationVerdict(
            status="no_feasible_tool",
            reason_codes=reason_codes or ["NO_FEASIBLE_TOOL"],
        )

    stable = not score_panel.stress_rankings.top1_flip
    reason_codes.append("STABLE_TOP1" if stable else "NO_STABLE_TOP1")

    bias_ok = diagnostics.category_bias_risk != "high"
    reason_codes.append("BIAS_RISK_ACCEPTABLE" if bias_ok else "HIGH_CATEGORY_BIAS_RISK")

    evidence_ok = top_score.evidence_level in _SUFFICIENT_EVIDENCE
    reason_codes.append("SUFFICIENT_EVIDENCE" if evidence_ok else "INSUFFICIENT_EVIDENCE")

    if stable and bias_ok and evidence_ok:
        return CertificationVerdict(
            status="certified_primary",
            certified_primary=top_score.tool,
            reason_codes=reason_codes,
        )

    return CertificationVerdict(
        status="candidate_set",
        candidate_set=_candidate_set(top_score.tool),
        reason_codes=reason_codes,
    )
