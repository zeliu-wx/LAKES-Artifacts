"""Feasibility gates for the SCREC tool dispatch pipeline."""

from __future__ import annotations

from toolrank.schemas import ContractFeatures, D1Metric, FeasibilityResult, ToolCard, TriState
from toolrank.schemas_v2 import BudgetProfile
from toolrank.solc_range import version_in_range


def _required_inputs(source_kind: str) -> list[str]:
    if source_kind == "sol":
        return ["sol"]
    if source_kind == "bytecode":
        return ["bytecode"]
    if source_kind == "runtime":
        return ["runtime"]
    if source_kind == "mixed":
        return ["sol", "bytecode", "runtime"]
    return []


def _metric_payload(card: ToolCard) -> D1Metric | None:
    if not card.d1_metrics:
        return None
    if "default" in card.d1_metrics:
        return card.d1_metrics["default"]
    first_key = sorted(card.d1_metrics.keys())[0]
    return card.d1_metrics[first_key]


def _estimated_runtime_minutes(card: ToolCard) -> float | None:
    metric = _metric_payload(card)
    if metric is None or metric.resolved_time_sec is None:
        return None
    return metric.resolved_time_sec / 60.0


def check_feasibility(
    card: ToolCard,
    features: ContractFeatures,
    budget: BudgetProfile,
) -> FeasibilityResult:
    reasons: list[str] = []

    required_inputs = _required_inputs(features.source_kind)
    if features.source_kind == "mixed":
        if required_inputs and not any(getattr(card.d7_input_support, item, False) for item in required_inputs):
            reasons.append("missing required input support: mixed")
    else:
        for input_type in required_inputs:
            if not getattr(card.d7_input_support, input_type, False):
                reasons.append(f"missing required input support: {input_type}")

    if (
        features.primary_solidity_version
        and not version_in_range(features.primary_solidity_version, card.d2_solidity_versions)
    ):
        reasons.append(f"target solidity version unsupported: {features.primary_solidity_version}")

    if features.is_multifile and card.d3_multifile_support == TriState.no:
        reasons.append("target is multifile but tool lacks multifile support")

    if features.is_multicontract and card.d3_multicontract_support == TriState.no:
        reasons.append("target is multicontract but tool lacks multicontract support")

    runtime_minutes = _estimated_runtime_minutes(card)
    if runtime_minutes is not None and runtime_minutes > budget.runtime_cap_minutes:
        reasons.append(
            f"estimated runtime exceeds budget: {runtime_minutes:.2f}m > {budget.runtime_cap_minutes:.2f}m"
        )

    return FeasibilityResult(feasible=not reasons, reasons=reasons)
