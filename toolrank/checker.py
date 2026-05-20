"""Checker: 4-rule legality auditor for CEGO decisions."""

from __future__ import annotations

from toolrank.assignment_evidence import is_assignment_eligible
from toolrank.ownership_evidence import (
    attention_categories,
    card_scene_tier,
    card_scope_text,
    category_capability_pro_refs,
    category_marginal_value,
    category_ownership_evidence,
    dataset_support_refs,
    rcov_evidence_id,
)
from toolrank.schemas import DASP10_CATEGORIES
from toolrank.schemas_v2 import (
    ActionByEvidenceMatrix,
    ActionEvidenceBlock,
    ActionEvidenceClaim,
    CandidateAction,
    CheckerVerdict,
    Step1EvidencePacket,
    Step2DecisionCertificate,
)


Failure = tuple[str, str]
ALL_CATEGORY = "ALL"


def _action_by_id(matrix: ActionByEvidenceMatrix) -> dict[str, CandidateAction]:
    return {action.action_id: action for action in matrix.actions}


def _selected_action(
    certificate: Step2DecisionCertificate,
    matrix: ActionByEvidenceMatrix,
) -> CandidateAction | None:
    return _action_by_id(matrix).get(certificate.selected_action_id)


def _all_claims(certificate: Step2DecisionCertificate) -> list[ActionEvidenceClaim]:
    block = certificate.action_evidence
    if block is None:
        return []
    return block.for_claims + block.against_claims + block.compare_claims + block.gap_claims


def _check_action_legality(
    certificate: Step2DecisionCertificate,
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
) -> list[Failure]:
    failures: list[Failure] = []
    action = _selected_action(certificate, matrix)
    if action is None:
        failures.append(
            ("ACTION_NOT_IN_MATRIX", "Selected action is not present in the action matrix.")
        )
    elif not action.legal:
        failures.append(
            ("ACTION_ILLEGAL_IN_MATRIX", "Selected action is marked illegal in the action matrix.")
        )

    feasible_by_tool = {entry.tool: entry.feasible for entry in packet.tool_table}
    seen_tools: set[str] = set()
    duplicate_tools: set[str] = set()
    for item in certificate.selected_plan:
        if not feasible_by_tool.get(item.tool, False):
            failures.append(
                (
                    f"TOOL_NOT_FEASIBLE:{item.tool}",
                    f"Selected tool is missing from the feasible tool table: {item.tool}.",
                )
            )
        if item.tool in seen_tools and item.tool not in duplicate_tools:
            duplicate_tools.add(item.tool)
            failures.append((f"DUPLICATE_TOOL:{item.tool}", f"Selected plan repeats tool: {item.tool}."))
        seen_tools.add(item.tool)

    if certificate.budget.estimated_use.tool_slots > certificate.budget.limit.tool_slots:
        failures.append(
            (
                "BUDGET_TOOL_SLOTS_EXCEEDED",
                "Estimated tool slots exceed the configured tool slot limit.",
            )
        )
    if (
        certificate.budget.estimated_use.runtime_cap_minutes
        > certificate.budget.limit.runtime_cap_minutes
    ):
        failures.append(
            (
                "BUDGET_RUNTIME_EXCEEDED",
                "Estimated runtime exceeds the configured runtime cap.",
            )
        )
    return failures


def _check_evidence_legality(
    certificate: Step2DecisionCertificate,
    matrix: ActionByEvidenceMatrix,
) -> list[Failure]:
    legal_refs = {card.evidence_id for card in matrix.evidence_cards}
    failures: list[Failure] = []
    for claim in _all_claims(certificate):
        for ref in claim.evidence_refs:
            if ref in legal_refs or ref.startswith("step1."):
                continue
            failures.append((f"UNRESOLVABLE_REF:{ref}", f"Evidence reference cannot be resolved: {ref}."))
    return failures


def _action_type(
    certificate: Step2DecisionCertificate,
    matrix: ActionByEvidenceMatrix,
) -> str:
    action = _selected_action(certificate, matrix)
    return action.action_type if action is not None else certificate.decision_type


def _selected_tools(certificate: Step2DecisionCertificate) -> set[str]:
    return {item.tool for item in certificate.selected_plan}


def _check_evidence_completeness(
    certificate: Step2DecisionCertificate,
    matrix: ActionByEvidenceMatrix,
) -> list[Failure]:
    action_type = _action_type(certificate, matrix)
    block = certificate.action_evidence or ActionEvidenceBlock(action_id=certificate.selected_action_id)

    if action_type == "PLAN_COMPOSITION":
        failures: list[Failure] = []
        for slot, claims in (
            ("FOR", block.for_claims),
            ("AGAINST", block.against_claims),
            ("COMPARE", block.compare_claims),
            ("GAP", block.gap_claims),
        ):
            if not claims:
                failures.append(
                    (
                        f"MISSING_{slot}_CLAIMS",
                        f"{action_type} is missing {slot.lower()} claims.",
                    )
                )
        return failures

    if action_type == "RUN_ROBUST_SINGLE":
        has_ranking_or_bias_ref = any(
            ref.startswith("ev_scene_")
            or ref.startswith("step1.score_panel")
            or ref.startswith("step1.category_diagnostics")
            for claim in _all_claims(certificate)
            for ref in claim.evidence_refs
        )
        if not has_ranking_or_bias_ref:
            return [
                (
                    "NO_RANKING_OR_BIAS_EVIDENCE",
                    "RUN_ROBUST_SINGLE requires ranking or category-bias evidence.",
                )
            ]
    return []


def _has_composition_fields(certificate: Step2DecisionCertificate) -> bool:
    return bool(
        certificate.primary_tool
        or certificate.tool_categories
    )


def _is_all_category(category: str) -> bool:
    return category.upper() == ALL_CATEGORY


def _claim_refs(certificate: Step2DecisionCertificate) -> set[str]:
    return {
        ref
        for claim in _all_claims(certificate)
        for ref in claim.evidence_refs
    }


def _rag_assignment_refs(matrix: ActionByEvidenceMatrix, *, category: str, tool: str) -> set[str]:
    return set(category_capability_pro_refs(matrix, category=category, tool=tool))


def _has_recall_assignment_card(
    refs: set[str],
    matrix: ActionByEvidenceMatrix,
    *,
    category: str,
    tool: str,
) -> bool:
    for card in matrix.evidence_cards:
        if card.evidence_id not in refs:
            continue
        if (
            card.evidence_type != "per_category_detected_total"
            or card.tool != tool
            or card.category != category
            or card.value is None
            or card.value.rate is None
        ):
            continue
        if is_assignment_eligible(card.value.detected, card.value.total, card.value.rate):
            return True
    return False


def _has_assignment_evidence(
    certificate: Step2DecisionCertificate,
    matrix: ActionByEvidenceMatrix,
    *,
    category: str,
    tool: str,
) -> bool:
    refs = _claim_refs(certificate)
    if _has_recall_assignment_card(refs, matrix, category=category, tool=tool):
        return True
    return bool(refs & _rag_assignment_refs(matrix, category=category, tool=tool))


def _owner_pool_status(
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
    *,
    category: str,
    tool: str,
):
    ownership = category_ownership_evidence(packet, matrix, category)
    strong_tools = {candidate.tool for candidate in ownership.strong_candidates}
    weak_tools = {candidate.tool for candidate in ownership.weak_candidates}
    if tool == ownership.preferred:
        return "preferred", ownership
    if tool in strong_tools:
        return "strong", ownership
    if tool in weak_tools:
        return "weak", ownership
    return "none", ownership


def _override_failures(
    certificate: Step2DecisionCertificate,
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
    *,
    category: str,
    tool: str,
) -> list[Failure]:
    status, ownership = _owner_pool_status(packet, matrix, category=category, tool=tool)
    if status not in {"strong", "weak"}:
        return []
    if (
        not ownership.rag_override_eligible
        or not ownership.preferred
        or tool not in ownership.override_targets
    ):
        return [
            (
                "ownership_override_not_authorized",
                "Override owner requires category rag_override_eligible=true and override_targets membership.",
            )
        ]

    preferred_ref = rcov_evidence_id(ownership.preferred, category)
    override_refs = set(ownership.override_refs)
    owner_rag_pro_refs = set(ownership.override_targets[tool])
    block = certificate.action_evidence or ActionEvidenceBlock(action_id=certificate.selected_action_id)
    against_ref_sets = [set(claim.evidence_refs) for claim in block.against_claims]
    has_against_chain = any(preferred_ref in refs and refs & override_refs for refs in against_ref_sets)
    failures: list[Failure] = []
    if not has_against_chain:
        has_rag_ref = any(refs & override_refs for refs in against_ref_sets)
        has_preferred_ref = any(preferred_ref in refs for refs in against_ref_sets)
        if not has_rag_ref:
            failures.append(
                (
                    "ownership_override_missing_rag_ref",
                    "Override owner requires an against_claim citing an override RAG ref.",
                )
            )
        if not has_preferred_ref:
            failures.append(
                (
                    "ownership_override_missing_preferred_ref",
                    "Override owner requires an against_claim citing the preferred owner's recall ref.",
                )
            )
        if has_rag_ref and has_preferred_ref:
            failures.append(
                (
                    "ownership_override_missing_against_chain",
                    "Override owner requires one against_claim to cite both override and preferred-owner refs.",
                )
            )

    support_refs = {rcov_evidence_id(tool, category)} | dataset_support_refs(matrix, tool=tool, category=category)
    has_owner_support = any(set(claim.evidence_refs) & support_refs for claim in block.for_claims)
    if not has_owner_support:
        failures.append(
            (
                "ownership_override_missing_owner_support_ref",
                "Override owner requires a for_claim citing its recall or external dataset ref.",
            )
        )
    has_owner_rag_pro = any(set(claim.evidence_refs) & owner_rag_pro_refs for claim in block.for_claims)
    if not has_owner_rag_pro:
        failures.append(
            (
                "ownership_override_missing_rag_pro_ref",
                "Override owner requires a for_claim citing a RAG pro reference for this tool/category.",
            )
        )
    return failures


def _check_composition_ownership_legality(
    certificate: Step2DecisionCertificate,
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
) -> list[Failure]:
    if not _has_composition_fields(certificate):
        return []
    failures: list[Failure] = []
    primary = certificate.primary_tool
    packet_attention_categories = set(attention_categories(packet))
    for tool, categories in certificate.tool_categories.items():
        if tool == primary:
            continue
        for category in categories:
            if _is_all_category(category):
                continue
            if category not in packet_attention_categories:
                continue
            status, _ownership = _owner_pool_status(packet, matrix, category=category, tool=tool)
            if status == "preferred":
                continue
            if status in {"strong", "weak"}:
                failures.extend(
                    _override_failures(
                        certificate,
                        packet,
                        matrix,
                        category=category,
                        tool=tool,
                    )
                )
                continue
            failures.append(
                (
                    "ownership_owner_not_in_eligible_pools",
                    f"Assigned owner is not preferred, strong-eligible, or weak-eligible for {category}: {tool}.",
                )
            )
    return failures


def _check_category_assignments_against_panel(
    certificate: Step2DecisionCertificate,
    matrix: ActionByEvidenceMatrix,
) -> list[Failure]:
    failures: list[Failure] = []
    for assignment in certificate.category_assignments:
        panel = matrix.ownership_panel.get(assignment.category)
        if panel is None:
            failures.append(
                (
                    f"category_assignment_not_in_ownership_panel:{assignment.category}",
                    f"Category assignment is missing from ownership_panel: {assignment.category}.",
                )
            )
            continue
        if assignment.assignment_type in {"gap", "stop_with_gap"}:
            continue
        if not assignment.owner_tool:
            failures.append(
                (
                    f"category_assignment_missing_owner:{assignment.category}",
                    f"Category assignment is missing owner_tool: {assignment.category}.",
                )
            )
            continue
        eligible = {candidate.tool for candidate in panel.strong_candidates + panel.weak_candidates}
        eligible.update(panel.override_targets)
        if assignment.owner_tool not in eligible:
            failures.append(
                (
                    f"category_assignment_owner_not_eligible:{assignment.owner_tool}/{assignment.category}",
                    f"Category assignment owner is not eligible for {assignment.category}: {assignment.owner_tool}.",
                )
            )
    return failures


def _check_rag_overrides(
    certificate: Step2DecisionCertificate,
    matrix: ActionByEvidenceMatrix,
) -> list[Failure]:
    failures: list[Failure] = []
    for override in certificate.rag_overrides:
        if not override.applied:
            continue
        panel = matrix.ownership_panel.get(override.category)
        if panel is None or not panel.rag_override_eligible:
            failures.append(
                (
                    f"rag_override_not_eligible:{override.category}",
                    f"RAG override is not eligible for category: {override.category}.",
                )
            )
        if not override.to_owner or override.to_owner not in (panel.override_targets if panel else {}):
            failures.append(
                (
                    f"rag_override_target_not_allowed:{override.to_owner}/{override.category}",
                    f"RAG override target is not allowed for {override.category}: {override.to_owner}.",
                )
            )
        if not override.for_refs or not override.against_refs or not override.compare_refs:
            failures.append(
                (
                    f"rag_override_missing_for_against_compare_refs:{override.category}",
                    f"Applied RAG override is missing for/against/compare refs: {override.category}.",
                )
            )
    return failures


def _check_assignment_caveats(certificate: Step2DecisionCertificate) -> list[Failure]:
    failures: list[Failure] = []
    for assignment in certificate.category_assignments:
        if assignment.assignment_type in {"gap", "stop_with_gap"} or not assignment.owner_tool:
            continue
        if assignment.unrelated_external_only and not assignment.caveat_refs:
            failures.append(
                (
                    f"unrelated_external_only_assignment_missing_caveat:{assignment.owner_tool}/{assignment.category}",
                    f"Unrelated-external-only assignment is missing caveat refs: {assignment.owner_tool}/{assignment.category}.",
                )
            )
    return failures


def _unrelated_external_caveat_reasons(
    certificate: Step2DecisionCertificate,
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
) -> list[str]:
    if not _has_composition_fields(certificate):
        return []
    block = certificate.action_evidence or ActionEvidenceBlock(action_id=certificate.selected_action_id)
    cards_by_id = {card.evidence_id: card for card in matrix.evidence_cards}
    reasons: list[str] = []
    primary = certificate.primary_tool
    attention = set(attention_categories(packet))
    for tool, categories in certificate.tool_categories.items():
        if tool == primary:
            continue
        for category in categories:
            if _is_all_category(category) or category not in attention:
                continue
            _status, ownership = _owner_pool_status(packet, matrix, category=category, tool=tool)
            if tool != ownership.preferred:
                continue
            cited_cards: list[EvidenceCard] = []
            for claim in block.for_claims:
                for ref in claim.evidence_refs:
                    card = cards_by_id.get(ref)
                    if (
                        card is not None
                        and card.evidence_type == "per_category_detected_total"
                        and card.tool == tool
                        and card.category == category
                    ):
                        cited_cards.append(card)
            if cited_cards and all(
                card_scope_text(card) == "external_dataset"
                and card_scene_tier(packet, card) == "unrelated"
                for card in cited_cards
            ):
                reasons.append(f"unrelated_external_only_for_owner: {tool}/{category}")
    return reasons


def _marginal_owner_reasons(
    certificate: Step2DecisionCertificate,
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
) -> list[str]:
    if not _has_composition_fields(certificate):
        return []
    reasons: list[str] = []
    primary = certificate.primary_tool
    attention = set(attention_categories(packet))
    for tool, categories in certificate.tool_categories.items():
        if tool == primary:
            continue
        for category in categories:
            if _is_all_category(category) or category not in attention:
                continue
            marginal_value = category_marginal_value(packet, matrix, category)
            if marginal_value == "low":
                reasons.append(f"low_marginal_owner: {tool}/{category}")
            elif marginal_value == "none":
                reasons.append(f"none_marginal_owner: {tool}/{category}")
    return reasons


def _check_composition_action_legality(
    certificate: Step2DecisionCertificate,
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
) -> list[Failure]:
    if not _has_composition_fields(certificate):
        return []
    action = _selected_action(certificate, matrix)
    if action is None:
        return []

    step1_anchor = packet.primary_attention.primary_tool
    if step1_anchor:
        pass
    elif packet.certification.certified_primary:
        step1_anchor = packet.certification.certified_primary
    elif packet.certification.candidate_set:
        step1_anchor = packet.certification.candidate_set[0]
    elif packet.score_panel.nominal_scores:
        step1_anchor = sorted(packet.score_panel.nominal_scores, key=lambda item: item.rank)[0].tool
    if step1_anchor and certificate.primary_tool != step1_anchor:
        return [
            (
                "PRIMARY_TOOL_NOT_STEP1_ANCHOR",
                "Composition primary_tool must equal the Step1 anchor.",
            )
        ]

    if action.action_type != "PLAN_COMPOSITION" and action.tools and certificate.primary_tool != action.tools[0]:
        return [
            (
                "PRIMARY_TOOL_NOT_ACTION_ANCHOR",
                "Composition primary_tool must equal the first tool in the selected action.",
            )
        ]
    return []


def _check_composition_completeness(
    certificate: Step2DecisionCertificate,
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
) -> list[Failure]:
    if not _has_composition_fields(certificate):
        return []
    action = _selected_action(certificate, matrix)
    if action is None:
        return []

    failures: list[Failure] = []
    primary = certificate.primary_tool
    selected_tools = _selected_tools(certificate)
    packet_primary_attention_categories = (
        set(packet.primary_attention.confirmed_weak_categories)
        | set(packet.primary_attention.low_support_categories)
    )
    if primary and primary not in selected_tools:
        failures.append(
            (
                "PRIMARY_TOOL_NOT_SELECTED",
                "Primary tool must be included in selected_plan.",
            )
        )
    primary_categories = certificate.tool_categories.get(primary or "", [])
    if primary and not any(_is_all_category(category) for category in primary_categories):
        failures.append(
            (
                "PRIMARY_TOOL_MISSING_ALL",
                "Primary tool must own ALL in tool_categories.",
            )
        )

    for tool, categories in certificate.tool_categories.items():
        if tool == primary:
            continue
        if any(_is_all_category(category) for category in categories):
            failures.append(
                (
                    f"COMPLEMENT_TOOL_HAS_ALL:{tool}",
                    "Complement tools must own specific vulnerability categories, not ALL.",
                )
            )

    for tool, categories in certificate.tool_categories.items():
        if tool == primary:
            continue
        for category in categories:
            if _is_all_category(category):
                continue
            if category not in DASP10_CATEGORIES:
                failures.append(
                    (
                        f"UNKNOWN_ASSIGNED_CATEGORY:{category}",
                        f"Assigned category is not a DASP10 category: {category}.",
                    )
                )
                continue
            if packet_primary_attention_categories and category not in packet_primary_attention_categories:
                failures.append(
                    (
                        f"ASSIGNED_CATEGORY_NOT_PRIMARY_ATTENTION:{category}",
                        "Assigned category is not in Step1 primary confirmed-weak or low-support set.",
                    )
                )
            if tool not in selected_tools:
                failures.append(
                    (
                        f"ASSIGNED_TOOL_NOT_SELECTED:{tool}",
                        "Assigned tool is not part of the selected action.",
                    )
                )
            owner_status, _ownership = _owner_pool_status(packet, matrix, category=category, tool=tool)
            if owner_status in {"strong", "weak"}:
                if not _override_failures(certificate, packet, matrix, category=category, tool=tool):
                    continue
                continue
            if owner_status == "none":
                continue
            if not _has_assignment_evidence(
                certificate,
                matrix,
                category=category,
                tool=tool,
            ):
                failures.append(
                    (
                        f"ASSIGNMENT_MISSING_EVIDENCE:{category}:{tool}",
                        "Assigned category must cite recall-coverage or RAG comparison/recommendation evidence.",
                    )
                )
    return failures


def _check_forbidden_claims(certificate: Step2DecisionCertificate) -> list[Failure]:
    failures: list[Failure] = []
    attestation = certificate.forbidden_claims_attestation
    for field_name in (
        "no_target_vulnerability_claim",
        "no_code_semantic_inference",
        "no_precision_from_detected_total",
        "absence_of_findings_not_treated_as_safe",
        "no_unsourced_numeric_gain",
    ):
        if not getattr(attestation, field_name):
            failures.append(
                (
                    f"ATTESTATION_VIOLATED:{field_name}",
                    f"Forbidden-claims attestation is false for {field_name}.",
                )
            )

    for item in _all_claims(certificate):
        claim = item.claim.lower()
        if "target" in claim and any(
            marker in claim
            for marker in ("vulnerable", "vulnerability", "has risk", "is at risk")
        ):
            failures.append(
                (
                    "FORBIDDEN_TARGET_VULNERABILITY_CLAIM",
                    "Claim asserts a target-contract vulnerability status.",
                )
            )
        if (
            ("no findings" in claim or "no vulnerabilities" in claim)
            and ("safe" in claim or "secure" in claim)
        ):
            failures.append(
                (
                    "FORBIDDEN_ABSENCE_MEANS_SAFE",
                    "Claim treats absence of findings as proof of safety.",
                )
            )
        if (
            ("precision" in claim or "f1" in claim)
            and "detected" in claim
            and "total" in claim
        ):
            failures.append(
                (
                    "FORBIDDEN_PRECISION_FROM_DETECTED_TOTAL",
                    "Claim derives precision or F1 from detected/total evidence.",
                )
            )
    return failures


def _verdict(
    status: str,
    certificate: Step2DecisionCertificate,
    failures: list[Failure],
) -> CheckerVerdict:
    verdict = CheckerVerdict(
        status=status,
        checked_action_id=certificate.selected_action_id,
        rule_failures=[code for code, _reason in failures],
        reasons=[reason for _code, reason in failures],
    )
    if status == "REJECT":
        verdict.hard_failures = list(verdict.rule_failures)
    elif status == "REQUEST_REGENERATION":
        verdict.regeneration_reasons = list(verdict.rule_failures)
    return verdict


def check_decision(
    certificate: Step2DecisionCertificate,
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
) -> CheckerVerdict:
    """对 CEGO 决策证书做 4 类合法性审计。"""
    action_failures = _check_action_legality(certificate, packet, matrix)
    evidence_failures = _check_evidence_legality(certificate, matrix)
    completeness_failures = _check_evidence_completeness(certificate, matrix)
    composition_action_failures = _check_composition_action_legality(certificate, packet, matrix)
    ownership_failures = _check_composition_ownership_legality(certificate, packet, matrix)
    category_assignment_failures = _check_category_assignments_against_panel(certificate, matrix)
    rag_override_failures = _check_rag_overrides(certificate, matrix)
    assignment_caveat_failures = _check_assignment_caveats(certificate)
    composition_failures = _check_composition_completeness(certificate, packet, matrix)
    forbidden_failures = _check_forbidden_claims(certificate)
    caveat_reasons = _unrelated_external_caveat_reasons(certificate, packet, matrix)
    marginal_reasons = _marginal_owner_reasons(certificate, packet, matrix)
    advisory_reasons = caveat_reasons + marginal_reasons

    all_failures = (
        action_failures
        + composition_action_failures
        + ownership_failures
        + category_assignment_failures
        + rag_override_failures
        + evidence_failures
        + completeness_failures
        + assignment_caveat_failures
        + composition_failures
        + forbidden_failures
    )
    if not all_failures:
        return CheckerVerdict(
            status="ACCEPT",
            checked_action_id=certificate.selected_action_id,
            reasons=advisory_reasons,
            advisory_reasons=advisory_reasons,
        )
    if (
        action_failures
        or composition_action_failures
        or ownership_failures
        or category_assignment_failures
        or rag_override_failures
        or forbidden_failures
    ):
        verdict = _verdict("REJECT", certificate, all_failures)
    else:
        verdict = _verdict("REQUEST_REGENERATION", certificate, all_failures)
    verdict.reasons.extend(advisory_reasons)
    verdict.advisory_reasons = advisory_reasons
    return verdict
