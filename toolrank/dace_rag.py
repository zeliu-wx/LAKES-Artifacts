"""DACE-RAG: action enumeration, vector retrieval, and evidence matrix construction."""

from __future__ import annotations

import os

from toolrank.assignment_evidence import is_assignment_eligible, is_close_local_margin
from toolrank.category_candidates import category_candidate_tools
from toolrank.ownership_evidence import (
    attention_categories,
    category_marginal_value,
    category_ownership_panel,
)
from toolrank.passage_store import PassageRetriever
from toolrank.precision_gate import candidate_passes_precision_gate
from toolrank.solc_range import version_in_range
from toolrank.schemas_v2 import (
    ActionByEvidenceMatrix,
    ActionEvidence,
    BudgetProfile,
    CandidateAction,
    EvidenceCard,
    EvidenceCardValue,
    EvidenceRef,
    KnowledgeKind,
    NominalToolScore,
    Passage,
    RagOverrideRecord,
    RecallCoverageEntry,
    RelationToOwner,
    Step1EvidencePacket,
    ToolRunSummary,
    ToolTableEntry,
)


__all__ = ["build_action_evidence_matrix"]


_ALERT_ORDER = {"low": 0, "medium": 1, "high": 2}
_SLOTS = ("FOR", "AGAINST", "COMPARE", "GAP")


def _empty_evidence() -> dict[str, list[ActionEvidence]]:
    return {slot: [] for slot in _SLOTS}


def _scene_card_id(tool: str) -> str:
    return f"ev_scene_{tool}"


def _rcov_card_id(tool: str, category: str) -> str:
    return f"ev_rcov_{tool}_{category}"


def _runtime_card_id(tool: str) -> str:
    return f"ev_runtime_{tool}"


def _score_by_tool(packet: Step1EvidencePacket) -> dict[str, NominalToolScore]:
    return {score.tool: score for score in packet.score_panel.nominal_scores}


def _tool_table_by_tool(packet: Step1EvidencePacket) -> dict[str, ToolTableEntry]:
    return {entry.tool: entry for entry in packet.tool_table}


def _feasible_tools(packet: Step1EvidencePacket) -> set[str]:
    return {entry.tool for entry in packet.tool_table if entry.feasible}


def _rcov_by_tool_category(packet: Step1EvidencePacket) -> dict[tuple[str, str], RecallCoverageEntry]:
    return {
        (entry.tool, entry.category): entry
        for entry in packet.recall_coverage.matrix
    }


def _dataset_rows_by_category(packet: Step1EvidencePacket) -> dict[str, list]:
    rows: dict[str, list] = {}
    for row in packet.performance_db_view:
        rows.setdefault(row.category, []).append(row)
    return rows


def _rcov_entries_for_tool(packet: Step1EvidencePacket, tool: str) -> list[RecallCoverageEntry]:
    return [entry for entry in packet.recall_coverage.matrix if entry.tool == tool]


def _alert_max(tools: list[str], table: dict[str, ToolTableEntry]) -> str:
    alert = "low"
    for tool in tools:
        risk = table[tool].tool_cost.alert_risk
        if _ALERT_ORDER[risk] > _ALERT_ORDER[alert]:
            alert = risk
    return alert


def _estimated_budget(tools: list[str], table: dict[str, ToolTableEntry]) -> BudgetProfile:
    if not tools:
        return BudgetProfile(tool_slots=0, runtime_cap_minutes=0.0, alert_cap="low")
    runtime = sum(table[tool].tool_cost.expected_runtime_minutes or 0.0 for tool in tools)
    return BudgetProfile(
        tool_slots=len(tools),
        runtime_cap_minutes=runtime,
        alert_cap=_alert_max(tools, table),
    )


def _legality(estimated: BudgetProfile, budget: BudgetProfile) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if estimated.tool_slots > budget.tool_slots:
        reasons.append("EXCEEDS_TOOL_SLOTS")
    if estimated.runtime_cap_minutes > budget.runtime_cap_minutes:
        reasons.append("EXCEEDS_RUNTIME_CAP")
    return not reasons, reasons


def _make_action(
    action_id: str,
    action_type: str,
    tools: list[str],
    evidence: dict[str, list[ActionEvidence]],
    budget: BudgetProfile,
    table: dict[str, ToolTableEntry],
) -> CandidateAction:
    estimated = _estimated_budget(tools, table)
    legal, reasons = _legality(estimated, budget)
    return CandidateAction(
        action_id=action_id,
        action_type=action_type,
        tools=tools,
        evidence=evidence,
        estimated_budget=estimated,
        legal=legal,
        legality_reasons=reasons,
    )


def _scene_metric_semantics(score: NominalToolScore) -> str:
    if score.F1_scene is not None:
        return "historical_f1"
    if score.R_scene is not None:
        return "historical_recall"
    return "qualitative"


def _build_evidence_cards(packet: Step1EvidencePacket) -> list[EvidenceCard]:
    cards: list[EvidenceCard] = []
    for score in packet.score_panel.nominal_scores:
        evidence_id = _scene_card_id(score.tool)
        cards.append(
            EvidenceCard(
                evidence_id=evidence_id,
                source=EvidenceRef(
                    evidence_id=evidence_id,
                    source_type="step1_field",
                    field_path="score_panel.nominal_scores",
                ),
                evidence_type="scene_metric",
                tool=score.tool,
                metric_semantics=_scene_metric_semantics(score),
                value=EvidenceCardValue(rate=score.S_scene),
            )
        )

    for entry in packet.recall_coverage.matrix:
        if entry.R_hat is None:
            continue
        evidence_id = _rcov_card_id(entry.tool, entry.category)
        cards.append(
            EvidenceCard(
                evidence_id=evidence_id,
                source=EvidenceRef(
                    evidence_id=evidence_id,
                    source_type="step1_field",
                    field_path="recall_coverage.matrix",
                ),
                evidence_type="per_category_detected_total",
                tool=entry.tool,
                category=entry.category,
                metric_semantics="recall_side_detection_rate",
                value=EvidenceCardValue(
                    detected=entry.detected,
                    total=entry.total,
                    rate=entry.R_hat,
                ),
                )
            )

    for row in packet.performance_db_view:
        cards.append(
            EvidenceCard(
                evidence_id=row.evidence_id,
                source=EvidenceRef(
                    evidence_id=row.evidence_id,
                    source_type="step1_field",
                    field_path="performance_db_view",
                ),
                evidence_type="per_category_detected_total",
                tool=row.tool,
                category=row.category,
                metric_semantics="recall_side_detection_rate",
                value=EvidenceCardValue(
                    detected=row.detected,
                    total=row.total,
                    rate=row.R_hat,
                ),
                scope={
                    "source_id": row.source_id,
                    "dataset_name": row.dataset_name,
                },
            )
        )

    for entry in packet.tool_table:
        evidence_id = _runtime_card_id(entry.tool)
        cards.append(
            EvidenceCard(
                evidence_id=evidence_id,
                source=EvidenceRef(
                    evidence_id=evidence_id,
                    source_type="step1_field",
                    field_path="tool_table",
                ),
                evidence_type="runtime_cost",
                tool=entry.tool,
                metric_semantics="runtime",
                value=EvidenceCardValue(value=entry.tool_cost.expected_runtime_minutes),
            )
        )

    for row in packet.tool_overall_metrics:
        cards.append(
            EvidenceCard(
                evidence_id=row.evidence_id,
                source=EvidenceRef(
                    evidence_id=row.evidence_id,
                    source_type="step1_field",
                    field_path="tool_overall_metrics",
                ),
                evidence_type="tool_scope",
                tool=row.tool,
                metric_semantics="historical_f1",
                value=EvidenceCardValue(value=row.f1),
                scope={
                    "source_id": row.source_id,
                    "dataset_name": row.dataset_name,
                    "precision": row.precision,
                    "recall": row.recall,
                    "f1": row.f1,
                    "execution_time_avg": row.execution_time_avg,
                },
            )
        )
    return cards


def _append_scene_claim(
    evidence: dict[str, list[ActionEvidence]],
    slot: str,
    score: NominalToolScore,
) -> None:
    evidence[slot].append(
        ActionEvidence(
            claim=f"{score.tool} ranks #{score.rank} with S_scene={score.S_scene:.3f}",
            evidence_refs=[_scene_card_id(score.tool)],
        )
    )


def _append_single_tool_rcov(
    packet: Step1EvidencePacket,
    evidence: dict[str, list[ActionEvidence]],
    tool: str,
) -> None:
    for entry in _rcov_entries_for_tool(packet, tool):
        if entry.R_hat is None:
            continue
        if entry.support_level in {"strong", "medium"}:
            evidence["FOR"].append(
                ActionEvidence(
                    claim=(
                        f"{tool} has {entry.support_level} coverage in {entry.category} "
                        f"(R_hat={entry.R_hat:.2f})"
                    ),
                    evidence_refs=[_rcov_card_id(tool, entry.category)],
                )
            )
        elif entry.support_level == "weak":
            evidence["AGAINST"].append(
                ActionEvidence(
                    claim=f"{tool} has weak coverage in {entry.category} (R_hat={entry.R_hat:.2f})",
                    evidence_refs=[_rcov_card_id(tool, entry.category)],
                )
            )


def _append_stress_against(
    packet: Step1EvidencePacket,
    evidence: dict[str, list[ActionEvidence]],
    tool: str,
) -> None:
    mixtures = {
        "local": packet.score_panel.stress_rankings.local,
        "global": packet.score_panel.stress_rankings.global_,
        "uniform_supported": packet.score_panel.stress_rankings.uniform_supported,
    }
    for mixture, ranking in mixtures.items():
        if ranking and ranking[0] != tool:
            evidence["AGAINST"].append(
                ActionEvidence(
                    claim=f"{tool} is not top-1 under {mixture} stress mixture",
                    evidence_refs=[],
                )
            )


def _single_tool_evidence(
    packet: Step1EvidencePacket,
    tool: str,
    *,
    robust_single: bool,
) -> dict[str, list[ActionEvidence]]:
    evidence = _empty_evidence()
    scores = _score_by_tool(packet)
    score = scores.get(tool)
    if score is None:
        return evidence

    _append_scene_claim(evidence, "FOR", score)
    _append_single_tool_rcov(packet, evidence, tool)
    _append_stress_against(packet, evidence, tool)

    top_score = min(packet.score_panel.nominal_scores, key=lambda item: item.rank, default=None)
    if robust_single and top_score is not None and top_score.tool != tool:
        evidence["COMPARE"].append(
            ActionEvidence(
                claim=(
                    f"Nominal top-1 is {top_score.tool} (S_scene={top_score.S_scene:.3f}) "
                    f"vs {tool} (S_scene={score.S_scene:.3f})"
                ),
                evidence_refs=[_scene_card_id(top_score.tool), _scene_card_id(tool)],
            )
        )

    if score.evidence_level != "local_strong":
        evidence["GAP"].append(
            ActionEvidence(
                claim=(
                    f"Evidence level for {tool} is {score.evidence_level} - "
                    "limited local benchmarks"
                ),
                evidence_refs=[_scene_card_id(tool)],
            )
        )
    return evidence


def _support_quality(entry: RecallCoverageEntry | None) -> str:
    if entry is None:
        return "none"
    if entry.total is None:
        return "unknown"
    if entry.total <= 0:
        return "none"
    if entry.total < 10:
        return "low"
    if entry.total < 30:
        return "medium"
    return "high"


def _assignable_categories(packet: Step1EvidencePacket) -> list[str]:
    categories = (
        list(packet.primary_attention.confirmed_weak_categories)
        + list(packet.primary_attention.low_support_categories)
    )
    return list(dict.fromkeys(categories))


def _is_better_category_candidate(
    entry: RecallCoverageEntry | None,
    primary_entry: RecallCoverageEntry | None,
    *,
    packet: Step1EvidencePacket | None = None,
    tool: str | None = None,
) -> bool:
    if entry is None or entry.R_hat is None or entry.R_hat <= 0:
        return False
    if packet is not None and tool is not None and not candidate_passes_precision_gate(packet, tool):
        return False
    if not is_assignment_eligible(entry.detected, entry.total, entry.R_hat):
        return False
    if entry.support_level not in {"medium", "strong"}:
        return False
    primary_rate = primary_entry.R_hat if primary_entry and primary_entry.R_hat is not None else 0.0
    return entry.R_hat > primary_rate


def _assignment_candidate_sort_key(item: tuple[str, RecallCoverageEntry]) -> tuple[float, int, str]:
    tool, entry = item
    return (-(entry.R_hat or -1.0), -(entry.total or 0), tool)


def _composition_planning_evidence(
    packet: Step1EvidencePacket,
    primary_tool: str,
) -> dict[str, list[ActionEvidence]]:
    evidence = _empty_evidence()
    scores = _score_by_tool(packet)
    if primary_tool in scores:
        _append_scene_claim(evidence, "FOR", scores[primary_tool])

    rcov_by_key = _rcov_by_tool_category(packet)
    dataset_rows = _dataset_rows_by_category(packet)
    feasible = _feasible_tools(packet)
    low_support_categories = set(packet.primary_attention.low_support_categories)
    for category in _assignable_categories(packet):
        primary_entry = rcov_by_key.get((primary_tool, category))
        if category in low_support_categories:
            rows = [
                row
                for row in dataset_rows.get(category, [])
                if is_assignment_eligible(row.detected, row.total, row.R_hat)
                and candidate_passes_precision_gate(packet, row.tool)
            ]
            if rows:
                evidence["FOR"].append(
                    ActionEvidence(
                        claim=(
                            f"category={category} primary local support insufficient (total<10); "
                            "cross-dataset evidence rows available for LLM judgment"
                        ),
                        evidence_refs=[row.evidence_id for row in rows[:12]],
                    )
                )
            else:
                evidence["GAP"].append(
                    ActionEvidence(
                        claim=(
                            f"category={category} primary local support insufficient (total<10); "
                            "no cross-dataset assignment-eligible evidence; rely on RAG Tool Knowledge if available"
                        ),
                        evidence_refs=[],
                    )
                )
            continue

        local_quality = _support_quality(primary_entry)
        refs = []
        if primary_entry is not None and primary_entry.R_hat is not None:
            refs.append(_rcov_card_id(primary_tool, category))
        stronger_entries: list[tuple[str, RecallCoverageEntry]] = []
        for tool in sorted(feasible):
            if tool == primary_tool:
                continue
            entry = rcov_by_key.get((tool, category))
            if not _is_better_category_candidate(entry, primary_entry, packet=packet, tool=tool):
                continue
            stronger_entries.append((tool, entry))
        stronger_entries.sort(key=_assignment_candidate_sort_key)
        if stronger_entries:
            refs.extend(_rcov_card_id(tool, category) for tool, _entry in stronger_entries)
            preferred_tool, preferred_entry = stronger_entries[0]
            stronger_tools = [
                f"{tool}={entry.R_hat:.2f}" for tool, entry in stronger_entries
            ]
            close_note = ""
            if len(stronger_entries) >= 2 and is_close_local_margin(
                preferred_entry.R_hat,
                stronger_entries[1][1].R_hat,
            ):
                close_note = (
                    " local candidates are close; compare cross-dataset performance evidence "
                    "and RAG evidence before assigning owner;"
                )
            evidence["FOR"].append(
                ActionEvidence(
                    claim=(
                        f"{category} has feasible complement evidence for primary "
                        f"{primary_tool}; preferred local evidence is "
                        f"{preferred_tool}={preferred_entry.R_hat:.2f};"
                        f"{close_note} "
                        f"candidates: {', '.join(stronger_tools)}"
                    ),
                    evidence_refs=refs,
                )
            )
        else:
            evidence["GAP"].append(
                ActionEvidence(
                    claim=(
                        f"local evidence for {category} is {local_quality}; "
                        f"no local feasible complement has detected/total support."
                    ),
                    evidence_refs=refs,
                )
            )

        if local_quality in {"none", "low", "unknown"}:
            rows = [
                row
                for row in dataset_rows.get(category, [])
                if is_assignment_eligible(row.detected, row.total, row.R_hat)
                and candidate_passes_precision_gate(packet, row.tool)
            ]
            if rows:
                evidence["FOR"].append(
                    ActionEvidence(
                        claim=(
                            f"local evidence for {category} is {local_quality}; "
                            "cross-dataset performance evidence has assignment-eligible refs for LLM judgment."
                        ),
                        evidence_refs=[row.evidence_id for row in rows[:12]],
                    )
                )
            else:
                evidence["GAP"].append(
                    ActionEvidence(
                        claim=(
                            f"local evidence for {category} is {local_quality}; "
                            "cross-dataset performance view has no assignment-eligible evidence."
                        ),
                        evidence_refs=[],
                    )
                )

    evidence["AGAINST"].append(
        ActionEvidence(
            claim="The model must account for each selected tool's runtime and alert burden.",
            evidence_refs=[_runtime_card_id(entry.tool) for entry in packet.tool_table if entry.feasible],
        )
    )
    evidence["AGAINST"].append(
        ActionEvidence(
            claim="RAG passages and recall hints do not prove union recall for a selected tool set.",
            evidence_refs=[],
        )
    )
    evidence["COMPARE"].append(
        ActionEvidence(
            claim=f"The LLM should compare feasible tools against weak categories of primary {primary_tool}.",
            evidence_refs=[],
        )
    )
    evidence["GAP"].append(
        ActionEvidence(
            claim="Cross-tool overlap and union recall are not identifiable from current evidence",
            evidence_refs=[],
        )
    )
    return evidence


def _stop_evidence(packet: Step1EvidencePacket) -> dict[str, list[ActionEvidence]]:
    evidence = _empty_evidence()
    evidence["FOR"].append(
        ActionEvidence(
            claim="Stop scheduling under current evidence and budget while unresolved gaps remain explicit",
            evidence_refs=[],
        )
    )
    weak_categories = sorted(
        {
            category
            for categories in packet.recall_coverage.weak_categories_by_tool.values()
            for category in categories
        }
    )
    if weak_categories:
        evidence["AGAINST"].append(
            ActionEvidence(
                claim=f"Uncovered weak categories remain: {', '.join(weak_categories)}",
                evidence_refs=[],
            )
        )
    return evidence


def _focus_tools(packet: Step1EvidencePacket, feasible: set[str]) -> list[str]:
    tools: list[str] = []
    for item in packet.dace_rag_focus:
        if item.tool in feasible and item.tool not in tools:
            tools.append(item.tool)
    return tools


def _add_run_primary(
    actions: list[CandidateAction],
    packet: Step1EvidencePacket,
    budget: BudgetProfile,
    table: dict[str, ToolTableEntry],
    feasible: set[str],
) -> str | None:
    tool = packet.primary_attention.primary_tool or packet.certification.certified_primary
    if (
        packet.certification.status == "certified_primary"
        and tool
        and tool in feasible
        and budget.tool_slots >= 1
    ):
        actions.append(
            _make_action(
                f"run_primary_{tool}",
                "RUN_PRIMARY",
                [tool],
                _single_tool_evidence(packet, tool, robust_single=False),
                budget,
                table,
            )
        )
        return tool
    return None


def _primary_candidate_tool(packet: Step1EvidencePacket, feasible: set[str]) -> str | None:
    if packet.primary_attention.primary_tool in feasible:
        return packet.primary_attention.primary_tool
    if packet.certification.certified_primary in feasible:
        return packet.certification.certified_primary
    for tool in packet.certification.candidate_set or []:
        if tool in feasible:
            return tool
    for score in sorted(packet.score_panel.nominal_scores, key=lambda item: item.rank):
        if score.tool in feasible:
            return score.tool
    return None


def _add_robust_single(
    actions: list[CandidateAction],
    packet: Step1EvidencePacket,
    budget: BudgetProfile,
    table: dict[str, ToolTableEntry],
    primary_tool: str | None,
) -> None:
    if primary_tool is None:
        return
    actions.append(
        _make_action(
            f"run_robust_single_{primary_tool}",
            "RUN_ROBUST_SINGLE",
            [primary_tool],
            _single_tool_evidence(packet, primary_tool, robust_single=True),
            budget,
            table,
        )
    )


def _add_composition_plan_action(
    actions: list[CandidateAction],
    packet: Step1EvidencePacket,
    budget: BudgetProfile,
    primary_tool: str | None,
    evidence_cards: list[EvidenceCard],
) -> None:
    if primary_tool is None or budget.tool_slots < 1:
        return
    marginal_matrix = ActionByEvidenceMatrix(
        budget_profile=budget,
        actions=[],
        evidence_cards=evidence_cards,
    )
    complement_slots = sum(
        1
        for category in attention_categories(packet)
        if category_marginal_value(packet, marginal_matrix, category) in {"high", "medium"}
    )
    estimated_budget = BudgetProfile(
        tool_slots=min(budget.tool_slots, 1 + complement_slots),
        runtime_cap_minutes=budget.runtime_cap_minutes,
        alert_cap=budget.alert_cap,
    )
    actions.append(
        CandidateAction(
            action_id="plan_tool_composition",
            action_type="PLAN_COMPOSITION",
            tools=[],
            evidence=_composition_planning_evidence(packet, primary_tool),
            estimated_budget=estimated_budget,
            legal=True,
            legality_reasons=[],
        )
    )


def _add_continue_hedges(
    actions: list[CandidateAction],
    packet: Step1EvidencePacket,
    budget: BudgetProfile,
    table: dict[str, ToolTableEntry],
    feasible: set[str],
    run_history: list[ToolRunSummary] | None,
) -> None:
    if not run_history:
        return
    already_run = {item.tool for item in run_history if item.run_status != "NOT_RUN"}
    for tool in _focus_tools(packet, feasible):
        if tool in already_run:
            continue
        actions.append(
            _make_action(
                f"continue_hedge_{tool}",
                "CONTINUE_HEDGE",
                [tool],
                _single_tool_evidence(packet, tool, robust_single=True),
                budget,
                table,
            )
        )


_RELATION_TO_DECISION_ROLE: dict[RelationToOwner, str] = {
    "supports_owner": "support",
    "owner_complements": "support",
    "opposes_owner": "oppose",
    "owner_ineligible": "constraint",
    "owner_stronger": "compare",
    "owner_weaker": "compare",
    "evidence_gap": "gap",
}

_KIND_TO_METRIC_SEMANTICS: dict[KnowledgeKind, str] = {
    "category_capability": "recall_side_detection_rate",
    "tool_complementarity": "recall_side_detection_rate",
    "fp_precision_risk": "historical_precision",
    "failure_mode": "qualitative",
    "hard_scheduling_rule": "scope",
}

_PASSAGE_TO_EVIDENCE_RELIABILITY: dict[str, str] = {
    "peer_reviewed": "peer_reviewed",
    "artifact": "artifact",
    "manual_curated": "manual_curated",
    "official_tool_doc": "documentation",
    "maintainer_issue": "documentation",
    "internal_eval": "experiment_log",
    "community_report": "unknown",
}

_TIER_TO_EXTRACTION_CONFIDENCE: dict[str, str] = {
    "hard": "high",
    "medium": "medium",
    "weak": "low",
}


def _passage_decision_role(passage: Passage) -> str:
    return _RELATION_TO_DECISION_ROLE[passage.relation_to_owner]


def _passage_metric_semantics(passage: Passage) -> str:
    return _KIND_TO_METRIC_SEMANTICS[passage.knowledge_kind]


def _passage_source_reliability(passage: Passage) -> str:
    return _PASSAGE_TO_EVIDENCE_RELIABILITY[passage.source_reliability]


def _passage_paper_id(passage: Passage) -> str | None:
    raw = passage.source_id
    if ":" in raw:
        prefix, _, rest = raw.partition(":")
        if prefix in {"paper", "issue", "artifact"}:
            return rest or None
    return None


def _passage_to_evidence_card(passage: Passage) -> EvidenceCard:
    """Project an owner-oriented Passage into an EvidenceCard."""

    return EvidenceCard(
        evidence_id=passage.passage_id,
        source=EvidenceRef(
            evidence_id=passage.passage_id,
            source_type="rag_passage",
            paper_id=_passage_paper_id(passage),
            field_path=passage.source_id,
            extraction_confidence=_TIER_TO_EXTRACTION_CONFIDENCE[passage.evidence_tier],
        ),
        evidence_type="rag_passage",
        tool=passage.owner_tool,
        category=passage.category,
        metric_semantics=_passage_metric_semantics(passage),
        value=None,
        scope={
            "owner_tool": passage.owner_tool,
            "category": passage.category,
            "knowledge_kind": passage.knowledge_kind,
            "relation_to_owner": passage.relation_to_owner,
            "applicability_tags": list(passage.applicability_tags),
            "counterpart_tool_ids": list(passage.counterpart_tool_ids),
            "action_scope": list(passage.action_scope),
            "evidence_basis": passage.evidence_basis,
            "evidence_tier": passage.evidence_tier,
            "claim_text": passage.claim_text,
            "source_excerpt": passage.source_excerpt,
        },
        limitations=[passage.limitations_text] if passage.limitations_text else [],
        aggregation_level="paper_level",
        decision_role=_passage_decision_role(passage),
        source_reliability=_passage_source_reliability(passage),
    )


def _slot_for_passage(passage: Passage) -> str:
    role = _passage_decision_role(passage)
    if role in {"oppose", "constraint"}:
        return "AGAINST"
    if role == "support":
        return "FOR"
    if role == "compare":
        return "COMPARE"
    return "GAP"


def _inject_retrieved_evidence(
    evidence: dict[str, list[ActionEvidence]],
    passages: list[Passage],
) -> None:
    for passage in passages:
        slot = _slot_for_passage(passage)
        evidence[slot].append(
            ActionEvidence(
                claim=passage.claim_text[:200],
                evidence_refs=[passage.passage_id],
            )
        )


def _scene_text(packet: Step1EvidencePacket) -> str:
    features = packet.target_contract.get("features")
    if not isinstance(features, dict):
        return ""
    pieces: list[str] = []
    for key, value in sorted(features.items()):
        if isinstance(value, (str, int, float, bool)):
            pieces.append(f"{key}={value}")
        elif isinstance(value, list):
            vals = [str(item) for item in value if isinstance(item, (str, int, float, bool))]
            if vals:
                pieces.append(f"{key}={','.join(vals[:5])}")
        if len(pieces) >= 5:
            break
    return " ".join(pieces)


def _build_scheduling_queries(
    packet: Step1EvidencePacket,
    feasible_tools: set[str],
    primary_tool: str,
    category: str,
    top_k: int,
) -> list[dict]:
    scene_text = _scene_text(packet)
    candidate_tools = [
        tool for tool in category_candidate_tools(packet, category, limit=3)
        if tool in feasible_tools
    ]
    if primary_tool not in candidate_tools and primary_tool in feasible_tools:
        candidate_tools.insert(0, primary_tool)

    queries: list[dict] = []
    queries.append({
        "query_text": f"{primary_tool} {category} complement owner scheduling {scene_text}".strip(),
        "tool_ids": [primary_tool],
        "categories": [category],
        "knowledge_kinds": ["tool_complementarity", "hard_scheduling_rule"],
        "top_k": top_k,
    })
    for tool in candidate_tools:
        queries.append({
            "query_text": f"{tool} {category} analyzer capability risk scheduling {scene_text}".strip(),
            "tool_ids": [tool],
            "categories": [category],
            "knowledge_kinds": [
                "category_capability",
                "fp_precision_risk",
                "failure_mode",
                "hard_scheduling_rule",
            ],
            "top_k": top_k,
        })
        queries.append({
            "query_text": f"{tool} {category} false positive precision risk scheduling",
            "tool_ids": [tool],
            "categories": [category],
            "knowledge_kinds": ["fp_precision_risk", "failure_mode"],
            "top_k": 1,
        })
    return queries


def _retrieve_for_tool_space(
    retriever: PassageRetriever,
    packet: Step1EvidencePacket,
    feasible: set[str],
    primary_tool: str | None,
    top_k: int = 3,
) -> list[Passage]:
    return _retrieve_for_evidence_needs(
        retriever,
        packet,
        feasible,
        primary_tool,
        top_k=top_k,
    )


def _retrieve_for_evidence_needs(
    retriever: PassageRetriever,
    packet: Step1EvidencePacket,
    feasible: set[str],
    primary_tool: str | None,
    top_k: int = 1,
) -> list[Passage]:
    weak_categories = _assignable_categories(packet)
    if not primary_tool or not weak_categories:
        return []

    queries: list[dict] = []
    for category in weak_categories:
        queries.extend(
            _build_scheduling_queries(
                packet,
                feasible_tools=feasible,
                primary_tool=primary_tool,
                category=category,
                top_k=top_k,
            )
        )

    try:
        all_results = retriever.batch_search_structured(queries)
    except Exception:
        if os.getenv("TOOLRANK_RAG_STRICT_ERRORS", "").strip().lower() in {"1", "true", "yes"}:
            raise
        all_results = [[] for _ in queries]

    passages: list[Passage] = []
    seen: set[str] = set()
    for results in all_results:
        for passage, _score in results:
            if passage.passage_id in seen:
                continue
            if not _passage_applies_to_target(packet, passage):
                continue
            passages.append(passage)
            seen.add(passage.passage_id)
    return passages


def _target_solidity_version(packet: Step1EvidencePacket) -> str | None:
    features = packet.target_contract.get("features")
    if isinstance(features, dict):
        value = features.get("primary_solidity_version")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = packet.target_contract.get("solidity_version")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _passage_applies_to_target(packet: Step1EvidencePacket, passage: Passage) -> bool:
    solc_tags = [
        tag.removeprefix("solc:")
        for tag in passage.applicability_tags
        if tag.startswith("solc:")
    ]
    if not solc_tags:
        return True
    version = _target_solidity_version(packet)
    return bool(version) and any(version_in_range(version, tag) for tag in solc_tags)


def build_action_evidence_matrix(
    packet: Step1EvidencePacket,
    budget: BudgetProfile,
    run_history: list[ToolRunSummary] | None = None,
    retriever: PassageRetriever | None = None,
    rag_top_k: int = 3,
) -> ActionByEvidenceMatrix:
    """顶层入口。枚举动作 → 向量检索 → 生成证据卡 → 填充证据槽 → 组装矩阵。"""
    table = _tool_table_by_tool(packet)
    feasible = _feasible_tools(packet)
    actions: list[CandidateAction] = []
    evidence_cards = _build_evidence_cards(packet)

    certified_tool = _add_run_primary(actions, packet, budget, table, feasible)
    primary_tool = certified_tool or _primary_candidate_tool(packet, feasible)
    _add_robust_single(actions, packet, budget, table, primary_tool)
    _add_composition_plan_action(actions, packet, budget, primary_tool, evidence_cards)
    _add_continue_hedges(actions, packet, budget, table, feasible, run_history)

    actions.append(
        _make_action(
            "stop_with_gaps",
            "STOP",
            [],
            _stop_evidence(packet),
            budget,
            table,
        )
    )

    rag_cards: list[EvidenceCard] = []
    if retriever is not None:
        seen_passages: set[str] = set()
        plan_action = next((action for action in actions if action.action_id == "plan_tool_composition"), None)
        if plan_action is not None:
            passages = _retrieve_for_tool_space(
                retriever,
                packet,
                feasible,
                primary_tool,
                top_k=rag_top_k,
            )
            _inject_retrieved_evidence(plan_action.evidence, passages)
            for passage in passages:
                if passage.passage_id not in seen_passages:
                    rag_cards.append(_passage_to_evidence_card(passage))
                    seen_passages.add(passage.passage_id)
            # Always surface hard-tier passages whose category overlaps any
            # weak/low_support category, regardless of which candidate tools
            # the per-tool retrieval queries asked about. Hard evidence is
            # authoritative and must not be missed because of a narrow
            # candidate_tools selection.
            attention_category_set = set(_assignable_categories(packet))
            if attention_category_set:
                for passage in getattr(retriever, "_passages", []):
                    if passage.evidence_tier != "hard":
                        continue
                    if passage.category not in attention_category_set:
                        continue
                    if not _passage_applies_to_target(packet, passage):
                        continue
                    if passage.passage_id in seen_passages:
                        continue
                    rag_cards.append(_passage_to_evidence_card(passage))
                    seen_passages.add(passage.passage_id)
    all_cards = evidence_cards + rag_cards
    matrix = ActionByEvidenceMatrix(
        budget_profile=budget,
        actions=actions,
        evidence_cards=all_cards,
    )
    ownership_panel = {
        category: category_ownership_panel(packet, matrix, category)
        for category in attention_categories(packet)
    }
    override_panel = {
        category: RagOverrideRecord(
            category=category,
            applied=False,
            from_owner=panel.preferred_owner,
            to_owner=None,
            for_refs=[],
            against_refs=panel.override_refs,
            compare_refs=[],
        )
        for category, panel in ownership_panel.items()
        if panel.rag_override_eligible or panel.override_refs
    }
    gap_categories = [
        category for category, panel in ownership_panel.items()
        if panel.assignment_status in {"gap", "stop_with_gap"}
    ]
    return ActionByEvidenceMatrix(
        budget_profile=budget,
        actions=actions,
        evidence_cards=all_cards,
        ownership_panel=ownership_panel,
        override_panel=override_panel,
        gap_categories=gap_categories,
    )
