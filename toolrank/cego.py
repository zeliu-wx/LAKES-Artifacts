"""CEGO: Constrained Evidence-Grounded Orchestration LLM decision protocol."""

from __future__ import annotations

import json

from pydantic import ValidationError

from toolrank.assignment_evidence import (
    count_text,
    is_assignment_eligible,
    is_close_local_margin,
    rate_text,
)
from toolrank.category_candidates import category_candidate_tools as _category_candidate_tools
from toolrank.evidence_packet import (
    LOW_SUPPORT_TOTAL,
    TOP_SCENE_CLEAR_WEAK_GAP,
    WEAK_RATE_THRESHOLD,
)
from toolrank.openai_compat import (
    DEFAULT_OPENAI_MODEL,
    OpenAICompatClient,
    OpenAICompatError,
    create_json_chat_completion,
)
from toolrank.precision_gate import candidate_passes_precision_gate, top_scene_source_ids
from toolrank.ownership_evidence import (
    OwnershipCandidate,
    attention_categories,
    card_scene_priority,
    card_scene_tier,
    card_scope_text,
    category_ownership_evidence,
    category_group,
    category_marginal_value,
    near_scene_strong_candidates,
    rcov_evidence_id,
    scheduling_evidence_priority,
    unrelated_strong_candidates,
)
from toolrank.schemas_v2 import (
    ActionByEvidenceMatrix,
    ActionEvidenceBlock,
    ActionEvidenceClaim,
    BudgetProfile,
    BudgetUsage,
    CandidateAction,
    CategoryAssignment,
    EvidenceCard,
    ForbiddenClaimsAttestation,
    RagOverrideRecord,
    SelectedToolEntry,
    Step1EvidencePacket,
    Step2DecisionCertificate,
    ToolRunSummary,
)


class CegoError(RuntimeError):
    """Raised when CEGO cannot produce a valid decision certificate."""


_STEP1_REF_PREFIXES = {
    "Certification.": "step1.certification.",
    "Stress Rankings.": "step1.score_panel.stress_rankings.",
    "Category Diagnostics.": "step1.category_diagnostics.",
}

_NON_EVIDENCE_REF_TAGS = {"unrelated_external_only"}


def _canonicalize_evidence_ref(ref: str) -> str:
    if ref.startswith("ev_") or ref.startswith("step1."):
        return ref
    for old_prefix, new_prefix in _STEP1_REF_PREFIXES.items():
        if ref.startswith(old_prefix):
            return new_prefix + ref[len(old_prefix) :]
    return ref


def _claim_evidence_refs(refs: list) -> list:
    cleaned: list = []
    for ref in refs:
        if not isinstance(ref, str):
            cleaned.append(ref)
            continue
        canonical_ref = _canonicalize_evidence_ref(ref)
        if canonical_ref in _NON_EVIDENCE_REF_TAGS:
            continue
        cleaned.append(canonical_ref)
    return cleaned


def _build_system_prompt() -> str:
    return """You are a smart-contract security tool scheduler.
Your job: choose a legal action envelope and plan the selected tools.

Rules:
1. selected_action_id MUST be one of the provided legal action IDs.
2. Every claim in your response MUST reference an evidence_id shown in the prompt or an exact step1.* ref shown in the prompt.
3. You MUST NOT claim that the target contract has or does not have any vulnerability.
4. You MUST NOT infer vulnerability types from function names or code semantics.
5. You MUST NOT interpret detected/total as precision or F1.
6. You MUST NOT treat absence of findings as proof of safety.
7. You MUST NOT cite numeric performance gains without a sourced evidence card.
8. Step1 anchor is the primary tool. You MUST NOT choose a different primary tool.
9. ALL is only valid for the primary tool. Complement tools MUST be assigned specific vulnerability categories.
10. tool_categories maps the primary tool to ["ALL"] and each complement tool to assigned vulnerability categories.
11. Use exact lower-case category ids from the prompt's step1.primary_attention.confirmed_weak_categories and step1.primary_attention.low_support_categories lines. Do not uppercase or rename categories.
12. RAG passages and cross-dataset performance refs are supporting evidence for your decision, not pre-applied constraints.
13. For PLAN_COMPOSITION, selected_tools may include any feasible tools within budget; RAG does not lock candidates or combinations.
14. Use matched dataset knowledge and RAG together to assign categories.
15. selected_tools MUST include the Step1 anchor primary tool for PLAN_COMPOSITION.
16. Complement tools may only own categories listed under step1.primary_attention.confirmed_weak_categories or step1.primary_attention.low_support_categories.
17. Only categories listed under Assignment-Eligible Evidence (including weak_candidates when rag_override_eligible=true) may appear in tool_categories for complement tools.
17a. If an Assignment-Eligible Evidence line shows top_scene_primary_observation=primary_top_scene_no_clear_weak_gap_no_complement_needed, do not assign a complement owner for that category. Leave it covered by the primary ALL envelope unless a hard RAG override target is explicitly shown.
18. Prefer the category's preferred= tool under Assignment-Eligible Evidence.
    Owner MAY be replaced with any other candidate (strong-eligible or
    weak-eligible) only when rag_override_eligible=true for that category and
    the new owner appears in override_targets. In that case:
    a. against_claims MUST contain at least one claim whose evidence_refs include
       at least one of the override_refs and the preferred tool's recall-coverage ref.
    b. for_claims MUST contain at least one claim whose evidence_refs include the
       new owner's recall-coverage ref or an external dataset ref for that tool/category.
    c. for_claims MUST contain at least one claim whose evidence_refs include
       a RAG pro ref supporting the new owner for that tool/category (drawn
       from override_targets).
    d. The new owner MUST appear in override_targets. A tool whose only
       rate > 0 evidence comes from unrelated datasets cannot be an
       override target.
    Otherwise, owner MUST come from the strong-eligible candidates (preferred
    by default; runtime may only break ties).
19. Assignment-eligible evidence has three tiers of descending importance:
    (a) local evidence (recall-coverage rows from the primary matched dataset),
    (b) near_scene evidence (rows from scene_pool neighbors ranked 2..N),
    (c) unrelated evidence (rows from datasets outside scene_pool).
    If local evidence contains a strong-eligible candidate, that candidate is
    the preferred owner; near_scene and unrelated rows cannot override it
    unless the RAG override pathway (rule 18) is satisfied. If local has no
    strong-eligible candidate, use near_scene; if near_scene also has none,
    fall back to unrelated. Override targets (rule 18) must have rate > 0 on
    a local or near_scene row; tools that only appear on unrelated rows are
    not eligible as override targets.
20. For low_support categories, primary local detected/total is unjudgeable. You MUST base assignment decisions on cross-dataset performance refs and RAG Tool Knowledge. If neither has assignment-eligible evidence, place the category in gap_claims and do not assign it in tool_categories.
21. assignment_eligible requires total>=10 and rate>=0.3 in the cited evidence card.
22. budget.tool_slots is a hard upper bound. Add complement slots only when Slot Marginal Value permits.
23. recall measures how many ground truths a tool detects, not whether the
    detections are correct. When the preferred tool has assignment-eligible
    strong recall but a RAG con passage flags a category-specific
    false-positive burden for it, treat that RAG signal as F1-side downside risk.
    In that case, the LLM SHOULD switch owner to a candidate from
    override_targets per rule 18, rather than keeping the recall leader and
    writing the RAG con only as a caveat.
24. If an Assignment-Eligible Evidence line has risk_refs for the preferred
    owner, those refs are category-specific fp_precision_risk evidence.
    detected/total, R_hat, local, or near_scene recall rows do not cancel
    that risk. Unless the prompt shows precision-side evidence that directly
    offsets the risk, do not assign any complement owner for that category.
    Do not switch to a different complement using recall rows or
    category_capability RAG alone. Otherwise leave the category to the primary
    ALL envelope or put it in gap_claims.
25. ev_overall_* rows are dataset-level overall metrics, not per-category. MUST NOT cite as category capability. May cite as coarse FP-risk signal when claim text acknowledges 'dataset-level overall'.
26. RAG passages carry an evidence_tier field rendered as [TIER=hard] or
    [TIER=weak] in ## RAG Tool Knowledge. tier=hard marks human-curated
    authoritative passages with explicit ownership instructions, rankings, or
    version constraints; tier=weak marks qualitative observations extracted
    from papers or benchmarks. Hard RAG does not rewrite preferred= before
    your decision. When applicable, it opens rag_override_eligible and places
    feasible alternatives in override_targets. If you choose such an
    alternative, output hard_rag_override plus an applied rag_overrides entry
    and cite the hard passage in the required for/against/compare chains. When
    no override target is listed, keep the preferred owner and cite the hard
    passage only as informational context.
    tier=weak passages must NOT trigger this hard-evidence override even if
    they argue against the preferred owner.

Output a JSON object with exactly these fields:
- selected_action_id: string (must match one of the legal action IDs)
- selected_tools: array of tool ids selected for execution, including the primary tool when planning a composition
- primary_tool: string (must equal the Step1 anchor)
- tool_categories: object mapping tool id to ["ALL"] for the primary or assigned category strings for complements
- category_assignments: [{category: string, owner_tool: string|null, assignment_type: string, evidence_refs: [string], caveat_refs: [string], unrelated_external_only: boolean}]
- rag_overrides: [{category: string, applied: boolean, from_owner: string|null, to_owner: string|null, for_refs: [string], against_refs: [string], compare_refs: [string]}]
- for_claims: [{claim: string, evidence_refs: [string]}]
- against_claims: [{claim: string, evidence_refs: [string]}]
- compare_claims: [{claim: string, evidence_refs: [string]}]
- gap_claims: [{claim: string, evidence_refs: [string]}]
- short_summary: string (1-2 sentences, no target vulnerability claims)"""


def _refs_text(refs: list[str]) -> str:
    return ", ".join(refs) if refs else "none"


def _claim_lines(claims: list) -> list[str]:
    if not claims:
        return ["- none [refs: none]"]
    return [f"- {item.claim} [refs: {_refs_text(item.evidence_refs)}]" for item in claims]


def _card_value_text(value) -> str:
    if value is None:
        return "None"
    if value.rate is not None:
        return str(value.rate)
    if value.value is not None:
        return str(value.value)
    return "None"


def _matched_source_ids(packet: Step1EvidencePacket) -> set[str]:
    ids = {neighbor.paper_id for neighbor in packet.scene_pool.neighbors if neighbor.paper_id}
    for neighbor in packet.scene_pool.neighbors:
        ids.update(neighbor.provenance_refs)
    return ids


def _card_assignment_eligible(card: EvidenceCard, packet: Step1EvidencePacket | None = None) -> bool:
    if card.value is None:
        return False
    if not is_assignment_eligible(card.value.detected, card.value.total, card.value.rate):
        return False
    if packet is not None and card.tool is not None:
        return candidate_passes_precision_gate(packet, card.tool)
    return True


def _top_cards_by_tool(cards: list[EvidenceCard], limit: int = 3) -> list[EvidenceCard]:
    selected: list[EvidenceCard] = []
    seen_tools: set[str] = set()
    for card in cards:
        if card.tool is None or card.tool in seen_tools:
            continue
        selected.append(card)
        seen_tools.add(card.tool)
        if len(selected) >= limit:
            break
    return selected


def _per_category_card_text(card: EvidenceCard, packet: Step1EvidencePacket | None = None) -> str:
    value = card.value
    detected = value.detected if value is not None else None
    total = value.total if value is not None else None
    rate = value.rate if value is not None else None
    eligible = str(_card_assignment_eligible(card, packet)).lower()
    source_id = card.scope.get("source_id")
    dataset_text = f", dataset={source_id}" if source_id else ""
    return (
        f"- {card.evidence_id}: {card.evidence_type}, tool={card.tool}, "
        f"category={card.category}{dataset_text}, tp/GT={count_text(detected, total)} "
        f"R_hat={rate_text(rate)} assignment_eligible={eligible}"
    )


def _matched_dataset_lines(packet: Step1EvidencePacket) -> list[str]:
    lines = ["## Matched Dataset Knowledge", "Scene neighbors:"]
    if packet.scene_pool.neighbors:
        for neighbor in packet.scene_pool.neighbors:
            lines.append(
                (
                    f"- slice={neighbor.slice_id} source={neighbor.paper_id} "
                    f"weight={neighbor.weight:.4f} distance={neighbor.distance:.4f}"
                )
            )
    else:
        lines.append("- none")

    lines.extend(["", "Primary scene tool scores:"])
    if packet.score_panel.nominal_scores:
        for score in sorted(packet.score_panel.nominal_scores, key=lambda item: item.rank):
            f1 = f"{score.F1_scene:.4f}" if score.F1_scene is not None else "None"
            lines.append(
                (
                    f"- {score.tool}: S_scene={score.S_scene:.4f} "
                    f"F1={f1} evidence_level={score.evidence_level}"
                )
            )
    else:
        lines.append("- none")

    lines.extend(["", "Per-category recall-side hints:"])
    by_category: dict[str, list] = {}
    for row in packet.recall_coverage.matrix:
        if row.R_hat is None:
            continue
        by_category.setdefault(row.category, []).append(row)
    if by_category:
        for category in sorted(by_category):
            values = [
                f"{row.tool}={row.R_hat:.4f}({row.support_level})"
                for row in sorted(by_category[category], key=lambda item: item.tool)
            ]
            lines.append(f"- {category}: {'; '.join(values)}")
    else:
        lines.append("- none")
    return lines


def _tool_overall_metrics_lines(packet: Step1EvidencePacket) -> list[str]:
    """Render per-tool overall precision/recall/F1 per cross-referenced dataset.

    Per-category precision is unavailable in the public sources, so this section
    surfaces the next-best signal: each candidate tool's overall precision,
    recall, and F1 on each dataset entry.  Use it together with per-category
    R_hat rows to gauge FP burden when picking owners.
    """

    lines = ["## Tool Overall Metrics (per dataset)"]
    rows = packet.tool_overall_metrics
    if not rows:
        lines.append("- none")
        return lines

    matched = _matched_source_ids(packet)
    by_tool: dict[str, list] = {}
    for row in rows:
        by_tool.setdefault(row.tool, []).append(row)

    def _fmt(v: float | None) -> str:
        return f"{v:.3f}" if isinstance(v, (int, float)) else "None"

    for tool in sorted(by_tool):
        for row in sorted(
            by_tool[tool],
            key=lambda r: (0 if r.source_id in matched else 1, r.source_id),
        ):
            scope = "matched" if row.source_id in matched else "external"
            lines.append(
                f"- {row.evidence_id}: tool={row.tool} dataset={row.source_id}"
                f"({row.dataset_name}) scope={scope} "
                f"precision={_fmt(row.precision)} recall={_fmt(row.recall)} "
                f"f1={_fmt(row.f1)}"
            )
    return lines


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
            rankings_by_tool.setdefault(row.tool, []).append(
                {
                    "row": row,
                    "rank": index,
                    "peer_count": len(ranked),
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


def _external_rank_summary(tool: str, rankings_by_tool: dict[str, list[dict]]) -> str:
    best = _best_external_ranking(rankings_by_tool.get(tool, []))
    if best is None:
        return "best_ext_rank=none"
    row = best["row"]
    return f"best_ext_rank={best['rank']}/{best['peer_count']}@{row.dataset_name}"


def _cross_dataset_support_summary(tool: str, rankings_by_tool: dict[str, list[dict]]) -> str:
    eligible = _eligible_external_rankings(rankings_by_tool.get(tool, []))
    eligible_datasets = {item["dataset_key"] for item in eligible}
    repeated = str(len(eligible_datasets) >= 2).lower()
    return f"cross_dataset={len(eligible_datasets)}eligible/repeated={repeated}"


def _category_candidate_ref_ids(packet: Step1EvidencePacket) -> set[str]:
    refs: set[str] = set()
    local_by_key = _rcov_by_tool_category(packet.recall_coverage)
    for category in attention_categories(packet):
        rankings_by_tool = _external_rankings_by_tool(packet, category)
        for tool in _category_candidate_tools(packet, category):
            entry = local_by_key.get((tool, category))
            if entry is not None and entry.R_hat is not None:
                refs.add(rcov_evidence_id(tool, category))
            best = _best_external_ranking(rankings_by_tool.get(tool, []))
            if best is not None:
                refs.add(best["row"].evidence_id)
            for item in _eligible_external_rankings(rankings_by_tool.get(tool, []))[:3]:
                refs.add(item["row"].evidence_id)
    return refs


def _primary_local_support(
    primary_tool: str,
    category: str,
    local_by_key: dict[tuple[str, str], object],
) -> str:
    entry = local_by_key.get((primary_tool, category))
    support_level = str(getattr(entry, "support_level", "") or "")
    if support_level in {"strong", "medium", "weak"}:
        return support_level
    return "none"


def _candidate_prompt_scope(packet: Step1EvidencePacket, card: EvidenceCard) -> str:
    if card_scope_text(card) == "local":
        return "local"
    scene_tier = card_scene_tier(packet, card)
    if scene_tier in {"primary", "near_scene", "matched_scene"}:
        return "near_scene"
    return "unrelated"


def _ownership_candidate_prompt_scope(candidate: OwnershipCandidate) -> str:
    if candidate.scope == "local" or candidate.scene_tier == "local":
        return "local"
    if candidate.scene_tier in {"primary", "near_scene", "matched_scene"}:
        return "near_scene"
    return "unrelated"


def _candidate_context_text(tool: str, rankings_by_tool: dict[str, list[dict]]) -> str:
    return (
        f"{_external_rank_summary(tool, rankings_by_tool)} "
        f"{_cross_dataset_support_summary(tool, rankings_by_tool)}"
    )


def _top_scene_category_rows_line(
    packet: Step1EvidencePacket,
    category: str,
    *,
    limit: int = 8,
) -> str:
    source_ids = top_scene_source_ids(packet)
    if not source_ids:
        return "  top_scene_category_rows: none"

    source_rank = {source_id: rank for rank, source_id in enumerate(source_ids)}
    by_tool = {}
    by_rank: dict[str, int] = {}
    for row in packet.performance_db_view:
        if row.category != category or row.source_id not in source_rank or row.R_hat is None:
            continue
        rank = source_rank[row.source_id]
        previous_rank = by_rank.get(row.tool)
        if previous_rank is None or rank < previous_rank:
            by_tool[row.tool] = row
            by_rank[row.tool] = rank
    rows = sorted(
        by_tool.values(),
        key=lambda row: (
            -(row.R_hat if row.R_hat is not None else -1.0),
            -(row.total or 0),
            row.tool,
            row.evidence_id,
        ),
    )
    if not rows:
        return "  top_scene_category_rows: none"

    primary_tool = packet.primary_attention.primary_tool
    visible = rows[:limit]
    primary_row = next((row for row in rows if row.tool == primary_tool), None)
    if primary_row is not None and all(row.tool != primary_tool for row in visible):
        visible = [*visible[: max(limit - 1, 0)], primary_row]

    parts: list[str] = []
    for index, row in enumerate(visible, start=1):
        role = ",primary" if row.tool == primary_tool else ""
        parts.append(
            f"rank={index}{role} {row.tool} {row.evidence_id} "
            f"source={row.source_id} tp/GT={count_text(row.detected, row.total)} "
            f"R_hat={rate_text(row.R_hat)}"
        )

    observation = "primary_missing_in_top_scene"
    if primary_row is not None:
        primary_rate = primary_row.R_hat
        best_other = next((row for row in rows if row.tool != primary_tool), None)
        clear_gap = (
            best_other is not None
            and best_other.R_hat is not None
            and best_other.R_hat >= WEAK_RATE_THRESHOLD
            and primary_rate is not None
            and best_other.R_hat - primary_rate >= TOP_SCENE_CLEAR_WEAK_GAP
        )
        primary_total = primary_row.total
        if clear_gap:
            observation = "primary_top_scene_clear_weak_gap_review_complement"
        elif (
            primary_total is not None
            and primary_total < LOW_SUPPORT_TOTAL
            and primary_rate is not None
            and primary_rate < WEAK_RATE_THRESHOLD
        ):
            observation = "primary_top_scene_low_sample_weak_review_topk"
        else:
            observation = "primary_top_scene_no_clear_weak_gap_no_complement_needed"

    return (
        f"  top_scene_category_rows: {'; '.join(parts)}; "
        f"top_scene_primary_observation={observation}"
    )


def _assignment_eligible_evidence_lines(
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
) -> list[str]:
    lines = [
        "## Assignment-Eligible Evidence",
        "Only these refs may support complement entries in tool_categories.",
        "The preferred= tool is the default owner for that category.",
        "RAG override targets are LLM decision options, not pre-applied owners.",
        "external_dataset refs are individual dataset rows, not globally aggregated scores.",
    ]
    primary_tool = packet.primary_attention.primary_tool
    if not primary_tool:
        lines.append("- none")
        return lines

    categories = attention_categories(packet)
    if not categories:
        lines.append("- none")
        return lines

    feasible_tools = {entry.tool for entry in packet.tool_table if entry.feasible}
    cards_by_id = {card.evidence_id: card for card in matrix.evidence_cards}
    local_by_key = _rcov_by_tool_category(packet.recall_coverage)
    by_category: dict[str, list[EvidenceCard]] = {category: [] for category in categories}
    for card in matrix.evidence_cards:
        if (
            card.evidence_type != "per_category_detected_total"
            or card.category not in by_category
            or card.tool == primary_tool
            or card.tool not in feasible_tools
            or not _card_assignment_eligible(card, packet)
        ):
            continue
        by_category[card.category].append(card)

    for category in categories:
        group = category_group(packet, category)
        ownership = category_ownership_evidence(packet, matrix, category)
        rankings_by_tool = _external_rankings_by_tool(packet, category)
        if group == "low_support":
            by_category[category] = [
                card for card in by_category[category] if card_scope_text(card) == "external_dataset"
            ]
        sorted_cards = sorted(
            by_category[category],
            key=lambda card: (
                card_scene_priority(packet, card),
                -(card.value.rate if card.value and card.value.rate is not None else -1.0),
                -(card.value.total if card.value and card.value.total is not None else 0),
                card.tool or "",
                card.evidence_id,
            ),
        )
        cards = _top_cards_by_tool(sorted_cards)
        local_cards = [card for card in sorted_cards if card_scope_text(card) == "local"]
        close_parts: list[str] = []
        if cards and group == "confirmed_weak" and len(local_cards) >= 2:
            top_value = local_cards[0].value
            next_value = local_cards[1].value
            margin = (
                (top_value.rate or 0.0) - (next_value.rate or 0.0)
                if top_value is not None and next_value is not None
                else None
            )
            if margin is not None and is_close_local_margin(
                top_value.rate if top_value is not None else None,
                next_value.rate if next_value is not None else None,
            ):
                close_parts.extend(
                    [
                        f"local_margin=close({margin:.4f})",
                        "review_performance_db_and_rag=true",
                    ]
                )
                global_cards = [
                    card for card in sorted_cards if card_scope_text(card) == "external_dataset"
                ]
                if global_cards:
                    global_card = global_cards[0]
                    global_value = global_card.value
                    close_parts.append(
                        (
                            f"external_dataset_preferred={global_card.tool} {global_card.evidence_id} "
                            f"tp/GT={count_text(global_value.detected, global_value.total)} "
                            f"R_hat={rate_text(global_value.rate)}"
                        )
                    )
        values = []
        for card in cards:
            value = card.value
            values.append(
                (
                    f"{card.tool} {card.evidence_id} "
                    f"tp/GT={count_text(value.detected, value.total)} "
                    f"R_hat={rate_text(value.rate)} scope={_candidate_prompt_scope(packet, card)} "
                    f"{_candidate_context_text(card.tool or '', rankings_by_tool)}"
                )
            )
        weak_values = []
        for candidate in ownership.weak_candidates:
            weak_values.append(
                (
                    f"{candidate.tool} {candidate.evidence_id} "
                    f"tp/GT={count_text(candidate.detected, candidate.total)} "
                    f"R_hat={rate_text(candidate.rate)} "
                    f"scope={_ownership_candidate_prompt_scope(candidate)} "
                    f"{_candidate_context_text(candidate.tool, rankings_by_tool)}"
                )
            )
        guidance = f"; {'; '.join(close_parts)}" if close_parts else ""
        candidates_text = "; ".join(values) if values else "none"
        weak_text = "; ".join(weak_values) if weak_values else "none"
        override_text = "rag_override_eligible=false"
        if ownership.rag_override_eligible:
            hard_refs = [
                ref
                for ref in ownership.override_refs
                if (
                    ref in cards_by_id
                    and scheduling_evidence_priority(cards_by_id[ref], category=category) == "strong"
                )
            ]
            hard_target_refs = [
                ref
                for refs in ownership.override_targets.values()
                for ref in refs
                if (
                    ref in cards_by_id
                    and scheduling_evidence_priority(cards_by_id[ref], category=category) == "strong"
                )
            ]
            hard_text = ""
            if hard_refs or hard_target_refs:
                hard_text = (
                    " hard_override=strong_scheduling_evidence"
                    f" hard_refs={','.join(hard_refs)}"
                    f" hard_targets={','.join(hard_target_refs)}"
                )
            override_text = (
                f"rag_override_eligible=true{hard_text} "
                f"override_refs={','.join(ownership.override_refs)}"
            )
        preferred_text = ownership.preferred or (cards[0].tool if cards else "none")
        risk_refs = _fp_precision_risk_refs(
            matrix,
            category=category,
            tool=preferred_text if preferred_text != "none" else None,
        )
        risk_text = ""
        if risk_refs:
            risk_text = (
                f"; risk_refs={','.join(risk_refs)} "
                "risk_action=prefer_gap_or_primary_all_when_only_recall_side_support"
            )
        primary_support = _primary_local_support(primary_tool, category, local_by_key)
        lines.append(
            f"- {category}: group={group}; primary_local_support={primary_support}; "
            f"preferred={preferred_text}{guidance}; candidates: {candidates_text}; "
            f"weak_candidates: {weak_text}; {override_text}{risk_text}"
        )
        lines.append(_top_scene_category_rows_line(packet, category))
        source_ids_by_evidence_id = {
            card.evidence_id: str(card.scope.get("source_id") or "")
            for card in matrix.evidence_cards
        }
        near_scene_text = _external_candidate_line_text(
            near_scene_strong_candidates(ownership),
            source_ids_by_evidence_id=source_ids_by_evidence_id,
        )
        unrelated_text = _external_candidate_line_text(
            unrelated_strong_candidates(ownership),
            source_ids_by_evidence_id=source_ids_by_evidence_id,
        )
        lines.append(f"  near_scene_dataset_candidates: {near_scene_text}")
        lines.append(f"  unrelated_dataset_candidates: {unrelated_text}")
        lines.append(f"  override_targets: {_override_targets_text(ownership.override_targets)}")
    return lines


def _ownership_panel_lines(matrix: ActionByEvidenceMatrix) -> list[str]:
    lines = ["## Ownership Panel"]
    if not matrix.ownership_panel:
        return lines + ["- none"]
    for category, panel in matrix.ownership_panel.items():
        lines.append(
            f"- {category}: assignment_status={panel.assignment_status}, "
            f"preferred_owner={panel.preferred_owner}, group={panel.group}, "
            f"rag_override_eligible={panel.rag_override_eligible}, "
            f"unrelated_external_only={panel.unrelated_external_only}, "
            f"required_refs={panel.required_claim_refs}"
        )
        strong = [candidate.tool for candidate in panel.strong_candidates]
        weak = [candidate.tool for candidate in panel.weak_candidates]
        if strong:
            lines.append(f"  strong_candidates={strong}")
        if weak:
            lines.append(f"  weak_candidates={weak}")
        if panel.override_refs:
            lines.append(f"  override_refs={panel.override_refs}")
        if panel.override_targets:
            lines.append(f"  override_targets={panel.override_targets}")
        if panel.caveats:
            lines.append(f"  caveat_tags={','.join(panel.caveats)}; not_an_evidence_ref")
    return lines


def _gap_category_lines(matrix: ActionByEvidenceMatrix) -> list[str]:
    lines = ["## Gap Categories"]
    if not matrix.gap_categories:
        return lines + ["- none"]
    for category in matrix.gap_categories:
        panel = matrix.ownership_panel.get(category)
        reason = panel.gap_reason if panel else "gap"
        lines.append(f"- {category}: {reason}")
    return lines


def _override_targets_text(override_targets: dict[str, list[str]]) -> str:
    if not override_targets:
        return "none"
    return "; ".join(
        f"{tool}(rag_pro={','.join(refs)})"
        for tool, refs in override_targets.items()
    )


def _slot_marginal_value_lines(
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
) -> list[str]:
    lines = [
        "## Slot Marginal Value",
        "Each attention category has a marginal_value tag indicating how much an",
        "extra tool slot would add. Use it together with budget.tool_slots to",
        "decide how many complement tools to select.",
    ]
    categories = attention_categories(packet)
    if not categories:
        lines.append("- none")
        return lines
    for category in categories:
        lines.append(
            f"- {category}: marginal_value={category_marginal_value(packet, matrix, category)}"
        )
    return lines


def _display_scene_tier(scene_tier: str) -> str:
    if scene_tier == "external_dataset":
        return "unrelated"
    if scene_tier == "matched_scene":
        return "near_scene"
    return scene_tier


def _external_candidate_line_text(
    candidates: list[OwnershipCandidate],
    *,
    source_ids_by_evidence_id: dict[str, str],
) -> str:
    if not candidates:
        return "none"
    values: list[str] = []
    for candidate in candidates:
        source_id = source_ids_by_evidence_id.get(candidate.evidence_id, "")
        dataset_text = f"dataset={source_id} " if source_id else ""
        values.append(
            (
                f"{candidate.tool} {candidate.evidence_id} "
                f"{dataset_text}"
                f"tp/GT={count_text(candidate.detected, candidate.total)} "
                f"R_hat={rate_text(candidate.rate)} scene_tier={_display_scene_tier(candidate.scene_tier)}"
            )
        )
    return "; ".join(values)


def _feasible_tool_space_lines(packet: Step1EvidencePacket) -> list[str]:
    lines = ["## Feasible Tool Space"]
    ranks = {score.tool: score.rank for score in packet.score_panel.nominal_scores}
    feasible_entries = [entry for entry in packet.tool_table if entry.feasible]
    if not feasible_entries:
        lines.append("- none")
        return lines
    for entry in sorted(feasible_entries, key=lambda item: (ranks.get(item.tool, 9999), item.tool)):
        cost = entry.tool_cost
        runtime = cost.expected_runtime_minutes
        runtime_text = "unknown" if runtime is None else f"{runtime:.1f}"
        rank_text = ranks.get(entry.tool, "unranked")
        lines.append(
            (
                f"- {entry.tool}: rank={rank_text} family={entry.family} "
                f"runtime_min={runtime_text} alert={cost.alert_risk}"
            )
        )
    return lines


def _scope_list(scope: dict, key: str) -> list[str]:
    value = scope.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _rag_scope_polarity(scope: dict) -> str:
    relation_to_owner = str(scope.get("relation_to_owner") or "")
    if relation_to_owner in {"opposes_owner", "owner_ineligible", "owner_weaker"}:
        return "con"
    if relation_to_owner == "evidence_gap":
        return "gap"
    if relation_to_owner == "owner_stronger":
        return "comparative"
    if relation_to_owner in {"supports_owner", "owner_complements"}:
        return "pro"
    return ""


def _fp_precision_risk_refs(
    matrix: ActionByEvidenceMatrix,
    *,
    category: str,
    tool: str | None,
) -> list[str]:
    if not tool:
        return []
    refs: list[str] = []
    for card in matrix.evidence_cards:
        if card.evidence_type != "rag_passage":
            continue
        if card.tool != tool:
            continue
        if category not in _scope_list(card.scope, "categories") and card.category != category:
            continue
        if str(card.scope.get("knowledge_kind") or "") != "fp_precision_risk":
            continue
        if _rag_scope_polarity(card.scope) != "con":
            continue
        refs.append(card.evidence_id)
    return refs


def _rag_scope_passage_type(scope: dict) -> str:
    passage_type = str(scope.get("passage_type") or "")
    if passage_type:
        return passage_type
    relation_to_owner = str(scope.get("relation_to_owner") or "")
    if relation_to_owner == "owner_ineligible":
        return "limitation"
    if relation_to_owner in {"owner_stronger", "owner_weaker"}:
        return "comparison"
    if relation_to_owner == "owner_complements":
        return "recommendation"
    if relation_to_owner == "evidence_gap":
        return "gap"
    if relation_to_owner:
        return "performance"
    return ""


def _rag_scope_scene_constraints(scope: dict) -> list[str]:
    scene_constraints = _scope_list(scope, "scene_constraints")
    if scene_constraints:
        return scene_constraints
    return [
        tag.removeprefix("scene:")
        for tag in _scope_list(scope, "applicability_tags")
        if tag.startswith("scene:")
    ]


def _rag_tool_knowledge_lines(matrix: ActionByEvidenceMatrix) -> list[str]:
    lines = [
        "## RAG Tool Knowledge",
        "RAG passages are decision evidence; applicable hard passages may justify LLM-declared overrides but are not pre-applied owners.",
    ]
    rag_cards = [card for card in matrix.evidence_cards if card.evidence_type == "rag_passage"]
    if not rag_cards:
        lines.append("- none")
        return lines
    for card in rag_cards:
        passage_type = _rag_scope_passage_type(card.scope)
        knowledge_kind = card.scope.get("knowledge_kind") or ""
        polarity = _rag_scope_polarity(card.scope)
        evidence_tier = card.scope.get("evidence_tier") or "weak"
        counterpart_tool_ids = _scope_list(card.scope, "counterpart_tool_ids")
        primary_tool = card.scope.get("primary_tool") or (
            counterpart_tool_ids[0]
            if card.scope.get("relation_to_owner") == "owner_complements" and counterpart_tool_ids
            else ""
        )
        complement_tool = card.scope.get("complement_tool") or (
            card.tool if card.scope.get("relation_to_owner") == "owner_complements" else ""
        )
        scene_constraints = _rag_scope_scene_constraints(card.scope)
        evidence_basis = card.scope.get("evidence_basis") or ""
        priority = scheduling_evidence_priority(card, category=card.category or "")
        limitations = _scope_list(card.scope, "limitations") or list(card.limitations)
        text = str(card.scope.get("text") or card.scope.get("claim_text") or "")
        risk_action = ""
        if knowledge_kind == "fp_precision_risk" and polarity == "con":
            risk_action = " risk_action=avoid_complement_when_only_recall_side_support"
        scene_text = ", ".join(scene_constraints)
        limitations_text = ", ".join(str(item) for item in limitations)
        lines.append(
            (
                f"- {card.evidence_id}: [TIER={evidence_tier} POLARITY={polarity}] kind={knowledge_kind} polarity={polarity} paper={card.source.paper_id} "
                f"primary={primary_tool} complement={complement_tool} tool={card.tool} category={card.category} "
                f"scene={scene_text} limitations={limitations_text} basis={evidence_basis} type={passage_type} "
                f"priority={priority} source_reliability={card.source_reliability}{risk_action}"
            )
        )
        if text:
            lines.append(f"  text={text}")
    return lines


def _build_user_prompt(
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
    run_history: list[ToolRunSummary] | None,
) -> str:
    budget = matrix.budget_profile
    primary_tool = _step1_anchor_tool(packet)
    lines = [
        f"## Budget: tool_slots={budget.tool_slots}, runtime_cap_minutes={budget.runtime_cap_minutes}, alert_cap={budget.alert_cap}",
        "",
        "## Category Ownership",
        f"Step1 anchor primary_tool={primary_tool}",
        f"step1.primary_attention.confirmed_weak_categories={packet.primary_attention.confirmed_weak_categories}",
        f"step1.primary_attention.low_support_categories={packet.primary_attention.low_support_categories}",
        "Confirmed-weak categories have primary local total>=10 with R_hat<0.3 (known-weak). Low-support categories have primary local total<10 (not locally judgeable). Complement tools may own categories from either group, decided by cross-dataset performance refs and RAG Tool Knowledge.",
        "",
        "## Step1 Field Refs",
        "Use the exact ref before '=' when citing these fields.",
        f"step1.certification.status={packet.certification.status}",
        f"step1.certification.certified_primary={packet.certification.certified_primary}",
        f"step1.certification.candidate_set={packet.certification.candidate_set}",
        f"step1.certification.reason_codes={packet.certification.reason_codes}",
        f"step1.score_panel.stress_rankings.local={packet.score_panel.stress_rankings.local}",
        f"step1.score_panel.stress_rankings.global={packet.score_panel.stress_rankings.global_}",
        f"step1.score_panel.stress_rankings.uniform_supported={packet.score_panel.stress_rankings.uniform_supported}",
        f"step1.score_panel.stress_rankings.top1_flip={packet.score_panel.stress_rankings.top1_flip}",
        f"step1.category_diagnostics.category_bias_risk={packet.category_diagnostics.category_bias_risk}",
        "",
    ]
    lines.extend(_matched_dataset_lines(packet))
    lines.extend([""])
    lines.extend(_tool_overall_metrics_lines(packet))
    lines.extend([""])
    lines.extend(_assignment_eligible_evidence_lines(packet, matrix))
    lines.extend([""])
    lines.extend(_ownership_panel_lines(matrix))
    lines.extend([""])
    lines.extend(_gap_category_lines(matrix))
    lines.extend([""])
    lines.extend(_slot_marginal_value_lines(packet, matrix))
    lines.extend([""])
    lines.extend(_feasible_tool_space_lines(packet))
    lines.extend([""])
    lines.extend(_rag_tool_knowledge_lines(matrix))
    lines.extend(["", "## Legal Action Envelopes"])

    for action in matrix.actions:
        if not action.legal:
            continue
        lines.extend(
            [
                f"### {action.action_id} ({action.action_type})",
                f"Tools: {action.tools}",
                (
                    f"Estimated: slots={action.estimated_budget.tool_slots}, "
                    f"runtime={action.estimated_budget.runtime_cap_minutes} min, "
                    f"alert={action.estimated_budget.alert_cap}"
                ),
            ]
        )
        for slot in ("FOR", "AGAINST", "COMPARE", "GAP"):
            lines.append(f"{slot}:")
            lines.extend(_claim_lines(action.evidence.get(slot, [])))
        lines.append("")

    lines.append("## Evidence Cards")
    performance_refs = _category_candidate_ref_ids(packet)
    if matrix.evidence_cards:
        for card in matrix.evidence_cards:
            if card.evidence_type == "rag_passage":
                continue
            if card.evidence_type == "per_category_detected_total":
                if card_scope_text(card) == "external_dataset" and card.evidence_id not in performance_refs:
                    continue
                lines.append(_per_category_card_text(card, packet))
                continue
            lines.append(
                (
                    f"- {card.evidence_id}: {card.evidence_type}, tool={card.tool}, "
                    f"metric={card.metric_semantics}, value={_card_value_text(card.value)}"
                )
            )
        if lines[-1] == "## Evidence Cards":
            lines.append("- none")
    else:
        lines.append("- none")

    if run_history:
        lines.extend(["", "## Run History"])
        for item in run_history:
            lines.append(
                (
                    f"- {item.tool}: status={item.run_status}, "
                    f"runtime={item.runtime_minutes} min, findings={item.total_findings}"
                )
            )

    return "\n".join(lines)


def _step1_anchor_tool(packet: Step1EvidencePacket) -> str | None:
    if packet.primary_attention.primary_tool:
        return packet.primary_attention.primary_tool
    if packet.certification.certified_primary:
        return packet.certification.certified_primary
    if packet.certification.candidate_set:
        return packet.certification.candidate_set[0]
    for score in sorted(packet.score_panel.nominal_scores, key=lambda item: item.rank):
        return score.tool
    return None


def _response_schema() -> dict:
    claim_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claim": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["claim", "evidence_refs"],
    }
    category_list_schema = {"type": "array", "items": {"type": "string"}}
    tool_categories_schema = {
        "type": "object",
        "additionalProperties": category_list_schema,
    }
    category_assignment_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "category": {"type": "string"},
            "owner_tool": {"type": ["string", "null"]},
            "assignment_type": {
                "type": "string",
                "enum": [
                    "primary_all",
                    "local_owner",
                    "near_scene_owner",
                    "external_weak_owner",
                    "hard_rag_override",
                    "gap",
                    "stop_with_gap",
                ],
            },
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "caveat_refs": {"type": "array", "items": {"type": "string"}},
            "unrelated_external_only": {"type": "boolean"},
        },
        "required": [
            "category",
            "owner_tool",
            "assignment_type",
            "evidence_refs",
            "caveat_refs",
            "unrelated_external_only",
        ],
    }
    override_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "category": {"type": "string"},
            "applied": {"type": "boolean"},
            "from_owner": {"type": ["string", "null"]},
            "to_owner": {"type": ["string", "null"]},
            "for_refs": {"type": "array", "items": {"type": "string"}},
            "against_refs": {"type": "array", "items": {"type": "string"}},
            "compare_refs": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "category",
            "applied",
            "from_owner",
            "to_owner",
            "for_refs",
            "against_refs",
            "compare_refs",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_action_id": {"type": "string"},
            "selected_tools": {"type": "array", "items": {"type": "string"}},
            "primary_tool": {"type": "string"},
            "tool_categories": tool_categories_schema,
            "category_assignments": {"type": "array", "items": category_assignment_schema},
            "rag_overrides": {"type": "array", "items": override_schema},
            "for_claims": {"type": "array", "items": claim_schema},
            "against_claims": {"type": "array", "items": claim_schema},
            "compare_claims": {"type": "array", "items": claim_schema},
            "gap_claims": {"type": "array", "items": claim_schema},
            "short_summary": {"type": "string"},
        },
        "required": [
            "selected_action_id",
            "selected_tools",
            "primary_tool",
            "tool_categories",
            "category_assignments",
            "rag_overrides",
            "for_claims",
            "against_claims",
            "compare_claims",
            "gap_claims",
            "short_summary",
        ],
    }


def _build_selected_plan(action: CandidateAction) -> list[SelectedToolEntry]:
    if action.action_type == "RUN_PRIMARY":
        return [SelectedToolEntry(tool=action.tools[0], role="STARTER", execution_order=1)]
    if action.action_type == "RUN_ROBUST_SINGLE":
        return [SelectedToolEntry(tool=action.tools[0], role="SINGLE", execution_order=1)]
    if action.action_type == "CONTINUE_HEDGE":
        return [SelectedToolEntry(tool=action.tools[0], role="CONTINUATION", execution_order=1)]
    return []


def _build_planned_selected_plan(
    selected_tools: list[str],
    primary_tool: str | None,
) -> list[SelectedToolEntry]:
    if not selected_tools:
        return []
    ordered_tools: list[str] = []
    if primary_tool and primary_tool in selected_tools:
        ordered_tools.append(primary_tool)
    for tool in selected_tools:
        if tool not in ordered_tools:
            ordered_tools.append(tool)
    entries: list[SelectedToolEntry] = []
    for index, tool in enumerate(ordered_tools, start=1):
        role = "STARTER" if index == 1 else "COMPLEMENT"
        entries.append(SelectedToolEntry(tool=tool, role=role, execution_order=index))
    return entries


def _build_budget_usage(action: CandidateAction, budget: BudgetProfile) -> BudgetUsage:
    return BudgetUsage(
        limit=budget,
        estimated_use=action.estimated_budget,
        remaining_after_plan=BudgetProfile(
            tool_slots=budget.tool_slots - action.estimated_budget.tool_slots,
            runtime_cap_minutes=budget.runtime_cap_minutes - action.estimated_budget.runtime_cap_minutes,
            alert_cap=budget.alert_cap,
        ),
    )


_ALERT_ORDER = {"low": 0, "medium": 1, "high": 2}


def _planned_budget_usage(
    selected_tools: list[str],
    packet: Step1EvidencePacket,
    budget: BudgetProfile,
) -> BudgetUsage:
    table = {entry.tool: entry for entry in packet.tool_table}
    runtime = 0.0
    alert = "low"
    for tool in selected_tools:
        entry = table.get(tool)
        if entry is None:
            continue
        runtime += entry.tool_cost.expected_runtime_minutes or 0.0
        if _ALERT_ORDER[entry.tool_cost.alert_risk] > _ALERT_ORDER[alert]:
            alert = entry.tool_cost.alert_risk
    estimated = BudgetProfile(
        tool_slots=len(selected_tools),
        runtime_cap_minutes=runtime,
        alert_cap=alert,
    )
    return BudgetUsage(
        limit=budget,
        estimated_use=estimated,
        remaining_after_plan=BudgetProfile(
            tool_slots=budget.tool_slots - estimated.tool_slots,
            runtime_cap_minutes=budget.runtime_cap_minutes - estimated.runtime_cap_minutes,
            alert_cap=budget.alert_cap,
        ),
    )


def _claims(raw: dict, key: str) -> list[ActionEvidenceClaim]:
    claims = []
    for item in raw.get(key, []):
        if isinstance(item, dict) and isinstance(item.get("evidence_refs"), list):
            item = {
                **item,
                "evidence_refs": _claim_evidence_refs(item["evidence_refs"]),
            }
        claims.append(ActionEvidenceClaim.model_validate(item))
    return claims


def _category_assignments(raw: dict) -> list[CategoryAssignment]:
    return [CategoryAssignment(**item) for item in raw.get("category_assignments", [])]


def _rag_overrides(raw: dict) -> list[RagOverrideRecord]:
    return [RagOverrideRecord(**item) for item in raw.get("rag_overrides", [])]


def _tool_categories_from_assignments(
    primary_tool: str | None,
    assignments: list[CategoryAssignment],
    fallback: dict,
) -> dict[str, list[str]]:
    if not assignments:
        return dict(fallback or {})
    categories: dict[str, list[str]] = {}
    if primary_tool:
        categories[primary_tool] = ["ALL"]
    for assignment in assignments:
        if not assignment.owner_tool or assignment.assignment_type in {"gap", "stop_with_gap"}:
            continue
        if assignment.assignment_type == "primary_all":
            continue
        categories.setdefault(assignment.owner_tool, [])
        if assignment.category not in categories[assignment.owner_tool]:
            categories[assignment.owner_tool].append(assignment.category)
    return categories


def _sync_selected_tools_with_categories(
    selected_tools: list[str],
    primary_tool: str | None,
    tool_categories: dict[str, list[str]],
) -> list[str]:
    if not tool_categories:
        return selected_tools
    required_tools = list(tool_categories)
    ordered: list[str] = []
    for tool in ([primary_tool] if primary_tool else []) + selected_tools + required_tools:
        if not tool or tool not in required_tools or tool in ordered:
            continue
        ordered.append(tool)
    return ordered


def _parse_and_assemble(
    raw: dict,
    matrix: ActionByEvidenceMatrix,
    budget: BudgetProfile,
    packet: Step1EvidencePacket | None = None,
) -> Step2DecisionCertificate:
    try:
        selected_action_id = raw["selected_action_id"]
    except KeyError as exc:
        raise CegoError("CEGO response missing selected_action_id.") from exc

    actions_by_id = {action.action_id: action for action in matrix.actions}
    action = actions_by_id.get(selected_action_id)
    if action is None:
        normalized = selected_action_id.lower().replace(" ", "_")
        action = actions_by_id.get(normalized)
        if action is not None:
            selected_action_id = normalized
    if action is None:
        # LLM sometimes returns the action_type (e.g. "PLAN_COMPOSITION") instead of
        # the action_id ("plan_tool_composition"). Match by action_type as a fallback.
        upper = selected_action_id.strip().upper().replace(" ", "_")
        for act in matrix.actions:
            if act.action_type.upper() == upper:
                action = act
                selected_action_id = act.action_id
                break
    if action is None:
        normalized = selected_action_id.lower().replace(" ", "_")
        for aid in actions_by_id:
            if normalized in aid or aid in normalized:
                action = actions_by_id[aid]
                selected_action_id = aid
                break
    if action is None:
        raise CegoError(f"CEGO selected unknown action_id: {selected_action_id}")
    if not action.legal:
        raise CegoError(f"CEGO selected illegal action_id: {selected_action_id}")

    try:
        primary_tool = raw.get("primary_tool")
        selected_tools = [
            str(tool)
            for tool in raw.get("selected_tools", [])
            if isinstance(tool, str)
        ]
        category_assignments = _category_assignments(raw)
        rag_overrides = _rag_overrides(raw)
        tool_categories = _tool_categories_from_assignments(
            primary_tool,
            category_assignments,
            dict(raw.get("tool_categories") or {}),
        )
        selected_tools = _sync_selected_tools_with_categories(
            selected_tools,
            primary_tool,
            tool_categories,
        )
        selected_plan = (
            _build_planned_selected_plan(selected_tools, primary_tool)
            if action.action_type == "PLAN_COMPOSITION"
            else _build_selected_plan(action)
        )
        budget_usage = (
            _planned_budget_usage(selected_tools, packet, budget)
            if action.action_type == "PLAN_COMPOSITION" and packet is not None
            else _build_budget_usage(action, budget)
        )
        evidence_block = ActionEvidenceBlock(
            action_id=selected_action_id,
            for_claims=_claims(raw, "for_claims"),
            against_claims=_claims(raw, "against_claims"),
            compare_claims=_claims(raw, "compare_claims"),
            gap_claims=_claims(raw, "gap_claims"),
        )
        return Step2DecisionCertificate(
            decision_type=action.action_type,
            selected_action_id=selected_action_id,
            selected_plan=selected_plan,
            primary_tool=primary_tool,
            tool_categories=tool_categories,
            category_assignments=category_assignments,
            rag_overrides=rag_overrides,
            action_evidence=evidence_block,
            budget=budget_usage,
            forbidden_claims_attestation=ForbiddenClaimsAttestation(),
            short_summary=str(raw.get("short_summary", "")),
        )
    except (IndexError, TypeError, ValidationError, ValueError) as exc:
        raise CegoError(f"CEGO response could not be assembled: {exc}") from exc


def run_cego(
    client: OpenAICompatClient,
    model: str,
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
    run_history: list[ToolRunSummary] | None = None,
) -> Step2DecisionCertificate:
    """顶层入口。构造 prompt → 调 LLM → 解析响应 → 组装证书。"""
    try:
        raw = create_json_chat_completion(
            client=client,
            model=model or DEFAULT_OPENAI_MODEL,
            system_prompt=_build_system_prompt(),
            user_prompt=_build_user_prompt(packet, matrix, run_history),
            schema=_response_schema(),
            raise_on_error=True,
        )
    except OpenAICompatError as exc:
        raise CegoError(f"CEGO LLM call failed: {exc}") from exc

    if raw is None:
        raise CegoError("CEGO LLM returned no data.")
    if not isinstance(raw, dict):
        raise CegoError(f"CEGO LLM returned invalid data: {json.dumps(raw, ensure_ascii=False)}")
    return _parse_and_assemble(raw, matrix, matrix.budget_profile, packet)
