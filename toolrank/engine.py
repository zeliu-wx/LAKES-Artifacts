"""End-to-end SCREC, DACE-RAG, CEGO, and checker pipeline orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

from toolrank import report_parser
from toolrank.cego import run_cego
from toolrank.certification import certify
from toolrank.checker import check_decision
from toolrank.contract_profile import analyze_target
from toolrank.dace_rag import build_action_evidence_matrix
from toolrank.dataset_kb import load_performance_db
from toolrank.evidence_packet import build_evidence_packet, build_tool_table
from toolrank.execution import build_execution_plan, execute_plan
from toolrank.fusion import compact_fused_report_payload, fuse_reports
from toolrank.openai_compat import load_openai_client
from toolrank.passage_store import PassageRetriever, load_passage_store
from toolrank.vector_store import VectorIndex
from toolrank.rcov import build_recall_coverage
from toolrank.retrieval import load_toolcards
from toolrank.scene_pool import build_scene_pool
from toolrank.scene_scoring import compute_scene_scores
from toolrank.schemas import (
    CompositionPlan,
    ContractFeatures,
    DASP10_CATEGORIES,
    ExecutionResult,
    Finding,
    FusedReport,
    PerformanceKnowledgeBase,
    ToolCard,
    ToolScore,
)
from toolrank.schemas_v2 import (
    ActionByEvidenceMatrix,
    BudgetProfile,
    BudgetUsage,
    CategoryDiagnostics,
    CheckerVerdict,
    ForbiddenClaimsAttestation,
    PrimaryAttention,
    RecallCoverageMatrix,
    ScenePool,
    ScorePanel,
    SelectedToolEntry,
    Step1EvidencePacket,
    Step2DecisionCertificate,
)
from toolrank.stress import run_stress_test


@dataclass
class PipelineResult:
    features: ContractFeatures
    packet: Step1EvidencePacket
    matrix: ActionByEvidenceMatrix
    certificate: Step2DecisionCertificate
    checker_verdict: CheckerVerdict
    execution: ExecutionResult | None = None
    fused_report: FusedReport | None = None
    lakes_output_dir: str | None = None
    warnings: list[str] = field(default_factory=list)


def _emit(enabled: bool, title: str, details: str) -> None:
    if not enabled:
        return
    sys.stderr.write(f"\n[{title}]\n{details}\n")
    sys.stderr.flush()


def _explain(enabled: bool, lines: list[str]) -> None:
    """Print indented detail lines under the previous [stage] header."""
    if not enabled or not lines:
        return
    for line in lines:
        sys.stderr.write(f"  {line}\n")
    sys.stderr.flush()


def _detail_scene_pool(pool: ScenePool) -> list[str]:
    if not pool.neighbors:
        return ["(empty scene pool)"]
    lines: list[str] = []
    for index, neighbor in enumerate(pool.neighbors):
        tag = "primary" if index == 0 else "near   "
        paper = neighbor.paper_id or "?"
        lines.append(
            f"{tag} slice={neighbor.slice_id}  paper={paper}  "
            f"weight={neighbor.weight:.3f}  distance={neighbor.distance:.3f}"
        )
    primary = pool.neighbors[0]
    if primary.category_profile:
        top = sorted(primary.category_profile.items(), key=lambda item: -item[1])[:6]
        profile = ", ".join(f"{cat}={mass:.2f}" for cat, mass in top)
        lines.append(f"primary scene category profile: {profile}")
    return lines


def _detail_tool_table(tool_table: list[Any]) -> list[str]:
    feasible = [entry.tool for entry in tool_table if entry.feasible]
    infeasible = [
        (entry.tool, list(entry.feasibility_reasons)) for entry in tool_table if not entry.feasible
    ]
    lines: list[str] = []
    if feasible:
        lines.append(f"feasible ({len(feasible)}): {', '.join(feasible)}")
    for tool, reasons in infeasible:
        reason_str = ", ".join(reasons) if reasons else "unknown"
        lines.append(f"infeasible: {tool} (reasons: {reason_str})")
    return lines


def _detail_scene_scoring(nominal_scores: list[Any], top: int = 5) -> list[str]:
    if not nominal_scores:
        return ["(no nominal scores)"]
    head = sorted(nominal_scores, key=lambda item: item.rank)[:top]
    lines = [f"top {len(head)} by S_scene (on primary neighbor):"]
    for score in head:
        f1 = f"{score.F1_scene:.3f}" if score.F1_scene is not None else "—"
        lines.append(
            f"  {score.rank}. {score.tool:14s}  S_scene={score.S_scene:.3f}  "
            f"F1={f1}  evidence={score.evidence_level}"
        )
    return lines


def _detail_diagnostics(diagnostics: CategoryDiagnostics) -> list[str]:
    lines: list[str] = []
    if diagnostics.scene_category_profile:
        top = sorted(diagnostics.scene_category_profile.items(), key=lambda item: -item[1])[:5]
        profile = ", ".join(f"{cat}={mass:.2f}" for cat, mass in top)
        lines.append(f"scene category profile (top): {profile}")
    if diagnostics.bias_signals:
        lines.append(f"bias_signals: {', '.join(diagnostics.bias_signals)}")
    return lines


def _detail_top_scene_support(
    kb: PerformanceKnowledgeBase | None,
    scene_pool: ScenePool,
    primary_attention: PrimaryAttention,
) -> tuple[list[str], dict[str, str]]:
    """Show tier-1 top-scene support and return per-category decisions.

    Mirrors evidence_packet._top_scene_primary_decision so display matches the
    actual two-tier weak/low_support partition (top-scene first, aggregate fallback).
    Returns (display_lines, decisions) where decisions[category] is one of:
      "weak" | "not_weak" | "fallback" | "no_data".
    """
    primary_tool = primary_attention.primary_tool
    if kb is None or primary_tool is None or not scene_pool.neighbors:
        return [], {}
    top = scene_pool.neighbors[0]
    top_ids: set[str] = set()
    if top.paper_id:
        top_ids.add(top.paper_id)
    top_ids.update(top.provenance_refs)
    if not top_ids:
        return [], {}

    def _tk(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    primary_key = _tk(primary_tool)
    dasp = set(DASP10_CATEGORIES)
    by_cat: dict[str, dict[str, tuple[int | None, int | None, float]]] = {}
    for entry in kb.entries:
        if entry.source_id not in top_ids:
            continue
        for obs in entry.tool_performance_data:
            counts = obs.vulnerability_score_counts or {}
            scores = obs.vulnerability_scores or {}
            for category in (set(counts) | set(scores)) & dasp:
                cnt = counts.get(category)
                if cnt is not None:
                    if cnt.total <= 0:
                        continue
                    by_cat.setdefault(category, {})[obs.tool_name] = (
                        cnt.detected,
                        cnt.total,
                        cnt.detected / cnt.total,
                    )
                else:
                    score = scores.get(category)
                    if score is None:
                        continue
                    by_cat.setdefault(category, {})[obs.tool_name] = (
                        None,
                        None,
                        float(score),
                    )

    decisions: dict[str, str] = {}
    lines = [
        f"primary={primary_tool} top-scene support "
        f"(tier-1: only scene_pool.neighbors[0]=\"{top.paper_id or top.slice_id}\" is consulted):",
    ]
    for category in sorted(dasp):
        by_tool = by_cat.get(category, {})
        primary_data: tuple[int | None, int | None, float] | None = None
        for tname, value in by_tool.items():
            if _tk(tname) == primary_key:
                primary_data = value
                break
        best_other: tuple[str, float] | None = None
        for tname, (_d, _t, rate) in by_tool.items():
            if _tk(tname) == primary_key:
                continue
            if best_other is None or rate > best_other[1]:
                best_other = (tname, rate)

        if primary_data is None:
            decisions[category] = "no_data"
            lines.append(
                f"  {category:26s} primary=(no data)                                   "
                f"→ tier-1 NO DATA → falls to tier-2 aggregate"
            )
            continue

        p_det, p_tot, p_rate = primary_data
        clear_gap = (
            best_other is not None
            and best_other[1] >= 0.3
            and best_other[1] - p_rate >= 0.2
        )
        if p_tot is not None and p_tot >= 10:
            label = "weak (clear gap)" if clear_gap else "not_weak"
            decisions[category] = "weak" if clear_gap else "not_weak"
        elif p_rate >= 0.3 and not clear_gap:
            label = "not_weak"
            decisions[category] = "not_weak"
        else:
            label = "fallback (small sample / no clear gap)"
            decisions[category] = "fallback"

        if p_det is not None and p_tot is not None:
            primary_cell = f"{p_det}/{p_tot}"
        else:
            primary_cell = f"rate={p_rate:.3f}"
        primary_cell = f"{primary_cell:<14s}"
        rate_text = f"R={p_rate:.3f}"
        comp_text = (
            f"  best_other={best_other[0]}={best_other[1]:.3f}"
            if best_other
            else "  no_competitor"
        )
        lines.append(
            f"  {category:26s} primary={primary_cell} {rate_text}{comp_text}  → tier-1 {label}"
        )
    return lines, decisions


def _detail_rcov(
    rcov: RecallCoverageMatrix,
    primary_attention: PrimaryAttention,
    tier1_decisions: dict[str, str] | None = None,
    *,
    top: int = 12,
) -> list[str]:
    """Show tier-2 aggregate rows ONLY for categories that tier-1 could not decide.

    If `tier1_decisions` is provided, filter to categories whose tier-1 decision
    is "fallback" or "no_data". Categories already decided by tier-1 are omitted.
    If `tier1_decisions` is None (no top-scene data at all), fall back to showing
    all DASP rows.
    """
    primary_tool = primary_attention.primary_tool
    if primary_tool is None:
        return ["(no primary tool to focus on)"]
    confirmed_weak = set(primary_attention.confirmed_weak_categories)
    low_support = set(primary_attention.low_support_categories)
    dasp = set(DASP10_CATEGORIES)
    rows = [row for row in rcov.matrix if row.tool == primary_tool and row.category in dasp]
    if not rows:
        return [f"(no DASP rcov rows for primary={primary_tool})"]

    if tier1_decisions:
        fallback_cats = {
            category
            for category, decision in tier1_decisions.items()
            if decision in {"fallback", "no_data"}
        }
        rows = [row for row in rows if row.category in fallback_cats]
        if not rows:
            return [
                f"primary={primary_tool} tier-2 aggregate fallback:",
                "  (tier-1 decided every DASP category; tier-2 not consulted)",
            ]

    rows.sort(key=lambda row: ((row.R_hat if row.R_hat is not None else 1.0), row.category))
    lines = [
        f"primary={primary_tool} tier-2 aggregate fallback",
        "(only categories where tier-1 could not decide; totals = GT counts summed across ALL matched neighbors):",
    ]
    for row in rows[:top]:
        r_hat = f"{row.R_hat:.3f}" if row.R_hat is not None else "—"
        det = row.detected if row.detected is not None else 0
        tot = row.total if row.total is not None else 0
        marker_parts: list[str] = []
        if row.category in confirmed_weak:
            marker_parts.append("confirmed_weak")
        elif row.category in low_support:
            marker_parts.append("low_support")
        marker = ("  ← " + " / ".join(marker_parts)) if marker_parts else ""
        lines.append(
            f"  {row.category:26s} {det:>4d}/{tot:<5d}  R_hat={r_hat}  support={row.support_level:11s}{marker}"
        )
    return lines


def _detail_certification(cert: Any) -> list[str]:
    lines: list[str] = []
    if cert.certified_primary:
        lines.append(f"certified_primary: {cert.certified_primary}")
    if cert.candidate_set:
        lines.append(f"candidate_set: [{', '.join(cert.candidate_set)}]")
    if cert.reason_codes:
        lines.append(f"reason_codes: {' → '.join(cert.reason_codes)}")
    return lines


def _detail_evidence_packet(packet: Step1EvidencePacket) -> list[str]:
    pa = packet.primary_attention
    lines: list[str] = []
    if pa.primary_tool:
        lines.append(f"primary_tool: {pa.primary_tool}")
    lines.append(
        f"confirmed_weak: [{', '.join(pa.confirmed_weak_categories) or 'none'}]"
    )
    lines.append(
        f"low_support:    [{', '.join(pa.low_support_categories) or 'none'}]"
    )
    if packet.dace_rag_focus:
        lines.append("dace_rag_focus:")
        for item in packet.dace_rag_focus[:6]:
            lines.append(f"  · {item.tool} / {item.category}: {item.reason}")
    return lines


def _detail_dace_actions(matrix: ActionByEvidenceMatrix, slot_cap: int = 3) -> list[str]:
    if not matrix.actions:
        return ["(no candidate actions)"]
    lines: list[str] = []
    for action in matrix.actions:
        flag = "✓" if action.legal else "✗"
        tools_str = ", ".join(action.tools) if action.tools else "—"
        runtime = action.estimated_budget.runtime_cap_minutes
        lines.append(
            f"{flag} action={action.action_id}  type={action.action_type}  "
            f"tools=[{tools_str}]  budget=slots={action.estimated_budget.tool_slots}, "
            f"runtime={runtime:.1f}min"
        )
        if not action.legal and action.legality_reasons:
            lines.append(f"    legality_reasons: {', '.join(action.legality_reasons)}")
        for slot in ("FOR", "AGAINST", "COMPARE", "GAP"):
            claims = action.evidence.get(slot, [])
            if not claims:
                continue
            lines.append(f"    {slot:7s} ({len(claims)}):")
            for claim in claims[:slot_cap]:
                refs = list(claim.evidence_refs or [])
                ref_text = ", ".join(refs[:2])
                if len(refs) > 2:
                    ref_text += f", +{len(refs) - 2}"
                if not ref_text:
                    ref_text = "—"
                claim_text = claim.claim
                if len(claim_text) > 100:
                    claim_text = claim_text[:97] + "..."
                lines.append(f"      - {claim_text}  [refs: {ref_text}]")
            if len(claims) > slot_cap:
                lines.append(f"      ...and {len(claims) - slot_cap} more")
    return lines


def _detail_ownership_panel(matrix: ActionByEvidenceMatrix) -> list[str]:
    if not matrix.ownership_panel and not matrix.gap_categories:
        return []
    lines: list[str] = ["ownership panel:"]
    for category, panel in matrix.ownership_panel.items():
        owner = panel.preferred_owner or "—"
        override = " [override_eligible]" if panel.rag_override_eligible else ""
        lines.append(
            f"  · {category:24s} group={panel.group:14s} preferred={owner:14s} "
            f"strong={len(panel.strong_candidates)} weak={len(panel.weak_candidates)} "
            f"status={panel.assignment_status}{override}"
        )
    if matrix.gap_categories:
        lines.append(f"gap_categories: [{', '.join(matrix.gap_categories)}]")
    return lines


def _detail_rag_passages(matrix: ActionByEvidenceMatrix, top: int = 20) -> list[str]:
    rag_cards = [card for card in matrix.evidence_cards if card.evidence_type == "rag_passage"]
    if not rag_cards:
        return ["(no RAG passages injected)"]
    lines = [f"retrieved {len(rag_cards)} RAG passage(s):"]
    for card in rag_cards[:top]:
        scope = card.scope or {}
        tier = scope.get("evidence_tier", "?")
        kind = scope.get("knowledge_kind", "?")
        rel = scope.get("relation_to_owner", "?")
        tool = card.tool or "?"
        category = card.category or "?"
        claim_text = scope.get("claim_text", "") or ""
        snippet = claim_text.strip().replace("\n", " ")
        if len(snippet) > 90:
            snippet = snippet[:87] + "..."
        lines.append(
            f"  [{tier:6s}] {card.evidence_id}"
        )
        lines.append(
            f"           kind={kind:22s} rel={rel:18s} owner={tool}@{category}  role={card.decision_role}"
        )
        if snippet:
            lines.append(f"           ▸ {snippet}")
    if len(rag_cards) > top:
        lines.append(f"  ...and {len(rag_cards) - top} more")
    return lines


def _detail_execution(execution: Any) -> list[str]:
    if execution is None:
        return []
    lines = [
        f"status: {execution.status}  return_code: {execution.return_code}",
        f"primary_tool: {execution.primary_tool}",
    ]
    if execution.tool_categories:
        cats = ", ".join(
            f"{tool}=[{', '.join(categories)}]"
            for tool, categories in execution.tool_categories.items()
        )
        lines.append(f"tool_categories: {cats}")
    if execution.runner_command:
        lines.append("runner_command:")
        for token in execution.runner_command:
            lines.append(f"  {token}")
    findings = execution.per_tool_findings or {}
    if findings:
        lines.append("per_tool_findings:")
        for tool, items in findings.items():
            cat_counts: Counter[str] = Counter(
                (item.get("category") if isinstance(item, dict) else getattr(item, "category", "?"))
                for item in items
            )
            summary = ", ".join(f"{cat}={count}" for cat, count in cat_counts.most_common(5))
            lines.append(f"  · {tool}: total={len(items)}  ({summary or 'no findings'})")
    return lines


def _detail_checker(verdict: CheckerVerdict) -> list[str]:
    lines: list[str] = []
    if verdict.status == "ACCEPT":
        lines.append(
            "all 10 sub-checks passed "
            "(action / evidence / completeness / composition_action / ownership / "
            "category_assignments / rag_overrides / assignment_caveats / "
            "composition_completeness / forbidden_claims)"
        )
    else:
        if verdict.hard_failures:
            lines.append(f"hard_failures: {', '.join(verdict.hard_failures)}")
        if verdict.regeneration_reasons:
            lines.append(f"regeneration_reasons: {', '.join(verdict.regeneration_reasons)}")
        if verdict.reasons:
            preview = "; ".join(verdict.reasons[:3])
            lines.append(f"reasons: {preview}")
    if verdict.advisory_reasons:
        preview = "; ".join(verdict.advisory_reasons[:3])
        lines.append(f"advisory: {preview}")
    return lines


def _certificate_selected_tools(certificate: Step2DecisionCertificate) -> list[str]:
    selected_tool_ids: list[str] = []
    for item in sorted(certificate.selected_plan, key=lambda entry: entry.execution_order):
        if item.tool and item.tool not in selected_tool_ids:
            selected_tool_ids.append(item.tool)
    if not selected_tool_ids:
        for tool_id in certificate.tool_categories:
            if tool_id not in selected_tool_ids:
                selected_tool_ids.append(tool_id)

    primary_tool = certificate.primary_tool or (selected_tool_ids[0] if selected_tool_ids else "")
    if primary_tool and primary_tool not in selected_tool_ids:
        selected_tool_ids.insert(0, primary_tool)
    return selected_tool_ids


def _decision_category_lines(
    certificate: Step2DecisionCertificate,
) -> tuple[list[str], list[str]]:
    owners: list[str] = []
    gaps: list[str] = []
    seen_owners: set[str] = set()
    seen_gaps: set[str] = set()

    def add_owner(category: str, tool: str | None) -> None:
        if not category or not tool:
            return
        line = f"{category}->{tool}"
        if line not in seen_owners:
            owners.append(line)
            seen_owners.add(line)

    def add_gap(category: str) -> None:
        if not category or category in seen_gaps:
            return
        gaps.append(category)
        seen_gaps.add(category)

    if certificate.primary_tool:
        add_owner("ALL", certificate.primary_tool)

    for assignment in certificate.category_assignments:
        category = assignment.category
        if not category or category.upper() == "ALL":
            continue
        if assignment.assignment_type in {"gap", "stop_with_gap"} or not assignment.owner_tool:
            add_gap(category)
            continue
        if assignment.assignment_type == "primary_all":
            continue
        add_owner(category, assignment.owner_tool)

    has_specific_owner = any(not line.startswith("ALL->") for line in owners)
    if not has_specific_owner:
        for tool_id, categories in certificate.tool_categories.items():
            for category in categories:
                if not category or category.upper() == "ALL":
                    continue
                add_owner(category, tool_id)

    return owners, gaps


def _format_decision_reason(certificate: Step2DecisionCertificate) -> str:
    lines = [
        f"action={certificate.selected_action_id}",
        f"decision_type={certificate.decision_type}",
    ]
    if certificate.primary_tool:
        lines.append(f"primary_tool={certificate.primary_tool}")

    selected_tools = _certificate_selected_tools(certificate)
    if selected_tools:
        lines.append(f"selected_tools={', '.join(selected_tools)}")

    category_owners, gaps = _decision_category_lines(certificate)
    if category_owners:
        lines.append(f"category_owners={', '.join(category_owners)}")
    if gaps:
        lines.append(f"gaps={', '.join(gaps)}")
    if certificate.engine_fallback_reason:
        lines.append(f"fallback_reason={certificate.engine_fallback_reason}")

    summary = certificate.short_summary.strip()
    if summary:
        lines.append(f"llm_summary={summary}")
    return "\n".join(lines)


def _emit_decision_reason(enabled: bool, certificate: Step2DecisionCertificate) -> None:
    details = _format_decision_reason(certificate)
    if not details:
        return
    title = "Decision Reason" if certificate.engine_fallback_reason else "LLM Reason"
    _emit(enabled, title, details)


def _zero_budget_usage(limit: BudgetProfile) -> BudgetUsage:
    """Return a BudgetUsage with zero estimated use against `limit`."""
    zero = BudgetProfile(tool_slots=0, runtime_cap_minutes=0.0, alert_cap=limit.alert_cap)
    return BudgetUsage(limit=limit, estimated_use=zero, remaining_after_plan=limit)


def _emit_stop_with_gaps_fallback(
    *,
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
    last_hard_failures: list[str],
) -> Step2DecisionCertificate:
    """Build a synthetic STOP_WITH_GAPS certificate for unresolved hard failures."""
    focus = list(packet.primary_attention.confirmed_weak_categories) + list(
        packet.primary_attention.low_support_categories
    )
    primary = packet.primary_attention.primary_tool or ""
    return Step2DecisionCertificate(
        decision_type="STOP",
        selected_action_id="stop_with_gaps",
        selected_plan=(
            [SelectedToolEntry(tool=primary, role="STARTER", execution_order=1)]
            if primary else []
        ),
        primary_tool=primary or None,
        tool_categories={primary: ["ALL"]} if primary else {},
        budget=_zero_budget_usage(matrix.budget_profile),
        forbidden_claims_attestation=ForbiddenClaimsAttestation(),
        short_summary=f"stop_with_gaps focus={','.join(focus)}",
        engine_fallback_reason=(
            "hard_failures_unresolved: " + "; ".join(last_hard_failures)
        ),
    )


def decide_with_ceiling_fallback(
    *,
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
    max_cego_retries: int,
    run_cego_fn: Callable[..., Step2DecisionCertificate],
    check_fn: Callable[
        [Step2DecisionCertificate, ActionByEvidenceMatrix, Step1EvidencePacket],
        CheckerVerdict,
    ],
) -> tuple[Step2DecisionCertificate, CheckerVerdict]:
    """Run CEGO with retry ceiling and apply deterministic fallback policy."""
    last_cert: Step2DecisionCertificate | None = None
    last_verdict: CheckerVerdict | None = None

    for _attempt in range(max_cego_retries + 1):
        cert = run_cego_fn(packet, matrix, last_verdict)
        verdict = check_fn(cert, matrix, packet)
        last_cert, last_verdict = cert, verdict
        if verdict.status == "ACCEPT":
            return cert, verdict

    assert last_cert is not None and last_verdict is not None

    if last_verdict.hard_failures:
        synthetic = _emit_stop_with_gaps_fallback(
            packet=packet,
            matrix=matrix,
            last_hard_failures=list(last_verdict.hard_failures),
        )
        synthetic_verdict = check_fn(synthetic, matrix, packet)
        if synthetic_verdict.status != "ACCEPT":
            raise RuntimeError(
                "synthetic stop_with_gaps certificate failed checker - "
                "this is a Task 6.4 invariant violation"
            )
        return synthetic, synthetic_verdict

    advisory = (
        list(last_verdict.advisory_reasons)
        + ["regen_ceiling_reached"]
        + [f"unresolved:{reason}" for reason in last_verdict.regeneration_reasons]
    )
    accepted_verdict = last_verdict.model_copy(
        update={"status": "ACCEPT", "advisory_reasons": advisory}
    )
    return last_cert, accepted_verdict


def _build_global_category_profile(kb: PerformanceKnowledgeBase | None) -> dict[str, float]:
    """从 performance KB 中所有 entry 的 vulnerability_categories 聚合全局类别分布。"""
    if kb is None:
        return {}

    counts: dict[str, int] = {}
    for entry in kb.entries:
        categories = getattr(entry.dataset_profile, "vulnerability_categories", None)
        if not categories and entry.dataset_profile.complexity_stats is not None:
            categories = entry.dataset_profile.complexity_stats.vulnerability_categories
        for category in categories or []:
            counts[category] = counts.get(category, 0) + 1

    total = sum(counts.values())
    if total <= 0:
        return {}
    return {category: count / total for category, count in counts.items()}


def _build_category_diagnostics(pool: ScenePool) -> CategoryDiagnostics:
    """从 scene pool 的 neighbor category profiles 加权聚合场景类别分布，判断 bias risk。"""
    profile: dict[str, float] = {}
    for neighbor in pool.neighbors:
        for category, value in neighbor.category_profile.items():
            profile[category] = profile.get(category, 0.0) + neighbor.weight * value

    total = sum(profile.values())
    if total > 0:
        profile = {key: value / total for key, value in profile.items()}

    max_share = max(profile.values()) if profile else 0.0
    signals: list[str] = []
    if max_share > 0.7:
        risk = "high"
        signals = [
            f"{category} dominates at {value:.0%}"
            for category, value in profile.items()
            if value > 0.7
        ]
    elif max_share > 0.5:
        risk = "medium"
        signals = [
            f"{category} at {value:.0%}"
            for category, value in profile.items()
            if value > 0.5
        ]
    else:
        risk = "low"
    return CategoryDiagnostics(
        scene_category_profile=profile,
        category_bias_risk=risk,
        bias_signals=signals,
    )


def _empty_kb() -> PerformanceKnowledgeBase:
    return PerformanceKnowledgeBase(knowledge_base_type="performance", entries=[])


def _normalize_raw_findings(tool_id: str, raw_findings: list[dict[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    for raw in raw_findings:
        if raw.get("ignored") is True:
            continue
        category = raw.get("category") or raw.get("vulnerability_type") or raw.get("type") or raw.get("check") or raw.get("name") or raw.get("title") or "unknown"
        category = str(category).strip()
        if category.upper() == "IGNORE":
            continue
        location = str(raw.get("location") or "")
        if not location:
            file_value = raw.get("file") or raw.get("filename") or raw.get("sourceFile") or ""
            line_value = raw.get("line") or raw.get("lineno") or raw.get("startLine") or ""
            if file_value:
                location = f"{file_value}:{line_value}" if line_value else str(file_value)
        confidence = raw.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
                if confidence > 1.0:
                    confidence = confidence / 100.0
            except (TypeError, ValueError):
                confidence = None
        findings.append(
            Finding(
                source_tool=str(raw.get("source_tool") or tool_id),
                category=category,
                location=location,
                severity=raw.get("severity") or raw.get("impact") or raw.get("level"),
                confidence=confidence,
                explanation=str(raw.get("explanation") or raw.get("description") or raw.get("message") or raw.get("info") or ""),
                raw=raw,
            )
        )
    return findings


def _collect_selected_tool_findings(
    execution: ExecutionResult,
    selected_tool_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        tool_id: findings
        for tool_id, findings in execution.per_tool_findings.items()
        if tool_id in selected_tool_ids and findings
    }


def _synthesize_toolcard_findings(
    cards: list[ToolCard],
    composition: CompositionPlan,
) -> dict[str, list[Finding]]:
    card_by_id = {card.tool_id: card for card in cards}
    findings_by_tool: dict[str, list[Finding]] = {}
    for category, tool_id in composition.category_assignments.items():
        card = card_by_id.get(tool_id)
        tool_name = card.tool_name if card else tool_id
        findings_by_tool.setdefault(tool_id, []).append(
            Finding(
                source_tool=tool_id,
                category=category,
                location="coverage-level",
                explanation=f"{tool_name} covers {category} by toolcard capability evidence",
            )
        )
    return findings_by_tool


def _build_fused_report(
    ranked: list[ToolScore],
    cards: list[ToolCard],
    composition: CompositionPlan,
    *,
    execution: ExecutionResult | None = None,
):
    if execution is not None:
        selected_raw = _collect_selected_tool_findings(execution, set(composition.selected_tool_ids))
        if selected_raw:
            anchor_findings = _normalize_raw_findings(
                composition.anchor_tool_id,
                selected_raw.get(composition.anchor_tool_id, []),
            )
            complement_findings = {
                tool_id: _normalize_raw_findings(tool_id, raw)
                for tool_id, raw in selected_raw.items()
                if tool_id != composition.anchor_tool_id
            }
            return fuse_reports(
                anchor_findings,
                complement_findings,
                composition,
                findings_source="execution",
            )
        if execution.status == "executed" and execution.results_root:
            synthesized = _synthesize_toolcard_findings(cards, composition)
            if synthesized:
                return fuse_reports(
                    synthesized.get(composition.anchor_tool_id, []),
                    {
                        tool_id: findings
                        for tool_id, findings in synthesized.items()
                        if tool_id != composition.anchor_tool_id
                    },
                    composition,
                    findings_source="synthesized",
                )
    return None


def _composition_from_certificate(certificate: Step2DecisionCertificate) -> CompositionPlan:
    selected_tool_ids: list[str] = []
    for item in sorted(certificate.selected_plan, key=lambda entry: entry.execution_order):
        if item.tool and item.tool not in selected_tool_ids:
            selected_tool_ids.append(item.tool)
    if not selected_tool_ids:
        for tool_id in certificate.tool_categories:
            if tool_id not in selected_tool_ids:
                selected_tool_ids.append(tool_id)

    primary_tool = certificate.primary_tool or (selected_tool_ids[0] if selected_tool_ids else "")
    if primary_tool and primary_tool not in selected_tool_ids:
        selected_tool_ids.insert(0, primary_tool)

    category_assignments: dict[str, str] = {}
    for assignment in certificate.category_assignments:
        if not assignment.owner_tool:
            continue
        if assignment.assignment_type in {"gap", "stop_with_gap", "primary_all"}:
            continue
        if not assignment.category or assignment.category.upper() == "ALL":
            continue
        category_assignments[assignment.category] = assignment.owner_tool

    if not category_assignments:
        for tool_id, categories in certificate.tool_categories.items():
            if tool_id == primary_tool:
                continue
            for category in categories:
                if category.upper() == "ALL":
                    continue
                category_assignments[category] = tool_id

    return CompositionPlan(
        selected_tool_ids=selected_tool_ids,
        anchor_tool_id=primary_tool,
        complementary_tool_ids=[tool_id for tool_id in selected_tool_ids if tool_id != primary_tool],
        category_assignments=category_assignments,
        rationale=certificate.short_summary,
    )


def _lakes_output_dir(results_root: str | Path) -> Path:
    root = Path(results_root)
    return root if root.name == "LAKES_out" else root / "LAKES_out"


def _contract_output_dir(results_root: str | Path, target_path: str | Path) -> Path:
    target = Path(target_path)
    contract_id = target.stem if target.suffix.lower() == ".sol" else target.name
    return _lakes_output_dir(results_root) / contract_id


def _write_lakes_fused_report(
    *,
    results_root: str | Path,
    target_path: str | Path,
    fused_report: FusedReport,
    composition: CompositionPlan,
    execution: ExecutionResult,
) -> Path:
    lakes_dir = _contract_output_dir(results_root, target_path)
    lakes_dir.mkdir(parents=True, exist_ok=True)
    (lakes_dir / "fused_report.json").write_text(
        json.dumps(compact_fused_report_payload(fused_report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (lakes_dir / "fusion_plan.json").write_text(
        composition.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (lakes_dir / "execution.json").write_text(
        execution.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return lakes_dir


def _run_execution_pipeline(
    *,
    target_path: str | Path,
    composition_plan: CompositionPlan,
    run_results_root: str | Path | None = None,
    runner_script: str | Path | None = None,
    runner_cwd: str | Path | None = None,
    execute_runner: bool = False,
    **execution_kwargs,
) -> tuple[ExecutionResult, Path]:
    root = Path(run_results_root or "LAKES_out")
    lakes_root = _lakes_output_dir(root)
    contract_dir = _contract_output_dir(root, target_path)
    raw_dir = contract_dir / "raw"
    if execute_runner:
        for filename in ("fused_report.json", "fusion_plan.json", "execution.json"):
            stale_file = lakes_root / filename
            if stale_file.exists():
                stale_file.unlink()
        for filename in ("fused_report.json", "fusion_plan.json", "execution.json"):
            stale_file = contract_dir / filename
            if stale_file.exists():
                stale_file.unlink()
        for stale_dir in lakes_root.glob("laskes_run_*"):
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    planned = build_execution_plan(
        target_path,
        raw_dir,
        composition_plan,
        runner_script=runner_script,
        runner_cwd=runner_cwd,
        write_lakes_output=False,
        **execution_kwargs,
    )
    execution = execute_plan(planned) if execute_runner else planned
    per_tool_findings = report_parser.load_per_tool_findings_from_run_dir(
        raw_dir,
        set(composition_plan.selected_tool_ids),
    )
    updates: dict[str, Any] = {"results_root": str(raw_dir)}
    if per_tool_findings:
        updates["per_tool_findings"] = per_tool_findings
    return execution.model_copy(update=updates), raw_dir


def _load_passage_retriever(
    *,
    passage_store_path: Path,
    vector_index_path: Path | None = None,
    enable_retrieval: bool = True,
) -> PassageRetriever | None:
    if not enable_retrieval:
        return None
    pstore = load_passage_store(passage_store_path)
    if not pstore or not pstore.passages:
        return None
    if vector_index_path is None or not vector_index_path.exists():
        return None
    index = VectorIndex.load(vector_index_path)
    return PassageRetriever(pstore, index=index)


def run_recommendation(
    *,
    target_path: str | None,
    toolcards_dir: str,
    tool_slots: int = 1,
    runtime_cap_minutes: float = 30.0,
    alert_cap: str = "medium",
    model: str = "",
    top_k: int = 5,
    max_cego_retries: int = 2,
    emit_stderr: bool = True,
    explain: bool = False,
    enable_retrieval: bool = True,
    passage_store_path: str | None = None,
    vector_index_path: str | None = None,
    execute: bool = False,
    run_results_root: str | None = None,
    runner_script: str | None = None,
    runner_cwd: str | None = None,
    tool_timeout_sec: int = 1200,
    gptscan_timeout_sec: int = 600,
    execution_jobs: int = 0,
    openai_api_key: str | None = None,
    openai_api_base: str | None = None,
) -> PipelineResult:
    """端到端管线：SCREC → DACE-RAG → CEGO → Checker。"""
    cards = load_toolcards(toolcards_dir)
    toolcards_path = Path(toolcards_dir)
    performance_db_path = toolcards_path / "performance_db.json"
    kb = load_performance_db(performance_db_path) if performance_db_path.exists() else None
    resolved_passage_store_path = Path(passage_store_path) if passage_store_path else toolcards_path / "passage_store.json"
    resolved_vector_index_path = (
        Path(vector_index_path) if vector_index_path else toolcards_path / "vector_index" / "index.json"
    )
    retriever = _load_passage_retriever(
        passage_store_path=resolved_passage_store_path,
        vector_index_path=resolved_vector_index_path,
        enable_retrieval=enable_retrieval,
    )
    features = analyze_target(target_path)
    budget = BudgetProfile(
        tool_slots=tool_slots,
        runtime_cap_minutes=runtime_cap_minutes,
        alert_cap=alert_cap,
    )
    warnings: list[str] = []
    if kb is None:
        warnings.append("performance_db.json not found; benchmark evidence is unavailable")
    if enable_retrieval and retriever is None:
        if not resolved_passage_store_path.exists():
            warnings.append("passage_store.json not found; RAG retrieval is disabled")
        elif not resolved_vector_index_path.exists():
            warnings.append("vector_index/index.json not found; RAG retrieval is disabled")
        else:
            warnings.append("passage_store.json is empty; RAG retrieval is disabled")
    _emit(
        emit_stderr,
        "Load",
        f"cards={len(cards)} kb_entries={len(kb.entries) if kb else 0} "
        f"retriever={'active' if retriever else 'off'} target={target_path or 'none'}",
    )

    pool = build_scene_pool(features, kb, top_k=top_k) if kb else ScenePool(neighbors=[])
    _emit(emit_stderr, "Scene Pool", f"neighbors={len(pool.neighbors)}")
    _explain(explain, _detail_scene_pool(pool))

    tool_table = build_tool_table(cards, features, budget)
    feasible_count = sum(1 for entry in tool_table if entry.feasible)
    _emit(emit_stderr, "Tool Table", f"tools={len(tool_table)} feasible={feasible_count}")
    _explain(explain, _detail_tool_table(tool_table))

    tool_ids = [card.tool_id for card in cards]
    if kb:
        nominal_scores, tool_slice_scores = compute_scene_scores(pool, kb, tool_ids)
    else:
        nominal_scores, tool_slice_scores = [], {}
    _emit(emit_stderr, "Scene Scoring", f"nominal_scores={len(nominal_scores)}")
    _explain(explain, _detail_scene_scoring(nominal_scores))

    global_profile = _build_global_category_profile(kb)
    stress = run_stress_test(pool, tool_slice_scores, global_profile)

    diagnostics = _build_category_diagnostics(pool)
    _emit(
        emit_stderr,
        "Diagnostics",
        f"category_bias_risk={diagnostics.category_bias_risk}",
    )
    _explain(explain, _detail_diagnostics(diagnostics))

    score_panel = ScorePanel(nominal_scores=nominal_scores, stress_rankings=stress)
    matched_ids = {neighbor.paper_id for neighbor in pool.neighbors if neighbor.paper_id} or None
    rcov = (
        build_recall_coverage(kb, tool_ids, matched_source_ids=matched_ids)
        if kb
        else RecallCoverageMatrix(taxonomy_level="parent")
    )
    _emit(emit_stderr, "Recall Coverage", f"entries={len(rcov.matrix)}")

    certification = certify(score_panel, diagnostics, rcov, tool_table)
    _emit(emit_stderr, "Certification", f"status={certification.status}")
    _explain(explain, _detail_certification(certification))

    packet = build_evidence_packet(
        features,
        cards,
        kb or _empty_kb(),
        budget,
        tool_table,
        pool,
        score_panel,
        diagnostics,
        rcov,
        certification,
    )
    _emit(emit_stderr, "Evidence Packet", f"focus={len(packet.dace_rag_focus)}")
    top_scene_lines, tier1_decisions = _detail_top_scene_support(
        kb, pool, packet.primary_attention
    )
    _explain(
        explain,
        _detail_evidence_packet(packet)
        + top_scene_lines
        + _detail_rcov(packet.recall_coverage, packet.primary_attention, tier1_decisions),
    )

    matrix = build_action_evidence_matrix(packet, budget, retriever=retriever)
    legal_actions = sum(1 for action in matrix.actions if action.legal)
    _emit(emit_stderr, "DACE-RAG", f"actions={len(matrix.actions)} legal={legal_actions}")
    _explain(explain, _detail_dace_actions(matrix))
    _explain(explain, _detail_ownership_panel(matrix))
    if explain:
        _emit(emit_stderr, "RAG Passages", f"injected={sum(1 for c in matrix.evidence_cards if c.evidence_type == 'rag_passage')}")
        _explain(explain, _detail_rag_passages(matrix))

    client = load_openai_client()
    if client is None:
        raise RuntimeError("LLM endpoint unavailable")

    attempt_counter = {"value": 0}

    def _run_cego_with_prev(
        pkt: Step1EvidencePacket,
        mtx: ActionByEvidenceMatrix,
        _prev: CheckerVerdict | None,
    ) -> Step2DecisionCertificate:
        return run_cego(client, model, pkt, mtx)

    def _check_with_emit(
        cert: Step2DecisionCertificate,
        mtx: ActionByEvidenceMatrix,
        pkt: Step1EvidencePacket,
    ) -> CheckerVerdict:
        verdict = check_decision(cert, pkt, mtx)
        attempt_counter["value"] += 1
        _emit(
            emit_stderr,
            "Checker",
            f"attempt={attempt_counter['value']} status={verdict.status}",
        )
        _explain(explain, _detail_checker(verdict))
        return verdict

    certificate, verdict = decide_with_ceiling_fallback(
        packet=packet,
        matrix=matrix,
        max_cego_retries=max_cego_retries,
        run_cego_fn=_run_cego_with_prev,
        check_fn=_check_with_emit,
    )
    _emit_decision_reason(emit_stderr, certificate)

    execution: ExecutionResult | None = None
    fused_report: FusedReport | None = None
    lakes_output_dir: str | None = None
    if execute:
        if not target_path:
            raise RuntimeError("--execute requires a target contract file or directory")
        if certificate.decision_type == "STOP":
            warnings.append("execution skipped because CEGO selected STOP")
        else:
            composition = _composition_from_certificate(certificate)
            results_root = Path(run_results_root or "LAKES_out")
            execution, _temp_dir = _run_execution_pipeline(
                target_path=target_path,
                composition_plan=composition,
                run_results_root=results_root,
                runner_script=runner_script,
                runner_cwd=runner_cwd,
                execute_runner=True,
                tool_timeout_sec=tool_timeout_sec,
                gptscan_timeout_sec=gptscan_timeout_sec,
                execution_jobs=execution_jobs,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
            )
            _emit(
                emit_stderr,
                "Execution",
                f"status={execution.status if execution else 'none'} "
                f"return_code={execution.return_code if execution else 'none'}",
            )
            _explain(explain, _detail_execution(execution))
            fused_report = _build_fused_report(
                nominal_scores,
                cards,
                composition,
                execution=execution,
            )
            if fused_report is not None:
                lakes_output_dir = str(
                    _write_lakes_fused_report(
                        results_root=results_root,
                        target_path=target_path,
                        fused_report=fused_report,
                        composition=composition,
                        execution=execution,
                    )
                )
                _emit(emit_stderr, "LAKES_out", f"fused_report={Path(lakes_output_dir) / 'fused_report.json'}")
            else:
                warnings.append("execution completed but no fused report could be built")

    return PipelineResult(
        features=features,
        packet=packet,
        matrix=matrix,
        certificate=certificate,
        checker_verdict=verdict,
        execution=execution,
        fused_report=fused_report,
        lakes_output_dir=lakes_output_dir,
        warnings=warnings,
    )
