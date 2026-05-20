"""Assemble Step 1 SCREC evidence packets."""

from __future__ import annotations

from dataclasses import dataclass

from toolrank.feasibility import check_feasibility
from toolrank.schemas import ContractFeatures, D1Metric, DASP10_CATEGORIES, PerformanceKnowledgeBase, ToolCard
from toolrank.schemas_v2 import (
    BudgetProfile,
    CategoryDiagnostics,
    CertificationVerdict,
    DACERAGFocusItem,
    PerformanceDBEvidenceRow,
    PrimaryAttention,
    RecallCoverageMatrix,
    ScenePool,
    ScorePanel,
    Step1EvidencePacket,
    ToolCostEntry,
    ToolOverallMetricsRow,
    ToolTableEntry,
)


LOW_SUPPORT_TOTAL = 10
WEAK_RATE_THRESHOLD = 0.3
TOP_SCENE_CLEAR_WEAK_GAP = 0.2
_DASP10_CATEGORIES = set(DASP10_CATEGORIES)


@dataclass(frozen=True)
class _CategorySupport:
    detected: int | None
    total: int | None
    rate: float
    source_rank: int


def _tool_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _evidence_id(source_id: str, dataset_name: str, tool: str, category: str) -> str:
    safe_source = "".join(ch if ch.isalnum() else "_" for ch in source_id.lower()).strip("_")
    safe_dataset = "".join(ch if ch.isalnum() else "_" for ch in dataset_name.lower()).strip("_")
    safe_tool = "".join(ch if ch.isalnum() else "_" for ch in tool.lower()).strip("_")
    safe_category = "".join(ch if ch.isalnum() else "_" for ch in category.lower()).strip("_")
    return f"ev_dataset_{safe_source}_{safe_dataset}_{safe_tool}_{safe_category}"


def _metric_payload(card: ToolCard) -> D1Metric | None:
    if not card.d1_metrics:
        return None
    if "default" in card.d1_metrics:
        return card.d1_metrics["default"]
    return card.d1_metrics[sorted(card.d1_metrics)[0]]


def _runtime_minutes(card: ToolCard) -> float | None:
    metric = _metric_payload(card)
    if metric is None or metric.resolved_time_sec is None:
        return None
    return metric.resolved_time_sec / 60.0


def _runtime_bucket(runtime_minutes: float | None) -> str:
    if runtime_minutes is None:
        return "unknown"
    if runtime_minutes < 5.0:
        return "low"
    if runtime_minutes < 30.0:
        return "medium"
    return "high"


def _family(card: ToolCard) -> str:
    mode = card.d8_mode.value
    if mode == "static":
        return "static_source" if card.d7_input_support.sol else "static_bytecode"
    if mode in {"symbolic", "fuzz", "ml", "llm"}:
        return mode
    return "unknown"


def build_tool_table(
    cards: list[ToolCard],
    features: ContractFeatures,
    budget: BudgetProfile,
) -> list[ToolTableEntry]:
    entries: list[ToolTableEntry] = []
    for card in cards:
        feasibility = check_feasibility(card, features, budget)
        runtime_minutes = _runtime_minutes(card)
        entries.append(
            ToolTableEntry(
                tool=card.tool_id,
                family=_family(card),
                feasible=feasibility.feasible,
                feasibility_reasons=feasibility.reasons,
                expected_runtime_bucket=_runtime_bucket(runtime_minutes),
                failure_risk_bucket="unknown",
                tool_cost=ToolCostEntry(
                    tool_slots=1,
                    expected_runtime_minutes=runtime_minutes,
                    alert_risk=budget.alert_cap,
                ),
            )
        )
    return entries


def _pick_eligible_in_scene(
    scene_support: dict[str, dict[str, _CategorySupport]] | None,
    source_tool: str,
    category: str,
) -> str | None:
    """Apply rate>=WEAK_RATE_THRESHOLD + (total>=LOW_SUPPORT_TOTAL OR rate>=0.5)."""
    if not scene_support:
        return None
    by_tool = scene_support.get(category, {})
    eligible: list[tuple[str, _CategorySupport]] = []
    for tool, sup in by_tool.items():
        if tool == source_tool:
            continue
        if sup.rate is None or sup.rate < WEAK_RATE_THRESHOLD:
            continue
        if (sup.total is not None and sup.total >= LOW_SUPPORT_TOTAL) or sup.rate >= 0.5:
            eligible.append((tool, sup))
    if not eligible:
        return None
    best = max(eligible, key=lambda item: (item[1].rate, item[1].total or 0, item[0]))
    return best[0]


def _best_hedge_tool(
    rcov: RecallCoverageMatrix,
    source_tool: str,
    category: str,
    top_scene_support: dict[str, dict[str, _CategorySupport]] | None = None,
    near_scene_support: dict[str, dict[str, _CategorySupport]] | None = None,
) -> tuple[str | None, str]:
    """Pick hedge tool: top-scene → near-scenes → full-aggregate fallback.

    Three-tier mirror of PA's scene-aware philosophy:
    - tier-1: best non-primary in scene_pool.neighbors[0]
    - tier-2: best non-primary aggregated over neighbors[1:] (excluding top-scene)
    - tier-3: best non-primary in full rcov aggregate (incl. top-scene)
    All tiers enforce the same eligibility threshold so a weak-everywhere hedge
    is never picked.

    Returns (hedge_tool, source_tier).
    source_tier ∈ {"top_scene", "near_scene", "aggregate_fallback", "no_strong_candidate"}.
    """
    pick = _pick_eligible_in_scene(top_scene_support, source_tool, category)
    if pick:
        return pick, "top_scene"

    pick = _pick_eligible_in_scene(near_scene_support, source_tool, category)
    if pick:
        return pick, "near_scene"

    eligible_rows = [
        row
        for row in rcov.matrix
        if row.category == category
        and row.tool != source_tool
        and row.R_hat is not None
        and row.R_hat >= WEAK_RATE_THRESHOLD
        and (
            (row.total is not None and row.total >= LOW_SUPPORT_TOTAL)
            or row.R_hat >= 0.5
        )
    ]
    if eligible_rows:
        best = max(
            eligible_rows,
            key=lambda row: (row.R_hat or 0.0, row.total or 0, row.tool),
        )
        return best.tool, "aggregate_fallback"

    return None, "no_strong_candidate"


def generate_dace_rag_focus(
    certification: CertificationVerdict,
    rcov: RecallCoverageMatrix,
    primary_attention: PrimaryAttention | None = None,
    *,
    top_scene_support: dict[str, dict[str, _CategorySupport]] | None = None,
    near_scene_support: dict[str, dict[str, _CategorySupport]] | None = None,
) -> list[DACERAGFocusItem]:
    source_tools: list[str] = []
    if certification.status == "certified_primary" and certification.certified_primary:
        source_tools = [certification.certified_primary]
    elif certification.status == "candidate_set" and certification.candidate_set:
        source_tools = list(certification.candidate_set)

    focus: list[DACERAGFocusItem] = []
    for source_tool in source_tools:
        categories = list(rcov.weak_categories_by_tool.get(source_tool, []))
        if primary_attention is not None and source_tool == primary_attention.primary_tool:
            categories = list(primary_attention.confirmed_weak_categories) + list(
                primary_attention.low_support_categories
            )
        for category in categories:
            hedge_tool, tier = _best_hedge_tool(
                rcov,
                source_tool,
                category,
                top_scene_support=top_scene_support,
                near_scene_support=near_scene_support,
            )
            if hedge_tool is None:
                continue
            focus.append(
                DACERAGFocusItem(
                    tool=hedge_tool,
                    category=category,
                    reason=f"hedge_for_{source_tool}_gap_via_{tier}",
                )
            )
    return focus


def _resolve_primary_tool(certification: CertificationVerdict, score_panel: ScorePanel) -> str | None:
    if certification.certified_primary:
        return certification.certified_primary
    if certification.candidate_set:
        return certification.candidate_set[0]
    for score in sorted(score_panel.nominal_scores, key=lambda item: item.rank):
        return score.tool
    return None


def _top_scene_source_ids(scene_pool: ScenePool | None) -> list[str]:
    if scene_pool is None or not scene_pool.neighbors:
        return []
    top = scene_pool.neighbors[0]
    ids: list[str] = []
    if top.paper_id:
        ids.append(top.paper_id)
    ids.extend(top.provenance_refs)
    return list(dict.fromkeys(ids))


def _observation_category_support(observation, category: str) -> tuple[int | None, int | None, float] | None:
    counts = observation.vulnerability_score_counts or {}
    count = counts.get(category)
    if count is not None:
        if count.total <= 0:
            return None
        return count.detected, count.total, count.detected / count.total

    scores = observation.vulnerability_scores or {}
    score = scores.get(category)
    if score is None:
        return None
    return None, None, float(score)


def _top_scene_support_by_category(
    kb: PerformanceKnowledgeBase | None,
    scene_pool: ScenePool | None,
    tool_ids: list[str] | None,
) -> dict[str, dict[str, _CategorySupport]]:
    if kb is None or tool_ids is None:
        return {}
    top_sources = _top_scene_source_ids(scene_pool)
    if not top_sources:
        return {}

    source_rank = {source_id: index for index, source_id in enumerate(top_sources)}
    requested_by_key = {_tool_key(tool_id): tool_id for tool_id in tool_ids}
    by_category: dict[str, dict[str, _CategorySupport]] = {}
    for entry in kb.entries:
        rank = source_rank.get(entry.source_id)
        if rank is None:
            continue
        for observation in entry.tool_performance_data:
            tool = requested_by_key.get(_tool_key(observation.tool_name))
            if tool is None:
                continue
            categories = set(observation.vulnerability_score_counts or {}) | set(
                observation.vulnerability_scores or {}
            )
            for category in categories & _DASP10_CATEGORIES:
                support = _observation_category_support(observation, category)
                if support is None:
                    continue
                detected, total, rate = support
                category_support = by_category.setdefault(category, {})
                existing = category_support.get(tool)
                if existing is not None and existing.source_rank <= rank:
                    continue
                category_support[tool] = _CategorySupport(
                    detected=detected,
                    total=total,
                    rate=rate,
                    source_rank=rank,
                )
    return by_category


def _near_scene_source_ids(scene_pool: ScenePool | None) -> list[str]:
    """Source ids from neighbors[1:] (excludes the top scene)."""
    if scene_pool is None or len(scene_pool.neighbors) <= 1:
        return []
    ids: list[str] = []
    for neighbor in scene_pool.neighbors[1:]:
        if neighbor.paper_id:
            ids.append(neighbor.paper_id)
        ids.extend(neighbor.provenance_refs)
    return list(dict.fromkeys(ids))


def _near_scene_support_by_category(
    kb: PerformanceKnowledgeBase | None,
    scene_pool: ScenePool | None,
    tool_ids: list[str] | None,
) -> dict[str, dict[str, _CategorySupport]]:
    """Per-category support aggregated across near scenes (neighbors[1:]).

    Aggregates detected/total/rate across all near sources for each (tool, category)
    so the hedge picker sees a clean cross-near-scene view that is not diluted by
    the top scene's potential zero rows.
    """
    if kb is None or tool_ids is None:
        return {}
    near_sources = _near_scene_source_ids(scene_pool)
    if not near_sources:
        return {}
    near_source_set = set(near_sources)
    requested_by_key = {_tool_key(tool_id): tool_id for tool_id in tool_ids}
    count_acc: dict[tuple[str, str], tuple[int, int]] = {}
    rate_acc: dict[tuple[str, str], tuple[float, int]] = {}
    for entry in kb.entries:
        if entry.source_id not in near_source_set:
            continue
        for observation in entry.tool_performance_data:
            tool = requested_by_key.get(_tool_key(observation.tool_name))
            if tool is None:
                continue
            counts = observation.vulnerability_score_counts or {}
            scores = observation.vulnerability_scores or {}
            for category in (set(counts) | set(scores)) & _DASP10_CATEGORIES:
                count = counts.get(category)
                if count is not None:
                    if count.total <= 0:
                        continue
                    detected, total = count_acc.get((tool, category), (0, 0))
                    count_acc[(tool, category)] = (
                        detected + count.detected,
                        total + count.total,
                    )
                else:
                    score = scores.get(category)
                    if score is None:
                        continue
                    rate_sum, rate_count = rate_acc.get((tool, category), (0.0, 0))
                    rate_acc[(tool, category)] = (rate_sum + float(score), rate_count + 1)
    by_category: dict[str, dict[str, _CategorySupport]] = {}
    for (tool, category), (detected, total) in count_acc.items():
        if total <= 0:
            continue
        by_category.setdefault(category, {})[tool] = _CategorySupport(
            detected=detected,
            total=total,
            rate=detected / total,
            source_rank=1,
        )
    for (tool, category), (rate_sum, rate_count) in rate_acc.items():
        if rate_count <= 0 or (tool, category) in count_acc:
            continue
        by_category.setdefault(category, {})[tool] = _CategorySupport(
            detected=None,
            total=None,
            rate=rate_sum / rate_count,
            source_rank=1,
        )
    return by_category


def _top_scene_primary_decision(
    top_scene_support: dict[str, dict[str, _CategorySupport]],
    *,
    primary_tool: str,
    category: str,
) -> str:
    by_tool = top_scene_support.get(category)
    if not by_tool:
        return "fallback"
    primary = by_tool.get(primary_tool)
    if primary is None:
        return "fallback"

    best_other = max(
        (support for tool, support in by_tool.items() if tool != primary_tool),
        key=lambda support: support.rate,
        default=None,
    )
    has_clear_gap = (
        best_other is not None
        and best_other.rate >= WEAK_RATE_THRESHOLD
        and best_other.rate - primary.rate >= TOP_SCENE_CLEAR_WEAK_GAP
    )
    if primary.total is not None and primary.total >= LOW_SUPPORT_TOTAL:
        return "weak" if has_clear_gap else "not_weak"
    if primary.rate is not None and primary.rate >= WEAK_RATE_THRESHOLD and not has_clear_gap:
        return "not_weak"
    return "fallback"


def _partition_primary_categories(
    rcov: RecallCoverageMatrix,
    primary_tool: str | None,
    *,
    kb: PerformanceKnowledgeBase | None = None,
    scene_pool: ScenePool | None = None,
    tool_ids: list[str] | None = None,
    top_scene_support: dict[str, dict[str, _CategorySupport]] | None = None,
) -> tuple[list[str], list[str]]:
    if primary_tool is None:
        return [], []

    if top_scene_support is None:
        top_scene_support = _top_scene_support_by_category(kb, scene_pool, tool_ids)
    confirmed_weak: set[str] = set()
    low_support: set[str] = set()
    for entry in rcov.matrix:
        if entry.tool != primary_tool or entry.category not in _DASP10_CATEGORIES:
            continue
        top_scene_decision = _top_scene_primary_decision(
            top_scene_support,
            primary_tool=primary_tool,
            category=entry.category,
        )
        if top_scene_decision == "not_weak":
            continue
        if top_scene_decision == "weak":
            confirmed_weak.add(entry.category)
            continue
        if entry.total is None or entry.total < LOW_SUPPORT_TOTAL:
            low_support.add(entry.category)
            continue
        if entry.R_hat is not None and entry.R_hat < WEAK_RATE_THRESHOLD:
            confirmed_weak.add(entry.category)

    return sorted(confirmed_weak), sorted(low_support)


def generate_performance_db_view(
    kb: PerformanceKnowledgeBase,
    primary_attention: PrimaryAttention,
    tool_ids: list[str],
    matched_source_ids: set[str] | None = None,
) -> list[PerformanceDBEvidenceRow]:
    categories = set(primary_attention.confirmed_weak_categories) | set(
        primary_attention.low_support_categories
    )
    if not categories:
        return []

    requested_by_key = {_tool_key(tool_id): tool_id for tool_id in tool_ids}
    rows: list[PerformanceDBEvidenceRow] = []
    seen_ids: set[str] = set()
    for entry in kb.entries:
        for observation in entry.tool_performance_data:
            tool = requested_by_key.get(_tool_key(observation.tool_name))
            if tool is None:
                continue
            counts = observation.vulnerability_score_counts or {}
            scores = observation.vulnerability_scores or {}
            for category in sorted(categories & (set(counts) | set(scores))):
                count = counts.get(category)
                detected = count.detected if count is not None else None
                total = count.total if count is not None else None
                if total is not None and total <= 0:
                    continue
                r_hat = detected / total if detected is not None and total else scores.get(category)
                evidence_id = _evidence_id(
                    entry.source_id,
                    entry.dataset_profile.dataset_name,
                    tool,
                    category,
                )
                if evidence_id in seen_ids:
                    continue
                seen_ids.add(evidence_id)
                rows.append(
                    PerformanceDBEvidenceRow(
                        evidence_id=evidence_id,
                        source_id=entry.source_id,
                        dataset_name=entry.dataset_profile.dataset_name,
                        tool=tool,
                        category=category,
                        detected=detected,
                        total=total,
                        R_hat=r_hat,
                    )
                )
    matched_sources = matched_source_ids or set()
    rows.sort(
        key=lambda row: (
            row.category,
            0 if row.source_id in matched_sources else 1,
            0 if row.detected is not None and row.total is not None else 1,
            -(row.R_hat if row.R_hat is not None else -1.0),
            -(row.total or 0),
            row.tool,
            row.source_id,
        )
    )
    return rows


def generate_tool_overall_metrics_view(
    kb: PerformanceKnowledgeBase,
    tool_ids: list[str],
) -> list[ToolOverallMetricsRow]:
    """Surface per-tool overall (precision/recall/F1) metrics per dataset.

    Per-category precision is unavailable in the public sources we ingest, but
    each entry typically reports tool-level overall precision/recall/F1.  Expose
    those so the LLM can weigh FP-burden alongside the per-category recall in
    ``performance_db_view``.
    """

    requested_by_key = {_tool_key(tool_id): tool_id for tool_id in tool_ids}
    rows: list[ToolOverallMetricsRow] = []
    seen: set[tuple[str, str]] = set()
    for entry in kb.entries:
        for observation in entry.tool_performance_data:
            tool = requested_by_key.get(_tool_key(observation.tool_name))
            if tool is None:
                continue
            metrics = getattr(observation, "metrics", None)
            if metrics is None:
                continue
            precision = getattr(metrics, "precision", None)
            recall = getattr(metrics, "recall", None)
            f1 = getattr(metrics, "f1_score", None)
            if f1 is None:
                f1 = getattr(metrics, "f1", None)
            exec_time = getattr(metrics, "execution_time_avg", None)
            if precision is None and recall is None and f1 is None:
                continue
            key = (tool, entry.source_id)
            if key in seen:
                continue
            seen.add(key)
            evidence_id = (
                f"ev_overall_{entry.source_id}_{_tool_key(tool)}"
            )
            rows.append(
                ToolOverallMetricsRow(
                    evidence_id=evidence_id,
                    source_id=entry.source_id,
                    dataset_name=entry.dataset_profile.dataset_name,
                    tool=tool,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    execution_time_avg=exec_time,
                )
            )
    rows.sort(key=lambda r: (r.tool, r.source_id))
    return rows


def build_evidence_packet(
    features: ContractFeatures,
    cards: list[ToolCard],
    kb: PerformanceKnowledgeBase,
    budget: BudgetProfile,
    tool_table: list[ToolTableEntry],
    scene_pool: ScenePool,
    score_panel: ScorePanel,
    diagnostics: CategoryDiagnostics,
    rcov: RecallCoverageMatrix,
    certification: CertificationVerdict,
) -> Step1EvidencePacket:
    tool_ids = [card.tool_id for card in cards]
    primary_tool = _resolve_primary_tool(certification, score_panel)
    top_scene_support = _top_scene_support_by_category(kb, scene_pool, tool_ids)
    near_scene_support = _near_scene_support_by_category(kb, scene_pool, tool_ids)
    confirmed_weak, low_support = _partition_primary_categories(
        rcov,
        primary_tool,
        kb=kb,
        scene_pool=scene_pool,
        tool_ids=tool_ids,
        top_scene_support=top_scene_support,
    )
    primary_attention = PrimaryAttention(
        primary_tool=primary_tool,
        confirmed_weak_categories=confirmed_weak,
        low_support_categories=low_support,
    )
    matched_source_ids = {neighbor.paper_id for neighbor in scene_pool.neighbors if neighbor.paper_id}
    for neighbor in scene_pool.neighbors:
        matched_source_ids.update(neighbor.provenance_refs)

    return Step1EvidencePacket(
        target_contract={
            "feature_source": "deterministic_parser",
            "features": features.model_dump(),
        },
        tool_table=tool_table,
        scene_pool=scene_pool,
        score_panel=score_panel,
        category_diagnostics=diagnostics,
        recall_coverage=rcov,
        performance_db_view=generate_performance_db_view(
            kb,
            primary_attention,
            tool_ids,
            matched_source_ids=matched_source_ids,
        ),
        tool_overall_metrics=generate_tool_overall_metrics_view(kb, tool_ids),
        certification=certification,
        primary_attention=primary_attention,
        dace_rag_focus=generate_dace_rag_focus(
            certification,
            rcov,
            primary_attention,
            top_scene_support=top_scene_support,
            near_scene_support=near_scene_support,
        ),
        provenance_index=[],
    )
