"""Pydantic models for the SCREC, DACE-RAG, CEGO, and checker pipeline."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AlertCap = Literal["low", "medium", "high"]
ActionType = Literal[
    "RUN_PRIMARY",
    "RUN_ROBUST_SINGLE",
    "PLAN_COMPOSITION",
    "CONTINUE_HEDGE",
    "STOP",
]
EvidenceSlot = Literal["FOR", "AGAINST", "COMPARE", "GAP"]
ConfidenceLevel = Literal["low", "medium", "high"]
EvidenceAggregationLevel = Literal[
    "category_level",
    "tool_level",
    "dataset_level",
    "scene_level",
    "paper_level",
    "unknown",
]
EvidenceDecisionRole = Literal[
    "support",
    "oppose",
    "compare",
    "gap",
    "constraint",
    "caveat",
]
EvidenceSourceReliability = Literal[
    "manual_curated",
    "benchmark",
    "artifact",
    "peer_reviewed",
    "documentation",
    "experiment_log",
    "unknown",
]
ActionId = Literal[
    "SINGLE_TOOL",
    "PLAN_COMPOSITION",
    "CONTINUE_HEDGE",
    "STOP_WITH_GAPS",
]
KnowledgeKind = Literal[
    "category_capability",
    "tool_complementarity",
    "fp_precision_risk",
    "failure_mode",
    "hard_scheduling_rule",
]
RelationToOwner = Literal[
    "supports_owner",
    "opposes_owner",
    "owner_stronger",
    "owner_weaker",
    "owner_complements",
    "evidence_gap",
    "owner_ineligible",
]
EvidenceBasis = Literal[
    "benchmark_result",
    "paired_ablation",
    "official_documentation",
    "reproducible_issue",
    "manual_curation",
]
SourceReliability = Literal[
    "peer_reviewed",
    "artifact",
    "manual_curated",
    "official_tool_doc",
    "maintainer_issue",
    "internal_eval",
    "community_report",
]
EvidenceTier = Literal["hard", "medium", "weak"]

ALLOWED_TAG_PREFIXES = (
    "scene:",
    "scale:",
    "input:",
    "solc:",
    "toolver:",
    "evm:",
    "analysis:",
)

SCHEDULING_ACTIONS: set[ActionId] = {"SINGLE_TOOL", "PLAN_COMPOSITION", "CONTINUE_HEDGE"}

ALLOWED_RELATIONS: dict[KnowledgeKind, set[RelationToOwner]] = {
    "category_capability": {"supports_owner", "owner_stronger", "owner_weaker", "evidence_gap"},
    "tool_complementarity": {"owner_complements", "owner_stronger", "owner_weaker", "evidence_gap"},
    "fp_precision_risk": {"opposes_owner", "owner_stronger", "owner_weaker"},
    "failure_mode": {"opposes_owner", "owner_ineligible", "evidence_gap"},
    "hard_scheduling_rule": {"owner_ineligible"},
}

GLOBAL_CATEGORY = "__GLOBAL__"


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_type: Literal[
        "paper_table_cell",
        "paper_text",
        "benchmark_metadata",
        "toolcard",
        "step1_field",
        "runtime_output",
        "rag_passage",
    ]
    paper_id: str | None = None
    page: int | None = Field(default=None, ge=0)
    table: str | None = None
    cell: str | None = None
    field_path: str | None = None
    extraction_confidence: ConfidenceLevel = "medium"
    taxonomy_mapping_confidence: ConfidenceLevel | None = None


class BudgetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_slots: int = Field(ge=0)
    runtime_cap_minutes: float = Field(ge=0.0)
    alert_cap: AlertCap = "medium"


class ToolCostEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_slots: int = Field(default=1, ge=0)
    expected_runtime_minutes: float | None = Field(default=None, ge=0.0)
    alert_risk: AlertCap = "medium"


class SceneNeighbor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice_id: str
    benchmark_family: str
    paper_id: str | None = None
    weight: float = Field(ge=0.0, le=1.0)
    distance: float = Field(ge=0.0)
    category_profile: dict[str, float] = Field(default_factory=dict)
    provenance_refs: list[str] = Field(default_factory=list)


class ScenePool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: Literal["benchmark_slice"] = "benchmark_slice"
    neighbors: list[SceneNeighbor] = Field(default_factory=list)


class NominalToolScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    S_scene: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    P_scene: float | None = Field(default=None, ge=0.0, le=1.0)
    R_scene: float | None = Field(default=None, ge=0.0, le=1.0)
    F1_scene: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_level: Literal[
        "unsupported",
        "global_weak",
        "global_moderate",
        "local_weak",
        "local_moderate",
        "local_strong",
    ] = "unsupported"


class StressRankings(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    local: list[str] = Field(default_factory=list)
    global_: list[str] = Field(default_factory=list, alias="global")
    uniform_supported: list[str] = Field(default_factory=list)
    top1_flip: bool = False
    support_failures: list[str] = Field(default_factory=list)


class ScorePanel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nominal_scores: list[NominalToolScore] = Field(default_factory=list)
    stress_rankings: StressRankings


class CategoryDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_category_profile: dict[str, float] = Field(default_factory=dict)
    category_bias_risk: Literal["low", "medium", "high"] = "low"
    bias_signals: list[str] = Field(default_factory=list)


class RecallCoverageEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    category: str
    detected: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    R_hat: float | None = Field(default=None, ge=0.0, le=1.0)
    support_level: Literal["unsupported", "weak", "medium", "strong"] = "unsupported"
    evidence_refs: list[str] = Field(default_factory=list)


class RecallCoverageMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taxonomy_level: Literal["parent", "leaf"] = "parent"
    matrix: list[RecallCoverageEntry] = Field(default_factory=list)
    weak_categories_by_tool: dict[str, list[str]] = Field(default_factory=dict)


class CertificationVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "certified_primary",
        "candidate_set",
        "no_feasible_tool",
        "insufficient_evidence",
    ]
    certified_primary: str | None = None
    candidate_set: list[str] | None = None
    reason_codes: list[str] = Field(default_factory=list)


class PrimaryAttention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_tool: str | None = None
    confirmed_weak_categories: list[str] = Field(default_factory=list)
    low_support_categories: list[str] = Field(default_factory=list)


class ToolTableEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    family: Literal[
        "static_source",
        "static_bytecode",
        "symbolic",
        "fuzz",
        "ml",
        "llm",
        "hybrid",
        "unknown",
    ] = "unknown"
    feasible: bool
    feasibility_reasons: list[str] = Field(default_factory=list)
    expected_runtime_bucket: Literal["unknown", "low", "medium", "high"] = "unknown"
    failure_risk_bucket: Literal["unknown", "low", "medium", "high"] = "unknown"
    tool_cost: ToolCostEntry = Field(default_factory=ToolCostEntry)


class DACERAGFocusItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    category: str
    reason: str


class PerformanceDBEvidenceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_id: str
    dataset_name: str
    tool: str
    category: str
    detected: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    R_hat: float | None = Field(default=None, ge=0.0, le=1.0)


class ToolOverallMetricsRow(BaseModel):
    """Per-tool overall (not per-category) metrics on a referenced dataset."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_id: str
    dataset_name: str
    tool: str
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)
    execution_time_avg: float | None = Field(default=None, ge=0.0)


class Step1EvidencePacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["screc_v1"] = "screc_v1"
    target_contract: dict[str, Any] = Field(default_factory=dict)
    tool_table: list[ToolTableEntry] = Field(default_factory=list)
    scene_pool: ScenePool
    score_panel: ScorePanel
    category_diagnostics: CategoryDiagnostics
    recall_coverage: RecallCoverageMatrix
    performance_db_view: list[PerformanceDBEvidenceRow] = Field(default_factory=list)
    tool_overall_metrics: list[ToolOverallMetricsRow] = Field(default_factory=list)
    certification: CertificationVerdict
    primary_attention: PrimaryAttention = Field(default_factory=PrimaryAttention)
    dace_rag_focus: list[DACERAGFocusItem] = Field(default_factory=list)
    provenance_index: list[EvidenceRef] = Field(default_factory=list)


class EvidenceCardValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detected: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    rate: float | None = Field(default=None, ge=0.0, le=1.0)
    value: float | None = None


class EvidenceCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source: EvidenceRef
    evidence_type: Literal[
        "per_category_detected_total",
        "tool_scope",
        "runtime_cost",
        "failure_mode",
        "scene_metric",
        "step1_field",
        "rag_passage",
    ]
    tool: str | None = None
    category: str | None = None
    metric_semantics: Literal[
        "recall_side_detection_rate",
        "historical_precision",
        "historical_recall",
        "historical_f1",
        "runtime",
        "scope",
        "qualitative",
    ]
    value: EvidenceCardValue | None = None
    scope: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    extraction_confidence: ConfidenceLevel = "medium"
    taxonomy_mapping_confidence: ConfidenceLevel | None = None
    aggregation_level: EvidenceAggregationLevel = "unknown"
    decision_role: EvidenceDecisionRole = "support"
    source_reliability: EvidenceSourceReliability = "unknown"


class ActionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    evidence_refs: list[str] = Field(default_factory=list)


class CandidateAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    action_type: ActionType
    tools: list[str] = Field(default_factory=list)
    evidence: dict[EvidenceSlot, list[ActionEvidence]] = Field(default_factory=dict)
    estimated_budget: BudgetProfile
    legal: bool = True
    legality_reasons: list[str] = Field(default_factory=list)


class OwnerCandidateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    category: str
    eligibility: Literal["strong", "weak", "ineligible"]
    evidence_scope: Literal["local", "primary", "near_scene", "unrelated_external", "rag_only"]
    evidence_refs: list[str] = Field(default_factory=list)
    caveat_refs: list[str] = Field(default_factory=list)
    detected: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    rate: float | None = Field(default=None, ge=0.0, le=1.0)


class CategoryOwnershipPanel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    group: Literal["confirmed_weak", "low_support"]
    preferred_owner: str | None = None
    assignment_status: Literal["assigned", "gap", "stop_with_gap"] = "gap"
    strong_candidates: list[OwnerCandidateEvidence] = Field(default_factory=list)
    weak_candidates: list[OwnerCandidateEvidence] = Field(default_factory=list)
    rejected_candidates: list[OwnerCandidateEvidence] = Field(default_factory=list)
    gap_reason: str = ""
    unrelated_external_only: bool = False
    rag_override_eligible: bool = False
    override_targets: dict[str, list[str]] = Field(default_factory=dict)
    override_refs: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    required_claim_refs: list[str] = Field(default_factory=list)


class CategoryAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    owner_tool: str | None = None
    assignment_type: Literal[
        "primary_all",
        "local_owner",
        "near_scene_owner",
        "external_weak_owner",
        "hard_rag_override",
        "gap",
        "stop_with_gap",
    ]
    evidence_refs: list[str] = Field(default_factory=list)
    caveat_refs: list[str] = Field(default_factory=list)
    unrelated_external_only: bool = False


class RagOverrideRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    applied: bool = False
    from_owner: str | None = None
    to_owner: str | None = None
    for_refs: list[str] = Field(default_factory=list)
    against_refs: list[str] = Field(default_factory=list)
    compare_refs: list[str] = Field(default_factory=list)


class ActionByEvidenceMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dace_rag_v1"] = "dace_rag_v1"
    budget_profile: BudgetProfile
    actions: list[CandidateAction] = Field(default_factory=list)
    evidence_cards: list[EvidenceCard] = Field(default_factory=list)
    ownership_panel: dict[str, CategoryOwnershipPanel] = Field(default_factory=dict)
    override_panel: dict[str, RagOverrideRecord] = Field(default_factory=dict)
    gap_categories: list[str] = Field(default_factory=list)


class SelectedToolEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    role: Literal["STARTER", "COMPLEMENT", "SINGLE", "CONTINUATION"]
    execution_order: int = Field(ge=1)
    reason_codes: list[str] = Field(default_factory=list)


class ActionEvidenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    evidence_refs: list[str] = Field(default_factory=list)


class ActionEvidenceBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    for_claims: list[ActionEvidenceClaim] = Field(default_factory=list)
    against_claims: list[ActionEvidenceClaim] = Field(default_factory=list)
    compare_claims: list[ActionEvidenceClaim] = Field(default_factory=list)
    gap_claims: list[ActionEvidenceClaim] = Field(default_factory=list)


class BudgetUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: BudgetProfile
    estimated_use: BudgetProfile
    remaining_after_plan: BudgetProfile


class ForbiddenClaimsAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    no_target_vulnerability_claim: bool = True
    no_code_semantic_inference: bool = True
    no_precision_from_detected_total: bool = True
    absence_of_findings_not_treated_as_safe: bool = True
    no_unsourced_numeric_gain: bool = True


class Step2DecisionCertificate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dace_orch_v1"] = "dace_orch_v1"
    decision_type: ActionType
    selected_action_id: str
    selected_plan: list[SelectedToolEntry] = Field(default_factory=list)
    primary_tool: str | None = None
    tool_categories: dict[str, list[str]] = Field(default_factory=dict)
    category_assignments: list[CategoryAssignment] = Field(default_factory=list)
    rag_overrides: list[RagOverrideRecord] = Field(default_factory=list)
    action_evidence: ActionEvidenceBlock | None = None
    budget: BudgetUsage
    forbidden_claims_attestation: ForbiddenClaimsAttestation
    short_summary: str = ""
    engine_fallback_reason: str = ""


class CheckerVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ACCEPT", "REJECT", "REQUEST_REGENERATION"]
    checked_action_id: str | None = None
    rule_failures: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    hard_failures: list[str] = Field(default_factory=list)
    regeneration_reasons: list[str] = Field(default_factory=list)
    advisory_reasons: list[str] = Field(default_factory=list)


class Passage(BaseModel):
    """Owner-oriented atomic scheduling evidence.

    One Passage = one (owner_tool, category, claim) with a deterministic
    relation to that owner. The 4 evidence slots (FOR/AGAINST/COMPARE/GAP)
    are a runtime projection of `relation_to_owner` and are NOT stored.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Identity ---
    passage_id: Annotated[str, Field(min_length=6, max_length=80)]
    source_id: Annotated[str, Field(min_length=3, max_length=120)]

    # --- Retrieval keys (metadata filter) ---
    owner_tool: Annotated[str, Field(min_length=1, max_length=64)]
    counterpart_tool_ids: list[str] = Field(default_factory=list)
    category: Annotated[str, Field(min_length=1, max_length=64)]
    knowledge_kind: KnowledgeKind
    action_scope: Annotated[list[ActionId], Field(min_length=1)]
    applicability_tags: list[str] = Field(default_factory=list)

    # --- Decision semantics ---
    relation_to_owner: RelationToOwner
    evidence_basis: EvidenceBasis
    evidence_tier: EvidenceTier
    source_reliability: SourceReliability

    # --- Prompt content ---
    claim_text: Annotated[str, Field(min_length=12, max_length=180)]
    limitations_text: Annotated[str, Field(default="", max_length=120)]

    # --- Audit only (not rendered to LLM by default) ---
    source_excerpt: Annotated[str, Field(min_length=12, max_length=320)]

    @property
    def tool_ids(self) -> list[str]:
        return _dedupe_keep_order([self.owner_tool, *self.counterpart_tool_ids])

    @property
    def categories(self) -> list[str]:
        return [] if self.category == GLOBAL_CATEGORY else [self.category]

    @property
    def text(self) -> str:
        return self.claim_text

    @property
    def passage_type(self) -> str:
        if self.relation_to_owner == "owner_ineligible":
            return "limitation"
        if self.relation_to_owner in {"owner_stronger", "owner_weaker"}:
            return "comparison"
        if self.relation_to_owner == "owner_complements":
            return "recommendation"
        if self.relation_to_owner == "evidence_gap":
            return "gap"
        return "performance"

    @property
    def scheduling_type(self) -> str:
        if self.knowledge_kind == "tool_complementarity":
            return "tool_complementarity"
        if self.knowledge_kind == "failure_mode":
            return "tool_weakness"
        if self.relation_to_owner == "evidence_gap":
            return "coverage_gap"
        if self.relation_to_owner in {"owner_stronger", "owner_weaker"}:
            return "category_comparison"
        return self.knowledge_kind

    @property
    def polarity(self) -> str:
        if self.relation_to_owner in {"opposes_owner", "owner_ineligible", "owner_weaker"}:
            return "con"
        if self.relation_to_owner == "evidence_gap":
            return "gap"
        if self.relation_to_owner == "owner_stronger":
            return "comparative"
        return "pro"

    @property
    def scenario(self) -> str:
        for tag in self.applicability_tags:
            if tag.startswith("scene:"):
                return tag.removeprefix("scene:")
        return ""

    @property
    def comparison_scope(self) -> str:
        return self.category

    @property
    def stronger_tool_ids(self) -> list[str]:
        return [self.owner_tool] if self.relation_to_owner == "owner_stronger" else []

    @property
    def weaker_tool_ids(self) -> list[str]:
        return [self.owner_tool] if self.relation_to_owner == "owner_weaker" else []

    @property
    def primary_tool(self) -> str:
        return self.counterpart_tool_ids[0] if self.counterpart_tool_ids else ""

    @property
    def complement_tool(self) -> str:
        return self.owner_tool

    @property
    def scene_constraints(self) -> list[str]:
        return [self.scenario] if self.scenario else []

    @property
    def limitations(self) -> list[str]:
        return [self.limitations_text] if self.limitations_text else []

    @field_validator("counterpart_tool_ids", "action_scope", "applicability_tags", mode="before")
    @classmethod
    def _default_list(cls, value):
        return [] if value is None else value

    @field_validator("counterpart_tool_ids", "action_scope", "applicability_tags")
    @classmethod
    def _dedupe(cls, value: list[str]) -> list[str]:
        return _dedupe_keep_order(value)

    @field_validator("applicability_tags")
    @classmethod
    def _validate_tags(cls, tags: list[str]) -> list[str]:
        for tag in tags:
            if ":" not in tag or not tag.startswith(ALLOWED_TAG_PREFIXES):
                raise ValueError(f"unsupported applicability tag: {tag!r}")
        return tags

    @model_validator(mode="after")
    def _validate_contract(self) -> "Passage":
        if self.owner_tool in self.counterpart_tool_ids:
            raise ValueError("owner_tool cannot appear in counterpart_tool_ids")

        if self.relation_to_owner not in ALLOWED_RELATIONS[self.knowledge_kind]:
            raise ValueError(
                f"relation_to_owner={self.relation_to_owner!r} is illegal for "
                f"knowledge_kind={self.knowledge_kind!r}"
            )

        if self.knowledge_kind == "tool_complementarity" and not self.counterpart_tool_ids:
            raise ValueError("tool_complementarity requires at least one counterpart tool")

        if self.relation_to_owner in {"owner_stronger", "owner_weaker", "owner_complements"}:
            if not self.counterpart_tool_ids:
                raise ValueError(f"{self.relation_to_owner} requires counterpart_tool_ids")

        if self.relation_to_owner == "owner_ineligible":
            if self.knowledge_kind not in {"failure_mode", "hard_scheduling_rule"}:
                raise ValueError("owner_ineligible only valid for failure_mode / hard_scheduling_rule")
            if self.evidence_tier != "hard":
                raise ValueError("owner_ineligible must be hard evidence")

        if self.knowledge_kind == "hard_scheduling_rule":
            if self.relation_to_owner != "owner_ineligible":
                raise ValueError("hard_scheduling_rule must map to owner_ineligible")
            if set(self.action_scope) != SCHEDULING_ACTIONS:
                raise ValueError("hard_scheduling_rule must apply to all scheduling actions")
            if self.evidence_basis not in {"official_documentation", "reproducible_issue"}:
                raise ValueError("hard_scheduling_rule needs official_documentation or reproducible_issue")
            if self.source_reliability not in {
                "official_tool_doc", "maintainer_issue", "peer_reviewed", "artifact", "manual_curated",
            }:
                raise ValueError("hard_scheduling_rule has insufficient source_reliability")

        if self.category == GLOBAL_CATEGORY and self.knowledge_kind not in {
            "failure_mode", "hard_scheduling_rule",
        }:
            raise ValueError("__GLOBAL__ category only allowed for failure_mode / hard_scheduling_rule")

        if self.evidence_tier == "hard":
            if self.knowledge_kind in {"category_capability", "tool_complementarity", "fp_precision_risk"}:
                if self.source_reliability not in {"peer_reviewed", "artifact", "manual_curated"}:
                    raise ValueError("hard metric evidence needs peer_reviewed / artifact / manual_curated source")
                if self.evidence_basis not in {"benchmark_result", "paired_ablation"}:
                    raise ValueError("hard metric evidence needs benchmark_result or paired_ablation basis")
            else:
                if self.source_reliability not in {
                    "official_tool_doc", "maintainer_issue", "peer_reviewed", "artifact", "manual_curated",
                }:
                    raise ValueError("hard operational evidence needs trusted source")
                if self.evidence_basis not in {"official_documentation", "reproducible_issue"}:
                    raise ValueError("hard operational evidence needs official_documentation or reproducible_issue")

        if self.relation_to_owner == "owner_complements" and "SINGLE_TOOL" in self.action_scope:
            raise ValueError("owner_complements is not legal for SINGLE_TOOL")

        if self.relation_to_owner == "evidence_gap":
            if self.evidence_tier == "hard":
                raise ValueError("evidence_gap cannot be hard evidence")
            if self.evidence_basis not in {"benchmark_result", "manual_curation"}:
                raise ValueError("evidence_gap needs benchmark_result or manual_curation basis")

        return self


class PassageStore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passages: list[Passage] = Field(default_factory=list)


class FindingCategorySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    count: int = Field(ge=0)
    max_severity: Literal["none", "low", "medium", "high", "critical", "unknown"] = "unknown"


class ToolRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    run_status: Literal["NOT_RUN", "SUCCESS", "FAIL", "PARTIAL", "TIMEOUT"] = "NOT_RUN"
    runtime_minutes: float | None = Field(default=None, ge=0.0)
    findings_by_category: dict[str, FindingCategorySummary] = Field(default_factory=dict)
    empty_categories: list[str] = Field(default_factory=list)
    total_findings: int = Field(default=0, ge=0)
