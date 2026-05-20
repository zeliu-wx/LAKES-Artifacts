from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from toolrank.schemas_v2 import Passage


class ObservationBasis(str, Enum):
    author_explicit_claim = "author_explicit_claim"
    table_explicit_entry = "table_explicit_entry"
    experiment_observation = "experiment_observation"
    caption_summary = "caption_summary"
    negative_statement = "negative_statement"


class CoverageSemantics(str, Enum):
    claimed_supported_range = "claimed_supported_range"
    evaluated_on_range = "evaluated_on_range"


class GateDecision(str, Enum):
    accept = "accept"
    skip = "skip"
    defer = "defer"
    manual_review = "manual_review"


class AssertionStatus(str, Enum):
    accepted = "accepted"
    deferred = "deferred"
    conflict = "conflict"
    provenance_only = "provenance_only"
    skipped = "skipped"
    manual_review = "manual_review"


class LedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    doc_id: str
    chunk_id: str
    candidate_local_id: str
    candidate_type: str
    candidate_data: dict[str, Any]
    status: AssertionStatus
    gate_decision: GateDecision
    reason_codes: list[str]
    merge_hint: str
    timestamp: str
    run_id: str
    tool_id: Optional[str] = None
    predicate: Optional[str] = None
    extraction_source: str = "unknown"
    extraction_rationale: str = ""
    evidence_binding_rationale: str = ""
    decision_rationale: str = ""


class ToolInventoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface_name: str
    canonical_id: Optional[str] = None
    role: str = "cited_only"
    sections_discussed: list[str] = Field(default_factory=list)
    abbreviation: Optional[str] = None


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str = ""
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    what_contracts: str = ""
    how_collected: str = ""
    scale: str = ""
    labeling: str = ""
    source_components: list[str] = Field(default_factory=list)
    solidity_versions: Optional[str] = None
    contract_count: Optional[int] = None
    labeled: Optional[bool] = None
    target_vulnerabilities: list[str] = Field(default_factory=list)
    label_granularity: str = ""
    label_count: Optional[int] = None
    label_count_basis: str = ""


class ExperimentalConfigEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_per_contract: Optional[str] = None
    environment: Optional[str] = None
    evaluation_methodology: Optional[str] = None


class SectionRoleEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_path: list[str] = Field(default_factory=list)
    role: str = "body"
    tool_knowledge_density: str = "low"


class CoreferenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface_form: str
    resolves_to: str | list[str]
    type: str = "entity"


class AggregateClaimEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    scope: str = "study-level"
    note: str = ""


class ToolAliasResolutionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias_surface: str
    canonical_surface: str
    rationale: str = ""


class GlobalStrengthHintEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    category: str
    strength: str
    evidence_excerpt: str = ""


class GlobalRankingHintEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    stronger_tool: str
    stronger_than: list[str] = Field(default_factory=list)
    evidence_excerpt: str = ""


class ToolCapabilityHintEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    predicate: str
    value: dict[str, Any]
    evidence_excerpt: str = ""


class CombinationHintEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: list[str] = Field(default_factory=list)
    scenario: str = ""
    metric: str = ""
    value: float = 0.0
    avg_runtime_sec: Optional[float] = None
    evidence_excerpt: str = ""


class PerformanceObservationHintEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    tool_name: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    vulnerability_scores: dict[str, float] = Field(default_factory=dict)


class WorkingMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_aliases: list[ToolAliasResolutionEntry] = Field(default_factory=list)
    capability_hints: list[ToolCapabilityHintEntry] = Field(default_factory=list)
    global_strength_hints: list[GlobalStrengthHintEntry] = Field(default_factory=list)
    global_ranking_hints: list[GlobalRankingHintEntry] = Field(default_factory=list)
    combination_hints: list[CombinationHintEntry] = Field(default_factory=list)
    performance_observations: list[PerformanceObservationHintEntry] = Field(default_factory=list)
    scheduling_evidence: list[Passage] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PaperDossier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    paper_title: str = ""
    publication_year: Optional[int] = None
    paper_type: str = "other"
    primary_contribution: str = ""
    tools: list[ToolInventoryEntry] = Field(default_factory=list)
    datasets: list[DatasetEntry] = Field(default_factory=list)
    final_datasets: list[DatasetEntry] = Field(default_factory=list)
    separate_eval_sets: list[DatasetEntry] = Field(default_factory=list)
    experimental_config: ExperimentalConfigEntry = Field(default_factory=ExperimentalConfigEntry)
    sections: list[SectionRoleEntry] = Field(default_factory=list)
    coreferences: list[CoreferenceEntry] = Field(default_factory=list)
    aggregate_claims: list[AggregateClaimEntry] = Field(default_factory=list)
    working_memory: WorkingMemory = Field(default_factory=WorkingMemory)


class PaperDossierManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    dossier_path: str
    paper_type: str = "other"
    tools_identified: int = 0
    aggregate_claims_identified: int = 0
    extraction_model: str = ""
    extraction_tokens_used: int = 0


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    document_hashes: dict[str, str]
    mineru_version: str = "unknown"
    extraction_prompt_version: str
    extraction_model: str
    gate_prompt_version: str
    gate_model: str
    schema_registry_version: str
    normalizer_version: str = "1.0.0"
    alias_registry_version: str = "1.0.0"
    llm_first_pipeline: bool = True
    blast_radius_triggered: bool = False
    papers_processed: int = 0
    total_candidates: int = 0
    accepted: int = 0
    deferred: int = 0
    skipped: int = 0
    manual_review: int = 0
    materialized: int = 0
    ledger_path: Optional[str] = None
    materialized_cards: list[str] = Field(default_factory=list)
    paper_dossiers: list[PaperDossierManifestEntry] = Field(default_factory=list)
