from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator

from toolrank.kb_extract.assets import data_url, load_full_asset_input
from toolrank.kb_extract.llm_json import DOSSIER_TIMEOUT_SEC, coerce_stream_content, parse_json_content
from toolrank.openai_compat import OpenAICompatClient, OpenAICompatError
from toolrank.schemas_v2 import Passage, SourceReliability


RELATION_FIRST_PROMPT_VERSION = "kb_relation_first_v1"
RELATION_FIRST_GATE_VERSION = "deterministic_relation_first_projection_v1"

CANONICAL_CATEGORIES = [
    "reentrancy",
    "access_control",
    "arithmetic",
    "unchecked_low_level_calls",
    "denial_of_service",
    "bad_randomness",
    "front_running",
    "time_manipulation",
    "short_addresses",
]

EMPIRICAL_RELATION_KINDS = {
    "performance_comparison",
    "efficiency_comparison",
    "coverage_comparison",
}

_LOCAL_NO_PROXY_OPENER = build_opener(ProxyHandler({}))


class SourcePointer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = ""
    section: str = ""
    table: str = ""
    figure: str = ""
    page: int | None = None
    note: str = ""


class DossierEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    aliases: list[str] = Field(default_factory=list)
    whitelist_tool_id: str = ""
    role_labels: list[str] = Field(default_factory=list)
    salient_for_own_results: bool = False
    provenance: list[SourcePointer] = Field(default_factory=list)

    @field_validator("aliases", "role_labels", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        return value

    @field_validator("whitelist_tool_id", mode="before")
    @classmethod
    def _coerce_str(cls, value: Any) -> str:
        return "" if value is None else str(value)


class EntityInventory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    systems_tools: list[DossierEntity] = Field(default_factory=list)
    datasets_benchmarks: list[DossierEntity] = Field(default_factory=list)
    vulnerability_tasks: list[DossierEntity] = Field(default_factory=list)
    metrics: list[DossierEntity] = Field(default_factory=list)
    experiment_settings: list[DossierEntity] = Field(default_factory=list)


class ProjectionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["keep", "drop"]
    reason_code: str
    explanation: str


class ExperimentRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation_id: str
    relation_kind: Literal[
        "performance_comparison",
        "efficiency_comparison",
        "coverage_comparison",
        "capability_scope",
        "failure_mode",
        "compatibility_constraint",
        "tool_combination",
        "exclusion_or_applicability",
    ]
    whitelist_tool_ids: list[str] = Field(default_factory=list)
    external_tool_names: list[str] = Field(default_factory=list)
    dataset_names: list[str] = Field(default_factory=list)
    dataset_profile: str = ""
    vulnerability_categories: list[str] = Field(default_factory=list)
    task_names: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    metric_basis: str = ""
    scenario: str = ""
    compared_tool_ids: list[str] = Field(default_factory=list)
    stronger_tool_ids: list[str] = Field(default_factory=list)
    weaker_tool_ids: list[str] = Field(default_factory=list)
    comparison_scope: str = ""
    result_summary: str = ""
    applicability_boundary: str = ""
    source_reliability: SourceReliability = "peer_reviewed"
    provenance: list[SourcePointer] = Field(default_factory=list)
    projection_decision: ProjectionDecision

    @field_validator(
        "whitelist_tool_ids",
        "external_tool_names",
        "dataset_names",
        "vulnerability_categories",
        "task_names",
        "metrics",
        "compared_tool_ids",
        "stronger_tool_ids",
        "weaker_tool_ids",
        mode="before",
    )
    @classmethod
    def _coerce_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        return value

    @field_validator(
        "dataset_profile",
        "metric_basis",
        "scenario",
        "comparison_scope",
        "result_summary",
        "applicability_boundary",
        mode="before",
    )
    @classmethod
    def _coerce_str(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("source_reliability", mode="before")
    @classmethod
    def _coerce_source_reliability(cls, value: Any) -> str:
        if value is None or str(value).strip() == "":
            return "peer_reviewed"
        return str(value)


class RelationFirstDossier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    paper_title: str = ""
    paper_type: str = "other"
    summary: str = ""
    entity_inventory: EntityInventory = Field(default_factory=EntityInventory)
    experiment_relations: list[ExperimentRelation] = Field(default_factory=list)


class RelationFirstResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_dossier: RelationFirstDossier


class CritiqueEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation_index: int
    relation_id: str
    status: Literal["kept", "dropped", "rejected"]
    reason_code: str
    explanation: str
    relation: dict[str, Any]


class RelationFirstProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passages: list[Passage] = Field(default_factory=list)
    critique_log: list[CritiqueEntry] = Field(default_factory=list)
    total_relations: int = 0
    kept_count: int = 0
    dropped_count: int = 0
    rejected_count: int = 0
    paper_dossier: RelationFirstDossier


class RelationFirstLlmParseError(RuntimeError):
    """Raised when the relation-first LLM response cannot be parsed as JSON."""


def _lookup_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _normalize_allowed_tool_ids(raw_ids: list[str], allowed_tool_ids: set[str]) -> tuple[list[str], list[str]]:
    allowed_by_key = {_lookup_key(tool_id): tool_id for tool_id in allowed_tool_ids}
    normalized: list[str] = []
    unsupported: list[str] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        key = _lookup_key(str(raw_id))
        tool_id = allowed_by_key.get(key)
        if tool_id is None:
            unsupported.append(str(raw_id))
            continue
        if tool_id in seen:
            continue
        seen.add(tool_id)
        normalized.append(tool_id)
    return normalized, unsupported


def _ordered_union(*groups: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def _has_source_context(provenance: list[SourcePointer]) -> bool:
    for pointer in provenance:
        if pointer.source_id or pointer.section or pointer.table or pointer.figure or pointer.note:
            return True
    return False


def _relation_requires_empirical_context(relation: ExperimentRelation) -> bool:
    if relation.relation_kind in EMPIRICAL_RELATION_KINDS:
        return True
    return bool(relation.compared_tool_ids or relation.stronger_tool_ids or relation.weaker_tool_ids)


def _knowledge_kind(relation_kind: str) -> str:
    if relation_kind in {"tool_combination"}:
        return "partnership"
    return "category_capability"


def _polarity(relation: ExperimentRelation) -> str:
    if relation.relation_kind in {"failure_mode", "compatibility_constraint", "exclusion_or_applicability"}:
        return "con"
    if relation.weaker_tool_ids and not relation.stronger_tool_ids:
        return "con"
    if relation.relation_kind in {
        "performance_comparison",
        "efficiency_comparison",
        "coverage_comparison",
        "capability_scope",
        "tool_combination",
    }:
        return "pro"
    if relation.compared_tool_ids or relation.stronger_tool_ids or relation.weaker_tool_ids:
        return "comparative"
    return "comparative"


_RELATION_KIND_TO_KNOWLEDGE_KIND: dict[str, str] = {
    "performance_comparison": "category_capability",
    "efficiency_comparison": "category_capability",
    "coverage_comparison": "category_capability",
    "capability_scope": "category_capability",
    "reliability": "category_capability",
    "tool_combination": "tool_complementarity",
    "compatibility_constraint": "failure_mode",
    "exclusion_or_applicability": "failure_mode",
    "failure_mode": "failure_mode",
    "coverage_gap": "category_capability",
    "fp_precision_risk": "fp_precision_risk",
    "hard_scheduling_rule": "hard_scheduling_rule",
}


def _classify_knowledge_kind(relation_kind: str) -> str:
    return _RELATION_KIND_TO_KNOWLEDGE_KIND.get(relation_kind, "category_capability")


def _split_owner_and_counterparts(
    *,
    knowledge_kind: str,
    tool_ids: list[str],
    stronger_tool_ids: list[str],
    weaker_tool_ids: list[str],
) -> tuple[str, list[str]]:
    if knowledge_kind == "failure_mode" and weaker_tool_ids:
        owner = weaker_tool_ids[0]
    elif stronger_tool_ids:
        owner = stronger_tool_ids[0]
    elif tool_ids:
        owner = tool_ids[0]
    elif weaker_tool_ids:
        owner = weaker_tool_ids[0]
    else:
        owner = "gap_owner"
    counterparts = [tool for tool in tool_ids + stronger_tool_ids + weaker_tool_ids if tool != owner]
    seen: set[str] = set()
    deduped: list[str] = []
    for tool in counterparts:
        if tool in seen:
            continue
        seen.add(tool)
        deduped.append(tool)
    return owner, deduped


def _classify_relation_to_owner(
    *,
    knowledge_kind: str,
    relation_kind: str,
    polarity: str,
    owner_tool: str,
    stronger_tool_ids: list[str],
    weaker_tool_ids: list[str],
    counterpart_tool_ids: list[str],
) -> str:
    if relation_kind == "coverage_gap":
        return "evidence_gap"
    if knowledge_kind == "failure_mode":
        return "opposes_owner"
    if knowledge_kind == "tool_complementarity":
        return "owner_complements" if counterpart_tool_ids else "evidence_gap"
    if relation_kind == "tool_combination" and not counterpart_tool_ids:
        return "evidence_gap"
    if owner_tool in stronger_tool_ids and counterpart_tool_ids:
        return "owner_stronger"
    if owner_tool in weaker_tool_ids and counterpart_tool_ids:
        return "owner_weaker"
    if knowledge_kind == "fp_precision_risk":
        return "opposes_owner"
    if polarity in {"negative", "con"} and knowledge_kind != "category_capability":
        return "opposes_owner"
    return "supports_owner"


def _classify_evidence_tier(relation: ExperimentRelation) -> str:
    has_dataset = bool(relation.dataset_names) or bool(relation.dataset_profile.strip())
    has_metric = bool(relation.metric_basis.strip())
    knowledge_kind = _classify_knowledge_kind(relation.relation_kind)
    if knowledge_kind in {"failure_mode", "hard_scheduling_rule"} and relation.relation_kind != "hard_scheduling_rule":
        return "medium" if has_dataset or has_metric else "weak"
    if has_dataset and has_metric:
        return "hard"
    if has_dataset or has_metric:
        return "medium"
    return "weak"


def _classify_evidence_basis(relation: ExperimentRelation) -> str:
    if relation.dataset_names or relation.metric_basis.strip():
        return "benchmark_result"
    if relation.relation_kind in {"compatibility_constraint", "failure_mode"}:
        return "reproducible_issue"
    if relation.relation_kind == "hard_scheduling_rule":
        return "official_documentation"
    return "manual_curation"


def _schema_valid_evidence_tier(
    *,
    evidence_tier: str,
    knowledge_kind: str,
    evidence_basis: str,
    source_reliability: str,
) -> str:
    if evidence_tier != "hard":
        return evidence_tier
    if knowledge_kind in {"category_capability", "tool_complementarity", "fp_precision_risk"}:
        if source_reliability in {"peer_reviewed", "artifact", "manual_curated"} and evidence_basis in {
            "benchmark_result",
            "paired_ablation",
        }:
            return evidence_tier
        return "medium"
    if source_reliability in {
        "official_tool_doc",
        "maintainer_issue",
        "peer_reviewed",
        "artifact",
        "manual_curated",
    } and evidence_basis in {"official_documentation", "reproducible_issue"}:
        return evidence_tier
    return "medium"


def _action_scope_for(knowledge_kind: str, relation_to_owner: str) -> list[str]:
    del knowledge_kind
    if relation_to_owner == "owner_complements":
        return ["PLAN_COMPOSITION", "CONTINUE_HEDGE"]
    return ["SINGLE_TOOL", "PLAN_COMPOSITION", "CONTINUE_HEDGE"]


def _category_field(categories: list[str], knowledge_kind: str) -> str:
    if categories:
        return categories[0]
    if knowledge_kind in {"failure_mode", "hard_scheduling_rule"}:
        return "__GLOBAL__"
    return "general"


def _trim(value: str, *, max_length: int, fallback: str) -> str:
    text = (value or "").strip() or fallback
    if len(text) > max_length:
        text = text[: max_length - 3].rstrip() + "..."
    return text


def _relation_to_passage(
    *,
    relation: ExperimentRelation,
    doc_id: str,
    index: int,
    tool_ids: list[str],
    stronger_tool_ids: list[str],
    weaker_tool_ids: list[str],
    contextual_text: str,
) -> Passage:
    knowledge_kind = _classify_knowledge_kind(relation.relation_kind)
    owner_tool, counterpart_tool_ids = _split_owner_and_counterparts(
        knowledge_kind=knowledge_kind,
        tool_ids=tool_ids,
        stronger_tool_ids=stronger_tool_ids,
        weaker_tool_ids=weaker_tool_ids,
    )
    if knowledge_kind == "tool_complementarity" and not counterpart_tool_ids:
        knowledge_kind = "category_capability"
    polarity = _polarity(relation)
    relation_to_owner = _classify_relation_to_owner(
        knowledge_kind=knowledge_kind,
        relation_kind=relation.relation_kind,
        polarity=polarity,
        owner_tool=owner_tool,
        stronger_tool_ids=stronger_tool_ids,
        weaker_tool_ids=weaker_tool_ids,
        counterpart_tool_ids=counterpart_tool_ids,
    )
    scenario = _sanitize_external_names(relation.scenario, relation.external_tool_names)
    claim_text = _trim(
        _sanitize_external_names(relation.result_summary, relation.external_tool_names),
        max_length=180,
        fallback="Scheduling evidence for tool selection.",
    )
    boundary = _sanitize_external_names(relation.applicability_boundary, relation.external_tool_names)
    if relation_to_owner in {"opposes_owner", "owner_ineligible", "evidence_gap"}:
        limitations_text = _trim(boundary, max_length=120, fallback="")
    else:
        limitations_text = ""
    source_excerpt = _trim(
        contextual_text or boundary or claim_text,
        max_length=320,
        fallback="Scheduling evidence source excerpt.",
    )
    evidence_basis = _classify_evidence_basis(relation)
    evidence_tier = _classify_evidence_tier(relation)
    if relation_to_owner == "evidence_gap" and evidence_tier == "hard":
        evidence_tier = "medium"
    evidence_tier = _schema_valid_evidence_tier(
        evidence_tier=evidence_tier,
        knowledge_kind=knowledge_kind,
        evidence_basis=evidence_basis,
        source_reliability=relation.source_reliability,
    )
    return Passage(
        passage_id=f"p_{doc_id}_{index}",
        source_id=doc_id,
        owner_tool=owner_tool,
        counterpart_tool_ids=counterpart_tool_ids,
        category=_category_field(relation.vulnerability_categories, knowledge_kind),
        knowledge_kind=knowledge_kind,
        action_scope=_action_scope_for(knowledge_kind, relation_to_owner),
        relation_to_owner=relation_to_owner,
        applicability_tags=[f"scene:{scenario}"] if scenario else [],
        evidence_basis=evidence_basis,
        evidence_tier=evidence_tier,
        source_reliability=relation.source_reliability,
        claim_text=claim_text,
        limitations_text=limitations_text,
        source_excerpt=source_excerpt,
    )


def _sanitize_external_names(text: str, external_tool_names: list[str]) -> str:
    sanitized = text
    for name in external_tool_names:
        stripped = name.strip()
        if not stripped:
            continue
        sanitized = re.sub(re.escape(stripped), "external comparator", sanitized, flags=re.IGNORECASE)
    return " ".join(sanitized.split())


def _source_text(provenance: list[SourcePointer], external_tool_names: list[str]) -> str:
    labels: list[str] = []
    for pointer in provenance:
        pieces = [pointer.source_id, pointer.section, pointer.table, pointer.figure, pointer.note]
        label = " ".join(piece for piece in pieces if piece)
        if label:
            labels.append(_sanitize_external_names(label, external_tool_names))
    return "; ".join(labels)


def _projected_text(
    relation: ExperimentRelation,
    *,
    tool_ids: list[str],
    stronger_tool_ids: list[str],
    weaker_tool_ids: list[str],
) -> str:
    parts: list[str] = [f"Relation: {relation.relation_kind.replace('_', ' ')}"]
    if relation.scenario:
        parts.append(f"Scenario: {_sanitize_external_names(relation.scenario, relation.external_tool_names)}")
    if tool_ids:
        parts.append(f"ToolRank tools: {', '.join(tool_ids)}")
    tasks = _ordered_union(relation.vulnerability_categories, relation.task_names)
    if tasks:
        parts.append(f"Tasks: {_sanitize_external_names(', '.join(tasks), relation.external_tool_names)}")
    if relation.dataset_names:
        datasets = _sanitize_external_names(", ".join(relation.dataset_names), relation.external_tool_names)
        parts.append(f"Datasets: {datasets}")
    if relation.dataset_profile:
        parts.append(f"Dataset profile: {_sanitize_external_names(relation.dataset_profile, relation.external_tool_names)}")
    if relation.metric_basis:
        parts.append(f"Metric basis: {_sanitize_external_names(relation.metric_basis, relation.external_tool_names)}")
    elif relation.metrics:
        parts.append(f"Metrics: {_sanitize_external_names(', '.join(relation.metrics), relation.external_tool_names)}")
    if relation.comparison_scope:
        parts.append(f"Scope: {_sanitize_external_names(relation.comparison_scope, relation.external_tool_names)}")
    if stronger_tool_ids:
        parts.append(f"Stronger tools: {', '.join(stronger_tool_ids)}")
    if weaker_tool_ids:
        parts.append(f"Weaker/baseline tools: {', '.join(weaker_tool_ids)}")
    if relation.result_summary:
        parts.append(f"Result: {_sanitize_external_names(relation.result_summary, relation.external_tool_names)}")
    if relation.applicability_boundary:
        parts.append(f"Boundary: {_sanitize_external_names(relation.applicability_boundary, relation.external_tool_names)}")
    source = _source_text(relation.provenance, relation.external_tool_names)
    if source:
        parts.append(f"Source: {source}")
    return " | ".join(parts)


def _reject(
    *,
    index: int,
    relation: ExperimentRelation,
    reason_code: str,
    explanation: str,
) -> CritiqueEntry:
    return CritiqueEntry(
        relation_index=index,
        relation_id=relation.relation_id,
        status="rejected",
        reason_code=reason_code,
        explanation=explanation,
        relation=relation.model_dump(),
    )


def _validate_keep_relation(
    *,
    index: int,
    relation: ExperimentRelation,
    doc_id: str,
    allowed_tool_ids: set[str],
) -> tuple[Passage | None, CritiqueEntry]:
    relation_tool_ids, unsupported_relation_ids = _normalize_allowed_tool_ids(
        relation.whitelist_tool_ids,
        allowed_tool_ids,
    )
    if unsupported_relation_ids:
        return None, _reject(
            index=index,
            relation=relation,
            reason_code="unsupported_whitelist_tool_id",
            explanation=f"whitelist_tool_ids contains unsupported tools: {', '.join(unsupported_relation_ids)}",
        )
    if not relation_tool_ids:
        return None, _reject(
            index=index,
            relation=relation,
            reason_code="missing_whitelist_tool_id",
            explanation="Kept relations must involve at least one schedulable whitelist tool.",
        )

    compared_tool_ids, unsupported_compared = _normalize_allowed_tool_ids(relation.compared_tool_ids, allowed_tool_ids)
    if unsupported_compared:
        return None, _reject(
            index=index,
            relation=relation,
            reason_code="unsupported_compared_tool_id",
            explanation=f"compared_tool_ids contains unsupported tools: {', '.join(unsupported_compared)}",
        )
    stronger_tool_ids, unsupported_stronger = _normalize_allowed_tool_ids(relation.stronger_tool_ids, allowed_tool_ids)
    if unsupported_stronger:
        return None, _reject(
            index=index,
            relation=relation,
            reason_code="unsupported_stronger_tool_id",
            explanation=f"stronger_tool_ids contains unsupported tools: {', '.join(unsupported_stronger)}",
        )
    weaker_tool_ids, unsupported_weaker = _normalize_allowed_tool_ids(relation.weaker_tool_ids, allowed_tool_ids)
    if unsupported_weaker:
        return None, _reject(
            index=index,
            relation=relation,
            reason_code="unsupported_weaker_tool_id",
            explanation=f"weaker_tool_ids contains unsupported tools: {', '.join(unsupported_weaker)}",
        )

    comparison_participants = _ordered_union(compared_tool_ids, stronger_tool_ids, weaker_tool_ids)
    missing_participants = [tool_id for tool_id in comparison_participants if tool_id not in relation_tool_ids]
    if missing_participants:
        return None, _reject(
            index=index,
            relation=relation,
            reason_code="comparison_participants_outside_relation_tool_set",
            explanation=(
                "Comparative relation fields mention whitelist tools outside relation-level "
                f"whitelist_tool_ids: {', '.join(missing_participants)}"
            ),
        )

    if _relation_requires_empirical_context(relation):
        has_metric_context = bool(relation.metric_basis.strip())
        has_dataset_or_source_context = bool(
            relation.dataset_names or relation.dataset_profile.strip() or _has_source_context(relation.provenance)
        )
        if not relation.scenario.strip() or not has_metric_context or not has_dataset_or_source_context:
            return None, _reject(
                index=index,
                relation=relation,
                reason_code="missing_empirical_context",
                explanation=(
                    "Empirical comparison relations require scenario, metric context, and either dataset "
                    "details or source/provenance context."
                ),
            )

    tool_ids = _ordered_union(relation_tool_ids, compared_tool_ids, stronger_tool_ids, weaker_tool_ids)
    projected_text = _projected_text(
        relation,
        tool_ids=tool_ids,
        stronger_tool_ids=stronger_tool_ids,
        weaker_tool_ids=weaker_tool_ids,
    )
    passage = _relation_to_passage(
        relation=relation,
        doc_id=doc_id,
        index=index,
        tool_ids=tool_ids,
        stronger_tool_ids=stronger_tool_ids,
        weaker_tool_ids=weaker_tool_ids,
        contextual_text=projected_text,
    )
    critique = CritiqueEntry(
        relation_index=index,
        relation_id=relation.relation_id,
        status="kept",
        reason_code=relation.projection_decision.reason_code,
        explanation=relation.projection_decision.explanation,
        relation=relation.model_dump(),
    )
    return passage, critique


def project_relation_first_response(
    raw_response: dict[str, Any],
    *,
    doc_id: str,
    allowed_tool_ids: set[str],
) -> RelationFirstProjection:
    response = RelationFirstResponse.model_validate(raw_response)
    passages: list[Passage] = []
    critique_log: list[CritiqueEntry] = []
    dropped_count = 0
    rejected_count = 0

    for index, relation in enumerate(response.paper_dossier.experiment_relations):
        if relation.projection_decision.decision == "drop":
            dropped_count += 1
            critique_log.append(
                CritiqueEntry(
                    relation_index=index,
                    relation_id=relation.relation_id,
                    status="dropped",
                    reason_code=relation.projection_decision.reason_code,
                    explanation=relation.projection_decision.explanation,
                    relation=relation.model_dump(),
                )
            )
            continue

        passage, critique = _validate_keep_relation(
            index=index,
            relation=relation,
            doc_id=doc_id,
            allowed_tool_ids=allowed_tool_ids,
        )
        critique_log.append(critique)
        if passage is None:
            rejected_count += 1
            continue
        passages.append(passage)

    return RelationFirstProjection(
        passages=passages,
        critique_log=critique_log,
        total_relations=len(response.paper_dossier.experiment_relations),
        kept_count=len(passages),
        dropped_count=dropped_count,
        rejected_count=rejected_count,
        paper_dossier=response.paper_dossier,
    )


def _create_relation_first_json_completion(
    *,
    client: OpenAICompatClient,
    model: str,
    system_prompt: str,
    user_content: list[dict[str, Any]],
    timeout_sec: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "stream": True,
    }
    request = Request(
        url=f"{client.base_url}/chat/completions",
        data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {client.api_key}",
        },
        method="POST",
    )
    parsed_url = urlparse(request.full_url)
    opener = _LOCAL_NO_PROXY_OPENER if parsed_url.hostname in {"127.0.0.1", "localhost"} else None
    try:
        if opener is not None:
            response = opener.open(request, timeout=timeout_sec)
        else:
            response = urlopen(request, timeout=timeout_sec)
        with response:
            raw_stream = response.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        raise OpenAICompatError(f"relation-first request failed: {exc}") from exc

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    stream_event_count = 0
    content_delta_count = 0
    reasoning_delta_count = 0
    finish_reasons: list[str] = []
    ignored_json_event_count = 0

    for line in raw_stream.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload_line = stripped[5:].strip()
        if payload_line == "[DONE]":
            break
        try:
            chunk = json.loads(payload_line)
        except json.JSONDecodeError:
            ignored_json_event_count += 1
            continue
        stream_event_count += 1
        try:
            choice = chunk["choices"][0]
            delta = choice.get("delta") or {}
        except Exception:
            ignored_json_event_count += 1
            continue
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            finish_reasons.append(finish_reason)
        content = coerce_stream_content(delta.get("content"))
        if content:
            content_delta_count += 1
            content_parts.append(content)
        reasoning = coerce_stream_content(delta.get("reasoning_content"))
        if reasoning:
            reasoning_delta_count += 1
            reasoning_parts.append(reasoning)

    assembled_content = "".join(content_parts)
    parsed = parse_json_content(assembled_content)
    parse_error = ""
    if parsed is None:
        parse_error = "empty_content" if not assembled_content.strip() else "json_parse_failed"

    diagnostics = {
        "assembled_content": assembled_content,
        "raw_stream": raw_stream,
        "reasoning_content": "".join(reasoning_parts),
        "parse_error": parse_error,
        "metadata": {
            "stream_event_count": stream_event_count,
            "content_delta_count": content_delta_count,
            "reasoning_delta_count": reasoning_delta_count,
            "ignored_json_event_count": ignored_json_event_count,
            "raw_stream_char_count": len(raw_stream),
            "assembled_content_char_count": len(assembled_content),
            "reasoning_content_char_count": sum(len(part) for part in reasoning_parts),
            "finish_reasons": finish_reasons,
        },
    }
    return parsed, diagnostics


def _write_llm_parse_diagnostics(
    *,
    diagnostics_dir: Path,
    doc_id: str,
    diagnostics: dict[str, Any],
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    assembled_content = str(diagnostics.get("assembled_content") or "")
    raw_stream = str(diagnostics.get("raw_stream") or "")
    payload = {
        "doc_id": doc_id,
        "parse_error": diagnostics.get("parse_error") or "unknown_parse_error",
        "metadata": diagnostics.get("metadata") or {},
        "reasoning_content": diagnostics.get("reasoning_content") or "",
        "raw_stream_preview": raw_stream[:12000],
    }
    (diagnostics_dir / "raw_llm_response.txt").write_text(assembled_content, encoding="utf-8")
    (diagnostics_dir / "raw_llm_parse_error.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def extract_relation_first_raw_response(
    *,
    client: OpenAICompatClient,
    model: str,
    mineru_output_dir: Path,
    doc_id: str,
    allowed_tool_ids: set[str],
    diagnostics_dir: Path | None = None,
) -> dict[str, Any]:
    asset_input = load_full_asset_input(mineru_output_dir)
    if asset_input is None:
        raise RuntimeError(f"No MinerU markdown/content assets found in {mineru_output_dir}")

    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "ToolRank schedules smart-contract vulnerability detectors.\n"
                "Read the full paper and return a relation-first document-level dossier for KB extraction.\n"
                "Do not write ToolCards, formal toolcards, per-tool JSON files, or performance_db data.\n"
                "Do not write passage candidates. Passage generation is a deterministic projection from accepted experiment relations.\n"
                "Return JSON only matching the RelationFirstResponse schema.\n\n"
                f"schema={json.dumps(RelationFirstResponse.model_json_schema(), ensure_ascii=False)}\n"
                f"prompt_version={RELATION_FIRST_PROMPT_VERSION}\n"
                f"doc_id={doc_id}\n"
                f"system_tool_whitelist={sorted(allowed_tool_ids)}\n"
                f"canonical_categories={CANONICAL_CATEGORIES}\n\n"
                "Build an entity inventory for named systems/tools, datasets/benchmarks, vulnerability tasks/categories, metrics, and experiment settings.\n"
                "For each entity, assign role_labels such as main_experiment_subject, baseline_or_comparator, related_work_only, candidate_pool_only, excluded_tool, dataset_main_eval, dataset_secondary_eval, metric_used_for_comparison, background_only, or pure_dataset_description.\n"
                "Record provenance pointers whenever available, including section, table, figure, page, or source_id.\n"
                "Set salient_for_own_results true only when the entity supports this paper's own experiment results.\n\n"
                "Then output document-level experimental relations. Each relation must bind relation_id, relation_kind, whitelist_tool_ids, external_tool_names, dataset_names, dataset_profile, vulnerability_categories or task_names, metrics, metric_basis, scenario, compared_tool_ids, stronger_tool_ids, weaker_tool_ids, comparison_scope, result_summary, applicability_boundary, source_reliability, provenance, and projection_decision.\n"
                "source_reliability must be one of peer_reviewed, artifact, manual_curated, official_tool_doc, maintainer_issue, internal_eval, or community_report.\n"
                "Use relation_kind values only from performance_comparison, efficiency_comparison, coverage_comparison, capability_scope, failure_mode, compatibility_constraint, tool_combination, and exclusion_or_applicability.\n"
                "whitelist_tool_ids, compared_tool_ids, stronger_tool_ids, and weaker_tool_ids must contain only IDs from system_tool_whitelist.\n"
                "Non-whitelist systems/tools may appear in entity inventory and relation external_tool_names so paper understanding stays complete.\n"
                "Non-whitelist systems/tools must never appear in whitelist_tool_ids or directional comparison ID fields.\n"
                "Candidate-pool-only, related-work-only, background-only, and pure dataset-description material should stay in the dossier but projection_decision.decision must be drop.\n"
                "For empirical performance, efficiency, coverage, and comparison relations, include scenario, metric_basis, and either dataset_names/dataset_profile or precise provenance context.\n"
                "For comparative relations, include every participating whitelist tool in whitelist_tool_ids as well as the compared/stronger/weaker fields.\n"
                "Use projection_decision to say whether the relation is eligible to project into scheduling evidence, with a concise reason_code and explanation.\n\n"
                "FULL_MARKDOWN_START\n"
                f"{asset_input.full_markdown}\n"
                "FULL_MARKDOWN_END"
            ),
        }
    ]

    for index, table in enumerate(asset_input.tables, start=1):
        user_content.append(
            {
                "type": "text",
                "text": f"FULL_TABLE_{index}\nCaption: {table.caption}\nHTML_TABLE:\n{table.html}",
            }
        )
        table_image = mineru_output_dir / table.img_path if table.img_path else None
        if table_image and table_image.exists():
            user_content.append({"type": "image_url", "image_url": {"url": data_url(table_image)}})

    for index, image in enumerate(asset_input.images, start=1):
        user_content.append(
            {
                "type": "text",
                "text": f"FULL_IMAGE_{index}\nCaption: {image.caption}\nCONTEXT:\n{image.context}",
            }
        )
        image_path = mineru_output_dir / image.img_path if image.img_path else None
        if image_path and image_path.exists():
            user_content.append({"type": "image_url", "image_url": {"url": data_url(image_path)}})

    payload, diagnostics = _create_relation_first_json_completion(
        client=client,
        model=model,
        system_prompt="Return JSON only. Extract document-level experiment relations before any scheduling passage projection.",
        user_content=user_content,
        timeout_sec=DOSSIER_TIMEOUT_SEC,
    )
    if not payload:
        if diagnostics_dir is not None:
            _write_llm_parse_diagnostics(
                diagnostics_dir=diagnostics_dir,
                doc_id=doc_id,
                diagnostics=diagnostics,
            )
        raise RelationFirstLlmParseError(
            "Relation-first LLM extraction returned an empty or unparseable response"
        )
    return payload


__all__ = [
    "CANONICAL_CATEGORIES",
    "RELATION_FIRST_GATE_VERSION",
    "RELATION_FIRST_PROMPT_VERSION",
    "CritiqueEntry",
    "DossierEntity",
    "EntityInventory",
    "ExperimentRelation",
    "ProjectionDecision",
    "RelationFirstDossier",
    "RelationFirstLlmParseError",
    "RelationFirstProjection",
    "RelationFirstResponse",
    "SourcePointer",
    "extract_relation_first_raw_response",
    "project_relation_first_response",
]
