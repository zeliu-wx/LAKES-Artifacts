"""Shared ownership evidence helpers for CEGO prompt rendering and checker rules."""

from __future__ import annotations

from dataclasses import dataclass

from toolrank.assignment_evidence import is_strong_eligible, is_weak_eligible
from toolrank.precision_gate import candidate_passes_precision_gate
from toolrank.solc_range import version_in_range
from toolrank.schemas_v2 import (
    ActionByEvidenceMatrix,
    CategoryOwnershipPanel,
    EvidenceCard,
    OwnerCandidateEvidence,
    Step1EvidencePacket,
)


@dataclass(frozen=True)
class OwnershipCandidate:
    tool: str
    evidence_id: str
    detected: int | None
    total: int | None
    rate: float | None
    scope: str
    scene_tier: str
    scene_priority: int


@dataclass(frozen=True)
class CategoryOwnershipEvidence:
    category: str
    group: str
    preferred: str | None
    strong_candidates: list[OwnershipCandidate]
    all_strong_candidates: list[OwnershipCandidate]
    weak_candidates: list[OwnershipCandidate]
    rag_override_eligible: bool
    override_refs: list[str]
    override_targets: dict[str, list[str]]


STRONG_SCHEDULING_EVIDENCE = "strong"
USEFUL_SCHEDULING_EVIDENCE = "useful"
ORDINARY_SCHEDULING_EVIDENCE = "ordinary"


def attention_categories(packet: Step1EvidencePacket) -> list[str]:
    categories = (
        list(packet.primary_attention.confirmed_weak_categories)
        + list(packet.primary_attention.low_support_categories)
    )
    return list(dict.fromkeys(categories))


def category_group(packet: Step1EvidencePacket, category: str) -> str:
    if category in packet.primary_attention.low_support_categories:
        return "low_support"
    return "confirmed_weak"


def rcov_evidence_id(tool: str, category: str) -> str:
    return f"ev_rcov_{tool}_{category}"


def card_scope_text(card: EvidenceCard) -> str:
    if card.source.field_path == "recall_coverage.matrix":
        return "local"
    if card.source.field_path == "performance_db_view":
        return "external_dataset"
    return card.source.field_path or card.source.source_type


def primary_source_id(packet: Step1EvidencePacket) -> str | None:
    if not packet.scene_pool.neighbors:
        return None
    primary = max(packet.scene_pool.neighbors, key=lambda neighbor: neighbor.weight)
    return primary.paper_id or None


def near_scene_source_ids(packet: Step1EvidencePacket) -> set[str]:
    if not packet.scene_pool.neighbors:
        return set()
    sorted_neighbors = sorted(
        packet.scene_pool.neighbors,
        key=lambda neighbor: neighbor.weight,
        reverse=True,
    )
    ids: set[str] = set()
    for neighbor in sorted_neighbors[1:]:
        if neighbor.paper_id:
            ids.add(neighbor.paper_id)
        ids.update(neighbor.provenance_refs or [])
    return ids


def matched_source_ids(packet: Step1EvidencePacket) -> set[str]:
    source_id = primary_source_id(packet)
    return {source_id} if source_id else set()



def card_scene_tier(packet: Step1EvidencePacket, card: EvidenceCard) -> str:
    if card_scope_text(card) == "local":
        return "local"
    if card_scope_text(card) != "external_dataset":
        return "unrelated"
    source_id = card.scope.get("source_id")
    if not isinstance(source_id, str):
        return "unrelated"
    if source_id == primary_source_id(packet):
        return "primary"
    if source_id in near_scene_source_ids(packet):
        return "near_scene"
    return "unrelated"


def card_scene_priority(packet: Step1EvidencePacket, card: EvidenceCard) -> int:
    tier = card_scene_tier(packet, card)
    if tier == "local":
        return 0
    if tier == "primary":
        return 1
    if tier == "near_scene":
        return 2
    return 3


def _scope_list(card: EvidenceCard, key: str) -> list[str]:
    value = card.scope.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _rag_categories(card: EvidenceCard) -> set[str]:
    categories = _scope_list(card, "categories")
    return set(categories or ([card.category] if card.category else []))


def is_strong_scheduling_evidence(card: EvidenceCard, *, category: str) -> bool:
    """Classify hard human-curated category ownership passages."""
    return (
        card.evidence_type == "rag_passage"
        and card.category == category
        and card.source_reliability == "manual_curated"
        and str(card.scope.get("evidence_tier") or "") == "hard"
        and str(card.scope.get("knowledge_kind") or "") == "category_capability"
        and str(card.scope.get("relation_to_owner") or "") == "owner_stronger"
    )


def scheduling_evidence_priority(card: EvidenceCard, *, category: str) -> str:
    if is_strong_scheduling_evidence(card, category=category):
        return STRONG_SCHEDULING_EVIDENCE
    if (
        card.evidence_type == "rag_passage"
        and card.category == category
        and str(card.scope.get("knowledge_kind") or "") == "category_capability"
    ):
        return USEFUL_SCHEDULING_EVIDENCE
    return ORDINARY_SCHEDULING_EVIDENCE


def _category_capability_pro_tools(card: EvidenceCard) -> set[str]:
    if str(card.scope.get("knowledge_kind") or "") != "category_capability":
        return set()
    relation_to_owner = str(card.scope.get("relation_to_owner") or "")
    if relation_to_owner in {"supports_owner", "owner_stronger"}:
        return {card.tool} if card.tool else set()
    if relation_to_owner:
        return set()
    return set()


def _category_capability_con_tools(card: EvidenceCard) -> set[str]:
    if str(card.scope.get("knowledge_kind") or "") != "category_capability":
        return set()
    relation_to_owner = str(card.scope.get("relation_to_owner") or "")
    counterpart_tool_ids = set(_scope_list(card, "counterpart_tool_ids"))
    if relation_to_owner == "owner_weaker":
        return {card.tool} if card.tool else set()
    if relation_to_owner == "owner_stronger":
        return counterpart_tool_ids
    if relation_to_owner:
        return set()
    return set()


def _category_capability_stronger_tools(card: EvidenceCard, *, weaker_tool: str) -> set[str]:
    relation_to_owner = str(card.scope.get("relation_to_owner") or "")
    counterpart_tool_ids = set(_scope_list(card, "counterpart_tool_ids"))
    if relation_to_owner == "owner_stronger" and weaker_tool in counterpart_tool_ids:
        return {card.tool} if card.tool else set()
    if relation_to_owner == "owner_weaker" and card.tool == weaker_tool:
        return counterpart_tool_ids
    if relation_to_owner:
        return set()
    return set()


def _candidate_from_card(packet: Step1EvidencePacket, card: EvidenceCard) -> OwnershipCandidate:
    value = card.value
    return OwnershipCandidate(
        tool=card.tool or "",
        evidence_id=card.evidence_id,
        detected=value.detected if value is not None else None,
        total=value.total if value is not None else None,
        rate=value.rate if value is not None else None,
        scope=card_scope_text(card),
        scene_tier=card_scene_tier(packet, card),
        scene_priority=card_scene_priority(packet, card),
    )


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


def _solc_tags_apply(packet: Step1EvidencePacket, card: EvidenceCard) -> bool:
    solc_tags = [
        tag.removeprefix("solc:")
        for tag in _scope_list(card, "applicability_tags")
        if tag.startswith("solc:")
    ]
    if not solc_tags:
        return True
    version = _target_solidity_version(packet)
    return bool(version) and any(version_in_range(version, tag) for tag in solc_tags)


def _is_osiris_unavailable_fallback(card: EvidenceCard) -> bool:
    return "scene:osiris-unavailable-fallback" in _scope_list(card, "applicability_tags")


def _strong_scheduling_card_applies(
    packet: Step1EvidencePacket,
    card: EvidenceCard,
    *,
    category: str,
    feasible_tools: set[str],
) -> bool:
    if not is_strong_scheduling_evidence(card, category=category):
        return False
    if card.tool not in feasible_tools:
        return False
    if not _solc_tags_apply(packet, card):
        return False
    if _is_osiris_unavailable_fallback(card) and "osiris" in feasible_tools:
        return False
    return True


def _candidate_sort_key(candidate: OwnershipCandidate) -> tuple[int, float, int, str, str]:
    return (
        candidate.scene_priority,
        -(candidate.rate if candidate.rate is not None else -1.0),
        -(candidate.total or 0),
        candidate.tool,
        candidate.evidence_id,
    )


def _top_candidates_by_tool(candidates: list[OwnershipCandidate]) -> list[OwnershipCandidate]:
    selected: list[OwnershipCandidate] = []
    seen_tools: set[str] = set()
    for candidate in sorted(candidates, key=_candidate_sort_key):
        if candidate.tool in seen_tools:
            continue
        selected.append(candidate)
        seen_tools.add(candidate.tool)
    return selected


def near_scene_strong_candidates(
    ownership: CategoryOwnershipEvidence,
) -> list[OwnershipCandidate]:
    return [
        candidate
        for candidate in ownership.all_strong_candidates
        if candidate.scope == "external_dataset" and candidate.scene_tier == "near_scene"
    ]


def unrelated_strong_candidates(
    ownership: CategoryOwnershipEvidence,
) -> list[OwnershipCandidate]:
    return [
        candidate
        for candidate in ownership.all_strong_candidates
        if candidate.scope == "external_dataset" and candidate.scene_tier == "unrelated"
    ]


def category_marginal_value(
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
    category: str,
) -> str:
    """Return one of: 'high' / 'medium' / 'low' / 'none'."""
    ownership = category_ownership_evidence(packet, matrix, category)
    if ownership.preferred:
        near = near_scene_strong_candidates(ownership)
        if any(candidate.tool == ownership.preferred for candidate in near):
            return "high"
        return "medium"
    if ownership.weak_candidates:
        return "low"
    return "none"


def _packet_support_level(packet: Step1EvidencePacket, *, tool: str, category: str) -> str:
    for entry in packet.recall_coverage.matrix:
        if entry.tool == tool and entry.category == category:
            return entry.support_level
    return "unsupported"


def _candidate_evidence(
    candidate: OwnershipCandidate,
    category: str,
    eligibility: str,
) -> OwnerCandidateEvidence:
    scope = "unrelated_external" if candidate.scene_tier == "unrelated" else candidate.scene_tier
    return OwnerCandidateEvidence(
        tool=candidate.tool,
        category=category,
        eligibility=eligibility,
        evidence_scope=scope,
        evidence_refs=[candidate.evidence_id],
        caveat_refs=[],
        detected=candidate.detected,
        total=candidate.total,
        rate=candidate.rate,
    )


def category_ownership_panel(
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
    category: str,
) -> CategoryOwnershipPanel:
    evidence = category_ownership_evidence(packet, matrix, category)
    strong_source = [
        candidate
        for candidate in evidence.strong_candidates
        if _packet_support_level(packet, tool=candidate.tool, category=category) == "strong"
    ]
    demoted = [
        candidate
        for candidate in evidence.strong_candidates
        if candidate not in strong_source
    ]
    weak_source = list(evidence.weak_candidates) + demoted
    strong = [_candidate_evidence(candidate, category, "strong") for candidate in strong_source]
    weak = [_candidate_evidence(candidate, category, "weak") for candidate in weak_source]
    preferred = evidence.preferred
    unrelated_external_only = bool(strong) and all(
        candidate.evidence_scope == "unrelated_external" for candidate in strong
    )
    required_refs: list[str] = []
    if preferred:
        for candidate in strong_source + weak_source:
            if candidate.tool == preferred:
                required_refs.append(candidate.evidence_id)
                break
    has_strong = bool(strong)
    if preferred and has_strong:
        assignment_status = "assigned"
        gap_reason = ""
    elif preferred and not has_strong:
        assignment_status = "gap"
        gap_reason = "only_weak_candidates_for_preferred_owner"
    else:
        assignment_status = "gap"
        gap_reason = "no_assignment_eligible_owner_evidence"
    return CategoryOwnershipPanel(
        category=category,
        group=evidence.group,
        preferred_owner=preferred,
        assignment_status=assignment_status,
        strong_candidates=strong,
        weak_candidates=weak,
        rejected_candidates=[],
        gap_reason=gap_reason,
        unrelated_external_only=unrelated_external_only,
        rag_override_eligible=evidence.rag_override_eligible,
        override_targets=evidence.override_targets,
        override_refs=evidence.override_refs,
        caveats=["unrelated_external_only"] if unrelated_external_only else [],
        required_claim_refs=required_refs,
    )


def _category_capability_con_refs(
    matrix: ActionByEvidenceMatrix,
    *,
    category: str,
    preferred: str | None,
) -> list[str]:
    if not preferred:
        return []
    refs: list[str] = []
    for card in matrix.evidence_cards:
        if card.evidence_type != "rag_passage":
            continue
        if category not in _rag_categories(card):
            continue
        if preferred in _category_capability_con_tools(card):
            refs.append(card.evidence_id)
    return refs


def _con_ref_stronger_tools(
    matrix: ActionByEvidenceMatrix,
    *,
    evidence_id: str,
    preferred: str,
) -> set[str]:
    for card in matrix.evidence_cards:
        if card.evidence_id != evidence_id:
            continue
        return _category_capability_stronger_tools(card, weaker_tool=preferred)
    return set()


def category_capability_pro_refs(
    matrix: ActionByEvidenceMatrix,
    *,
    category: str,
    tool: str,
) -> list[str]:
    refs: list[str] = []
    for card in matrix.evidence_cards:
        if card.evidence_type != "rag_passage":
            continue
        if category not in _rag_categories(card):
            continue
        if tool in _category_capability_pro_tools(card):
            refs.append(card.evidence_id)
    return refs


def category_ownership_evidence(
    packet: Step1EvidencePacket,
    matrix: ActionByEvidenceMatrix,
    category: str,
) -> CategoryOwnershipEvidence:
    primary_tool = packet.primary_attention.primary_tool
    feasible_tools = {entry.tool for entry in packet.tool_table if entry.feasible}
    group = category_group(packet, category)
    all_strong: list[OwnershipCandidate] = []
    weak_candidates: list[OwnershipCandidate] = []

    for card in matrix.evidence_cards:
        if (
            card.evidence_type != "per_category_detected_total"
            or card.category != category
            or card.tool is None
            or card.tool == primary_tool
            or card.tool not in feasible_tools
            or card.value is None
        ):
            continue
        if not candidate_passes_precision_gate(packet, card.tool):
            continue
        candidate = _candidate_from_card(packet, card)
        if is_weak_eligible(card.value.detected, card.value.total, card.value.rate, True):
            weak_candidates.append(candidate)
        if group == "low_support" and card_scope_text(card) != "external_dataset":
            continue
        if is_strong_eligible(card.value.detected, card.value.total, card.value.rate):
            all_strong.append(candidate)

    local_strong = [
        candidate for candidate in all_strong if candidate.scene_tier in {"local", "primary"}
    ]
    near_strong = [
        candidate for candidate in all_strong if candidate.scene_tier == "near_scene"
    ]
    unrelated_strong = [
        candidate for candidate in all_strong if candidate.scene_tier == "unrelated"
    ]
    if local_strong:
        strong_candidates = local_strong
    elif near_strong:
        strong_candidates = near_strong
    else:
        strong_candidates = unrelated_strong
    all_strong_candidates = sorted(all_strong, key=_candidate_sort_key)
    strong_candidates = sorted(strong_candidates, key=_candidate_sort_key)
    weak_candidates = _top_candidates_by_tool(weak_candidates)

    preferred = strong_candidates[0].tool if strong_candidates else None
    override_refs = _category_capability_con_refs(matrix, category=category, preferred=preferred)
    override_targets: dict[str, list[str]] = {}
    seen_candidate_tools: set[str] = set()
    strong_override_tools = {
        candidate.tool
        for candidate in all_strong_candidates
        if candidate.scene_tier in {"local", "primary", "near_scene"}
    }
    for candidate in all_strong_candidates:
        if candidate.tool == preferred or candidate.tool in seen_candidate_tools:
            continue
        seen_candidate_tools.add(candidate.tool)
        if candidate.tool not in strong_override_tools:
            continue
        pro_refs = category_capability_pro_refs(matrix, category=category, tool=candidate.tool)
        if pro_refs:
            override_targets[candidate.tool] = pro_refs
    override_target_tools = set(override_targets)
    override_refs = [
        ref
        for ref in override_refs
        if preferred
        and _con_ref_stronger_tools(matrix, evidence_id=ref, preferred=preferred)
        & override_target_tools
    ]

    # Hard-tier scheduling evidence is exposed as an LLM-visible override
    # pathway. It must not rewrite the default preferred owner before CEGO
    # decides, but it can authorize a feasible owner named by the passage.
    hard_con_refs: list[str] = []
    hard_pro_targets: dict[str, list[str]] = {}
    for card in matrix.evidence_cards:
        if not _strong_scheduling_card_applies(
            packet,
            card,
            category=category,
            feasible_tools=feasible_tools,
        ):
            continue
        if preferred and preferred in _category_capability_con_tools(card):
            hard_con_refs.append(card.evidence_id)
        for tool_id in _category_capability_pro_tools(card):
            if tool_id == preferred or tool_id not in feasible_tools:
                continue
            if tool_id not in strong_override_tools:
                continue
            hard_pro_targets.setdefault(tool_id, []).append(card.evidence_id)
    if hard_con_refs and hard_pro_targets:
        override_targets = {
            tool_id: list(dict.fromkeys(refs))
            for tool_id, refs in hard_pro_targets.items()
        }
        override_refs = list(dict.fromkeys(hard_con_refs))

    return CategoryOwnershipEvidence(
        category=category,
        group=group,
        preferred=preferred,
        strong_candidates=strong_candidates,
        all_strong_candidates=all_strong_candidates,
        weak_candidates=weak_candidates,
        rag_override_eligible=bool(override_refs),
        override_refs=override_refs,
        override_targets=override_targets,
    )


def dataset_support_refs(
    matrix: ActionByEvidenceMatrix,
    *,
    tool: str,
    category: str,
) -> set[str]:
    refs: set[str] = set()
    for card in matrix.evidence_cards:
        if (
            card.evidence_type == "per_category_detected_total"
            and card.tool == tool
            and card.category == category
            and card_scope_text(card) == "external_dataset"
        ):
            refs.add(card.evidence_id)
    return refs
