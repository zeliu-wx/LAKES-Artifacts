"""Passage store: load passages and retrieve via vector similarity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolrank.schemas_v2 import Passage, PassageStore
from toolrank.vector_store import VectorIndex, resolve_embedding_config


def load_passage_store(path: Path) -> PassageStore | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return PassageStore.model_validate(data)


def _format_field_value(value: str | list[str]) -> str:
    if isinstance(value, list):
        return ", ".join(item for item in value if item)
    return value.strip()


def passage_to_embedding_text(passage: Passage) -> str:
    fields: list[tuple[str, str | list[str]]] = [
        ("claim_text", passage.claim_text),
        ("owner_tool", passage.owner_tool),
        ("counterpart_tool_ids", passage.counterpart_tool_ids),
        ("category", passage.category),
        ("knowledge_kind", passage.knowledge_kind),
        ("relation_to_owner", passage.relation_to_owner),
        ("action_scope", passage.action_scope),
        ("applicability_tags", passage.applicability_tags),
        ("evidence_basis", passage.evidence_basis),
        ("evidence_tier", passage.evidence_tier),
        ("source_reliability", passage.source_reliability),
        ("limitations_text", passage.limitations_text),
        ("source_id", passage.source_id),
    ]
    lines = []
    for name, value in fields:
        formatted = _format_field_value(value)
        if formatted:
            lines.append(f"{name}: {formatted}")
    return "\n".join(lines)


def passage_store_embedding_texts(store: PassageStore) -> list[str]:
    return [passage_to_embedding_text(passage) for passage in store.passages]


def _norm_key(value: str) -> str:
    return str(value).strip().lower()


class PassageGraphIndex:
    def __init__(self, passages: list[Passage]) -> None:
        self.tool_to_indexes: dict[str, set[int]] = {}
        self.category_to_indexes: dict[str, set[int]] = {}
        self.dataset_name_to_indexes: dict[str, set[int]] = {}
        self.scheduling_type_to_indexes: dict[str, set[int]] = {}
        self.source_id_to_indexes: dict[str, set[int]] = {}
        self.knowledge_kind_to_indexes: dict[str, set[int]] = {}
        for index, passage in enumerate(passages):
            for tool_id in [passage.owner_tool, *passage.counterpart_tool_ids]:
                self._add(self.tool_to_indexes, tool_id, index)
            self._add(self.category_to_indexes, passage.category, index)
            self._add(self.source_id_to_indexes, passage.source_id, index)
            self._add(self.scheduling_type_to_indexes, passage.scheduling_type, index)
            self._add(self.knowledge_kind_to_indexes, passage.knowledge_kind, index)

    @staticmethod
    def _add(index: dict[str, set[int]], key: str, passage_index: int) -> None:
        normalized = _norm_key(key)
        if normalized:
            index.setdefault(normalized, set()).add(passage_index)

    @staticmethod
    def _union(index: dict[str, set[int]], keys: list[str] | None) -> set[int]:
        candidates: set[int] = set()
        for key in keys or []:
            candidates.update(index.get(_norm_key(key), set()))
        return candidates

    def candidate_indexes(
        self,
        *,
        tool_ids: list[str] | None = None,
        categories: list[str] | None = None,
        dataset_names: list[str] | None = None,
        scheduling_types: list[str] | None = None,
        source_ids: list[str] | None = None,
        knowledge_kinds: list[str] | None = None,
    ) -> set[int]:
        candidates: set[int] | None = None
        for index, keys in (
            (self.tool_to_indexes, tool_ids),
            (self.dataset_name_to_indexes, dataset_names),
            (self.scheduling_type_to_indexes, scheduling_types),
            (self.source_id_to_indexes, source_ids),
            (self.knowledge_kind_to_indexes, knowledge_kinds),
        ):
            if not keys:
                continue
            matched = self._union(index, keys)
            candidates = matched if candidates is None else candidates & matched
        category_candidates = self._union(self.category_to_indexes, categories)
        if categories:
            return category_candidates if candidates is None else candidates & category_candidates
        if candidates is None:
            return category_candidates
        return candidates


def _has_structured_filter(*groups: list[str] | None) -> bool:
    return any(bool(group) for group in groups)


def project_passage_store_for_purpose_rag(store: PassageStore) -> PassageStore:
    """No-op preserved for backward compatibility with scripts and tests.

    Under the new owner-oriented Passage schema, every entry already carries a
    canonical `knowledge_kind` plus `relation_to_owner`, so no purpose-time
    re-projection is needed. The legacy implementation built copies via
    `_copy_passage`; that helper has been removed.
    """
    return store


def build_passage_vector_index(
    store: PassageStore,
    *,
    batch_size: int | None = None,
    **embedding_kwargs: Any,
) -> VectorIndex:
    index = VectorIndex()
    index.build(passage_store_embedding_texts(store), batch_size=batch_size, **embedding_kwargs)
    return index


def save_passage_vector_index(
    store: PassageStore,
    output_dir: Path,
    *,
    batch_size: int | None = None,
    **embedding_kwargs: Any,
) -> Path:
    index = build_passage_vector_index(store, batch_size=batch_size, **embedding_kwargs)
    config = resolve_embedding_config(
        base_url=embedding_kwargs.get("base_url"),
        api_key=embedding_kwargs.get("api_key"),
        model=embedding_kwargs.get("model"),
    )
    index_path = output_dir / "index.json"
    index.save(
        index_path,
        metadata={
            "provider": config.provider,
            "base_url": config.base_url,
            "model": config.model,
            "passage_ids": [passage.passage_id for passage in store.passages],
            "text_fields": [
                "claim_text",
                "owner_tool",
                "counterpart_tool_ids",
                "category",
                "knowledge_kind",
                "relation_to_owner",
                "action_scope",
                "applicability_tags",
                "evidence_basis",
                "evidence_tier",
                "source_reliability",
                "limitations_text",
                "source_id",
            ],
        },
    )
    return index_path


class PassageRetriever:
    def __init__(
        self,
        store: PassageStore,
        *,
        index: VectorIndex | None = None,
        **embedding_kwargs: Any,
    ) -> None:
        self._passages = store.passages
        self._graph = PassageGraphIndex(self._passages)
        self._index = index or build_passage_vector_index(store, **embedding_kwargs)
        if len(self._index) != len(self._passages):
            raise ValueError("Vector index size does not match passage store size")

    def search_text(
        self,
        query_text: str,
        top_k: int = 3,
        **embedding_kwargs: Any,
    ) -> list[tuple[Passage, float]]:
        if not query_text.strip():
            return []
        hits = self._index.query(query_text, top_k=top_k, **embedding_kwargs)
        return [(self._passages[i], score) for i, score in hits]

    def _fallback_candidates_without_knowledge_kind(
        self,
        *,
        tool_ids: list[str] | None = None,
        categories: list[str] | None = None,
        dataset_names: list[str] | None = None,
        scheduling_types: list[str] | None = None,
        source_ids: list[str] | None = None,
        knowledge_kinds: list[str] | None = None,
    ) -> set[int] | range | None:
        if not knowledge_kinds:
            return None
        candidates = self._graph.candidate_indexes(
            tool_ids=tool_ids,
            categories=categories,
            dataset_names=dataset_names,
            scheduling_types=scheduling_types,
            source_ids=source_ids,
        )
        if candidates:
            return candidates
        if _has_structured_filter(tool_ids, categories, dataset_names, scheduling_types, source_ids):
            return None
        return range(len(self._passages))

    def search_structured(
        self,
        query_text: str,
        *,
        tool_ids: list[str] | None = None,
        categories: list[str] | None = None,
        dataset_names: list[str] | None = None,
        scheduling_types: list[str] | None = None,
        source_ids: list[str] | None = None,
        knowledge_kinds: list[str] | None = None,
        top_k: int = 3,
        candidate_multiplier: int = 8,
        **embedding_kwargs: Any,
    ) -> list[tuple[Passage, float]]:
        if not query_text.strip():
            return []
        candidates = self._graph.candidate_indexes(
            tool_ids=tool_ids,
            categories=categories,
            dataset_names=dataset_names,
            scheduling_types=scheduling_types,
            source_ids=source_ids,
            knowledge_kinds=knowledge_kinds,
        )
        if not candidates:
            fallback_candidates = self._fallback_candidates_without_knowledge_kind(
                tool_ids=tool_ids,
                categories=categories,
                dataset_names=dataset_names,
                scheduling_types=scheduling_types,
                source_ids=source_ids,
                knowledge_kinds=knowledge_kinds,
            )
            if fallback_candidates is None:
                if _has_structured_filter(
                    tool_ids,
                    categories,
                    dataset_names,
                    scheduling_types,
                    source_ids,
                    knowledge_kinds,
                ):
                    return []
                return self.search_text(query_text, top_k=top_k, **embedding_kwargs)
            candidates = fallback_candidates
        candidate_limit = max(top_k, top_k * max(candidate_multiplier, 1))
        hits = self._index.query_candidates(
            query_text,
            candidates,
            top_k=min(candidate_limit, len(candidates)),
            **embedding_kwargs,
        )
        return [(self._passages[i], score) for i, score in hits[:top_k]]

    def batch_search_structured(
        self,
        queries: list[dict[str, Any]],
        **embedding_kwargs: Any,
    ) -> list[list[tuple[Passage, float]]]:
        if not queries:
            return []
        batch_inputs: list[tuple[str, set[int] | list[int] | range, int]] = []
        query_indexes: list[int] = []
        fallback_results: dict[int, list[tuple[Passage, float]]] = {}
        for qi, q in enumerate(queries):
            query_text = q.get("query_text", "")
            top_k = q.get("top_k", 3)
            candidate_multiplier = q.get("candidate_multiplier", 8)
            if not query_text.strip():
                fallback_results[qi] = []
                continue
            candidates = self._graph.candidate_indexes(
                tool_ids=q.get("tool_ids"),
                categories=q.get("categories"),
                dataset_names=q.get("dataset_names"),
                scheduling_types=q.get("scheduling_types"),
                source_ids=q.get("source_ids"),
                knowledge_kinds=q.get("knowledge_kinds"),
            )
            if not candidates:
                fallback_candidates = self._fallback_candidates_without_knowledge_kind(
                    tool_ids=q.get("tool_ids"),
                    categories=q.get("categories"),
                    dataset_names=q.get("dataset_names"),
                    scheduling_types=q.get("scheduling_types"),
                    source_ids=q.get("source_ids"),
                    knowledge_kinds=q.get("knowledge_kinds"),
                )
                if fallback_candidates is None:
                    has_filter = _has_structured_filter(
                        q.get("tool_ids"), q.get("categories"), q.get("dataset_names"),
                        q.get("scheduling_types"), q.get("source_ids"), q.get("knowledge_kinds"),
                    )
                    if has_filter:
                        fallback_results[qi] = []
                        continue
                    candidates = range(len(self._passages))
                else:
                    candidates = fallback_candidates
            candidate_limit = max(top_k, top_k * max(candidate_multiplier, 1))
            batch_inputs.append((query_text, candidates, min(candidate_limit, len(candidates))))
            query_indexes.append(qi)
        if batch_inputs:
            batch_hits = self._index.batch_query_candidates(batch_inputs, **embedding_kwargs)
        else:
            batch_hits = []
        results: list[list[tuple[Passage, float]]] = [[] for _ in queries]
        for qi in range(len(queries)):
            if qi in fallback_results:
                results[qi] = fallback_results[qi]
        for bi, qi in enumerate(query_indexes):
            top_k = queries[qi].get("top_k", 3)
            results[qi] = [(self._passages[i], score) for i, score in batch_hits[bi][:top_k]]
        return results

    def retrieve(
        self,
        tool_ids: list[str],
        categories: list[str],
        top_k: int = 3,
        **embedding_kwargs: Any,
    ) -> list[Passage]:
        query_text = " ".join(tool_ids) + " " + " ".join(categories)
        return [
            passage
            for passage, _ in self.search_structured(
                query_text,
                tool_ids=tool_ids,
                categories=categories,
                top_k=top_k,
                **embedding_kwargs,
            )
        ]
