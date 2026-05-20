from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolrank.kb_extract import relation_first
from toolrank.kb_extract.audit import build_run_manifest, project_extraction_audit, write_run_manifest
from toolrank.kb_extract.tool_whitelist import load_toolcard_whitelist
from toolrank.openai_compat import load_openai_client
from toolrank.schemas_v2 import Passage, PassageStore


def _log_progress(message: str) -> None:
    print(f"[kb_extract][pipeline] {message}", flush=True)


def _load_schema_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def _derive_paper_identity(root: Path) -> tuple[str, str]:
    markdown_files = sorted(root.glob("*.md"))
    if markdown_files and markdown_files[0].stem != "full":
        paper_name = markdown_files[0].stem
    elif root.name.lower() == "hybrid_auto" and root.parent.name:
        paper_name = root.parent.name
    else:
        paper_name = root.stem
    return paper_name, _slug(paper_name)


def _paper_dossier_entry(doc_id: str, paper_dossier: dict[str, Any], dossier_path: Path, model: str) -> dict[str, Any]:
    entity_inventory = paper_dossier.get("entity_inventory", {})
    tools = entity_inventory.get("systems_tools", []) if isinstance(entity_inventory, dict) else []
    experiment_relations = paper_dossier.get("experiment_relations", [])
    return {
        "doc_id": doc_id,
        "dossier_path": str(dossier_path),
        "paper_type": str(paper_dossier.get("paper_type", "other")),
        "tools_identified": len(tools) if isinstance(tools, list) else 0,
        "aggregate_claims_identified": len(experiment_relations) if isinstance(experiment_relations, list) else 0,
        "extraction_model": model,
        "extraction_tokens_used": 0,
    }


def run_kb_extract_pipeline(
    *,
    toolcards_dir: str | Path,
    out_dir: str | Path,
    model: str,
    dry_run: bool,
    mineru_output_dirs: list[Path],
    skip_existing: bool = False,
) -> dict[str, Any]:
    _ = toolcards_dir
    _ = skip_existing
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    client = load_openai_client()
    if client is None:
        raise RuntimeError("LLM client unavailable for KB extraction")

    package_root = Path(__file__).resolve().parent
    allowed_tool_ids = load_toolcard_whitelist(package_root / "config" / "toolcard_whitelist.json")
    schema_registry = _load_schema_registry(package_root / "config" / "schema_registry.json")

    source_roots = [Path(root).expanduser().resolve() for root in mineru_output_dirs]
    source_paths = list(source_roots)
    all_passages: list[Passage] = []
    raw_documents: list[dict[str, Any]] = []
    dossier_documents: list[dict[str, Any]] = []
    critique_entries: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    paper_dossier_entries: list[dict[str, Any]] = []

    total_candidates = 0
    total_dropped = 0
    total_rejected = 0

    _log_progress(f"starting documents={len(source_roots)} dry_run={dry_run} relation_first=True")
    for root in source_roots:
        doc_id, _source_prefix = _derive_paper_identity(root)
        _log_progress(f"doc={doc_id} extracting_experiment_relations source={root}")
        raw_response = relation_first.extract_relation_first_raw_response(
            client=client,
            model=model,
            mineru_output_dir=root,
            doc_id=doc_id,
            allowed_tool_ids=allowed_tool_ids,
            diagnostics_dir=out_root,
        )
        projection = relation_first.project_relation_first_response(
            raw_response,
            doc_id=doc_id,
            allowed_tool_ids=allowed_tool_ids,
        )

        all_passages.extend(projection.passages)
        total_candidates += projection.total_relations
        total_dropped += projection.dropped_count
        total_rejected += projection.rejected_count
        raw_documents.append({"doc_id": doc_id, "source": str(root), "raw_response": raw_response})
        dossier_documents.append({"doc_id": doc_id, "paper_dossier": projection.paper_dossier.model_dump()})
        for item in projection.critique_log:
            critique_entries.append({"doc_id": doc_id, **item.model_dump()})
        paper_rows.append(
            {
                "source": str(root),
                "candidates": projection.total_relations,
                "kept": projection.kept_count,
                "dropped": projection.dropped_count,
                "rejected": projection.rejected_count,
            }
        )
        _log_progress(
            f"doc={doc_id} relation_first relations={projection.total_relations} "
            f"kept={projection.kept_count} dropped={projection.dropped_count} rejected={projection.rejected_count}"
        )

    paper_dossier_path = out_root / "paper_dossier.json"
    raw_response_path = out_root / "raw_response.json"
    critique_log_path = out_root / "critique_log.json"
    passage_store_path = out_root / "passage_store.json"

    _write_json(paper_dossier_path, {"papers": dossier_documents})
    _write_json(raw_response_path, {"documents": raw_documents})
    _write_json(critique_log_path, {"entries": critique_entries})
    _write_json(passage_store_path, PassageStore(passages=all_passages).model_dump())

    for dossier in dossier_documents:
        paper_dossier_entries.append(
            _paper_dossier_entry(
                dossier["doc_id"],
                dossier["paper_dossier"],
                paper_dossier_path,
                model,
            )
        )

    skipped = total_dropped + total_rejected
    manifest = build_run_manifest(
        source_paths=source_paths or source_roots,
        extraction_model=model,
        gate_model=model,
        schema_registry_version=str(schema_registry.get("version", "unknown")),
        papers_processed=len(source_roots),
        total_candidates=total_candidates,
        accepted=len(all_passages),
        deferred=0,
        skipped=skipped,
        manual_review=0,
        materialized=len(all_passages),
        blast_radius_triggered=False,
        extraction_prompt_version=relation_first.RELATION_FIRST_PROMPT_VERSION,
        gate_prompt_version=relation_first.RELATION_FIRST_GATE_VERSION,
        ledger_path=None,
        materialized_cards=[],
        llm_first_pipeline=True,
        paper_dossiers=paper_dossier_entries,
    )
    write_run_manifest(out_root, manifest)

    audit = project_extraction_audit(
        paper_rows=paper_rows,
        merged_candidates=total_candidates,
        applied=len(all_passages),
        skipped=skipped,
        dry_run=dry_run,
        model=model,
        llm_used=client is not None,
        prompt_versions={
            "relation_first": relation_first.RELATION_FIRST_PROMPT_VERSION,
            "gate": relation_first.RELATION_FIRST_GATE_VERSION,
        },
        decisions=critique_entries,
    )
    return {
        "audit": audit,
        "manifest": manifest.model_dump(),
        "ledger_path": "",
        "materialized_cards": [],
        "blast_radius_triggered": False,
    }


__all__ = ["run_kb_extract_pipeline", "_derive_paper_identity"]
