from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from toolrank.kb_extract.models import RunManifest


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_path(path: Path) -> str:
    if path.is_file():
        return _hash_bytes(path.read_bytes())
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def build_run_manifest(
    *,
    source_paths: list[Path],
    extraction_model: str,
    gate_model: str,
    schema_registry_version: str,
    papers_processed: int,
    total_candidates: int,
    accepted: int,
    deferred: int,
    skipped: int,
    manual_review: int,
    materialized: int,
    blast_radius_triggered: bool,
    extraction_prompt_version: str,
    gate_prompt_version: str,
    ledger_path: str,
    materialized_cards: list[str],
    llm_first_pipeline: bool = True,
    paper_dossiers: list[dict[str, Any]] | list[Any] | None = None,
) -> RunManifest:
    return RunManifest(
        run_id=str(uuid4()),
        document_hashes={str(path): hash_path(path) for path in source_paths},
        extraction_prompt_version=extraction_prompt_version,
        extraction_model=extraction_model,
        gate_prompt_version=gate_prompt_version,
        gate_model=gate_model,
        schema_registry_version=schema_registry_version,
        llm_first_pipeline=llm_first_pipeline,
        blast_radius_triggered=blast_radius_triggered,
        papers_processed=papers_processed,
        total_candidates=total_candidates,
        accepted=accepted,
        deferred=deferred,
        skipped=skipped,
        manual_review=manual_review,
        materialized=materialized,
        ledger_path=ledger_path,
        materialized_cards=materialized_cards,
        paper_dossiers=paper_dossiers or [],
    )


def write_run_manifest(out_dir: Path, manifest: RunManifest) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def project_extraction_audit(
    *,
    paper_rows: list[dict[str, Any]],
    merged_candidates: int,
    applied: int,
    skipped: int,
    dry_run: bool,
    model: str,
    llm_used: bool,
    prompt_versions: dict[str, str],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "paper_count": len(paper_rows),
        "papers": paper_rows,
        "merged_candidates": merged_candidates,
        "applied": applied,
        "skipped": skipped,
        "dry_run": dry_run,
        "model": model,
        "llm_used": llm_used,
        "prompt_versions": prompt_versions,
        "decisions": decisions,
    }


__all__ = ["build_run_manifest", "write_run_manifest", "project_extraction_audit", "hash_path"]
