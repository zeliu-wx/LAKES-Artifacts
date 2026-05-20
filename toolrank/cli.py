from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from toolrank.cego import CegoError
from toolrank.engine import run_recommendation

app = typer.Typer(help="Deterministic SmartBugs-style tool recommendation CLI", no_args_is_help=True)

_DEFAULT_TOOLCARDS_DIR = str((Path(__file__).resolve().parent.parent / "toolcards"))


@app.callback()
def root() -> None:
    """Command group root."""


@app.command("recommend")
def recommend(
    target_path: Optional[str] = typer.Argument(None, help="Target contract file or directory."),
    tool_slots: int = typer.Option(5, "--tool-slots", min=0, help="Budget: max concurrent tools."),
    runtime_cap_minutes: float = typer.Option(30.0, "--runtime-cap-minutes", help="Budget: runtime cap in minutes."),
    alert_cap: str = typer.Option("medium", "--alert-cap", help="Budget: alert cap (low|medium|high)."),
    model: str = typer.Option("", "--model", help="LLM model name."),
    top_k: int = typer.Option(5, "--top-k", min=1, help="Scene pool top-K neighbors."),
    emit: str = typer.Option("summary", "--emit", help="Output format: json|summary (default: summary)."),
    toolcards_dir: str = typer.Option(_DEFAULT_TOOLCARDS_DIR, "--toolcards-dir", help="ToolCards folder path (default: packaged toolcards/)."),
    passage_store: Optional[str] = typer.Option(None, "--passage-store", help="Passage store JSON path."),
    vector_index: Optional[str] = typer.Option(None, "--vector-index", help="Vector index JSON path."),
    no_retrieval: bool = typer.Option(False, "--no-retrieval", help="Disable passage retrieval (ablation)."),
    execute: bool = typer.Option(False, "--execute", help="Run selected tools and write LAKES_out/<contract>/fused_report.json."),
    results_root: str = typer.Option("LAKES_out", "--results-root", help="Root for execution outputs; per-contract folders are created under LAKES_out."),
    runner_script: Optional[str] = typer.Option(None, "--runner-script", help="Analyzer runner script path."),
    runner_cwd: Optional[str] = typer.Option(None, "--runner-cwd", help="Working directory for the analyzer runner."),
    tool_timeout_sec: int = typer.Option(600, "--tool-timeout-sec", min=1, help="Per-tool timeout in seconds (default: 600)."),
    gptscan_timeout_sec: int = typer.Option(600, "--gptscan-timeout-sec", min=1, help="GPTScan LLM timeout in seconds."),
    execution_jobs: int = typer.Option(
        0,
        "--execution-jobs",
        min=0,
        help="Max tools to run in parallel during --execute; 0 means one job per selected tool.",
    ),
    openai_api_key: Optional[str] = typer.Option(None, "--openai-api-key", help="OpenAI API key forwarded to GPTScan."),
    openai_api_base: Optional[str] = typer.Option(None, "--openai-api-base", help="OpenAI-compatible API base forwarded to GPTScan."),
    explain: bool = typer.Option(
        True,
        "--explain/--no-explain",
        "-x",
        help="Print per-stage detail (default: on). Use --no-explain to suppress for scripted/parseable output.",
    ),
) -> None:
    emit = emit.lower().strip()
    alert_cap = alert_cap.lower().strip()
    if emit not in {"json", "summary"}:
        raise typer.BadParameter("--emit must be json or summary")
    if alert_cap not in {"low", "medium", "high"}:
        raise typer.BadParameter("--alert-cap must be low|medium|high")

    try:
        result = run_recommendation(
            target_path=target_path,
            toolcards_dir=toolcards_dir,
            tool_slots=tool_slots,
            runtime_cap_minutes=runtime_cap_minutes,
            alert_cap=alert_cap,
            model=model,
            top_k=top_k,
            explain=explain,
            enable_retrieval=not no_retrieval,
            passage_store_path=passage_store,
            vector_index_path=vector_index,
            execute=execute,
            run_results_root=results_root,
            runner_script=runner_script,
            runner_cwd=runner_cwd,
            tool_timeout_sec=tool_timeout_sec,
            gptscan_timeout_sec=gptscan_timeout_sec,
            execution_jobs=execution_jobs,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
        )
    except (CegoError, RuntimeError) as exc:
        typer.echo(f"recommendation failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    if emit == "summary":
        selected_tools = [item.tool for item in result.certificate.selected_plan]
        lines = [
            f"certification: {result.packet.certification.status}",
            f"selected_action: {result.certificate.selected_action_id}",
            f"checker: {result.checker_verdict.status}",
            f"selected_tools: {', '.join(selected_tools) if selected_tools else 'none'}",
        ]
        execution_result = getattr(result, "execution", None)
        lakes_output_dir = getattr(result, "lakes_output_dir", None)
        if execution_result is not None:
            lines.append(f"execution: {execution_result.status}")
        if lakes_output_dir:
            lines.append(f"LAKES_out: {Path(lakes_output_dir) / 'fused_report.json'}")
        typer.echo("\n".join(lines))
    else:
        payload = {
            "packet": result.packet.model_dump(),
            "certificate": result.certificate.model_dump(),
            "checker_verdict": result.checker_verdict.model_dump(),
            "execution": result.execution.model_dump() if getattr(result, "execution", None) else None,
            "fused_report": result.fused_report.model_dump() if getattr(result, "fused_report", None) else None,
            "lakes_output_dir": getattr(result, "lakes_output_dir", None),
            "warnings": result.warnings,
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("kb-extract")
def kb_extract(
    mineru_dirs: list[Path] = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="One or more MinerU hybrid_auto output directories.",
    ),
    toolcards_dir: Path = typer.Option(
        Path("toolcards"),
        "--toolcards-dir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory containing ToolCards.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Output directory for KB extraction artifacts.",
    ),
    model: str = typer.Option("gpt-5.5", "--model", help="LLM model name."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run extraction without materializing ToolCards."),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Reuse existing paper_dossier.json files that already contain scheduling evidence."),
    verbose: bool = typer.Option(False, "--verbose", help="Print accepted entry details and skip reason distribution."),
) -> None:
    from toolrank.kb_extract.pipeline import run_kb_extract_pipeline

    result = run_kb_extract_pipeline(
        toolcards_dir=toolcards_dir,
        out_dir=output_dir,
        model=model,
        dry_run=dry_run,
        mineru_output_dirs=mineru_dirs,
        skip_existing=skip_existing,
    )

    manifest = result.get("manifest", {})
    typer.echo("KB Extract Manifest")
    typer.echo("=" * 72)
    typer.echo(f"papers_processed: {manifest.get('papers_processed', 0)}")
    typer.echo(f"total_candidates: {manifest.get('total_candidates', 0)}")
    typer.echo(f"accepted: {manifest.get('accepted', 0)}")
    typer.echo(f"skipped: {manifest.get('skipped', 0)}")
    typer.echo(f"manual_review: {manifest.get('manual_review', 0)}")
    typer.echo(f"materialized: {manifest.get('materialized', 0)}")

    if not verbose:
        return

    audit = result.get("audit", {})
    paper_rows = audit.get("papers", []) if isinstance(audit, dict) else []
    paper_dossiers = manifest.get("paper_dossiers", [])
    typer.echo("\nPaper Dossiers")
    typer.echo("=" * 72)
    if not paper_dossiers:
        typer.echo("(none)")
        return
    for index, dossier in enumerate(paper_dossiers, start=1):
        row = paper_rows[index - 1] if index - 1 < len(paper_rows) and isinstance(paper_rows[index - 1], dict) else {}
        typer.echo(
            f"{index}. doc_id={dossier.get('doc_id', '(unknown)')} | "
            f"paper_type={dossier.get('paper_type', '(unknown)')} | "
            f"tools_identified={dossier.get('tools_identified', 0)} | "
            f"passage_candidates={row.get('candidates', 0)}"
        )
        typer.echo(f"   dossier_path={dossier.get('dossier_path', '')}")


@app.command("kb-audit")
def kb_audit(
    passage_store_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to passage_store.json.",
    ),
    toolcards_dir: str = typer.Option(
        "toolcards",
        "--toolcards-dir",
        help="Directory containing ToolCards (for whitelist reference).",
    ),
) -> None:
    from collections import Counter

    from toolrank.kb_extract.tool_whitelist import load_toolcard_whitelist
    from toolrank.schemas_v2 import PassageStore

    _ = toolcards_dir
    payload = json.loads(passage_store_path.read_text(encoding="utf-8"))
    store = PassageStore.model_validate(payload)
    allowed_tools = load_toolcard_whitelist(Path(__file__).resolve().parent / "kb_extract" / "config" / "toolcard_whitelist.json")
    expected_categories = {
        "reentrancy",
        "access_control",
        "arithmetic",
        "unchecked_low_level_calls",
        "denial_of_service",
        "bad_randomness",
        "front_running",
        "time_manipulation",
        "short_addresses",
    }
    source_counts = Counter(passage.source_id or "(unknown)" for passage in store.passages)
    kind_counts = Counter(passage.knowledge_kind or "(missing)" for passage in store.passages)
    tool_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for passage in store.passages:
        tool_counts[passage.owner_tool] += 1
        for tool_id in passage.counterpart_tool_ids:
            tool_counts[tool_id] += 1
        if passage.category and passage.category != "__GLOBAL__":
            category_counts[passage.category] += 1

    typer.echo(f"Total passages: {len(store.passages)}")
    typer.echo("\nBy source")
    for source, count in source_counts.most_common():
        typer.echo(f"  {source}: {count}")
    typer.echo("\nBy knowledge_kind")
    for kind, count in kind_counts.most_common():
        typer.echo(f"  {kind}: {count}")

    typer.echo("\nTool coverage (whitelist)")
    for tool_id in sorted(allowed_tools):
        count = tool_counts.get(tool_id, 0)
        marker = "" if count >= 3 else " <-- LOW" if count > 0 else " <-- MISSING"
        typer.echo(f"  {tool_id}: {count}{marker}")

    typer.echo("\nCategory coverage")
    for category in sorted(expected_categories):
        count = category_counts.get(category, 0)
        marker = "" if count >= 3 else " <-- LOW" if count > 0 else " <-- MISSING"
        typer.echo(f"  {category}: {count}{marker}")

    issues: list[str] = []
    for index, passage in enumerate(store.passages, start=1):
        text = passage.claim_text.lower()
        passage_tools = {passage.owner_tool, *passage.counterpart_tool_ids}
        for tool_id in sorted(allowed_tools):
            if tool_id.lower() in text and tool_id not in passage_tools:
                issues.append(
                    f"  passage #{index} [{passage.source_id}]: claim mentions '{tool_id}' "
                    f"but owner_tool={passage.owner_tool!r}, counterpart_tool_ids={passage.counterpart_tool_ids}"
                )

    typer.echo()
    if issues:
        typer.echo(f"Potential issues ({len(issues)})")
        for issue in issues[:20]:
            typer.echo(issue)
        if len(issues) > 20:
            typer.echo(f"  ... and {len(issues) - 20} more")
    else:
        typer.echo("No obvious issues found.")


@app.command("kb-vector-build")
def kb_vector_build(
    passage_store_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to passage_store.json.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory for vector index files. Defaults to passage_store parent/vector_index.",
    ),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Embedding API base URL."),
    model: Optional[str] = typer.Option(None, "--model", help="Embedding model name."),
    batch_size: int = typer.Option(0, "--batch-size", min=0, help="Embedding batch size; 0 sends one request."),
    no_proxy: bool = typer.Option(False, "--no-proxy", help="Bypass proxy variables for embedding requests."),
) -> None:
    from toolrank.passage_store import load_passage_store, save_passage_vector_index

    store = load_passage_store(passage_store_path)
    if store is None:
        typer.echo("passage_store.json not found", err=True)
        raise typer.Exit(1)
    output_dir = output_dir or (passage_store_path.parent / "vector_index")
    embedding_kwargs: dict[str, object] = {"no_proxy": no_proxy}
    if base_url:
        embedding_kwargs["base_url"] = base_url
    if model:
        embedding_kwargs["model"] = model
    index_path = save_passage_vector_index(
        store,
        output_dir,
        batch_size=batch_size or None,
        **embedding_kwargs,
    )
    typer.echo(f"wrote {index_path} passages={len(store.passages)}")


@app.command("kb-vector-query")
def kb_vector_query(
    passage_store_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to passage_store.json.",
    ),
    query: str = typer.Argument(..., help="Query text to embed and search."),
    index_dir: Path = typer.Option(
        ...,
        "--index-dir",
        exists=True,
        readable=True,
        resolve_path=True,
        help="Vector index directory or index.json path.",
    ),
    top_k: int = typer.Option(3, "--top-k", min=1, help="Number of passages to return."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Embedding API base URL."),
    model: Optional[str] = typer.Option(None, "--model", help="Embedding model name."),
    no_proxy: bool = typer.Option(False, "--no-proxy", help="Bypass proxy variables for embedding requests."),
) -> None:
    from toolrank.passage_store import PassageRetriever, load_passage_store
    from toolrank.vector_store import VectorIndex

    store = load_passage_store(passage_store_path)
    if store is None:
        typer.echo("passage_store.json not found", err=True)
        raise typer.Exit(1)
    index_path = index_dir / "index.json" if index_dir.is_dir() else index_dir
    index = VectorIndex.load(index_path)
    retriever = PassageRetriever(store, index=index)
    embedding_kwargs: dict[str, object] = {"no_proxy": no_proxy}
    if base_url:
        embedding_kwargs["base_url"] = base_url
    if model:
        embedding_kwargs["model"] = model
    results = retriever.search_text(query, top_k=top_k, **embedding_kwargs)
    payload = {
        "query": query,
        "results": [
            {
                "score": score,
                "passage_id": passage.passage_id,
                "source_id": passage.source_id,
                "owner_tool": passage.owner_tool,
                "counterpart_tool_ids": passage.counterpart_tool_ids,
                "category": passage.category,
                "knowledge_kind": passage.knowledge_kind,
                "relation_to_owner": passage.relation_to_owner,
                "evidence_tier": passage.evidence_tier,
                "claim_text": passage.claim_text,
            }
            for passage, score in results
        ],
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("refresh-kb")
def refresh_kb(
    performance_artifacts: Optional[list[str]] = typer.Option(
        None,
        "--performance",
        help="Paths to new performance_db JSON artifact(s) to merge.",
    ),
    toolcards_dir: str = typer.Option(
        "toolcards",
        "--toolcards-dir",
        help="Directory containing performance_db.json.",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for refreshed DBs. Defaults to --toolcards-dir (in-place).",
    ),
) -> None:
    """Refresh the dataset/benchmark knowledge base.

    Merges new performance benchmark artifacts into the existing
    performance_db.json knowledge base.
    """
    from toolrank.dataset_kb import refresh_performance_db

    tc_path = Path(toolcards_dir)
    out_path = Path(output_dir) if output_dir else tc_path
    out_path.mkdir(parents=True, exist_ok=True)

    refreshed: list[str] = []

    if performance_artifacts:
        perf_db = tc_path / "performance_db.json"
        perf_out = out_path / "performance_db.json"
        result = refresh_performance_db(
            existing_path=perf_db,
            new_artifact_paths=[Path(p) for p in performance_artifacts],
            output_path=perf_out,
        )
        refreshed.append(f"performance_db: {len(result.entries)} entries -> {perf_out}")

    if not refreshed:
        typer.echo("No artifacts provided. Use --performance.")
        raise typer.Exit(1)

    for line in refreshed:
        typer.echo(f"[OK] {line}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
