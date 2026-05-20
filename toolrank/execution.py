from __future__ import annotations

from collections import deque
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set

from toolrank.report_parser import load_per_tool_findings
from toolrank.schemas import CompositionPlan, ExecutionResult

logger = logging.getLogger(__name__)

DEFAULT_RUNNER_SCRIPT = (
    Path(os.environ["TOOLRANK_RUNNER_SCRIPT"])
    if os.getenv("TOOLRANK_RUNNER_SCRIPT")
    else Path(__file__).resolve().parent / "runner.py"
)
DEFAULT_RUNNER_CWD = Path(os.getenv("TOOLRANK_RUNNER_CWD", "")) if os.getenv("TOOLRANK_RUNNER_CWD") else None
DEFAULT_SMARTBUGS_DIR = Path(os.getenv("TOOLRANK_SMARTBUGS_DIR", "")) if os.getenv("TOOLRANK_SMARTBUGS_DIR") else None


def _stream_runner_output(
    command: List[str],
    *,
    cwd: Optional[str],
    tail_chars: int = 4000,
) -> tuple[int, Optional[str], Optional[str]]:
    """Run a command while streaming stdout/stderr to the terminal.

    The process output is also retained in bounded buffers so the caller can
    persist `stdout_tail` / `stderr_tail` into `ExecutionResult`.
    """
    proc = subprocess.Popen(
        command,
        cwd=cwd or None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_buf: deque[str] = deque()
    stderr_buf: deque[str] = deque()
    stdout_len = 0
    stderr_len = 0
    stdout_lock = threading.Lock()
    stderr_lock = threading.Lock()

    def _append(buf: deque[str], lock: threading.Lock, current_len: int, chunk: str) -> int:
        with lock:
            buf.append(chunk)
            current_len += len(chunk)
            while current_len > tail_chars and buf:
                removed = buf.popleft()
                current_len -= len(removed)
        return current_len

    def _pump(stream, sink, buf: deque[str], lock: threading.Lock, is_stdout: bool) -> None:
        nonlocal stdout_len, stderr_len
        if stream is None:
            return
        try:
            for line in stream:
                sink.write(line)
                sink.flush()
                if is_stdout:
                    stdout_len = _append(buf, lock, stdout_len, line)
                else:
                    stderr_len = _append(buf, lock, stderr_len, line)
        finally:
            stream.close()

    t_out = threading.Thread(
        target=_pump,
        # Stream runner stdout to the parent's stderr so shell redirection of
        # stdout can still capture a clean JSON payload from the CLI.
        args=(proc.stdout, sys.stderr, stdout_buf, stdout_lock, True),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_pump,
        args=(proc.stderr, sys.stderr, stderr_buf, stderr_lock, False),
        daemon=True,
    )
    t_out.start()
    t_err.start()
    return_code = proc.wait()
    t_out.join()
    t_err.join()

    stdout_tail = "".join(stdout_buf) or None
    stderr_tail = "".join(stderr_buf) or None
    return return_code, stdout_tail, stderr_tail


def _tool_category_mapping(plan: CompositionPlan) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for category, tool_id in plan.category_assignments.items():
        if tool_id == plan.anchor_tool_id:
            continue
        mapping.setdefault(tool_id, []).append(category)
    for tool_id in mapping:
        mapping[tool_id] = sorted(set(mapping[tool_id]))
    return dict(sorted(mapping.items()))


def _tool_category_arg(mapping: Dict[str, List[str]]) -> str:
    parts: List[str] = []
    for tool_id, categories in mapping.items():
        if not categories:
            continue
        parts.append(f"{tool_id}:{'|'.join(category.upper() for category in categories)}")
    return ",".join(parts)


def build_execution_plan(
    target_path: str | Path,
    results_root: str | Path,
    composition: CompositionPlan,
    *,
    runner_script: str | Path | None = None,
    runner_cwd: str | Path | None = None,
    tool_timeout_sec: int = 1200,
    gptscan_timeout_sec: int = 600,
    execution_jobs: int = 0,
    openai_api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    write_lakes_output: bool = True,
) -> ExecutionResult:
    runner_script_path = Path(runner_script) if runner_script else DEFAULT_RUNNER_SCRIPT
    if runner_script_path is None:
        raise ValueError("No runner script configured. Set TOOLRANK_RUNNER_SCRIPT or pass --runner-script.")
    runner_cwd_path = Path(runner_cwd) if runner_cwd else DEFAULT_RUNNER_CWD
    selected = composition.selected_tool_ids
    if execution_jobs < 0:
        raise ValueError("execution_jobs must be non-negative")
    effective_jobs = len(selected) if execution_jobs == 0 else execution_jobs
    primary = composition.anchor_tool_id or (selected[0] if selected else None)
    category_mapping = _tool_category_mapping(composition)
    command = [
        "python",
        str(runner_script_path),
        str(Path(target_path).resolve()),
        str(Path(results_root).resolve()),
        "--tools",
        ",".join(selected),
        "--primary_tool",
        primary or "",
        "--timeout",
        str(tool_timeout_sec),
        "--gptscan_timeout",
        str(gptscan_timeout_sec),
    ]
    if execution_jobs != 0 or effective_jobs > 1:
        command.extend(["--jobs", str(effective_jobs)])
    tool_categories = _tool_category_arg(category_mapping)
    if tool_categories:
        command.extend(["--tool_categories", tool_categories])
    if openai_api_key:
        command.extend(["--openai_api_key", openai_api_key])
    if openai_api_base:
        command.extend(["--openai_api_base", openai_api_base])
    if not write_lakes_output:
        command.append("--no-lakes-output")

    fusion_summary = (
        f"primary={primary}; category_overrides={category_mapping}"
        if primary
        else "no primary tool selected"
    )
    return ExecutionResult(
        status="planned",
        execution_mode="manual_runner",
        target_path=str(Path(target_path).resolve()),
        results_root=str(Path(results_root).resolve()),
        runner_script=str(runner_script_path.resolve()),
        runner_cwd=str(runner_cwd_path.resolve()) if runner_cwd_path else None,
        runner_command=command,
        primary_tool=primary,
        tool_categories=category_mapping,
        fusion_summary=fusion_summary,
    )


def _native_smartbugs_execute(plan: ExecutionResult) -> ExecutionResult:
    target_path = Path(plan.target_path or "")
    results_root = Path(plan.results_root or "")
    smartbugs_dir = DEFAULT_SMARTBUGS_DIR
    if smartbugs_dir is None:
        return plan.model_copy(
            update={"status": "failed", "stderr_tail": "No SmartBugs directory configured. Set TOOLRANK_SMARTBUGS_DIR."}
        )
    results_root.mkdir(parents=True, exist_ok=True)
    native_commands: List[str] = []
    combined_stdout: List[str] = []
    combined_stderr: List[str] = []
    failures: List[str] = []

    for tool_id in ([plan.primary_tool] if plan.primary_tool else []) + [
        tool for tool in plan.tool_categories.keys() if tool != plan.primary_tool
    ]:
        if not tool_id:
            continue
        out_dir = results_root / tool_id / target_path.name
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "./smartbugs",
            "-t",
            tool_id,
            "-f",
            str(target_path),
            "--timeout",
            "1200",
            "--continue-on-errors",
            "--results",
            str(out_dir),
            "--sarif",
            "--json",
        ]
        native_commands.append(" ".join(cmd))
        proc = subprocess.run(cmd, cwd=smartbugs_dir, capture_output=True, text=True)
        if proc.stdout:
            combined_stdout.append(f"[{tool_id}]\n{proc.stdout[-2000:]}")
        if proc.stderr:
            combined_stderr.append(f"[{tool_id}]\n{proc.stderr[-2000:]}")
        if proc.returncode != 0:
            failures.append(f"{tool_id}:{proc.returncode}")

    fusion_manifest = {
        "primary_tool": plan.primary_tool,
        "category_assignments": plan.tool_categories,
        "fusion_summary": plan.fusion_summary,
    }
    (results_root / "fusion_plan.json").write_text(json.dumps(fusion_manifest, indent=2), encoding="utf-8")

    status = "executed" if not failures else "failed"
    stderr_tail = "\n".join(combined_stderr) or None
    if failures:
        extra = f"native_smartbugs_failures={','.join(failures)}"
        stderr_tail = f"{stderr_tail}\n{extra}" if stderr_tail else extra

    # Harvest findings from results directory
    per_tool_findings = _harvest_findings(plan, results_root)

    return plan.model_copy(
        update={
            "status": status,
            "execution_mode": "native_smartbugs",
            "native_commands": native_commands,
            "stdout_tail": "\n".join(combined_stdout)[-4000:] or None,
            "stderr_tail": stderr_tail[-4000:] if stderr_tail else None,
            "return_code": 0 if not failures else 1,
            "per_tool_findings": per_tool_findings,
        }
    )


def execute_plan(plan: ExecutionResult) -> ExecutionResult:
    if not plan.runner_command:
        return plan.model_copy(update={"status": "failed", "stderr_tail": "missing runner command"})

    return_code, stdout_tail, stderr_tail = _stream_runner_output(
        plan.runner_command,
        cwd=plan.runner_cwd,
    )
    if return_code == 0:
        results_root = Path(plan.results_root) if plan.results_root else None
        per_tool_findings = _harvest_findings(plan, results_root) if results_root else {}
        return plan.model_copy(
            update={
                "status": "executed",
                "execution_mode": "manual_runner",
                "return_code": return_code,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "per_tool_findings": per_tool_findings,
            }
        )

    plan = plan.model_copy(
        update={
            "status": "failed",
            "execution_mode": "manual_runner",
            "return_code": return_code,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    )
    if stderr_tail and "Report not found" in stderr_tail:
        return _native_smartbugs_execute(plan)
    return plan


def _harvest_findings(
    plan: ExecutionResult,
    results_root: Optional[Path],
) -> Dict[str, list]:
    """Scan results directory and parse findings for selected tools only."""
    if results_root is None or not results_root.exists():
        return {}

    # Derive selected tool IDs from the execution plan
    selected: Set[str] = set()
    if plan.primary_tool:
        selected.add(plan.primary_tool)
    selected.update(plan.tool_categories.keys())

    if not selected:
        return {}

    try:
        return load_per_tool_findings(results_root, selected)
    except Exception as exc:
        logger.warning("Failed to harvest findings from %s: %s", results_root, exc)
        return {}
