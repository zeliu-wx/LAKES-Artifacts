"""Packaged analyzer runner for ToolRank execution plans.

The runner follows the command contract produced by ``toolrank.execution``:

    python -m toolrank.runner TARGET RESULTS --tools slither,mythril \
        --primary_tool slither --tool_categories mythril:ARITHMETIC

It executes the selected tools, stores per-tool JSON reports, and enriches
findings with canonical DASP categories so the ToolRank fusion layer can apply
category ownership.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from toolrank.fusion import compact_fused_report_payload, fuse_reports
from toolrank.report_parser import load_per_tool_findings_from_run_dir
from toolrank.runner_adapter import (
    _enrich_report_with_categories,
    _load_mapping,
    _map_tool_name,
    _parse_tools,
)
from toolrank.schemas import CompositionPlan, FusedReport

SECURIFY2_RUNNER = Path(
    os.getenv("TOOLRANK_SECURIFY2_RUNNER", "/Users/liuze/Downloads/securify2-master/securifyjson.py")
)
SECURIFY2_PLATFORM = os.getenv("TOOLRANK_SECURIFY2_PLATFORM", "linux/amd64")
SECURIFY2_DEFAULT_SOLC = os.getenv("TOOLRANK_SECURIFY2_DEFAULT_SOLC", "0.5.12")
SECURIFY2_IMAGE_TEMPLATE = os.getenv("TOOLRANK_SECURIFY2_IMAGE_TEMPLATE", "securify:{version}")
GPTSCAN_ROOT = Path(os.getenv("TOOLRANK_GPTSCAN_ROOT", "/Users/liuze/Downloads/GPTScan-main"))
GPTSCAN_PY = GPTSCAN_ROOT / ".venv" / "bin" / "python"
GPTSCAN_MAIN = GPTSCAN_ROOT / "src" / "main.py"
GPTSCAN_DEFAULT_API_BASE = os.getenv("TOOLRANK_GPTSCAN_DEFAULT_API_BASE", "https://api.whatai.cc/v1")
GPTSCAN_DEFAULT_MODEL_GPT4 = os.getenv("TOOLRANK_GPTSCAN_DEFAULT_MODEL_GPT4", "gpt-5.4")
SAILFISH_RUNNER = Path(os.getenv("TOOLRANK_SAILFISH_RUNNER", "/Users/liuze/Downloads/QuantifyX/run_sailfish.py"))
SAILFISH_DEFAULT_SOLC = os.getenv("TOOLRANK_SAILFISH_DEFAULT_SOLC", "0.4.25")
SMARTIAN_RUNNER = Path(os.getenv("TOOLRANK_SMARTIAN_RUNNER", "/Users/liuze/Downloads/QuantifyX/run_smartian.py"))

_PRAGMA_SOLIDITY_RE = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)
_SECURIFY2_IMAGE_CACHE: set[str] = set()
_CATEGORY_ALIASES = {
    "DENIAL_OF_SERVICE": "DENIAL_SERVICE",
    "UNCHECKED_LOW_LEVEL_CALLS": "UNCHECKED_LOW_CALLS",
    "UNCHECKED_LL_CALLS": "UNCHECKED_LOW_CALLS",
}


def _die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _stream_process(
    command: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    display_command: Optional[list[str]] = None,
) -> int:
    logged_command = display_command or command
    if cwd is not None:
        print(f"[run] (cwd={cwd}) {shlex.join(logged_command)}")
    else:
        print(f"[run] {shlex.join(logged_command)}")
    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()


def _normalize_category_label(label: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", label.strip().upper())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return _CATEGORY_ALIASES.get(normalized, normalized)


def _parse_tool_category_filters(raw: str) -> dict[str, list[str] | None]:
    if not raw:
        return {}
    parsed: dict[str, list[str] | None] = {}
    for part in raw.replace("\uFF0C", ",").split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            tool_part, categories_part = item.split(":", 1)
        else:
            tool_part, categories_part = item, "ALL"
        tool_key = _map_tool_name(tool_part)
        if not tool_key:
            continue
        if not categories_part.strip() or categories_part.strip().upper() == "ALL":
            parsed[tool_key] = None
            continue
        categories = [
            _normalize_category_label(token).lower()
            for token in re.split(r"[|;/]", categories_part)
            if token.strip()
        ]
        parsed[tool_key] = sorted(dict.fromkeys(categories)) or None
    return parsed


def _iter_contracts(target_path: Path) -> list[Path]:
    if target_path.is_file():
        return [target_path] if target_path.suffix.lower() == ".sol" else []
    if target_path.is_dir():
        return sorted(path for path in target_path.rglob("*.sol") if path.is_file())
    return []


def _relative_contract_path(contract_path: Path, target_root: Path) -> Path:
    if target_root.is_file():
        return Path(".")
    try:
        return contract_path.relative_to(target_root)
    except ValueError:
        return Path(contract_path.name)


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _find_result_json(out_dir: Path) -> Optional[Path]:
    if out_dir.is_file() and out_dir.name == "result.json":
        return out_dir
    if not out_dir.is_dir():
        return None
    direct = out_dir / "result.json"
    if direct.is_file():
        return direct
    for root, _, files in os.walk(out_dir):
        if "result.json" in files:
            return Path(root) / "result.json"
    return None


def _load_report(report_path: Path) -> Optional[dict]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_enriched_report(report_path: Path, tool_id: str, mapping: dict[tuple[str, str], str]) -> bool:
    report = _load_report(report_path)
    if report is None:
        return False
    enriched = _enrich_report_with_categories(report, tool_id, mapping)
    report_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def _candidate_known_reports_dirs(target_path: Path) -> list[Path]:
    candidates: list[Path] = []
    if os.getenv("TOOLRANK_KNOWN_REPORTS_DIR"):
        candidates.append(Path(os.environ["TOOLRANK_KNOWN_REPORTS_DIR"]))
    for base in [Path.cwd(), target_path if target_path.is_dir() else target_path.parent, Path(__file__).resolve()]:
        for parent in [base, *base.parents]:
            candidates.append(parent / "smartbugsout")
    return _dedupe_paths([path.expanduser() for path in candidates])


def _resolve_known_reports_dir(target_path: Path, known_reports_dir: str | Path | None) -> Optional[Path]:
    if known_reports_dir is not None:
        raw = str(known_reports_dir).strip()
        if not raw:
            return None
        path = Path(raw).expanduser().resolve()
        return path if path.is_dir() else None
    for candidate in _candidate_known_reports_dirs(target_path):
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _known_report_candidates(
    known_reports_root: Path,
    tool_id: str,
    contract_path: Path,
    target_root: Path,
) -> list[Path]:
    tool_dir = known_reports_root / tool_id
    raw_tool_dir = known_reports_root / "_raw" / tool_id
    rel = _relative_contract_path(contract_path, target_root)

    direct_candidates = [
        tool_dir / f"{contract_path.stem}.json",
        tool_dir / contract_path.name / "result.json",
    ]
    if rel != Path("."):
        direct_candidates.extend(
            [
                tool_dir / rel.with_suffix(".json"),
                tool_dir / rel / "result.json",
            ]
        )
    glob_candidates: list[Path] = []
    if tool_dir.is_dir():
        glob_candidates.extend(sorted(tool_dir.rglob(f"{contract_path.name}/result.json")))
    if raw_tool_dir.is_dir():
        glob_candidates.extend(sorted(raw_tool_dir.glob(f"*/{contract_path.name}/result.json")))
        if rel != Path("."):
            glob_candidates.extend(sorted(raw_tool_dir.glob(f"*/{rel.as_posix()}/result.json")))

    return _dedupe_paths([path for path in [*direct_candidates, *glob_candidates] if path.is_file()])


def _copy_known_report(report_path: Path, out_dir: Path) -> Optional[Path]:
    if report_path.name == "result.json":
        source_dir = report_path.parent
        if source_dir.resolve() != out_dir.resolve():
            if out_dir.exists():
                shutil.rmtree(out_dir)
            shutil.copytree(source_dir, out_dir)
        return out_dir / "result.json"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = out_dir / "result.json"
    shutil.copy2(report_path, copied)
    return copied


def _reuse_known_report(
    *,
    known_reports_root: Optional[Path],
    tool_id: str,
    contract_path: Path,
    target_root: Path,
    out_dir: Path,
    mapping: dict[tuple[str, str], str],
) -> bool:
    if known_reports_root is None:
        return False
    for candidate in _known_report_candidates(known_reports_root, tool_id, contract_path, target_root):
        copied = _copy_known_report(candidate, out_dir)
        if copied and _write_enriched_report(copied, tool_id, mapping):
            print(f"[reuse] known report found: {candidate}")
            return True
        print(f"[warn] known report unusable tool={tool_id} path={candidate}", file=sys.stderr)
    return False


def _collect_pragma_specs(contract_path: Path) -> list[str]:
    try:
        text = contract_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    stripped = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    stripped = re.sub(r"//.*", "", stripped)
    return [match.group(1).strip() for match in _PRAGMA_SOLIDITY_RE.finditer(stripped)]


def _lower_bound_from_spec(raw: str) -> Optional[str]:
    for pattern in (r"[\^~]\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", r">=\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", r"([0-9]+\.[0-9]+(?:\.[0-9]+)?)"):
        match = re.search(pattern, raw)
        if match:
            version = match.group(1)
            return f"{version}.0" if version.count(".") == 1 else version
    return None


def _parse_version_tuple(raw: str) -> tuple[int, int, int]:
    parts = [int(part) for part in raw.split(".") if part.isdigit()]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _max_lower_bound_version(contract_path: Path) -> Optional[tuple[int, int, int]]:
    versions = [
        _parse_version_tuple(version)
        for spec in _collect_pragma_specs(contract_path)
        if (version := _lower_bound_from_spec(spec))
    ]
    return max(versions) if versions else None


def _use_get_src_for_slither(contract_path: Path) -> bool:
    version = _max_lower_bound_version(contract_path)
    return bool(version and version >= (0, 8, 0))


def _run_slither_via_get_src(contract_path: Path, out_dir: Path) -> int:
    try:
        import get_src as get_src_module
    except Exception as exc:
        print(f"[warn] get_src import failed: {exc}", file=sys.stderr)
        return 1
    try:
        raw = get_src_module.get_src(str(contract_path))
    except Exception as exc:
        print(f"[warn] get_src failed: {exc}", file=sys.stderr)
        return 1
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"[warn] get_src output is not JSON: {exc}", file=sys.stderr)
            return 1
    elif isinstance(raw, dict):
        payload = raw
    else:
        print("[warn] get_src returned unexpected type", file=sys.stderr)
        return 1
    if payload.get("error"):
        print(f"[warn] get_src error: {payload.get('error')}", file=sys.stderr)
        return 1

    findings: list[dict] = []
    for item in payload.get("slither") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("ruleId") or item.get("id") or "slither_finding"
        finding: dict[str, object] = {"name": str(name)}
        message = item.get("message")
        if isinstance(message, dict) and message.get("text"):
            finding["message"] = str(message["text"])
        locations = item.get("locations") or []
        if locations and isinstance(locations[0], dict):
            phys = locations[0].get("physicalLocation") or {}
            region = phys.get("region") or {}
            artifact = phys.get("artifactLocation") or {}
            if region.get("startLine") is not None:
                finding["line"] = int(region["startLine"])
                finding["line_end"] = int(region.get("endLine") or region["startLine"])
            if artifact.get("uri"):
                finding["filename"] = str(artifact["uri"])
        findings.append(finding)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps({"errors": [], "fails": [], "findings": findings, "infos": [], "parser": {"tool": "get_src"}}, indent=2),
        encoding="utf-8",
    )
    return 0


def _solc_lower_bound(contract_path: Path) -> Optional[str]:
    version = _max_lower_bound_version(contract_path)
    return ".".join(str(part) for part in version) if version else None


def _run_solc_select(contract_path: Path, tool_label: str) -> None:
    version = _solc_lower_bound(contract_path)
    if not version:
        print(f"[warn] pragma lower-bound not found; keep current solc for {tool_label}", file=sys.stderr)
        return
    try:
        rc = _stream_process(["solc-select", "use", version])
    except FileNotFoundError:
        print("[warn] solc-select not found in PATH", file=sys.stderr)
        return
    if rc != 0:
        print(f"[warn] solc-select use {version} failed (rc={rc})", file=sys.stderr)


def _run_gptscan(
    contract_path: Path,
    out_dir: Path,
    api_key: str,
    gptscan_timeout: int,
    openai_api_base: str,
) -> int:
    if not GPTSCAN_MAIN.exists():
        print(f"[warn] GPTScan main.py not found: {GPTSCAN_MAIN}", file=sys.stderr)
        return 1
    _run_solc_select(contract_path, "gptscan")
    scan_source = contract_path
    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    if contract_path.is_file():
        temp_dir = tempfile.TemporaryDirectory(prefix="gptscan_src_")
        scan_source = Path(temp_dir.name)
        shutil.copy2(contract_path, scan_source / contract_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpt_out = out_dir / "gptscan_output.json"
    key = api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    command = [str(GPTSCAN_PY if GPTSCAN_PY.exists() else Path(sys.executable)), str(GPTSCAN_MAIN), "-s", str(scan_source), "-o", str(gpt_out), "-k", key]
    env = os.environ.copy()
    env["OPENAI_API_BASE"] = openai_api_base.strip() or env.get("OPENAI_API_BASE", "") or env.get("OPENAI_BASE_URL", "") or GPTSCAN_DEFAULT_API_BASE
    env["OPENAI_BASE_URL"] = env["OPENAI_API_BASE"]
    env["GPTSCAN_USE_GPT4"] = env.get("GPTSCAN_USE_GPT4", "1")
    env["GPTSCAN_MODEL_GPT4"] = env.get("GPTSCAN_MODEL_GPT4") or env.get("GPTSCAN_MODEL") or GPTSCAN_DEFAULT_MODEL_GPT4
    env["GPTSCAN_TIMEOUT_SECONDS"] = str(gptscan_timeout)
    masked = command[:]
    if "-k" in masked:
        key_index = masked.index("-k") + 1
        if key_index < len(masked):
            masked[key_index] = "***"
    try:
        rc = _stream_process(command, cwd=GPTSCAN_MAIN.parent, env=env, display_command=masked)
        if rc != 0:
            return rc
        report = _normalize_gptscan_report(_load_report(gpt_out) or {})
        (out_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _normalize_gptscan_report(payload: dict) -> dict:
    report = {"errors": [], "fails": [], "findings": [], "infos": [], "parser": {"tool": "gptscan"}}
    if payload.get("success") is False:
        report["errors"].append("GPTSCAN_FAILED")
        if payload.get("message"):
            report["fails"].append(str(payload["message"]))
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("code") or item.get("title") or "gptscan_issue"
        message = item.get("description") or ""
        recommendation = item.get("recommendation") or ""
        if recommendation:
            message = f"{message}\nRecommendation: {recommendation}" if message else f"Recommendation: {recommendation}"
        files = item.get("affectedFiles") or []
        if not files:
            report["findings"].append({"name": str(name), "message": message})
            continue
        for file_item in files:
            if not isinstance(file_item, dict):
                continue
            finding: dict[str, object] = {"name": str(name)}
            if message:
                finding["message"] = message
            if file_item.get("filePath"):
                finding["filename"] = str(file_item["filePath"])
            range_item = file_item.get("range") or {}
            start = (range_item.get("start") or {}).get("line")
            end = (range_item.get("end") or {}).get("line")
            if start is not None:
                finding["line"] = int(start)
                finding["line_end"] = int(end) if end is not None else int(start)
            report["findings"].append(finding)
    return report


def _run_securify2(contract_path: Path, out_dir: Path) -> int:
    if not SECURIFY2_RUNNER.exists():
        print(f"[warn] securify2 runner not found: {SECURIFY2_RUNNER}", file=sys.stderr)
        return 1
    version = _solc_lower_bound(contract_path) or SECURIFY2_DEFAULT_SOLC
    image = SECURIFY2_IMAGE_TEMPLATE.format(version=version)
    if image not in _SECURIFY2_IMAGE_CACHE:
        dockerfile = SECURIFY2_RUNNER.parent / "Dockerfile"
        if dockerfile.exists():
            rc = _stream_process(["docker", "build", "--platform", SECURIFY2_PLATFORM, "--build-arg", f"SOLC={version}", "-t", image, "."], cwd=SECURIFY2_RUNNER.parent)
            if rc != 0:
                return rc
        _SECURIFY2_IMAGE_CACHE.add(image)
    out_dir.mkdir(parents=True, exist_ok=True)
    return _stream_process(
        [
            "python3",
            str(SECURIFY2_RUNNER),
            str(contract_path),
            "-o",
            str(out_dir / "result.json"),
            "--image",
            image,
            "--platform",
            SECURIFY2_PLATFORM,
            "--sudo",
            "--debug-cmd",
        ],
        cwd=SECURIFY2_RUNNER.parent,
    )


def _run_sailfish(contract_path: Path, out_dir: Path, timeout: int) -> int:
    if not SAILFISH_RUNNER.exists():
        print(f"[warn] sailfish runner not found: {SAILFISH_RUNNER}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    solc_ver = _solc_lower_bound(contract_path) or SAILFISH_DEFAULT_SOLC
    return _stream_process(
        [
            sys.executable,
            str(SAILFISH_RUNNER),
            str(contract_path.parent),
            contract_path.name,
            solc_ver,
            "--artifacts_root",
            str(out_dir),
            "--timeout",
            str(timeout),
        ],
    )


def _run_smartian(contract_path: Path, out_dir: Path, timeout: int) -> int:
    if not SMARTIAN_RUNNER.exists():
        print(f"[warn] smartian runner not found: {SMARTIAN_RUNNER}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smartian_") as tmp:
        rc = _stream_process(
            [
                sys.executable,
                str(SMARTIAN_RUNNER),
                str(contract_path),
                tmp,
                "--timeout",
                str(timeout),
            ],
        )
        produced = next(Path(tmp).rglob("result.json"), None)
        if produced is not None:
            shutil.copy2(produced, out_dir / "result.json")
        else:
            print("[warn] smartian produced no result.json", file=sys.stderr)
        return rc


def _compile_runtime_hex_for_vandal(contract_path: Path) -> Optional[str]:
    _run_solc_select(contract_path, "vandal")
    try:
        proc = subprocess.run(
            ["solc", "--combined-json", "bin-runtime", str(contract_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print("[warn] solc not found in PATH for vandal runtime compile", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"[warn] solc compile failed for vandal: {proc.stderr.strip()}", file=sys.stderr)
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    contracts = payload.get("contracts")
    if not isinstance(contracts, dict):
        return None
    preferred = [key for key in contracts if str(key).endswith(f":{contract_path.stem}")]
    for key in preferred + [key for key in contracts if key not in preferred]:
        runtime_hex = str((contracts.get(key) or {}).get("bin-runtime") or "").strip()
        if runtime_hex:
            return runtime_hex
    return None


def _prepare_runtime_hex_file_for_vandal(contract_path: Path) -> tuple[Optional[Path], Optional[tempfile.TemporaryDirectory[str]]]:
    runtime_hex = _compile_runtime_hex_for_vandal(contract_path)
    if not runtime_hex:
        return None, None
    temp_dir = tempfile.TemporaryDirectory(prefix="vandal_rt_")
    path = Path(temp_dir.name) / f"{contract_path.stem}.rt.hex"
    path.write_text(runtime_hex, encoding="utf-8")
    return path, temp_dir


def _candidate_smartbugs_dirs(target_path: Path) -> list[Path]:
    candidates: list[Path] = []
    if os.getenv("TOOLRANK_SMARTBUGS_DIR"):
        candidates.append(Path(os.environ["TOOLRANK_SMARTBUGS_DIR"]))
    for base in [Path.cwd(), target_path if target_path.is_dir() else target_path.parent, Path(__file__).resolve()]:
        for parent in [base, *base.parents]:
            candidates.append(parent / "smartbugs")
    deduped: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in deduped:
            deduped.append(resolved)
    return deduped


def _resolve_smartbugs_dir(target_path: Path) -> Path:
    for candidate in _candidate_smartbugs_dirs(target_path):
        executable = candidate / "smartbugs"
        if candidate.is_dir() and executable.exists() and os.access(executable, os.X_OK):
            return candidate
    checked = ", ".join(str(path) for path in _candidate_smartbugs_dirs(target_path)[:5])
    raise RuntimeError(f"SmartBugs executable not found. Set TOOLRANK_SMARTBUGS_DIR. Checked: {checked}")


def _run_smartbugs_tool(
    tool_id: str,
    contract_path: Path,
    out_dir: Path,
    *,
    target_root: Path,
    smartbugs_dir: Optional[Path],
    timeout: int,
    gptscan_timeout: int,
    openai_api_key: str,
    openai_api_base: str,
) -> int:
    if tool_id == "securify2":
        return _run_securify2(contract_path, out_dir)
    if tool_id == "gptscan":
        return _run_gptscan(contract_path, out_dir, openai_api_key, gptscan_timeout, openai_api_base)
    if tool_id == "sailfish":
        return _run_sailfish(contract_path, out_dir, timeout)
    if tool_id == "smartian":
        return _run_smartian(contract_path, out_dir, timeout)
    if tool_id == "slither" and _use_get_src_for_slither(contract_path):
        print("[run] slither via get_src.py (pragma >= 0.8.x)")
        return _run_slither_via_get_src(contract_path, out_dir)

    temp_runtime_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    target_input = contract_path
    runtime_mode = False
    if tool_id == "vandal":
        target_input, temp_runtime_dir = _prepare_runtime_hex_file_for_vandal(contract_path)
        if target_input is None:
            return 1
        runtime_mode = True
        print(f"[vandal] runtime_input={target_input}")

    try:
        resolved_smartbugs_dir = smartbugs_dir or _resolve_smartbugs_dir(target_root)
        command = [
            "./smartbugs",
            "-t",
            tool_id,
            "-f",
            str(target_input),
            "--timeout",
            str(timeout),
            "--continue-on-errors",
            "--results",
            str(out_dir),
            "--sarif",
            "--json",
        ]
        if runtime_mode:
            command.append("--runtime")
        return _stream_process(command, cwd=resolved_smartbugs_dir)
    finally:
        if temp_runtime_dir is not None:
            temp_runtime_dir.cleanup()


def _run_one_tool_for_contract(
    tool_name: str,
    contract_path: Path,
    *,
    target_root: Path,
    results_root: Path,
    mapping: dict[tuple[str, str], str],
    known_reports_root: Optional[Path],
    smartbugs_dir: Optional[Path],
    timeout: int,
    gptscan_timeout: int,
    openai_api_key: str,
    openai_api_base: str,
) -> tuple[str, int]:
    rel = _relative_contract_path(contract_path, target_root)
    tool_id = _map_tool_name(tool_name)
    out_dir = results_root / tool_id / rel
    result_path = _find_result_json(out_dir)
    if result_path and _write_enriched_report(result_path, tool_id, mapping):
        print(f"[reuse] report found: {result_path}")
        return tool_id, 0

    if _reuse_known_report(
        known_reports_root=known_reports_root,
        tool_id=tool_id,
        contract_path=contract_path,
        target_root=target_root,
        out_dir=out_dir,
        mapping=mapping,
    ):
        return tool_id, 0

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = _run_smartbugs_tool(
        tool_id,
        contract_path,
        out_dir,
        target_root=target_root,
        smartbugs_dir=smartbugs_dir,
        timeout=timeout,
        gptscan_timeout=gptscan_timeout,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
    )
    if rc != 0:
        print(f"[warn] tool run failed tool={tool_id} exit_code={rc}", file=sys.stderr)
        return tool_id, rc
    result_path = _find_result_json(out_dir)
    if result_path is None or not _write_enriched_report(result_path, tool_id, mapping):
        print(f"[warn] result.json missing or invalid for tool={tool_id}", file=sys.stderr)
        return tool_id, 1
    return tool_id, 0


def _run_one_contract(
    contract_path: Path,
    *,
    target_root: Path,
    results_root: Path,
    selected_tools: list[str],
    mapping: dict[tuple[str, str], str],
    known_reports_root: Optional[Path],
    smartbugs_dir: Optional[Path],
    timeout: int,
    gptscan_timeout: int,
    openai_api_key: str,
    openai_api_base: str,
    jobs: int,
) -> int:
    exit_code = 0
    if jobs <= 1 or len(selected_tools) <= 1:
        for tool_name in selected_tools:
            _tool_id, rc = _run_one_tool_for_contract(
                tool_name,
                contract_path,
                target_root=target_root,
                results_root=results_root,
                mapping=mapping,
                known_reports_root=known_reports_root,
                smartbugs_dir=smartbugs_dir,
                timeout=timeout,
                gptscan_timeout=gptscan_timeout,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
            )
            if rc != 0:
                exit_code = rc
        return exit_code

    max_workers = min(jobs, len(selected_tools))
    print(f"[parallel] jobs={max_workers} tools={','.join(selected_tools)}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_one_tool_for_contract,
                tool_name,
                contract_path,
                target_root=target_root,
                results_root=results_root,
                mapping=mapping,
                known_reports_root=known_reports_root,
                smartbugs_dir=smartbugs_dir,
                timeout=timeout,
                gptscan_timeout=gptscan_timeout,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
            ): tool_name
            for tool_name in selected_tools
        }
        for future in as_completed(futures):
            tool_name = futures[future]
            try:
                _tool_id, rc = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard around external tools.
                print(f"[warn] tool run crashed tool={tool_name} error={exc}", file=sys.stderr)
                exit_code = 1
                continue
            if rc != 0:
                exit_code = rc
    return exit_code


def _write_fusion_plan(
    results_root: Path,
    *,
    selected_tools: list[str],
    primary_tool: str,
    tool_categories: dict[str, list[str] | None],
) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_tools": selected_tools,
        "primary_tool": primary_tool,
        "tool_categories": tool_categories,
    }
    (results_root / "fusion_plan.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _lakes_output_dir(results_root: Path) -> Path:
    return results_root if results_root.name == "LAKES_out" else results_root / "LAKES_out"


def _contract_output_dir(results_root: Path, target_path: Path) -> Path:
    contract_id = target_path.stem if target_path.suffix.lower() == ".sol" else target_path.name
    return _lakes_output_dir(results_root) / contract_id


def _tool_results_root(results_root: Path, target_path: Path, *, write_lakes_output: bool) -> Path:
    if not write_lakes_output:
        return results_root
    return _contract_output_dir(results_root, target_path) / "raw"


def _composition_from_runner_inputs(
    *,
    selected_tools: list[str],
    primary_tool: str,
    tool_categories: dict[str, list[str] | None],
) -> CompositionPlan:
    category_assignments: dict[str, str] = {}
    for tool_id, categories in tool_categories.items():
        if tool_id == primary_tool or categories is None:
            continue
        for category in categories:
            category_assignments[category] = tool_id
    return CompositionPlan(
        selected_tool_ids=selected_tools,
        anchor_tool_id=primary_tool,
        complementary_tool_ids=[tool for tool in selected_tools if tool != primary_tool],
        category_assignments=category_assignments,
    )


def _raw_findings_to_finding(tool_id: str, raw_findings: list[dict]) -> list:
    from toolrank.engine import _normalize_raw_findings

    return _normalize_raw_findings(tool_id, raw_findings)


def _build_fused_report_from_run(
    *,
    results_root: Path,
    composition: CompositionPlan,
) -> FusedReport:
    raw_by_tool = load_per_tool_findings_from_run_dir(results_root, set(composition.selected_tool_ids))
    anchor_findings = _raw_findings_to_finding(
        composition.anchor_tool_id,
        raw_by_tool.get(composition.anchor_tool_id, []),
    )
    complement_findings = {
        tool_id: _raw_findings_to_finding(tool_id, raw_findings)
        for tool_id, raw_findings in raw_by_tool.items()
        if tool_id != composition.anchor_tool_id
    }
    return fuse_reports(
        anchor_findings,
        complement_findings,
        composition,
        findings_source="execution",
    )


def _write_lakes_outputs(
    results_root: Path,
    *,
    target_path: Path,
    composition: CompositionPlan,
    fused_report: FusedReport,
) -> Path:
    lakes_dir = _contract_output_dir(results_root, target_path)
    lakes_dir.mkdir(parents=True, exist_ok=True)
    (lakes_dir / "fused_report.json").write_text(
        json.dumps(compact_fused_report_payload(fused_report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (lakes_dir / "fusion_plan.json").write_text(
        json.dumps(composition.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return lakes_dir


def run_targets(
    target_path: str | Path,
    results_root: str | Path,
    selected_tools: Sequence[str],
    *,
    primary_tool: str = "",
    tool_categories: str = "",
    smartbugs_dir: str | Path | None = None,
    timeout: int = 1200,
    gptscan_timeout: int = 600,
    openai_api_key: str = "",
    openai_api_base: str = "",
    jobs: int = 0,
    write_lakes_output: bool = True,
    known_reports_dir: str | Path | None = None,
) -> int:
    target = Path(target_path).resolve()
    root = Path(results_root).resolve()
    if not target.exists():
        _die(f"Target path not found: {target}", 1)
    if timeout <= 0:
        _die("--timeout must be a positive integer", 2)
    if gptscan_timeout <= 0:
        _die("--gptscan_timeout must be a positive integer", 2)
    if jobs < 0:
        _die("--jobs must be non-negative", 2)
    tools = [tool for tool in selected_tools if str(tool).strip()]
    if not tools:
        _die("No tools selected. Use --tool or --tools.", 1)

    contracts = _iter_contracts(target)
    if not contracts:
        _die(f"No .sol files found: {target}", 1)

    mapped_primary = _map_tool_name(primary_tool) if primary_tool else _map_tool_name(tools[0])
    mapped_tools = [_map_tool_name(tool) for tool in tools]
    effective_jobs = len(mapped_tools) if jobs == 0 else jobs
    parsed_tool_categories = _parse_tool_category_filters(tool_categories)
    composition = _composition_from_runner_inputs(
        selected_tools=mapped_tools,
        primary_tool=mapped_primary,
        tool_categories=parsed_tool_categories,
    )
    if write_lakes_output:
        _write_fusion_plan(_contract_output_dir(root, target), selected_tools=mapped_tools, primary_tool=mapped_primary, tool_categories=parsed_tool_categories)

    mapping = _load_mapping()
    resolved_smartbugs_dir = Path(smartbugs_dir).resolve() if smartbugs_dir else None
    known_reports_root = _resolve_known_reports_dir(target, known_reports_dir)
    if known_reports_root is not None:
        print(f"[known_reports] {known_reports_root}")
    tool_results_root = _tool_results_root(root, target, write_lakes_output=write_lakes_output)
    exit_code = 0
    for index, contract in enumerate(contracts, start=1):
        print(f"\n[batch] {index}/{len(contracts)} {contract}")
        rc = _run_one_contract(
            contract,
            target_root=target,
            results_root=tool_results_root,
            selected_tools=mapped_tools,
            mapping=mapping,
            known_reports_root=known_reports_root,
            smartbugs_dir=resolved_smartbugs_dir,
            timeout=timeout,
            gptscan_timeout=gptscan_timeout,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            jobs=effective_jobs,
        )
        if rc != 0:
            exit_code = rc
    if write_lakes_output:
        fused_report = _build_fused_report_from_run(results_root=tool_results_root, composition=composition)
        lakes_dir = _write_lakes_outputs(root, target_path=target, composition=composition, fused_report=fused_report)
        print(f"[lakes_out] {lakes_dir / 'fused_report.json'}")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ToolRank-selected SmartBugs-style analyzers.")
    parser.add_argument("contract_or_dir", help="Path to a .sol file or directory.")
    parser.add_argument("results_root", nargs="?", default="LAKES_out", help="Output root for analyzer reports.")
    parser.add_argument("--tool", default="", help="Single tool name to run.")
    parser.add_argument("--tools", default="", help="Comma-separated tool names to run.")
    parser.add_argument("--primary_tool", default="", help="Primary tool whose categories are kept by default.")
    parser.add_argument("--tool_categories", default="", help="Per-tool category ownership, e.g. slither:REENTRANCY|TIME_MANIPULATION.")
    parser.add_argument("--smartbugs-dir", default="", help="SmartBugs checkout directory. Defaults to TOOLRANK_SMARTBUGS_DIR or auto-discovery.")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-tool timeout in seconds.")
    parser.add_argument("--gptscan_timeout", type=int, default=600, help="GPTScan LLM timeout in seconds.")
    parser.add_argument("--openai_api_key", default="", help="OpenAI API key for GPTScan.")
    parser.add_argument("--openai_api_base", default="", help="OpenAI-compatible API base for GPTScan.")
    parser.add_argument("--known-reports-dir", default=None, help="Reuse existing SmartBugs reports from this directory before running tools.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Max tools to run in parallel per contract; 0 means one job per selected tool.",
    )
    parser.add_argument(
        "--no-lakes-output",
        action="store_true",
        help="Run tools only. Used by ToolRank when the engine writes the final LAKES_out files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tools = _parse_tools(args.tools)
    if not tools and args.tool:
        tools = [args.tool]
    return run_targets(
        args.contract_or_dir,
        args.results_root,
        tools,
        primary_tool=args.primary_tool,
        tool_categories=args.tool_categories,
        smartbugs_dir=args.smartbugs_dir or None,
        timeout=args.timeout,
        gptscan_timeout=args.gptscan_timeout,
        openai_api_key=args.openai_api_key,
        openai_api_base=args.openai_api_base,
        jobs=args.jobs,
        write_lakes_output=not args.no_lakes_output,
        known_reports_dir=args.known_reports_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
