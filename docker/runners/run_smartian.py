#!/usr/bin/env python3
"""
Run Smartian from Solidity source and export unified result.json format.

Output layout:
  <results_root>/smartian/<relative_contract_path>/
    result.json
    testcase/
    bug/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_SMARTIAN_DLL = Path(os.getenv("TOOLRANK_SMARTIAN_DLL", "/work/docker/vendor/smartian/build/Smartian.dll"))
SOLC_ARTIFACTS_DIR = Path.home() / ".solc-select" / "artifacts"
SOLCX_DIR = Path.home() / ".solcx"


def _die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _stream_process(cmd: List[str], cwd: Optional[Path] = None) -> int:
    if cwd is not None:
        print(f"[run] (cwd={cwd}) {' '.join(cmd)}")
    else:
        print(f"[run] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()


def _iter_contracts(path: Path) -> List[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".sol" else []
    if path.is_dir():
        return sorted([p for p in path.rglob("*.sol") if p.is_file()])
    return []


def _strip_comments(text: str) -> str:
    out: List[str] = []
    i = 0
    n = len(text)
    in_block = False
    while i < n:
        if in_block:
            if i + 1 < n and text[i] == "*" and text[i + 1] == "/":
                in_block = False
                i += 2
            else:
                i += 1
            continue
        if i + 1 < n and text[i] == "/" and text[i + 1] == "*":
            in_block = True
            i += 2
            continue
        if i + 1 < n and text[i] == "/" and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _collect_pragma_specs(sol_file: Path) -> List[str]:
    try:
        text = sol_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    text = _strip_comments(text)
    return [m.group(1).strip() for m in re.finditer(r"pragma\s+solidity\s+([^;]+);", text, re.IGNORECASE)]


def _version_from_token(token: str) -> Optional[str]:
    token = token.strip()
    m = re.search(r"([0-9]+\.[0-9]+(?:\.[0-9]+)?)", token)
    if not m:
        return None
    v = m.group(1)
    if v.count(".") == 1:
        v += ".0"
    return v


def _parse_version_tuple(raw: str) -> Tuple[int, int, int]:
    parts = raw.split(".")
    nums = [int(p) for p in parts if p.isdigit() or p.isnumeric()]
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def _parse_constraints_from_spec(spec: str) -> List[Tuple[str, Tuple[int, int, int]]]:
    s = spec.strip()
    constraints: List[Tuple[str, Tuple[int, int, int]]] = []
    if re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", s):
        v = _version_from_token(s)
        if v:
            constraints.append(("==", _parse_version_tuple(v)))
        return constraints

    m = re.search(r"\^\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", s)
    if m:
        v = _version_from_token(m.group(1))
        if v:
            lo = _parse_version_tuple(v)
            hi = (lo[0] + 1, 0, 0) if lo[0] > 0 else (0, lo[1] + 1, 0)
            constraints.append((">=", lo))
            constraints.append(("<", hi))
        return constraints

    m = re.search(r"~\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", s)
    if m:
        v = _version_from_token(m.group(1))
        if v:
            lo = _parse_version_tuple(v)
            hi = (lo[0], lo[1] + 1, 0)
            constraints.append((">=", lo))
            constraints.append(("<", hi))
        return constraints

    for op, raw_v in re.findall(r"(>=|<=|>|<|=)\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", s):
        v = _version_from_token(raw_v)
        if not v:
            continue
        parsed = _parse_version_tuple(v)
        constraints.append(("==", parsed) if op == "=" else (op, parsed))

    if constraints:
        return constraints

    v = _version_from_token(s)
    if v:
        constraints.append(("==", _parse_version_tuple(v)))
    return constraints


def _satisfy(version: Tuple[int, int, int], constraints: List[Tuple[str, Tuple[int, int, int]]]) -> bool:
    for op, rhs in constraints:
        if op == "==" and version != rhs:
            return False
        if op == ">=" and version < rhs:
            return False
        if op == "<=" and version > rhs:
            return False
        if op == ">" and version <= rhs:
            return False
        if op == "<" and version >= rhs:
            return False
    return True


def _discover_local_solc_bins() -> Dict[str, str]:
    bins: Dict[str, str] = {}
    if SOLC_ARTIFACTS_DIR.exists():
        for p in SOLC_ARTIFACTS_DIR.glob("solc-*"):
            if not p.is_dir():
                continue
            version = p.name.replace("solc-", "", 1)
            bin_path = p / f"solc-{version}"
            if bin_path.exists():
                bins.setdefault(version, str(bin_path))

    if SOLCX_DIR.exists():
        for p in SOLCX_DIR.glob("solc-v*"):
            if not p.is_file():
                continue
            m = re.fullmatch(r"solc-v([0-9]+\.[0-9]+\.[0-9]+)", p.name)
            if not m:
                continue
            bins.setdefault(m.group(1), str(p))
    return bins


def _pick_local_solc(sol_file: Path, local_bins: Dict[str, str], fallback_solc: str) -> Tuple[Optional[str], str]:
    specs = _collect_pragma_specs(sol_file)
    if not specs:
        return None, fallback_solc
    constraints: List[Tuple[str, Tuple[int, int, int]]] = []
    for s in specs:
        constraints.extend(_parse_constraints_from_spec(s))
    if not constraints:
        return None, fallback_solc

    candidates: List[Tuple[Tuple[int, int, int], str]] = [(_parse_version_tuple(ver), ver) for ver in local_bins]
    candidates.sort(reverse=True)
    for parsed, raw in candidates:
        if _satisfy(parsed, constraints):
            return raw, local_bins[raw]
    return None, fallback_solc


def _ensure_executable(cmd: str, flag: str = "--version") -> None:
    try:
        proc = subprocess.run([cmd, flag], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        _die(f"command not found: {cmd}", 1)
    if proc.returncode != 0:
        _die(f"failed command: {cmd} {flag}\n{proc.stdout}", 1)


def _compile_source(
    sol_file: Path,
    contract_root: Path,
    local_solc_bins: Dict[str, str],
    fallback_solc: str,
    bytecode_kind: str,
) -> Tuple[bool, Dict[str, object], Optional[str], str, str]:
    selected_solc_ver, solc_bin = _pick_local_solc(sol_file, local_solc_bins, fallback_solc)
    allow_paths = f".,{contract_root},{sol_file.parent}"
    proc = subprocess.run(
        [
            solc_bin,
            "--optimize",
            "--combined-json",
            "abi,bin,bin-runtime",
            "--allow-paths",
            allow_paths,
            str(sol_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return False, {}, selected_solc_ver, solc_bin, proc.stderr[-4000:]
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False, {}, selected_solc_ver, solc_bin, "invalid solc combined-json output"

    contracts = payload.get("contracts")
    if not isinstance(contracts, dict) or not contracts:
        return False, {}, selected_solc_ver, solc_bin, "no contracts in combined-json output"

    stem = sol_file.stem
    preferred = [k for k in contracts if str(k).endswith(f":{stem}")]
    ordered = preferred + [k for k in contracts if k not in preferred]
    for key in ordered:
        data = contracts.get(key)
        if not isinstance(data, dict):
            continue
        abi_raw = str(data.get("abi") or "")
        bin_raw = str(data.get("bin") or "")
        runtime_raw = str(data.get("bin-runtime") or "")
        target_bin = runtime_raw if bytecode_kind == "runtime" else bin_raw
        if not abi_raw or not target_bin:
            continue
        return True, {"contract_key": key, "abi": abi_raw, "bin": bin_raw, "bin-runtime": runtime_raw}, selected_solc_ver, solc_bin, ""

    return False, {}, selected_solc_ver, solc_bin, f"no contract with abi+{bytecode_kind} bytecode"


def _write_compile_inputs(tmp_dir: Path, compiled: Dict[str, object], bytecode_kind: str) -> Tuple[Path, Path]:
    abi_raw = str(compiled.get("abi") or "[]")
    bin_raw = str(compiled.get("bin") or "")
    runtime_raw = str(compiled.get("bin-runtime") or "")

    try:
        abi_obj = json.loads(abi_raw)
    except json.JSONDecodeError:
        abi_obj = []

    abi_path = tmp_dir / "contract.abi.json"
    bytecode_path = tmp_dir / ("contract.runtime.bin" if bytecode_kind == "runtime" else "contract.bin")
    abi_path.write_text(json.dumps(abi_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    bytecode_path.write_text(runtime_raw if bytecode_kind == "runtime" else bin_raw, encoding="utf-8")
    return abi_path, bytecode_path


SMARTIAN_TAG_NAME_MAP: Dict[str, str] = {
    "AF": "assertion-failure",
    "AW": "arbitrary-write",
    "BD": "blockstate-dependency",
    "CH": "control-hijack",
    "EL": "ether-leak",
    "IB": "integer-bug",
    "ME": "mishandled-exception",
    "MS": "multiple-send",
    "RE": "reentrancy",
    "SC": "suicidal-contract",
    "TO": "transaction-origin-use",
    "FE": "freezing-ether",
    "RV": "requirement-violation",
}


def _extract_bug_tags_from_filename(filename: str) -> List[str]:
    base = Path(filename).name
    m = re.match(r"id-\d+-(.+?)_\d+$", base)
    tag_part = m.group(1) if m else base
    tags: List[str] = []
    for raw in tag_part.split("-"):
        token = raw.strip()
        if not token:
            continue
        prefix = token.split("_", 1)[0].upper()
        tags.append(prefix)
    return sorted(set(tags))


def _normalize_smartian_report(
    contract_path: Path,
    out_dir: Path,
    rc: int,
    compile_err: str,
    selected_solc_ver: Optional[str],
    selected_solc_bin: str,
    contract_key: str,
    bytecode_kind: str,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "errors": [],
        "fails": [],
        "findings": [],
        "infos": [],
        "parser": {
            "tool": "smartian",
            "bytecode_kind": bytecode_kind,
            "contract_key": contract_key,
            "selected_solc": selected_solc_ver,
            "selected_solc_bin": selected_solc_bin,
        },
    }

    if compile_err:
        report["errors"].append("COMPILE_FAILED")
        report["fails"].append(compile_err)
        return report

    if rc != 0:
        report["errors"].append(f"EXIT_CODE_{rc}")

    bug_dir = out_dir / "bug"
    testcase_dir = out_dir / "testcase"
    if not bug_dir.exists():
        report["infos"].append("bug directory not found")
    if not testcase_dir.exists():
        report["infos"].append("testcase directory not found")

    findings: List[Dict[str, object]] = []
    if bug_dir.exists() and bug_dir.is_dir():
        for p in sorted([x for x in bug_dir.iterdir() if x.is_file()]):
            tags = _extract_bug_tags_from_filename(p.name)
            if not tags:
                tags = ["UNKNOWN"]
            for tag in tags:
                name = SMARTIAN_TAG_NAME_MAP.get(tag, f"smartian-{tag.lower()}")
                findings.append(
                    {
                        "name": name,
                        "message": f"smartian bug tag={tag} file={p.name}",
                        "filename": str(contract_path),
                    }
                )

    report["findings"] = findings
    return report


def _run_one_contract(
    contract_path: Path,
    contract_root: Path,
    out_dir: Path,
    smartian_dll: Path,
    dotnet_cmd: str,
    timeout: int,
    local_solc_bins: Dict[str, str],
    fallback_solc: str,
    bytecode_kind: str,
) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, compiled, selected_solc_ver, selected_solc_bin, compile_err = _compile_source(
        sol_file=contract_path,
        contract_root=contract_root,
        local_solc_bins=local_solc_bins,
        fallback_solc=fallback_solc,
        bytecode_kind=bytecode_kind,
    )

    rc = 1
    contract_key = ""
    if ok:
        contract_key = str(compiled.get("contract_key") or "")
        with tempfile.TemporaryDirectory(prefix="smartian_in_") as td:
            tmp_dir = Path(td)
            abi_path, bytecode_path = _write_compile_inputs(tmp_dir, compiled, bytecode_kind)
            cmd = [
                dotnet_cmd,
                str(smartian_dll),
                "fuzz",
                "-p",
                str(bytecode_path),
                "-a",
                str(abi_path),
                "-t",
                str(timeout),
                "-o",
                str(out_dir),
            ]
            rc = _stream_process(cmd, cwd=smartian_dll.parent.parent)
    else:
        print(f"[warn] compile failed: {compile_err}", file=sys.stderr)

    report = _normalize_smartian_report(
        contract_path=contract_path,
        out_dir=out_dir,
        rc=rc,
        compile_err=compile_err if not ok else "",
        selected_solc_ver=selected_solc_ver,
        selected_solc_bin=selected_solc_bin,
        contract_key=contract_key,
        bytecode_kind=bytecode_kind,
    )
    (out_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[smartian] findings={len(report.get('findings') or [])} "
        f"errors={len(report.get('errors') or [])} out={out_dir / 'result.json'}"
    )
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Smartian on Solidity source and output unified result.json")
    parser.add_argument("contract_or_dir", help=".sol file or directory")
    parser.add_argument("results_root", help="results root path")
    parser.add_argument("--smartian_dll", default=str(DEFAULT_SMARTIAN_DLL), help="path to Smartian.dll")
    parser.add_argument("--dotnet", default="dotnet", help="dotnet command path (default: dotnet)")
    parser.add_argument("--solc", default="solc", help="fallback solc binary path/name (default: solc)")
    parser.add_argument("--timeout", type=int, default=1200, help="fuzz timeout seconds (default: 1200)")
    parser.add_argument(
        "--bytecode_kind",
        choices=["bin", "runtime"],
        default="bin",
        help="Smartian -p input kind: deployment bin or runtime bin (default: bin)",
    )
    args = parser.parse_args()

    contract_or_dir = Path(args.contract_or_dir).resolve()
    results_root = Path(args.results_root).resolve()
    smartian_dll = Path(args.smartian_dll).resolve()

    if not contract_or_dir.exists():
        _die(f"contract path not found: {contract_or_dir}", 1)
    if not smartian_dll.exists():
        _die(f"Smartian.dll not found: {smartian_dll}", 1)
    if args.timeout <= 0:
        _die("--timeout must be positive", 2)

    _ensure_executable(args.dotnet, "--version")
    _ensure_executable(args.solc, "--version")

    contracts = _iter_contracts(contract_or_dir)
    if not contracts:
        _die(f"no .sol files found under: {contract_or_dir}", 1)

    try:
        results_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _die(f"results root not writable: {results_root} ({e})", 1)

    if contract_or_dir.is_file():
        contract_root = contract_or_dir.parent
    else:
        contract_root = contract_or_dir

    local_solc_bins = _discover_local_solc_bins()
    print(f"[start] contracts={len(contracts)}")
    print(f"[start] smartian_dll={smartian_dll}")
    print(f"[start] local_solc_versions={len(local_solc_bins)}")

    exit_code = 0
    for idx, contract_path in enumerate(contracts, start=1):
        print(f"\n[batch] {idx}/{len(contracts)} {contract_path}")
        try:
            rel = contract_path.relative_to(contract_root)
        except ValueError:
            rel = Path(contract_path.name)
        out_dir = results_root / "smartian" / rel
        rc = _run_one_contract(
            contract_path=contract_path,
            contract_root=contract_root,
            out_dir=out_dir,
            smartian_dll=smartian_dll,
            dotnet_cmd=args.dotnet,
            timeout=args.timeout,
            local_solc_bins=local_solc_bins,
            fallback_solc=args.solc,
            bytecode_kind=args.bytecode_kind,
        )
        if rc != 0:
            exit_code = rc

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
