from __future__ import annotations

import re
from typing import Optional, Tuple

VERSION_UNKNOWN_RANGE = "0.0.0-0.0.0"
DOMAIN_START = (0, 4, 0)
DOMAIN_END = (0, 8, 99)
_VERSION_RE = re.compile(r"0\.(\d+)(?:\.(\d+|x))?$")

VersionTuple = Tuple[int, int, int]
VersionRange = Tuple[VersionTuple, VersionTuple]


def _parse_version_token(token: str, *, upper: bool) -> Optional[VersionTuple]:
    text = token.strip().lower()
    if not text:
        return None
    match = _VERSION_RE.fullmatch(text)
    if not match:
        return None
    minor = int(match.group(1))
    patch_text = match.group(2)
    if patch_text in {None, "x"}:
        patch = 99 if upper else 0
    else:
        patch = int(patch_text)
    return 0, minor, patch


def _clamp_domain(version: VersionTuple) -> VersionTuple:
    if version < DOMAIN_START:
        return DOMAIN_START
    if version > DOMAIN_END:
        return DOMAIN_END
    return version


def normalize_solc_range(spec: Optional[str]) -> str:
    text = (spec or "").strip().lower()
    if not text or text in {"unknown", "n/a", "na", "none"}:
        return VERSION_UNKNOWN_RANGE
    if text == VERSION_UNKNOWN_RANGE:
        return VERSION_UNKNOWN_RANGE

    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    ranges: list[VersionRange] = []
    for piece in pieces:
        if "-" in piece:
            start_text, end_text = [part.strip() for part in piece.split("-", 1)]
            start = _parse_version_token(start_text, upper=False)
            end = _parse_version_token(end_text, upper=True)
            if start is None or end is None:
                continue
            ranges.append((_clamp_domain(start), _clamp_domain(end)))
            continue

        exact = _parse_version_token(piece, upper=False)
        if exact is None:
            continue
        if piece.endswith(".x") or re.fullmatch(r"0\.\d+", piece):
            end = _parse_version_token(piece, upper=True)
            if end is None:
                continue
            ranges.append((_clamp_domain(exact), _clamp_domain(end)))
        else:
            ranges.append((_clamp_domain(exact), _clamp_domain(exact)))

    if not ranges:
        return VERSION_UNKNOWN_RANGE

    start = min(start for start, _ in ranges)
    end = max(end for _, end in ranges)
    if end < start:
        start, end = end, start
    return f"{start[0]}.{start[1]}.{start[2]}-{end[0]}.{end[1]}.{end[2]}"


def parse_solc_range(spec: Optional[str]) -> Optional[VersionRange]:
    normalized = normalize_solc_range(spec)
    if normalized == VERSION_UNKNOWN_RANGE:
        return None
    start_text, end_text = normalized.split("-", 1)
    start = _parse_version_token(start_text, upper=False)
    end = _parse_version_token(end_text, upper=False)
    if start is None or end is None:
        return None
    return start, end


def version_in_range(version: Optional[str], spec: Optional[str]) -> bool:
    if not version:
        return True
    version_exact = _parse_version_token(version.strip().lower(), upper=False)
    if version_exact is None:
        normalized_version = normalize_solc_range(version)
        parsed_version = parse_solc_range(normalized_version)
        parsed_spec = parse_solc_range(spec)
        if parsed_version is None or parsed_spec is None:
            return False
        return parsed_spec[0] <= parsed_version[0] and parsed_spec[1] >= parsed_version[1]
    parsed_spec = parse_solc_range(spec)
    if parsed_spec is None:
        return False
    start, end = parsed_spec
    return start <= version_exact <= end


def coverage_width_score(spec: Optional[str]) -> float:
    parsed = parse_solc_range(spec)
    if parsed is None:
        return 0.0
    start, end = parsed
    domain_start = DOMAIN_START[1] * 100 + DOMAIN_START[2]
    domain_end = DOMAIN_END[1] * 100 + DOMAIN_END[2]
    start_scalar = start[1] * 100 + start[2]
    end_scalar = end[1] * 100 + end[2]
    width = max(end_scalar - start_scalar + 1, 0)
    domain_width = domain_end - domain_start + 1
    return max(0.0, min(width / domain_width, 1.0))
