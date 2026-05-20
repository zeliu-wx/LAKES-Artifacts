"""Shared assignment-evidence rules for tool category ownership."""

from __future__ import annotations


MIN_ASSIGNMENT_TOTAL = 10
MIN_ASSIGNMENT_RATE = 0.3
LOCAL_CLOSE_RATE_MARGIN = 0.05


def is_strong_eligible(
    detected: int | None,
    total: int | None,
    rate: float | None,
) -> bool:
    return (
        detected is not None
        and total is not None
        and total >= MIN_ASSIGNMENT_TOTAL
        and rate is not None
        and rate >= MIN_ASSIGNMENT_RATE
    )


is_assignment_eligible = is_strong_eligible


def is_weak_eligible(
    detected: int | None,
    total: int | None,
    rate: float | None,
    feasible: bool,
) -> bool:
    _ = (detected, total)
    return feasible and rate is not None and rate > 0.0


def count_text(detected: int | None, total: int | None) -> str:
    if detected is None or total is None:
        return "unknown"
    return f"{detected}/{total}"


def rate_text(rate: float | None) -> str:
    if rate is None:
        return "None"
    return f"{rate:.4f}"


def is_close_local_margin(top_rate: float | None, next_rate: float | None) -> bool:
    if top_rate is None or next_rate is None:
        return False
    return abs(top_rate - next_rate) <= LOCAL_CLOSE_RATE_MARGIN
