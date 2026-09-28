"""Check primitives: one result per claim, with the evidence that decided it."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
BLOCKED = "BLOCKED"

VALID_STATUSES: tuple[str, ...] = (PASS, FAIL, NOT_RUN, BLOCKED)
VALID_CATEGORIES: tuple[str, ...] = (
    "manuscript",
    "execution",
    "claim",
    "provenance",
    "hygiene",
    "gate",
)


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check, as it will be written into the report."""

    name: str
    category: str
    status: str
    expected: Any = None
    observed: Any = None
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")

    def as_record(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "category": self.category,
            "status": self.status,
            "expected": self.expected,
            "observed": self.observed,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


Check = Callable[[], CheckResult]


def compare(
    name: str,
    expected: object,
    observed: object,
    *,
    category: str = "execution",
    detail: str = "",
    evidence: Mapping[str, Any] | None = None,
    tolerance: float | None = None,
) -> CheckResult:
    """PASS when a numeric observation is inside tolerance of its expectation."""
    if tolerance is not None and isinstance(expected, float) and isinstance(observed, float):
        passed = abs(observed - expected) <= tolerance
    else:
        passed = expected == observed
    return CheckResult(
        name=name,
        category=category,
        status=PASS if passed else FAIL,
        expected=expected,
        observed=observed,
        detail=detail if detail else ("values agree" if passed else "values disagree"),
        evidence=dict(evidence or {}),
    )


def within(
    name: str,
    expected: float,
    observed: float,
    tolerance: float,
    *,
    category: str = "execution",
    detail: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> CheckResult:
    return compare(
        name,
        round(float(expected), 6),
        round(float(observed), 6),
        category=category,
        detail=detail,
        evidence={"tolerance": tolerance, **dict(evidence or {})},
        tolerance=float(tolerance),
    )


def truthy(
    name: str,
    condition: bool,
    *,
    category: str = "execution",
    expected: object = True,
    observed: object = None,
    detail: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(
        name=name,
        category=category,
        status=PASS if condition else FAIL,
        expected=expected,
        observed=bool(condition) if observed is None else observed,
        detail=detail,
        evidence=dict(evidence or {}),
    )


def not_run(
    name: str,
    reason: str,
    *,
    category: str = "execution",
    evidence: Mapping[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(
        name=name,
        category=category,
        status=NOT_RUN,
        expected="a measured value",
        observed=None,
        detail=reason,
        evidence=dict(evidence or {}),
    )


def blocked(
    name: str,
    reason: str,
    *,
    category: str = "execution",
    evidence: Mapping[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(
        name=name,
        category=category,
        status=BLOCKED,
        expected="an external resource",
        observed=None,
        detail=reason,
        evidence=dict(evidence or {}),
    )


def guard(
    name: str, body: Callable[[], CheckResult], *, category: str = "execution"
) -> CheckResult:
    """Run a check and turn an unexpected exception into an honest failure."""
    try:
        return body()
    except Exception as error:  # noqa: BLE001 - a check must never take the run down
        return CheckResult(
            name=name,
            category=category,
            status=FAIL,
            expected="the check to complete",
            observed=f"{type(error).__name__}: {error}",
            detail="the check raised before it could decide",
            evidence={},
        )


def summarise(results: Sequence[CheckResult]) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(VALID_STATUSES, 0)
    for result in results:
        counts[result.status] += 1
    counts["total"] = len(results)
    return counts


def family(results: Sequence[CheckResult], category: str) -> list[CheckResult]:
    return [result for result in results if result.category == category]


def failures(results: Sequence[CheckResult]) -> list[CheckResult]:
    return [result for result in results if result.status == FAIL]


def overall_status(results: Sequence[CheckResult]) -> str:
    """PARTIALLY_VERIFIED when anything is outstanding, VERIFIED only when nothing is."""
    counts = summarise(results)
    if counts[FAIL] or counts[NOT_RUN] or counts[BLOCKED]:
        return "PARTIALLY_VERIFIED"
    return "VERIFIED"


def code_status(results: Sequence[CheckResult]) -> str:
    """The verdict over the release's own code, excluding the manuscript's arithmetic."""
    code = [result for result in results if result.category != "manuscript"]
    if not code:
        return "UNVERIFIED"
    counts = summarise(code)
    if counts[FAIL]:
        return "PARTIALLY_VERIFIED"
    if counts[NOT_RUN] or counts[BLOCKED]:
        return "PARTIALLY_VERIFIED"
    return "VERIFIED"


def manuscript_discrepancies(results: Sequence[CheckResult]) -> list[dict[str, Any]]:
    return [result.as_record() for result in family(results, "manuscript") if result.status == FAIL]
