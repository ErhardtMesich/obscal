"""Multiplicity control over a test family.

Ref: Sec. Evaluation metrics and statistical analysis; SI Table S3 footnote.
P values are unadjusted in the tables and any count across a test family carries
its Holm-corrected count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HolmResult:
    """Holm step-down over one family, with the raw and adjusted values paired."""

    raw: NDArray[np.float64]
    adjusted: NDArray[np.float64]
    rejected: NDArray[np.bool_]
    level: float
    stopped_at: int | None

    @property
    def n_rejected(self) -> int:
        return int(np.sum(self.rejected))

    @property
    def n_tests(self) -> int:
        return int(self.raw.size)

    def as_record(self) -> dict[str, object]:
        return {
            "level": self.level,
            "n_tests": self.n_tests,
            "n_rejected": self.n_rejected,
            "n_retained": self.n_tests - self.n_rejected,
            "stopped_at_rank": self.stopped_at,
            "raw": self.raw.tolist(),
            "adjusted": self.adjusted.tolist(),
            "rejected": self.rejected.tolist(),
        }


def holm_step_down(p_values: NDArray[np.float64], level: float = 0.05) -> HolmResult:
    """Holm-Bonferroni step-down, carrying the running maximum of the scaled values.

    The adjusted value at rank i is the maximum of the scaled values at ranks up
    to and including i, capped at one; taking a minimum instead reports every
    adjusted value at the family Bonferroni level.
    """
    raw = np.asarray(p_values, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("a multiplicity family must be a non-empty vector")
    if np.any(raw < 0.0) or np.any(raw > 1.0):
        raise ValueError("p values must lie in [0, 1]")
    n = raw.size
    order = np.argsort(raw, kind="stable")
    sorted_raw = raw[order]
    scaled = sorted_raw * (n - np.arange(n))
    running = np.maximum.accumulate(scaled)
    adjusted_sorted = np.minimum(running, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    rejected_sorted = running <= level
    if not np.all(rejected_sorted):
        stop = int(np.argmin(rejected_sorted))
        rejected_sorted = np.zeros(n, dtype=np.bool_)
        rejected_sorted[:stop] = True
        stopped_at: int | None = stop
    else:
        rejected_sorted = np.ones(n, dtype=np.bool_)
        stopped_at = None
    rejected = np.empty(n, dtype=np.bool_)
    rejected[order] = rejected_sorted
    return HolmResult(
        raw=raw,
        adjusted=adjusted,
        rejected=rejected,
        level=float(level),
        stopped_at=stopped_at,
    )


def bonferroni(p_values: NDArray[np.float64], level: float = 0.05) -> dict[str, object]:
    """The family-wise level Holm's first rank matches, for contrast."""
    raw = np.asarray(p_values, dtype=np.float64)
    if raw.size == 0:
        raise ValueError("a multiplicity family must be non-empty")
    return {
        "level": float(level),
        "family_threshold": float(level) / float(raw.size),
        "n_rejected": int(np.sum(raw <= float(level) / float(raw.size))),
        "n_tests": int(raw.size),
    }


def family_from_names(
    names: list[str], p_values: NDArray[np.float64], level: float = 0.05
) -> dict[str, object]:
    """Name the members of a family and their fate, so a count is checkable."""
    if len(names) != int(np.asarray(p_values).size):
        raise ValueError("every family member needs one p value")
    result = holm_step_down(p_values, level)
    return {
        "members": names,
        "result": result.as_record(),
        "retained": [name for name, keep in zip(names, result.rejected, strict=True) if keep],
        "dropped": [name for name, keep in zip(names, result.rejected, strict=True) if not keep],
    }


def unadjusted_clearance(p_values: NDArray[np.float64], level: float = 0.05) -> dict[str, object]:
    raw = np.asarray(p_values, dtype=np.float64)
    return {
        "n_clearing_zero": int(np.sum(raw < level)),
        "n_tests": int(raw.size),
        "level": float(level),
    }
