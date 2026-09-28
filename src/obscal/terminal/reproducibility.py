"""Multi-batch, multi-run and multi-operator reproducibility by channel.

Ref: Sec. Terminal-assay error, attenuation and the variance decomposition;
Table 3; SI Table S18.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

CV_EXCELLENT_MAX = 5.0
CV_ACCEPTABLE_MAX = 20.0
ICC_EXCELLENT_MIN = 0.8
ICC_ACCEPTABLE_MIN = 0.2


def cv_percent(values: NDArray[np.float64]) -> float:
    """Coefficient of variation of a group-level quantity, in percent."""
    if values.size < 2:
        raise ValueError("a coefficient of variation needs at least two groups")
    mean = float(np.mean(values))
    if mean == 0.0:
        raise ValueError("the group mean must be non-zero")
    return 100.0 * float(np.std(values, ddof=1)) / mean


def pooled_within_cv(groups: Sequence[NDArray[np.float64]]) -> float:
    """Pooled within-group coefficient of variation, as a within-run figure.

    Each group is one run; the pooled spread is the within-run scatter and the
    denominator is the grand mean, so the figure is not a within-assay replicate.
    """
    usable = [np.asarray(group, dtype=np.float64) for group in groups if group.size >= 2]
    if not usable:
        raise ValueError("a pooled within-group CV needs at least one group of two readings")
    numerator = 0.0
    denominator_df = 0
    total = 0.0
    count = 0
    for group in usable:
        numerator += float(np.sum((group - float(np.mean(group))) ** 2))
        denominator_df += group.size - 1
        total += float(np.sum(group))
        count += group.size
    if denominator_df == 0:
        raise ValueError("no residual degrees of freedom in the pooled spread")
    grand_mean = total / count
    if grand_mean == 0.0:
        raise ValueError("the grand mean must be non-zero")
    pooled = float(np.sqrt(numerator / denominator_df))
    return 100.0 * pooled / grand_mean


def icc_two_way_random_absolute(matrix: NDArray[np.float64]) -> float:
    """Two-way random-effects absolute-agreement ICC for a complete subjects-by-raters table.

    Ref: Table 3 footnote. The estimator is the McGraw-Wong absolute-agreement
    form, whose two-rater case the agreement module also computes by its own
    route.
    """
    if matrix.ndim != 2:
        raise ValueError("the ICC table must be two-dimensional")
    n, k = matrix.shape
    if n < 2 or k < 2:
        raise ValueError("a two-way ICC needs at least two subjects and two raters")
    grand = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    msr = float(k) * float(np.sum((row_means - grand) ** 2)) / (n - 1)
    msc = float(n) * float(np.sum((column_means - grand) ** 2)) / (k - 1)
    residual = matrix - row_means.reshape(-1, 1) - column_means.reshape(1, -1) + grand
    mse = float(np.sum(residual**2)) / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    if denominator == 0.0:
        return 1.0 if msr == mse else float("nan")
    agreement: float = (msr - mse) / denominator
    return agreement


def icc_from_records(
    subjects: Sequence[str], raters: Sequence[str], values: Sequence[float]
) -> float:
    """Build the subjects-by-raters table from long records and score it."""
    subject_index = {name: index for index, name in enumerate(dict.fromkeys(subjects))}
    rater_index = {name: index for index, name in enumerate(dict.fromkeys(raters))}
    table = np.full((len(subject_index), len(rater_index)), np.nan, dtype=np.float64)
    for subject, rater, value in zip(subjects, raters, values, strict=True):
        table[subject_index[subject], rater_index[rater]] = float(value)
    if not np.all(np.isfinite(table)):
        raise ValueError("the ICC table must be complete; a missing rater is not a zero")
    return icc_two_way_random_absolute(table)


def grade_cv(between_run_cv: float) -> str:
    """Grade on the between-run coefficient, the platform's outermost axis."""
    if between_run_cv <= CV_EXCELLENT_MAX:
        return "excellent"
    if between_run_cv < CV_ACCEPTABLE_MAX:
        return "acceptable"
    return "poor"


def grade_icc(icc: float) -> str:
    if icc >= ICC_EXCELLENT_MIN:
        return "excellent"
    if icc >= ICC_ACCEPTABLE_MIN:
        return "acceptable"
    return "poor"


@dataclass(frozen=True)
class ChannelReproducibility:
    """One channel's three axis-specific coefficients plus its reliability."""

    channel: str
    within_run_cv: float
    between_run_cv: float
    between_operator_cv: float
    icc_within_run: float
    icc_between_runs: float
    grade: str

    def as_record(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "within_run_cv_pct": self.within_run_cv,
            "between_run_cv_pct": self.between_run_cv,
            "between_operator_cv_pct": self.between_operator_cv,
            "icc_within_run": self.icc_within_run,
            "icc_between_runs": self.icc_between_runs,
            "grade_between_runs": self.grade,
        }


def channel_reproducibility(
    channel: str,
    within_run_groups: Sequence[NDArray[np.float64]],
    between_run_means: NDArray[np.float64],
    between_operator_means: NDArray[np.float64],
    icc_within_run: float,
    icc_between_runs: float,
) -> ChannelReproducibility:
    between_run = cv_percent(np.asarray(between_run_means, dtype=np.float64))
    return ChannelReproducibility(
        channel=channel,
        within_run_cv=pooled_within_cv(within_run_groups),
        between_run_cv=between_run,
        between_operator_cv=cv_percent(np.asarray(between_operator_means, dtype=np.float64)),
        icc_within_run=icc_within_run,
        icc_between_runs=icc_between_runs,
        grade=grade_cv(between_run),
    )


def cv_spread_ratio(values: Mapping[str, float]) -> dict[str, Any]:
    """Best-to-worst spread of a family of coefficients, printed as a ratio."""
    finite = {name: value for name, value in values.items() if np.isfinite(value) and value > 0.0}
    if len(finite) < 2:
        raise ValueError("a spread ratio needs at least two positive coefficients")
    best = min(finite, key=lambda name: finite[name])
    worst = max(finite, key=lambda name: finite[name])
    report: dict[str, Any] = {
        "best_channel": str(best),
        "best_value": finite[best],
        "worst_channel": str(worst),
        "worst_value": finite[worst],
        "ratio": finite[worst] / finite[best],
        "n_channels": float(len(finite)),
    }
    return report


def channels_grading_better_than(
    records: Sequence[ChannelReproducibility], grade: str
) -> list[str]:
    order = {"poor": 0, "acceptable": 1, "excellent": 2}
    if grade not in order:
        raise ValueError(f"unknown grade {grade!r}")
    threshold = order[grade]
    return [record.channel for record in records if order[record.grade] >= threshold]


def channels_excellent_on_both_axes(
    records: Sequence[ChannelReproducibility],
    cv_threshold: float = CV_EXCELLENT_MAX,
    icc_threshold: float = ICC_EXCELLENT_MIN,
) -> list[str]:
    """The channels that reach the top grade on the coefficient and the reliability alike."""
    excellent: list[str] = [
        record.channel
        for record in records
        if record.between_run_cv <= cv_threshold and record.icc_between_runs >= icc_threshold
    ]
    return excellent


def terminal_endpoints(records: Sequence[ChannelReproducibility]) -> list[ChannelReproducibility]:
    """Destructive endpoints are the ones whose channel name carries a terminal tag."""
    return [record for record in records if record.channel.startswith("terminal/")]


def single_half_counterpart(halves_averaged_cv: float, ratio: float = 0.885) -> float:
    """The single-half coefficient, necessarily the larger of the two constructs."""
    return float(halves_averaged_cv) / float(ratio)


def run_batch_coefficient_identity(between_run_cv: float, between_batch_cv: float) -> bool:
    """On this design every run sits inside the batch layout, so the two coincide."""
    return bool(abs(between_run_cv - between_batch_cv) < 1e-9)


def half_mean_basis(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Collapse paired reads to the half-mean basis every contrast is read on."""
    if values.size % 2 != 0:
        raise ValueError("a half-mean basis needs an even number of reads")
    collapsed = values.reshape(-1, 2).mean(axis=1)
    return np.asarray(collapsed, dtype=np.float64)


def variance_split(within_sd: float, between_sd: float) -> dict[str, float]:
    """Within-chip against between-chip split of an endpoint's variance."""
    within_variance = float(within_sd) ** 2
    between_variance = float(between_sd) ** 2
    total = within_variance + between_variance
    if total <= 0.0:
        raise ValueError("a variance split needs a positive total")
    return {
        "within_chip_sd": float(within_sd),
        "between_chip_sd": float(between_sd),
        "within_share": within_variance / total,
        "between_share": between_variance / total,
        "total_sd": float(np.sqrt(total)),
    }
