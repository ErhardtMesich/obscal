"""Contrasts between arms, with batch as a crossed factor.

Ref: Sec. Evaluation metrics and statistical analysis. The manuscript names mixed
models with batch as a crossed factor; the reduction here is a batch-adjusted
fixed-effects linear contrast, which is an engineering choice because the run log
that would carry the random-effect structure is not deposited.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class Contrast:
    """One arm-against-arm contrast with the test that was actually run."""

    name: str
    test: str
    estimate: float
    standard_error: float
    statistic: float
    statistic_name: str
    degrees_of_freedom: float
    p_value: float
    ci_low: float
    ci_high: float
    n_left: int
    n_right: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_record(self) -> dict[str, float | int | str | bool]:
        return {
            "contrast": self.name,
            "test": self.test,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "statistic": self.statistic,
            "statistic_name": self.statistic_name,
            "degrees_of_freedom": self.degrees_of_freedom,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_left": self.n_left,
            "n_right": self.n_right,
            "significant_at_0p05": self.significant,
        }


def welch_t_test(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    level: float = 0.95,
    name: str | None = None,
) -> Contrast:
    """Welch's unequal-variance t test, the two-group contrast the manuscript uses."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("a Welch contrast needs at least two units per arm")
    mean_difference = float(np.mean(a) - np.mean(b))
    variance_a = float(np.var(a, ddof=1)) / a.size
    variance_b = float(np.var(b, ddof=1)) / b.size
    combined = variance_a + variance_b
    if combined <= 0.0:
        standard_error = 0.0
        statistic = float("nan")
        dof = float("nan")
        p_value = float("nan")
    else:
        standard_error = float(np.sqrt(combined))
        statistic = mean_difference / standard_error
        dof = combined**2 / (variance_a**2 / (a.size - 1) + variance_b**2 / (b.size - 1))
        p_value = float(2.0 * stats.t.sf(abs(statistic), dof))
    critical = float(stats.t.ppf(0.5 + level / 2.0, dof)) if np.isfinite(dof) else float("nan")
    return Contrast(
        name=name or "welch",
        test="welch_t",
        estimate=mean_difference,
        standard_error=standard_error,
        statistic=statistic,
        statistic_name="t",
        degrees_of_freedom=float(dof),
        p_value=p_value,
        ci_low=mean_difference - critical * standard_error,
        ci_high=mean_difference + critical * standard_error,
        n_left=int(a.size),
        n_right=int(b.size),
    )


def batch_adjusted_contrast(
    values: NDArray[np.float64],
    group: Sequence[str],
    batch: Sequence[str],
    *,
    treated: str,
    control: str,
    name: str | None = None,
) -> Contrast:
    """Arm effect after removing the batch means, with batch crossed and not confounded.

    The pool carries a balanced fraction of every state-by-timepoint cell in each
    batch, so batch is crossed with state and time rather than confounded with
    either, which is what makes an additive batch adjustment adequate here.
    """
    y = np.asarray(values, dtype=np.float64)
    groups = np.asarray(list(group), dtype=object)
    batches = np.asarray(list(batch), dtype=object)
    if not (y.size == groups.size == batches.size):
        raise ValueError("values, groups and batches must align")
    keep = np.isin(groups, [treated, control])
    y = y[keep]
    groups = groups[keep]
    batches = batches[keep]
    if np.unique(groups).size != 2:
        raise ValueError("both arms of the contrast must be present")
    treated_indicator = (groups == treated).astype(np.float64)
    batch_levels = list(dict.fromkeys(batches.tolist()))
    columns = [np.ones(y.size, dtype=np.float64), treated_indicator]
    for level in batch_levels[1:]:
        columns.append((batches == level).astype(np.float64))
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("the batch-adjusted design is rank deficient")
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    dof = y.size - design.shape[1]
    if dof <= 0:
        raise ValueError("no residual degrees of freedom for the contrast")
    sigma_squared = float(residual @ residual) / dof
    precision = np.linalg.inv(design.T @ design)
    estimate = float(coefficients[1])
    standard_error = float(np.sqrt(sigma_squared * precision[1, 1]))
    statistic = estimate / standard_error if standard_error > 0.0 else float("nan")
    p_value = (
        float(2.0 * stats.t.sf(abs(statistic), dof)) if np.isfinite(statistic) else float("nan")
    )
    critical = float(stats.t.ppf(0.975, dof))
    return Contrast(
        name=name or f"{treated}_vs_{control}_batch_adjusted",
        test="batch_adjusted_linear_contrast",
        estimate=estimate,
        standard_error=standard_error,
        statistic=statistic,
        statistic_name="t",
        degrees_of_freedom=float(dof),
        p_value=p_value,
        ci_low=estimate - critical * standard_error,
        ci_high=estimate + critical * standard_error,
        n_left=int(np.sum(groups == treated)),
        n_right=int(np.sum(groups == control)),
    )


def relative_reduction(treated_mean: float, control_mean: float) -> dict[str, float]:
    """Reduction of an endpoint in the treated arm, as a fraction and a percentage."""
    if control_mean == 0.0:
        raise ValueError("the control mean must be non-zero")
    reduction = (control_mean - treated_mean) / control_mean
    return {
        "treated_mean": float(treated_mean),
        "control_mean": float(control_mean),
        "absolute_reduction": float(control_mean) - float(treated_mean),
        "relative_reduction": reduction,
        "relative_reduction_pct": 100.0 * reduction,
    }


def arm_against_untreated(
    treated: NDArray[np.float64], untreated: NDArray[np.float64], arm: str
) -> dict[str, float | str]:
    """The reduction a treated arm achieves against the untreated injury arm."""
    summary = relative_reduction(float(np.mean(treated)), float(np.mean(untreated)))
    record: dict[str, float | str] = {"arm": arm}
    record.update(summary)
    return record


def dispersion(values: NDArray[np.float64]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise ValueError("a dispersion needs at least two units")
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
        "n": float(array.size),
        "sem": float(np.std(array, ddof=1) / np.sqrt(array.size)),
    }


def mann_whitney(
    left: NDArray[np.float64], right: NDArray[np.float64], name: str = "mann_whitney"
) -> Contrast:
    """Rank-sum contrast, used by the perfused-against-static platform validation."""
    result = stats.mannwhitneyu(left, right, alternative="two-sided")
    return Contrast(
        name=name,
        test="mann_whitney_u",
        estimate=float(np.median(left) - np.median(right)),
        standard_error=float("nan"),
        statistic=float(result.statistic),
        statistic_name="u",
        degrees_of_freedom=float("nan"),
        p_value=float(result.pvalue),
        ci_low=float("nan"),
        ci_high=float("nan"),
        n_left=int(np.asarray(left).size),
        n_right=int(np.asarray(right).size),
    )


def pooling_is_licensed(bias: float, bias_p: float, threshold: float = 0.05) -> dict[str, object]:
    """Whether replicate differences can be pooled: they must be centred on zero."""
    return {
        "bias": bias,
        "p_value": bias_p,
        "exchangeable": bool(bias_p > threshold),
        "verdict": (
            "halves_or_operators_are_exchangeable"
            if bias_p > threshold
            else "systematic_offset_detected"
        ),
    }
