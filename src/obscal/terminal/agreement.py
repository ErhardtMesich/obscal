"""Agreement between the non-destructive estimate and the destructive standard.

Ref: Sec. Agreement against the gold-standard assay; Table 4; SI Table S5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from obscal.runtime.seed import numpy_generator

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Agreement:
    """One agreement row: agreement coefficient, bias and the two limits."""

    n: int
    icc: float
    icc_low: float
    icc_high: float
    bias: float
    loa_low: float
    loa_high: float
    proportional_bias_slope: float
    proportional_bias_p: float
    rmse_percent: float
    reference_mean: float

    @property
    def limits_of_agreement_width(self) -> float:
        return self.loa_high - self.loa_low

    def as_record(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "icc": self.icc,
            "icc_ci_low": self.icc_low,
            "icc_ci_high": self.icc_high,
            "bias": self.bias,
            "loa_low": self.loa_low,
            "loa_high": self.loa_high,
            "proportional_bias_slope": self.proportional_bias_slope,
            "proportional_bias_p": self.proportional_bias_p,
            "rmse_percent": self.rmse_percent,
            "reference_mean": self.reference_mean,
        }


def icc_absolute_agreement(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Two-way random-effects absolute-agreement ICC for two readings per unit.

    Ref: Table 3 footnote, "2-way random-effects absolute-agreement estimates".
    """
    if first.shape != second.shape:
        raise ValueError("the two readings must align")
    n = first.size
    if n < 2:
        raise ValueError("an ICC needs at least two units")
    matrix = np.column_stack([first, second])
    grand = float(np.mean(matrix))
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    k = 2.0
    msr = k * float(np.sum((row_means - grand) ** 2)) / (n - 1)
    msc = n * float(np.sum((column_means - grand) ** 2)) / (k - 1.0)
    residual = matrix - row_means.reshape(-1, 1) - column_means.reshape(1, -1) + grand
    mse = float(np.sum(residual**2)) / ((n - 1) * (k - 1.0))
    denominator = msr + (k - 1.0) * mse + k * (msc - mse) / n
    if denominator == 0.0:
        return 1.0 if msr == mse else float("nan")
    return (msr - mse) / denominator


def icc_bootstrap_ci(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap interval on the absolute-agreement coefficient."""
    rng = numpy_generator(seed)
    n = first.size
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = rng.integers(0, n, size=n)
        values[index] = icc_absolute_agreement(first[draw], second[draw])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def bland_altman(first: NDArray[np.float64], second: NDArray[np.float64]) -> dict[str, float]:
    """Bias, limits of agreement and the proportional-bias regression."""
    if first.shape != second.shape:
        raise ValueError("the two readings must align")
    differences = first - second
    averages = 0.5 * (first + second)
    bias = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    slope, intercept, r_value, p_value, stderr = stats.linregress(averages, differences)
    return {
        "bias": bias,
        "difference_sd": sd,
        "loa_low": bias - Z_95 * sd,
        "loa_high": bias + Z_95 * sd,
        "proportional_bias_slope": float(slope),
        "proportional_bias_intercept": float(intercept),
        "proportional_bias_r": float(r_value),
        "proportional_bias_r_squared": float(r_value**2),
        "proportional_bias_p": float(p_value),
        "proportional_bias_stderr": float(stderr),
    }


def rmse_from_loa(bias: float, loa_low: float, loa_high: float, reference_mean: float) -> float:
    """Root mean square of the paired difference, rebuilt from the printed row.

    Ref: SI Table S5 convention. The row's own bias and limits of agreement
    determine it: the width divided by two times 1.96 is the difference standard
    deviation, and the root mean square is the quadrature sum with the bias.
    """
    if reference_mean <= 0.0:
        raise ValueError("the reference mean must be positive")
    difference_sd = (loa_high - loa_low) / (2.0 * Z_95)
    return 100.0 * float(np.sqrt(bias**2 + difference_sd**2)) / reference_mean


def rmse_direct(
    first: NDArray[np.float64], second: NDArray[np.float64], reference_mean: float
) -> float:
    """Root mean square of the paired difference, as a percentage of the reference."""
    if reference_mean <= 0.0:
        raise ValueError("the reference mean must be positive")
    differences = first - second
    variance = float(np.var(differences, ddof=1)) if differences.size > 1 else 0.0
    return 100.0 * float(np.sqrt(float(np.mean(differences)) ** 2 + variance)) / reference_mean


def agreement(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    reference_mean: float,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 0,
    proportional_bias_fitted: bool = True,
) -> Agreement:
    """One agreement row, reported with limits rather than a correlation alone."""
    summary = bland_altman(first, second)
    icc = icc_absolute_agreement(first, second)
    low, high = icc_bootstrap_ci(first, second, bootstrap_replicates, seed)
    if proportional_bias_fitted:
        slope = summary["proportional_bias_slope"]
        p_value = summary["proportional_bias_p"]
    else:
        slope = float("nan")
        p_value = float("nan")
    return Agreement(
        n=int(first.size),
        icc=icc,
        icc_low=low,
        icc_high=high,
        bias=summary["bias"],
        loa_low=summary["loa_low"],
        loa_high=summary["loa_high"],
        proportional_bias_slope=slope,
        proportional_bias_p=p_value,
        rmse_percent=rmse_from_loa(
            summary["bias"], summary["loa_low"], summary["loa_high"], reference_mean
        ),
        reference_mean=reference_mean,
    )


def agreement_ordering_is_licensed(
    sensor_against_assay: float, assay_against_itself: float
) -> dict[str, float | bool]:
    """An estimate cannot legitimately agree with an assay better than it agrees with itself.

    A non-destructive estimate agreeing better than the assay agrees with itself
    would be reporting leakage rather than sensitivity.
    """
    return {
        "sensor_against_assay": sensor_against_assay,
        "assay_against_itself": assay_against_itself,
        "gap": assay_against_itself - sensor_against_assay,
        "ordering_licensed": sensor_against_assay <= assay_against_itself,
    }


def common_mass_basis(
    values: NDArray[np.float64], source_mean: float, target_mean: float
) -> NDArray[np.float64]:
    """Place one assay on the other's pooled mean basis.

    Ref: Table 4 footnote. Placing one assay on the other's pooled mean sets the
    mean difference between them to zero by construction, so the bias on that row
    is zero and its limits of agreement are symmetric about it.
    """
    if source_mean <= 0.0:
        raise ValueError("the source mean must be positive")
    return values * (float(target_mean) / float(source_mean))


def agreement_table(rows: list[Agreement]) -> list[dict[str, float | int]]:
    return [row.as_record() for row in rows]
