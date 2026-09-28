"""Split-chip replication and the terminal assay's own error.

Ref: Sec. Split-chip replication and terminal-assay error; Eq. (1); SI Table S9.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class BlockSpread:
    """Mean and delivered spread of one endpoint over the calibration block."""

    mean: float
    within_sd: float
    between_sd: float
    total_sd: float
    cv_percent: float
    n_units: int
    n_cells: int


def block_spread(
    cell_means: Sequence[float], cell_sds: Sequence[float], cell_counts: Sequence[int]
) -> BlockSpread:
    """Pool the trajectory cells to the block summary Table 3 reports.

    Ref: SI Table S7 footnote; the 29 arm-by-timepoint cells pool on their own
    chip counts as divisor, so the mean is chip-weighted and the delivered
    spread is the quadrature sum of the pooled within-chip term and the spread
    of the cell means about that mean.
    """
    means = np.asarray(cell_means, dtype=np.float64)
    sds = np.asarray(cell_sds, dtype=np.float64)
    counts = np.asarray(cell_counts, dtype=np.float64)
    if not (means.shape == sds.shape == counts.shape):
        raise ValueError("cell means, sds and counts must align")
    if counts.sum() <= 0.0:
        raise ValueError("the block must carry at least one unit")
    weights = counts / counts.sum()
    mean = float(np.sum(weights * means))
    within = float(np.sqrt(np.sum(weights * sds**2)))
    between = float(np.sqrt(np.sum(weights * (means - mean) ** 2)))
    total = float(np.sqrt(within**2 + between**2))
    return BlockSpread(
        mean=mean,
        within_sd=within,
        between_sd=between,
        total_sd=total,
        cv_percent=100.0 * total / mean,
        n_units=int(counts.sum()),
        n_cells=int(means.size),
    )


@dataclass(frozen=True)
class SplitChipError:
    """The reference assay's own error, bounded from inside a destructive design."""

    n_devices: int
    half_difference_sd: float
    single_half_read_error: float
    half_mean_error: float
    mean_difference: float
    loa_low: float
    loa_high: float
    mean_difference_p: float
    within_chip_fraction: float
    rho_denominator_df: int

    @property
    def limits_of_agreement_width(self) -> float:
        return self.loa_high - self.loa_low

    @property
    def is_exchangeable(self) -> bool:
        return self.mean_difference_p > 0.05

    def as_record(self) -> dict[str, float | int | bool]:
        return {
            "n_devices": self.n_devices,
            "half_difference_sd": self.half_difference_sd,
            "single_half_read_error": self.single_half_read_error,
            "half_mean_error": self.half_mean_error,
            "mean_difference": self.mean_difference,
            "loa_low": self.loa_low,
            "loa_high": self.loa_high,
            "mean_difference_p": self.mean_difference_p,
            "within_chip_fraction": self.within_chip_fraction,
            "exchangeable": self.is_exchangeable,
        }


def split_chip_error(
    half_one: NDArray[np.float64], half_two: NDArray[np.float64]
) -> SplitChipError:
    """Bound the destructive standard's own error variance from above.

    Both halves of a matched-mass device enter the identical assay chain, so the
    half difference identifies the reference error and the half mean carries
    half of it.
    """
    if half_one.shape != half_two.shape:
        raise ValueError("the two half reads must align")
    if half_one.size < 2:
        raise ValueError("a split-chip bound needs at least two devices")
    differences = half_one - half_two
    difference_sd = float(np.std(differences, ddof=1))
    mean_difference = float(np.mean(differences))
    loa_low = mean_difference - Z_95 * difference_sd
    loa_high = mean_difference + Z_95 * difference_sd
    test = stats.ttest_1samp(differences, 0.0)
    single_half = difference_sd / np.sqrt(2.0)
    half_mean = single_half / np.sqrt(2.0)
    rho = within_chip_fraction_from_halves(half_one, half_two)
    return SplitChipError(
        n_devices=int(half_one.size),
        half_difference_sd=difference_sd,
        single_half_read_error=float(single_half),
        half_mean_error=float(half_mean),
        mean_difference=mean_difference,
        loa_low=loa_low,
        loa_high=loa_high,
        mean_difference_p=float(test.pvalue),
        within_chip_fraction=rho,
        rho_denominator_df=int(half_one.size - 1),
    )


def within_chip_fraction_from_halves(
    half_one: NDArray[np.float64], half_two: NDArray[np.float64]
) -> float:
    """Within-chip share of the endpoint's total variance, from paired halves."""
    differences = half_one - half_two
    half_means = 0.5 * (half_one + half_two)
    error_variance = float(np.var(differences, ddof=1)) / 2.0
    mean_variance = float(np.var(half_means, ddof=1))
    denominator = mean_variance + 0.5 * error_variance
    if denominator <= 0.0:
        raise ValueError("the half-mean variance is degenerate")
    return error_variance / denominator


def within_chip_fraction_from_cv_ratio(ratio: float) -> float:
    """Within-chip share implied by the half-mean over single-half CV ratio.

    Ref: SI Table S8 footnote. Averaging two halves divides the assay term by
    two and leaves the between-device term alone, so the squared ratio is one
    minus half the within-chip share.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError("the CV ratio must lie in (0, 1]")
    return 2.0 * (1.0 - ratio**2)


def attenuation_factor(sigma_x: float, sigma_u: float) -> float:
    """Per-axis errors-in-variables attenuation lambda = sigma_x^2/(sigma_x^2+sigma_u^2)."""
    if sigma_x <= 0.0:
        raise ValueError("the state spread must be positive")
    if sigma_u < 0.0:
        raise ValueError("the assay error cannot be negative")
    return sigma_x**2 / (sigma_x**2 + sigma_u**2)


def attenuation_from_design(design_cv: float, assay_cv: float) -> float:
    """Design-ladder value of lambda, computed rather than fitted."""
    return attenuation_factor(float(design_cv), float(assay_cv))


def error_covariance(
    matrix_single_read_error: float,
    activation_design_value: float,
    matrix_design_value: float,
) -> NDArray[np.float64]:
    """Sigma_u, with the matrix entry identified and the activation entry a design value.

    The replicate index is carried by the matrix component because both halves
    supply it; the activation component is read once per device, so its entry is
    a design value at the spread achieved rather than an identified one.
    """
    covariance = np.zeros((2, 2), dtype=np.float64)
    covariance[0, 0] = float(activation_design_value) ** 2
    covariance[1, 1] = float(matrix_single_read_error) ** 2
    covariance[0, 1] = covariance[1, 0] = float(matrix_design_value)
    return covariance


def per_axis_attenuation(
    state_spread: NDArray[np.float64], error_covariance_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Diagonal of the attenuation matrix Lambda = Sigma_x (Sigma_x + Sigma_u)^-1."""
    sigma_x = np.diag(np.asarray(state_spread, dtype=np.float64) ** 2)
    combined = sigma_x + error_covariance_matrix
    lam = sigma_x @ np.linalg.inv(combined)
    return np.asarray(np.diag(lam), dtype=np.float64)
