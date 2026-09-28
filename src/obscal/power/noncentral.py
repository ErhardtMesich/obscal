"""Detectability floors and the exact noncentral-t power coefficient.

Ref: Sec. Attenuation correction and the detectability floor, Eq. (5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


def noncentral_t_critical(df: int, alpha: float) -> float:
    """Two-sided critical value of the central t distribution."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    return float(stats.t.ppf(1.0 - alpha / 2.0, df))


def noncentrality_for_power(df: int, alpha: float, power: float) -> float:
    """Noncentrality that a two-sided t test needs to reach a target power."""
    critical = noncentral_t_critical(df, alpha)

    def gap(ncp: float) -> float:
        return float(stats.nct.sf(critical, df, ncp)) - float(power)

    if gap(0.0) >= 0.0:
        return 0.0
    upper = 1.0
    while gap(upper) < 0.0:
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("no noncentrality reaches the requested power")
    return float(optimize.brentq(gap, 0.0, upper, xtol=1e-13, rtol=1e-14))


def two_group_coefficient(
    n: int, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER
) -> float:
    """Exact coefficient k(n) for two groups of n units at P < alpha and the given power.

    The standard error of a two-group difference on n per group is sigma times
    sqrt(2/n), so k(n) is the required noncentrality divided by that factor.
    """
    if n < 2:
        raise ValueError("a two-group contrast needs at least two units per group")
    df = 2 * n - 2
    ncp = noncentrality_for_power(df, alpha, power)
    return ncp / float(np.sqrt(n / 2.0))


def coefficient_grid(
    sizes: Sequence[int], alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER
) -> dict[int, float]:
    return {int(n): two_group_coefficient(int(n), alpha, power) for n in sizes}


def minimum_detectable_difference(coefficient: float, dispersion_percent: float) -> float:
    """Smallest difference two groups of n separate, expressed as a percentage."""
    return float(coefficient) * float(dispersion_percent)


def replication_requirement(
    effect_percent: float,
    dispersion_percent: float,
    sizes: Sequence[int],
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> dict[str, Any]:
    """Smallest tabulated n whose detectable difference is inside the target effect."""
    if effect_percent <= 0.0:
        raise ValueError("the target effect must be positive")
    grid = coefficient_grid(sizes, alpha, power)
    feasible = [
        (size, coefficient * dispersion_percent)
        for size, coefficient in sorted(grid.items())
        if coefficient * dispersion_percent <= effect_percent
    ]
    if not feasible:
        return {
            "n": None,
            "detectable_percent": min(
                coefficient * dispersion_percent for coefficient in grid.values()
            ),
            "grid": grid,
            "reached": False,
        }
    size, detectable = feasible[0]
    return {"n": size, "detectable_percent": detectable, "grid": grid, "reached": True}


@dataclass(frozen=True)
class DetectabilityFloor:
    """Smallest change along one named axis that n units per group can resolve."""

    axis: str
    unit: str
    sigma: float
    n: int
    coefficient: float
    floor: float

    def as_record(self) -> dict[str, float | int | str]:
        return {
            "axis": self.axis,
            "unit": self.unit,
            "sigma": self.sigma,
            "n": self.n,
            "coefficient": self.coefficient,
            "floor": self.floor,
        }


def floor_from_sigma(
    axis: str,
    unit: str,
    sigma: float,
    n: int,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> DetectabilityFloor:
    """A floor built from a measured per-chip estimate standard deviation."""
    if sigma <= 0.0:
        raise ValueError("a detectability floor needs a positive per-chip standard deviation")
    coefficient = two_group_coefficient(n, alpha, power)
    return DetectabilityFloor(
        axis=axis,
        unit=unit,
        sigma=float(sigma),
        n=int(n),
        coefficient=coefficient,
        floor=minimum_detectable_difference(coefficient, sigma),
    )


def floor_grid(
    axis: str,
    unit: str,
    sigma: float,
    sizes: Sequence[int],
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> list[DetectabilityFloor]:
    return [floor_from_sigma(axis, unit, sigma, int(n), alpha, power) for n in sizes]


def estimator_covariance(
    corrected_sensitivity: NDArray[np.float64], residual_covariance: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Covariance of the delivered generalised least-squares state estimator.

    Ref: Eq. (5), the inner factor J_corr^T Sigma_eps^-1 J_corr.
    """
    if corrected_sensitivity.ndim != 2:
        raise ValueError("the sensitivity matrix must be two-dimensional")
    precision = np.linalg.inv(residual_covariance)
    information = corrected_sensitivity.T @ precision @ corrected_sensitivity
    return np.asarray(np.linalg.inv(information), dtype=np.float64)


def direction_vector(axis: str, axes: Sequence[str]) -> NDArray[np.float64]:
    """Unit direction e for one named axis inside the state plane."""
    vector = np.zeros(len(axes), dtype=np.float64)
    if axis not in axes:
        raise KeyError(f"axis {axis!r} is not in {tuple(axes)!r}")
    vector[list(axes).index(axis)] = 1.0
    return vector


def directional_floor(
    covariance: NDArray[np.float64],
    direction: NDArray[np.float64],
    n: int,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    """Eq. (5): k(n) times the standard error along the named direction."""
    quadratic = float(direction @ covariance @ direction)
    if quadratic <= 0.0:
        raise ValueError("the estimator covariance is not positive along that direction")
    return two_group_coefficient(n, alpha, power) * float(np.sqrt(quadratic))


def floor_from_estimator(
    corrected_sensitivity: NDArray[np.float64],
    residual_covariance: NDArray[np.float64],
    axes: Sequence[str],
    axis: str,
    n: int,
    unit: str,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> DetectabilityFloor:
    covariance = estimator_covariance(corrected_sensitivity, residual_covariance)
    direction = direction_vector(axis, axes)
    value = directional_floor(covariance, direction, n, alpha, power)
    sigma = float(np.sqrt(float(direction @ covariance @ direction)))
    return DetectabilityFloor(
        axis=axis,
        unit=unit,
        sigma=sigma,
        n=int(n),
        coefficient=two_group_coefficient(n, alpha, power),
        floor=value,
    )


def floor_ratio(floor: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("the reference level must be non-zero")
    return float(floor) / float(reference)
