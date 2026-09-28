"""The cell-free artefact channel, measured rather than assumed inert.

Ref: Sec. External validation and the artefact-confound control; SI Table S13.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SchedulePrediction:
    """How much of a quantity the cell-free channel alone recovers."""

    r_squared: float
    variance_share_of_state_estimate: float
    residual_contribution_after_covariates: float
    n_observations: int


def coefficient_of_determination(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> float:
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted series must align")
    total = float(np.sum((observed - float(np.mean(observed))) ** 2))
    if total <= 0.0:
        raise ValueError("cannot score an invariant series")
    residual = float(np.sum((observed - predicted) ** 2))
    return 1.0 - residual / total


def residual_contribution(
    full_residual_ss: float, reduced_residual_ss: float, total_ss: float
) -> float:
    """Share of the state variance the cell-free channel takes once covariates are in."""
    if total_ss <= 0.0:
        raise ValueError("total sum of squares must be positive")
    return (float(reduced_residual_ss) - float(full_residual_ss)) / float(total_ss)


def schedule_prediction(
    culture_day: NDArray[np.float64],
    cell_free_channels: NDArray[np.float64],
    *,
    modelled_covariates: NDArray[np.float64] | None = None,
    state_estimate_variance_share: float = float("nan"),
) -> SchedulePrediction:
    """Fit culture day on the cell-free channel, with and without the covariates.

    A cell-free channel predicting nothing would be a rigged control rather than
    a clean one, so the unconstrained figure is reported beside the constrained
    one instead of being assumed away.
    """
    design = np.column_stack([np.ones_like(culture_day), cell_free_channels])
    fitted = _least_squares_fit(design, culture_day)
    r_squared = coefficient_of_determination(culture_day, design @ fitted)

    if modelled_covariates is None:
        return SchedulePrediction(
            r_squared=r_squared,
            variance_share_of_state_estimate=state_estimate_variance_share,
            residual_contribution_after_covariates=float("nan"),
            n_observations=int(culture_day.size),
        )

    full_design = np.column_stack([design, modelled_covariates])
    full_fit = _least_squares_fit(full_design, culture_day)
    full_residual = float(np.sum((culture_day - full_design @ full_fit) ** 2))
    reduced_design = np.column_stack([np.ones_like(culture_day), modelled_covariates])
    reduced_fit = _least_squares_fit(reduced_design, culture_day)
    reduced_residual = float(np.sum((culture_day - reduced_design @ reduced_fit) ** 2))
    total = float(np.sum((culture_day - float(np.mean(culture_day))) ** 2))
    return SchedulePrediction(
        r_squared=r_squared,
        variance_share_of_state_estimate=state_estimate_variance_share,
        residual_contribution_after_covariates=residual_contribution(
            full_residual, reduced_residual, total
        ),
        n_observations=int(culture_day.size),
    )


def _least_squares_fit(
    design: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(coefficients, dtype=np.float64).reshape(-1)


def drift_against_signal(
    blank_drift_ohm: NDArray[np.float64],
    biological_change_ohm: NDArray[np.float64],
) -> dict[str, list[float]]:
    """Percentage drift and percentage matrix-driven change on a shared grid."""
    if blank_drift_ohm.shape != biological_change_ohm.shape:
        raise ValueError("blank and biological series must share a frequency grid")
    return {
        "blank_drift_pct": blank_drift_ohm.tolist(),
        "biological_change_pct": biological_change_ohm.tolist(),
        "drift_to_signal_ratio": (blank_drift_ohm / biological_change_ohm).tolist(),
    }


def excursed_frequency_flags(
    drift_pct: NDArray[np.float64], biological_pct: NDArray[np.float64], threshold: float
) -> NDArray[np.int64]:
    """Frequencies where measured drift is not small next to the biological change."""
    return (drift_pct >= threshold * biological_pct).astype(np.int64)


def artefact_covariate_loss(reference_rmse: float, perturbed_rmse: float) -> dict[str, float]:
    """Change in held-out error and its relative size against the reference."""
    delta = float(perturbed_rmse) - float(reference_rmse)
    return {
        "reference_rmse": float(reference_rmse),
        "perturbed_rmse": float(perturbed_rmse),
        "delta_rmse": delta,
        "relative_worse_fraction": delta / float(reference_rmse),
    }


def rank_perturbations(records: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    """Order perturbations by how much held-out error they add, largest first."""
    return sorted(records, key=lambda item: item[1], reverse=True)
