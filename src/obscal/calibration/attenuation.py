"""The errors-in-variables moment correction.

Ref: Sec. Attenuation correction and the detectability floor, Eq. (4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

AXES: tuple[str, ...] = ("activation", "matrix")


@dataclass(frozen=True)
class AttenuationResult:
    """Corrected sensitivity matrices, or the failed precondition that stopped them."""

    status: str
    corrected: NDArray[np.float64] | None
    ols: NDArray[np.float64]
    lambda_per_axis: NDArray[np.float64]
    precondition_positive_definite: bool
    precondition_min_eigenvalue: float

    @property
    def corrected_or_ols(self) -> NDArray[np.float64]:
        return self.ols if self.corrected is None else self.corrected

    def axis_lambda(self, axis: str) -> float:
        return float(self.lambda_per_axis[AXES.index(axis)])

    def as_record(self) -> dict[str, object]:
        return {
            "status": self.status,
            "precondition_positive_definite": self.precondition_positive_definite,
            "precondition_min_eigenvalue": self.precondition_min_eigenvalue,
            "lambda_activation": self.axis_lambda("activation"),
            "lambda_matrix": self.axis_lambda("matrix"),
            "corrected_sensitivity": None if self.corrected is None else self.corrected.tolist(),
        }


def sample_state_covariance(state_measured: NDArray[np.float64]) -> NDArray[np.float64]:
    """S_ww: the sample covariance of the delivered measured state, axes by axes."""
    if state_measured.ndim != 2:
        raise ValueError("the measured state must be a units-by-axis matrix")
    return np.asarray(np.cov(state_measured, rowvar=False, ddof=1), dtype=np.float64)


def attenuation_precondition(
    sample_covariance: NDArray[np.float64], error_covariance_matrix: NDArray[np.float64]
) -> tuple[bool, float, NDArray[np.float64]]:
    """Whether S_ww - Sigma_u is positive definite, and its smallest eigenvalue."""
    gap = np.asarray(sample_covariance, dtype=np.float64) - np.asarray(
        error_covariance_matrix, dtype=np.float64
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (gap + gap.T))
    return bool(np.all(eigenvalues > 0.0)), float(eigenvalues.min()), eigenvalues


def per_axis_lambda(
    sample_covariance: NDArray[np.float64], error_covariance_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Lambda = Sigma_x (Sigma_x + Sigma_u)^-1, reported per axis."""
    sigma_x = np.asarray(sample_covariance, dtype=np.float64) - np.asarray(
        error_covariance_matrix, dtype=np.float64
    )
    return np.asarray(
        np.diag(sigma_x @ np.linalg.inv(sigma_x + error_covariance_matrix)), dtype=np.float64
    )


def moment_correction(
    ols_sensitivity: NDArray[np.float64],
    sample_covariance: NDArray[np.float64],
    error_covariance_matrix: NDArray[np.float64],
) -> AttenuationResult:
    """Eq. (4): J_corr = J_OLS S_ww (S_ww - Sigma_u)^-1.

    The correction is classical and is cited, not claimed. Where the
    precondition fails, the attenuated slope is reported with the failed
    precondition rather than corrected silently.
    """
    ols = np.asarray(ols_sensitivity, dtype=np.float64)
    sample = np.asarray(sample_covariance, dtype=np.float64)
    sigma_u = np.asarray(error_covariance_matrix, dtype=np.float64)
    ok, smallest, _ = attenuation_precondition(sample, sigma_u)
    lambdas = per_axis_lambda(sample, sigma_u)
    if not ok:
        return AttenuationResult(
            status="attenuated_slope_reported_with_failed_precondition",
            corrected=None,
            ols=ols,
            lambda_per_axis=lambdas,
            precondition_positive_definite=False,
            precondition_min_eigenvalue=smallest,
        )
    gap = sample - sigma_u
    corrected = ols @ sample @ np.linalg.inv(gap)
    return AttenuationResult(
        status="corrected",
        corrected=np.asarray(corrected, dtype=np.float64),
        ols=ols,
        lambda_per_axis=lambdas,
        precondition_positive_definite=True,
        precondition_min_eigenvalue=smallest,
    )


def correction_moves_away_from_zero(
    ols_sensitivity: NDArray[np.float64], corrected: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Dividing by a factor below one must move every slope away from zero."""
    return np.abs(corrected) >= np.abs(ols_sensitivity) - 1e-12


def attenuation_ordering(
    lambda_by_axis: NDArray[np.float64], design_by_axis: NDArray[np.float64]
) -> dict[str, float]:
    """Averaging two halves lifts attenuation; a single half lowers it."""
    measured = float(np.mean(lambda_by_axis))
    design = float(np.mean(design_by_axis))
    return {
        "measured_mean_lambda": measured,
        "design_mean_lambda": design,
        "gap": measured - design,
    }


def report_slopes(
    ols_sensitivity: NDArray[np.float64],
    result: AttenuationResult,
    feature_names: tuple[str, ...],
) -> list[dict[str, float | str]]:
    """One record per feature per axis, naive and corrected, with its factor.

    The sensitivity matrix is features by axes, as the notation registry defines
    it, so a slope is read at [feature, axis].
    """
    effective = result.corrected_or_ols
    records: list[dict[str, float | str]] = []
    for feature_index, feature in enumerate(feature_names):
        for axis_index, axis in enumerate(AXES):
            naive = float(ols_sensitivity[feature_index, axis_index])
            corrected = float(effective[feature_index, axis_index])
            records.append(
                {
                    "feature": feature,
                    "axis": axis,
                    "naive_slope": naive,
                    "corrected_slope": corrected,
                    "lambda": result.axis_lambda(axis),
                    "status": result.status,
                }
            )
    return records


def corrected_slope_matrix(
    ols_by_feature: NDArray[np.float64],
    sample_covariance: NDArray[np.float64],
    error_covariance_matrix: NDArray[np.float64],
) -> AttenuationResult:
    """Convenience wrapper for a per-axis-covariates feature block."""
    return moment_correction(ols_by_feature, sample_covariance, error_covariance_matrix)
