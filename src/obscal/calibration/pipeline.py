"""The calibration step: pooled fit, moment correction, audit and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from obscal.calibration.attenuation import (
    AttenuationResult,
    moment_correction,
    report_slopes,
    sample_state_covariance,
)
from obscal.calibration.bootstrap import BootstrapResult, bootstrap
from obscal.calibration.diagnostics import (
    IdentifiabilityAudit,
    cascade_directional_derivative,
    identifiability_audit,
    identifiability_from_arm_mix,
)
from obscal.calibration.regression import (
    PooledFit,
    build_nuisance_design,
    fit_pooled,
    plugin_inflation,
)
from obscal.runtime.config import CalibrationConfig

AXES: tuple[str, ...] = ("activation", "matrix")


@dataclass(frozen=True)
class CalibrationInputs:
    """Everything the calibration step reads, as arrays on one unit axis."""

    features: NDArray[np.float64]
    state_measured: NDArray[np.float64]
    coverage: NDArray[np.float64]
    times: NDArray[np.float64]
    nuisance: NDArray[np.float64]
    feature_names: tuple[str, ...]
    strata: tuple[str, ...]
    error_covariance: NDArray[np.float64]
    axis_units: tuple[str, ...] = ("fold_change", "ug_hydroxyproline_per_chip")

    def __post_init__(self) -> None:
        n = self.features.shape[0]
        for name, array in (
            ("state_measured", self.state_measured),
            ("coverage", self.coverage),
            ("times", self.times),
            ("nuisance", self.nuisance),
        ):
            if array.shape[0] != n:
                raise ValueError(f"{name} must share the unit axis")
        if len(self.strata) != n:
            raise ValueError("every unit needs a stratum label")
        if self.error_covariance.shape != (len(AXES), len(AXES)):
            raise ValueError("the error covariance must be axis by axis")


@dataclass(frozen=True)
class CalibrationOutcome:
    """The calibrated front end and every number printed beside it."""

    pooled: PooledFit
    attenuation: AttenuationResult
    audit: IdentifiabilityAudit
    lambda_by_axis: dict[str, float]
    sample_covariance: NDArray[np.float64]
    error_covariance: NDArray[np.float64]
    inflation: float
    slope_records: list[dict[str, Any]]
    identifiability: dict[str, Any]
    bootstrap: BootstrapResult | None

    @property
    def corrected_sensitivity(self) -> NDArray[np.float64]:
        return self.attenuation.corrected_or_ols

    def summary(self) -> dict[str, Any]:
        return {
            "n_units": self.pooled.n_units,
            "feature_dimension": len(self.pooled.feature_names),
            "design_columns": self.pooled.design.width,
            "covariance_inflation": self.inflation,
            "lambda": self.lambda_by_axis,
            "identifiability": self.identifiability,
            "slopes": self.slope_records,
            "attenuation_status": self.attenuation.status,
        }


def time_basis_columns(config: CalibrationConfig) -> int:
    """Number of smooth time-basis columns the pooled design carries."""
    basis = config.pooled.get("time_basis")
    if isinstance(basis, dict):
        return int(basis.get("basis_columns", 3))
    return 3


def run_calibration(
    inputs: CalibrationInputs,
    config: CalibrationConfig,
    *,
    include_coverage: bool = True,
    include_interaction: bool = True,
    has_acellular_rows: bool = True,
    bootstrap_replicates: int | None = None,
    seed: int = 0,
) -> CalibrationOutcome:
    """Fit the pooled calibration, correct it and audit it before reporting."""
    pooled = fit_pooled(
        features=inputs.features,
        state_measured=inputs.state_measured,
        coverage=inputs.coverage,
        times=inputs.times,
        nuisance=inputs.nuisance,
        feature_names=inputs.feature_names,
        time_basis_columns=time_basis_columns(config),
        include_coverage=include_coverage,
        include_interaction=include_interaction,
        n_biological=int(inputs.features.shape[0]),
    )
    sample = sample_state_covariance(inputs.state_measured)
    attenuation = moment_correction(pooled.sensitivity_ols, sample, inputs.error_covariance)
    audit = identifiability_audit(pooled.design.matrix, pooled.design.state_columns)
    lambdas = {axis: attenuation.axis_lambda(axis) for axis in AXES}
    records = report_slopes(pooled.sensitivity_ols, attenuation, pooled.feature_names)
    identifiability = identifiability_from_arm_mix(
        has_acellular_rows=has_acellular_rows,
        has_coverage_rows=include_coverage,
    )
    replicates = (
        int(config.bootstrap["replicates"])
        if bootstrap_replicates is None
        else int(bootstrap_replicates)
    )
    result = _bootstrap_slopes(inputs, config, pooled, attenuation, replicates, seed)
    return CalibrationOutcome(
        pooled=pooled,
        attenuation=attenuation,
        audit=audit,
        lambda_by_axis=lambdas,
        sample_covariance=sample,
        error_covariance=inputs.error_covariance,
        inflation=plugin_inflation(int(inputs.features.shape[0]), config.feature_dimension),
        slope_records=records,
        identifiability=identifiability,
        bootstrap=result,
    )


def _bootstrap_slopes(
    inputs: CalibrationInputs,
    config: CalibrationConfig,
    pooled: PooledFit,
    attenuation: AttenuationResult,
    replicates: int,
    seed: int,
) -> BootstrapResult | None:
    if replicates < 2:
        return None

    def statistic(indices: NDArray[np.int64]) -> NDArray[np.float64]:
        refit = fit_pooled(
            features=inputs.features[indices],
            state_measured=inputs.state_measured[indices],
            coverage=inputs.coverage[indices],
            times=inputs.times[indices],
            nuisance=inputs.nuisance[indices],
            feature_names=inputs.feature_names,
            time_basis_columns=time_basis_columns(config),
            include_coverage=pooled.design.coverage_column >= 0,
            include_interaction=pooled.design.interaction_columns[0] >= 0,
            n_biological=int(indices.size),
        )
        sample = sample_state_covariance(inputs.state_measured[indices])
        corrected = moment_correction(refit.sensitivity_ols, sample, inputs.error_covariance)
        matrix_row = pooled.feature_names.index("log_rb")
        return np.array(
            [
                corrected.corrected_or_ols[matrix_row, 1],
                corrected.corrected_or_ols[matrix_row, 0],
            ],
            dtype=np.float64,
        )

    return bootstrap(
        statistic,
        inputs.strata,
        n_units=int(inputs.features.shape[0]),
        replicates=replicates,
        seed=seed,
        level=float(config.bootstrap.get("level", 0.95)),
        preserve_strata=bool(config.bootstrap.get("strata_preserved", True)),
    )


def cascade_report(
    ols_sensitivity: NDArray[np.float64], weights: tuple[float, float] = (1.0, 1.0)
) -> dict[str, float]:
    """The derivative the design still supports when the ladders are absent."""
    derivative = cascade_directional_derivative(ols_sensitivity, weights)
    return {
        "cascade_derivative_mean": float(np.mean(derivative)),
        "cascade_derivative_max": float(np.max(np.abs(derivative))),
        "n_features": float(derivative.size),
    }


def nuisance_summary(
    numeric: dict[str, NDArray[np.float64]], categorical: dict[str, list[str]]
) -> dict[str, object]:
    design = build_nuisance_design(numeric, categorical)
    return {"width": design.width, "columns": list(design.columns)}
