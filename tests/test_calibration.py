"""Pooled fit, moment correction, identifiability diagnostics and the bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from obscal.calibration import (
    CalibrationInputs,
    assemble_design,
    bootstrap,
    build_nuisance_design,
    condition_number,
    confidence_interval,
    fit_pooled,
    joint_region_area,
    moment_correction,
    pairwise_variance_inflation,
    plugin_inflation,
    residual_covariance,
    sample_state_covariance,
    standardise_design,
    variance_inflation_factors,
    wald_test,
)
from obscal.calibration.regression import TimeBasis

FEATURES = ("log_rb", "log_cm", "alpha", "matrix_texture", "effluent")


def _planted(n: int = 90, seed: int = 5) -> CalibrationInputs:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, 2)) @ np.array([[1.0, 0.85], [0.0, 0.55]])
    coverage = rng.normal(88.0, 3.4, size=n)
    times = rng.choice([3.0, 7.0, 14.0, 21.0], size=n)
    truth = np.array([[0.4, 0.0], [-0.02, 0.01], [-0.03, -0.02], [1.2, 2.4], [5.0, 1.6]])
    coverage_truth = np.array([0.9, -0.7, -0.05, 0.7, 0.4])
    noise = rng.normal(scale=0.05, size=(n, 5))
    features = state @ truth.T + np.outer(coverage - 88.0, coverage_truth) + noise + 5.0
    nuisance = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    return CalibrationInputs(
        features=features,
        state_measured=state,
        coverage=coverage,
        times=times,
        nuisance=nuisance,
        feature_names=FEATURES,
        strata=tuple(rng.choice(["a", "b", "c"], size=n)),
        error_covariance=np.diag([0.01, 0.04]),
    )


def test_plugin_inflation_is_the_printed_formula() -> None:
    assert plugin_inflation(116, 5) == pytest.approx(1.0642, abs=5e-5)
    with pytest.raises(ValueError):
        plugin_inflation(5, 5)


def test_residual_covariance_uses_the_inflated_denominator() -> None:
    residuals = np.eye(3)[:2]
    covariance = residual_covariance(residuals, n_units=5, feature_dimension=2)
    assert covariance.shape == (3, 3)
    with pytest.raises(ValueError):
        residual_covariance(residuals, 4, 5)


def test_design_assembly_names_every_block() -> None:
    inputs = _planted(24)
    basis = TimeBasis.fit(inputs.times, 3)
    design = assemble_design(
        times=inputs.times,
        time_basis=basis,
        state_measured=inputs.state_measured,
        coverage=inputs.coverage,
        nuisance=inputs.nuisance,
    )
    assert "intercept" in design.columns
    assert "state::activation" in design.columns
    assert "coverage" in design.columns
    assert design.block(["state::activation", "state::matrix"]).shape[1] == 2
    assert design.matrix.shape[0] == 24


def test_design_drops_the_time_basis_when_time_does_not_vary() -> None:
    inputs = _planted(20)
    constant = np.full_like(inputs.times, 21.0)
    basis = TimeBasis.fit(constant, 3)
    assert not basis.varies
    design = assemble_design(
        times=constant,
        time_basis=basis,
        state_measured=inputs.state_measured,
        coverage=inputs.coverage,
        nuisance=np.zeros((20, 0)),
    )
    assert not any(name.startswith("basis::") for name in design.columns)
    assert not any(name.startswith("state_by_time::") for name in design.columns)


def test_pooled_fit_recovers_a_planted_sensitivity() -> None:
    inputs = _planted()
    fit = fit_pooled(
        features=inputs.features,
        state_measured=inputs.state_measured,
        coverage=inputs.coverage,
        times=inputs.times,
        nuisance=inputs.nuisance,
        feature_names=FEATURES,
    )
    assert fit.sensitivity_ols.shape == (5, 2)
    assert fit.sensitivity_ols[0, 0] == pytest.approx(0.4, abs=0.05)
    assert fit.sensitivity_ols[3, 1] == pytest.approx(2.4, abs=0.15)
    assert fit.coverage_sensitivity.shape == (5,)
    assert fit.residual_dof > 0


def test_reference_coding_drops_one_level_per_covariate() -> None:
    design = build_nuisance_design(
        {"temperature": np.linspace(36.0, 38.0, 6)},
        {"batch": ["B1", "B2", "B3"] * 2, "operator": ["OP1", "OP2"] * 3},
    )
    assert design.width == 1 + 2 + 1
    assert all(column.startswith("z::") for column in design.columns)
    assert design.drop(["z::batch=B2"]).width == design.width - 1


def test_moment_correction_undoes_a_known_attenuation() -> None:
    rng = np.random.default_rng(9)
    n = 4000
    truth = rng.normal(size=(n, 2)) * np.array([1.0, 1.2])
    error = rng.normal(scale=[0.5, 0.6], size=(n, 2))
    measured = truth + error
    slopes = np.array([[0.4, 0.5], [0.2, -0.3]])
    response = measured @ slopes.T + rng.normal(scale=0.01, size=(n, 2))
    fit = fit_pooled(
        features=response,
        state_measured=measured,
        coverage=np.zeros(n),
        times=np.linspace(1.0, 20.0, n),
        nuisance=np.zeros((n, 0)),
        feature_names=("a", "b"),
        include_coverage=False,
    )
    sample = sample_state_covariance(measured)
    sigma_u = np.diag([0.25, 0.36])
    result = moment_correction(fit.sensitivity_ols, sample, sigma_u)
    assert result.status == "corrected"
    assert np.abs(result.corrected).mean() > np.abs(fit.sensitivity_ols).mean()
    assert 0.0 < result.axis_lambda("activation") < 1.0


def test_moment_correction_reports_a_failed_precondition_instead_of_correcting() -> None:
    ols = np.array([[1.0, 2.0]])
    sample = np.array([[0.01, 0.0], [0.0, 0.01]])
    result = moment_correction(ols, sample, np.eye(2) * 0.5)
    assert result.status == "attenuated_slope_reported_with_failed_precondition"
    assert result.corrected is None
    assert result.corrected_or_ols is not None


def test_diagnostics_match_their_closed_forms_on_two_columns() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(size=(500, 2))
    base[:, 1] = 0.6 * base[:, 0] + np.sqrt(1 - 0.36) * base[:, 1]
    design = base + rng.normal(scale=0.1, size=base.shape)
    correlation = float(np.corrcoef(design, rowvar=False)[0, 1])
    vifs = variance_inflation_factors(design)
    assert vifs[0] == pytest.approx(pairwise_variance_inflation(correlation), rel=1e-6)
    closed = np.sqrt((1 + abs(correlation)) / (1 - abs(correlation)))
    assert condition_number(design) == pytest.approx(closed, rel=1e-3)


def test_joint_region_area_is_the_information_determinant() -> None:
    rng = np.random.default_rng(6)
    design = standardise_design(rng.normal(size=(300, 2)))
    singular = np.linalg.svd(design, compute_uv=False)
    assert joint_region_area(design, [0, 1]) == pytest.approx((singular[0] * singular[1]) ** 2)
    with pytest.raises(ValueError):
        joint_region_area(design, [0])


def test_standardising_a_constant_column_is_refused() -> None:
    with pytest.raises(ValueError):
        standardise_design(np.ones((4, 2)))


def test_bootstrap_brackets_its_point_estimate() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=60)
    strata = tuple("abc"[index % 3] for index in range(60))

    def statistic(indices: np.ndarray) -> np.ndarray:
        return np.array([float(np.mean(values[indices]))])

    result = bootstrap(statistic, strata, n_units=60, replicates=120, seed=2)
    assert result.replicates.shape == (120, 1)
    assert result.low[0] <= result.point[0] <= result.high[0]
    assert result.standard_error[0] > 0.0
    assert set(result.as_record()) >= {"level", "n_replicates", "point", "low", "high"}


def test_wald_test_and_interval_are_consistent() -> None:
    report = wald_test(0.4, 0.1, residual_dof=50)
    assert report["t"] == pytest.approx(4.0)
    assert report["p"] < 0.001
    low, high = confidence_interval(0.4, 0.1, 50)
    assert low < 0.4 < high
    assert np.isnan(wald_test(0.4, 0.0, 50)["p"])
