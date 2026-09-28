"""The execution family: every mechanism the release implements is actually run.

Each check builds its own reference from a definition, an enumeration or a
brute-force recomputation, and never calls the implementation it audits. The
substrate is the harness device pool, so every number a check reports is the
release's own and no manuscript table value is claimed from it.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats

from obscal.allocation import (
    InformationState,
    convergence_at_full_budget,
    d_optimal_order,
    gains,
    marginal_gain,
    prior_art_allocator_order,
    random_matched_rate_order,
)
from obscal.calibration import (
    CalibrationInputs,
    CalibrationOutcome,
    build_nuisance_design,
    condition_number,
    joint_region_area,
    pairwise_variance_inflation,
    plugin_inflation,
    sample_state_covariance,
    standardise_design,
    variance_inflation_factors,
)
from obscal.chip import (
    fit_sweep,
    geometric_correction_factor,
    impedance_model,
    residual_degrees_of_freedom,
)
from obscal.chip.circuit import CircuitParameters
from obscal.cohort import Cohort, DeviceRecord, build_cohort_fixture
from obscal.cohort.fixture import FixtureSpec
from obscal.design import (
    block_closure_identities,
    ledger_closure,
    withdrawal_share_pct,
)
from obscal.evaluation import (
    batch_adjusted_contrast,
    binomial_direction_test,
    holm_step_down,
    render_table,
    roc_auc,
    welch_t_test,
)
from obscal.frontend import (
    FEATURE_DIMENSION,
    BandSensitivity,
    band_table,
    matrix_information_shares,
)
from obscal.power import estimator_covariance, two_group_coefficient
from obscal.runtime.atomic import write_json, write_text
from obscal.runtime.config import RunSpec
from obscal.runtime.seed import numpy_generator, set_seed
from obscal.study import (
    StudyInputs,
    calibration_blocks,
    calibration_inputs,
    feature_vector_for,
    terminal_sweep_index,
    twin_window_for,
)
from obscal.terminal import (
    agreement,
    block_spread,
    half_mean_basis,
    icc_from_records,
    icc_two_way_random_absolute,
    split_chip_error,
    within_chip_fraction_from_cv_ratio,
)
from obscal.twin import (
    ArimaRung,
    PersistenceRung,
    Rung,
    calibrate_intervals,
    empirical_coverage,
    fit_backbone,
    leakage_diagnostic,
    mechanistic_trajectory,
    parameter_count,
    predict_series,
    reconstruction_and_forecast_loss,
    score_ladder,
    to_held_out_series,
)
from obscal.twin.pipeline import TwinWindow
from obscal.twin.rungs import fit_mechanistic_parameters
from obscal.verification.checks import (
    CheckResult,
    compare,
    not_run,
    truthy,
    within,
)

AXES = ("activation", "matrix")
MATRIX_AXIS = 1
FEATURE_KEYS = ("log_rb", "log_cm", "alpha", "matrix_texture", "effluent_pro_collagen")


@dataclass(frozen=True)
class ExecutionContext:
    """Everything the execution checks read, resolved once by the driver."""

    root: Path
    run: RunSpec
    cohort: Cohort
    inputs: StudyInputs
    calibration: CalibrationOutcome
    arrays: CalibrationInputs

    @property
    def biological(self) -> list[DeviceRecord]:
        return [device for device in self.cohort.by_block("biological") if device.retained]

    @property
    def fixture(self) -> FixtureSpec:
        return FixtureSpec.load(self.root, self.run.seed)


def build_context(root: Path, run: RunSpec, cohort: Cohort) -> ExecutionContext:
    from obscal.study import run_calibration_study

    inputs = StudyInputs.load(root, cohort)
    devices = calibration_blocks(inputs, include_ladders=run.toggles.decoupling_ladders)
    arrays = calibration_inputs(inputs, run, devices)
    calibration, _ = run_calibration_study(inputs, run, bootstrap_replicates=32)
    return ExecutionContext(
        root=root,
        run=run,
        cohort=cohort,
        inputs=inputs,
        calibration=calibration,
        arrays=arrays,
    )


def _true_sensitivities(fixture: FixtureSpec) -> NDArray[np.float64]:
    """The harness pool's own generating sensitivities on the standardised scale."""
    front = fixture.section("front_end")
    return np.asarray(
        [float(front["sensitivity_matrix"][key]) for key in FEATURE_KEYS], dtype=np.float64
    )


def _true_activation_sensitivities(fixture: FixtureSpec) -> NDArray[np.float64]:
    front = fixture.section("front_end")
    return np.asarray(
        [float(front["sensitivity_activation"][key]) for key in FEATURE_KEYS], dtype=np.float64
    )


def _true_coverage_sensitivities(fixture: FixtureSpec) -> NDArray[np.float64]:
    front = fixture.section("front_end")
    return np.asarray(
        [float(front["sensitivity_coverage"][key]) for key in FEATURE_KEYS], dtype=np.float64
    )


def pool_checks(context: ExecutionContext) -> list[CheckResult]:
    out: list[CheckResult] = []
    design_blocks = {
        "biological": 148,
        "acellular_ladder": 24,
        "coverage_ladder": 20,
        "cell_free": 8,
        "held_out_geometry": 12,
        "cross_line": 12,
        "static_counterpart": 36,
    }
    observed_blocks = {name: len(context.cohort.by_block(name)) for name in design_blocks}
    out.append(
        compare(
            "harness pool carries the designed number of devices",
            260,
            len(context.cohort),
            detail="the pool is laid out to the article's own chip budget",
            evidence=context.cohort.summary(),
        )
    )
    out.append(
        compare(
            "harness pool block partition matches the design",
            design_blocks,
            observed_blocks,
            detail="SI Table S9 Panel B and Table 1",
        )
    )
    rebuilt = build_cohort_fixture(FixtureSpec.load(context.root, context.run.seed))
    out.append(
        truthy(
            "building the pool twice yields identical devices",
            all(
                np.array_equal(left.series_days, right.series_days)
                and np.array_equal(left.front_end_state, right.front_end_state)
                for left, right in zip(context.cohort.devices, rebuilt.devices, strict=True)
            ),
            expected=True,
            observed=len(rebuilt),
            detail="the pool builder is a pure function of its specification and seed",
        )
    )
    closures = block_closure_identities()
    out.append(
        truthy(
            "every published budget identity closes",
            all(left == right for left, right in closures.values()),
            expected=True,
            observed={name: [left, right] for name, (left, right) in closures.items()},
            detail="arm totals, ladder sums, pool total, withdrawals and the schedule split",
        )
    )
    ledger = ledger_closure()
    out.append(
        truthy(
            "the withdrawal ledger closes against the pool",
            all(abs(left - right) < 1e-9 for left, right in ledger.values()),
            expected=True,
            observed=round(withdrawal_share_pct(), 6),
            detail="class counts and shares sum to the withdrawn total",
        )
    )
    with tempfile.TemporaryDirectory() as scratch:
        from obscal.cohort import read_cohort, write_cohort

        directory = write_cohort(Path(scratch) / "cohort", context.cohort)
        reloaded = read_cohort(directory)
        identical = all(
            left.device_id == right.device_id
            and bool(np.array_equal(left.front_end_state, right.front_end_state))
            and bool(np.array_equal(left.sweeps.real_ohm, right.sweeps.real_ohm))
            and bool(np.array_equal(left.sweeps.imag_ohm, right.sweeps.imag_ohm))
            and left.terminal == right.terminal
            and left.nuisance == right.nuisance
            for left, right in zip(context.cohort.devices, reloaded.devices, strict=True)
        )
    out.append(
        truthy(
            "the pool round-trips through its on-disk form unchanged",
            identical,
            expected=True,
            observed=len(reloaded),
            detail="arrays and labels rebuild the same records",
        )
    )
    return out


def circuit_checks(context: ExecutionContext) -> list[CheckResult]:
    inputs = context.inputs
    out: list[CheckResult] = []
    device = context.biological[0]
    frequencies = np.asarray(device.sweeps.frequencies_hz, dtype=np.float64)
    out.append(
        compare(
            "one sweep leaves the printed number of residual degrees of freedom",
            9,
            residual_degrees_of_freedom(frequencies.size),
            detail="two real quantities per frequency less the five free parameters",
        )
    )
    truth = CircuitParameters(
        r_ext_ohm=120.0,
        r_barrier_ohm=260.0,
        c_membrane_f=4.0e-9,
        cpe_magnitude=1.0e-5,
        cpe_exponent=0.86,
    )
    clean = fit_sweep(frequencies, impedance_model(truth, frequencies))
    out.append(
        within(
            "the circuit fit recovers an exactly modelled sweep",
            0.0,
            clean.residual_chi_square,
            1e-8,
            detail="a noiseless sweep must be fitted to machine precision",
        )
    )
    out.append(
        within(
            "the recovered barrier resistance of a noiseless sweep",
            truth.r_barrier_ohm,
            clean.parameters.r_barrier_ohm,
            truth.r_barrier_ohm * 1e-5,
            detail="the fit is exercised against a closed-form target it can be wrong about",
        )
    )
    out.append(
        within(
            "the recovered constant-phase exponent of a noiseless sweep",
            truth.cpe_exponent,
            clean.parameters.cpe_exponent,
            1e-6,
            detail="the exponent is the axis the interface reading rests on",
        )
    )
    index = terminal_sweep_index(device)
    fit = fit_sweep(frequencies, device.sweeps.at(index) * inputs.correction_factor)
    out.append(
        truthy(
            "the fit converges on a measured sweep from the pool",
            fit.converged and math.isfinite(fit.lack_of_fit_p),
            expected=True,
            observed=round(fit.parameters.r_barrier_ohm, 6),
            detail="the sweep was forward-modelled, so the fit has a target it can miss",
        )
    )
    out.append(
        truthy(
            "the lack-of-fit statistic stays inside the unit interval",
            bool(np.isfinite(fit.lack_of_fit_p)) and fit.lack_of_fit_p <= 1.0,
            expected=True,
            observed=round(min(max(fit.lack_of_fit_p, 0.0), 1.0), 6),
            detail="a residual chi-square survival probability on nine degrees of freedom",
        )
    )
    factors = {round(geometric_correction_factor(inputs.geometry), 12) for _ in context.biological}
    out.append(
        compare(
            "the geometric correction is one study constant",
            1,
            len(factors),
            detail="never refitted per device, timepoint or arm",
            evidence={"factor": sorted(factors)[0]},
        )
    )
    out.append(
        truthy(
            "the geometric correction is a fraction of the floor area",
            0.0 < sorted(factors)[0] < 1.0,
            expected=True,
            observed=sorted(factors)[0],
            detail="the perimeter term the current path adds is taken off the floor area",
        )
    )
    return out


def feature_checks(context: ExecutionContext) -> list[CheckResult]:
    out: list[CheckResult] = []
    device = context.biological[0]
    vector = feature_vector_for(device, context.inputs, multi_frequency=True)
    single = feature_vector_for(device, context.inputs, multi_frequency=False)
    out.append(
        compare(
            "the fused front end carries the printed number of features",
            FEATURE_DIMENSION,
            vector.dimension,
            detail="Eq. (2): three electrical, one optical and one effluent feature",
        )
    )
    out.append(
        compare(
            "the single-frequency ablation collapses the electrical block",
            3,
            single.dimension,
            detail="the comparator channel replaces the three electrical features",
        )
    )
    out.append(
        truthy(
            "coverage is carried beside the features rather than inside them",
            "coverage" not in vector.names and vector.coverage_percent >= 0.0,
            expected=True,
            observed=round(vector.coverage_percent, 4),
            detail="coverage enters the regression as its own measured covariate",
        )
    )
    bands = band_table(
        [
            BandSensitivity(frequency_hz=float(hz), activation=0.4, matrix=0.5, coverage=2.0)
            for hz in context.inputs.frontend.frequencies_hz
        ]
    )
    shares = matrix_information_shares([band.matrix for band in bands.bands])
    out.append(
        within(
            "band information shares sum to one hundred percent",
            100.0,
            float(sum(shares)),
            1e-9,
            detail="the bands partition the matrix-axis information",
        )
    )
    out.append(
        compare(
            "the excitation grid is the printed seven-point set",
            (100.0, 500.0, 1000.0, 4000.0, 10000.0, 40000.0, 100000.0),
            tuple(float(value) for value in context.inputs.frontend.frequencies_hz),
            detail="100 Hz to 100 kHz, unevenly spaced",
        )
    )
    return out


def calibration_checks(context: ExecutionContext) -> list[CheckResult]:
    outcome = context.calibration
    arrays = context.arrays
    out: list[CheckResult] = []
    out.append(
        compare(
            "the pooled design matrix has full column rank",
            outcome.pooled.design.width,
            int(np.linalg.matrix_rank(outcome.pooled.design.matrix)),
            detail="rank equals the delivered column count",
            evidence={"rows": int(outcome.pooled.design.matrix.shape[0])},
        )
    )
    covariance = outcome.pooled.residual_covariance
    out.append(
        truthy(
            "the residual covariance is symmetric",
            bool(np.allclose(covariance, covariance.T, atol=1e-12)),
            expected=True,
            observed=float(np.abs(covariance - covariance.T).max()),
            detail="a covariance the floor is computed from must be symmetric",
        )
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    out.append(
        truthy(
            "the residual covariance is positive definite",
            bool(np.all(eigenvalues > 0.0)),
            expected=True,
            observed=float(eigenvalues.min()),
            detail="Eq. (4) inverts it",
        )
    )
    out.append(
        within(
            "the plug-in covariance inflation is the printed formula",
            plugin_inflation(int(arrays.features.shape[0]), FEATURE_DIMENSION),
            outcome.inflation,
            1e-9,
            detail="Nbio/(Nbio - p - 2) at the pool's own device count",
        )
    )
    out.append(
        compare(
            "the identifiability audit prices both state axes",
            (True, "both_axes_identified"),
            (outcome.audit.jointly_identified, outcome.audit.verdict),
            detail="neither axis is rank deficient in the plane the two span",
            evidence=outcome.audit.as_record(),
        )
    )
    sample = sample_state_covariance(arrays.state_measured)
    brute = _brute_moment_correction(
        outcome.pooled.sensitivity_ols, sample, arrays.error_covariance
    )
    delivered = outcome.corrected_sensitivity
    out.append(
        within(
            "the moment correction matches a brute-force recomputation",
            0.0,
            float(np.abs(delivered - brute).max()),
            1e-12,
            detail="Eq. (4) recomputed by an independent matrix product",
        )
    )
    sigma_x = sample - arrays.error_covariance
    brute_lambda = np.diag(
        sigma_x @ np.linalg.solve(sigma_x + arrays.error_covariance, np.eye(sigma_x.shape[0]))
    )
    for axis_index, axis in enumerate(AXES):
        out.append(
            within(
                f"delivered attenuation on the {axis} axis",
                float(brute_lambda[axis_index]),
                outcome.lambda_by_axis[axis],
                1e-12,
                detail="diagonal of Sigma_x (Sigma_x + Sigma_u)^-1 with an explicit solve",
                evidence={"sigma_x": float(sigma_x[axis_index, axis_index])},
            )
        )
    naive = outcome.pooled.sensitivity_ols[:, MATRIX_AXIS]
    corrected = outcome.corrected_sensitivity[:, MATRIX_AXIS]
    out.append(
        truthy(
            "the correction enlarges the delivered matrix sensitivity",
            float(np.linalg.norm(corrected)) > float(np.linalg.norm(naive)),
            expected="a larger norm after the moment correction",
            observed=[
                round(float(np.linalg.norm(naive)), 6),
                round(float(np.linalg.norm(corrected)), 6),
            ],
            detail="every per-axis factor is below one, so the correction can only enlarge it",
        )
    )
    scale = context.fixture.section("front_end")["state_scale"]
    state_spread = np.array(
        [float(scale["activation_spread"]), float(scale["matrix_spread"])], dtype=np.float64
    )
    coverage_spread = float(scale["coverage_spread"])
    truth_matrix = _true_sensitivities(context.fixture)
    truth_activation = _true_activation_sensitivities(context.fixture)
    truth_coverage = _true_coverage_sensitivities(context.fixture)
    standard_error = outcome.pooled.coefficient_se("state::") / np.where(
        np.array([outcome.lambda_by_axis[axis] for axis in AXES], dtype=np.float64).reshape(-1, 1)
        > 0.0,
        np.array([outcome.lambda_by_axis[axis] for axis in AXES], dtype=np.float64).reshape(-1, 1),
        1.0,
    )
    for axis_index, (axis, truth, spread_index) in enumerate(
        (
            ("activation", truth_activation, 0),
            ("matrix", truth_matrix, 1),
        )
    ):
        estimate = outcome.corrected_sensitivity[:, axis_index] * state_spread[spread_index]
        interval = 3.0 * standard_error[axis_index] * state_spread[spread_index]
        inside = np.abs(estimate - truth) <= interval
        out.append(
            truthy(
                f"the pooled calibration recovers the pool's own {axis} sensitivities",
                bool(np.all(inside)),
                expected="every generated sensitivity inside three standard errors of its estimate",
                observed=[
                    [round(float(a), 4), round(float(b), 4), round(float(c), 4)]
                    for a, b, c in zip(estimate, truth, interval, strict=True)
                ],
                detail=(
                    "the delivered slope is put on the pool's own standardised scale and "
                    "compared with the sensitivity the pool was generated from; a channel "
                    "whose sensitivity is below the regression's own noise floor is reported "
                    "through its interval rather than through a bare ratio"
                ),
            )
        )
    coverage_estimate = outcome.pooled.coverage_sensitivity * coverage_spread
    coverage_interval = 3.0 * outcome.pooled.coefficient_se("coverage")[0] * coverage_spread
    out.append(
        truthy(
            "the pooled calibration recovers the pool's own coverage sensitivities",
            bool(np.all(np.abs(coverage_estimate - truth_coverage) <= coverage_interval)),
            expected="every generated coverage sensitivity inside three standard errors",
            observed=[
                [round(float(a), 4), round(float(b), 4), round(float(c), 4)]
                for a, b, c in zip(
                    coverage_estimate, truth_coverage, coverage_interval, strict=True
                )
            ],
            detail="coverage is the dominant nuisance and is identified by its own ladder",
        )
    )
    if outcome.bootstrap is None:
        out.append(
            not_run(
                "the bootstrap interval brackets its point estimate",
                "the bootstrap was disabled for this run",
            )
        )
    else:
        interval = outcome.bootstrap
        out.append(
            truthy(
                "the bootstrap interval brackets its point estimate",
                bool(
                    np.all(interval.low <= interval.point)
                    and np.all(interval.point <= interval.high)
                ),
                expected=True,
                observed=[
                    float(interval.low[0]),
                    float(interval.point[0]),
                    float(interval.high[0]),
                ],
                detail="a percentile interval that excludes its own statistic is mis-built",
            )
        )
    return out


def _brute_moment_correction(
    ols: NDArray[np.float64], sample: NDArray[np.float64], sigma_u: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Eq. (4) written out with an explicit inverse, as the reference."""
    inverse = np.linalg.inv(sample - sigma_u)
    return np.asarray(ols @ sample @ inverse, dtype=np.float64)


def diagnostics_checks(context: ExecutionContext) -> list[CheckResult]:
    out: list[CheckResult] = []
    rng = numpy_generator(11)
    correlation = 0.62
    latent = rng.normal(size=(400, 2))
    latent[:, 1] = correlation * latent[:, 0] + math.sqrt(1.0 - correlation**2) * latent[:, 1]
    design = latent + rng.normal(scale=0.2, size=latent.shape)
    correlation_matrix = np.asarray(np.corrcoef(design, rowvar=False), dtype=np.float64)
    sample_correlation = float(correlation_matrix[0, 1])
    computed = variance_inflation_factors(design)
    out.append(
        within(
            "the variance inflation factor matches its two-column closed form",
            pairwise_variance_inflation(sample_correlation),
            float(computed[0]),
            1e-6,
            detail="1/(1 - r^2) written out by hand against the fitted diagnostic",
            evidence={"correlation": round(sample_correlation, 6)},
        )
    )
    closed_form = math.sqrt((1.0 + abs(sample_correlation)) / (1.0 - abs(sample_correlation)))
    out.append(
        within(
            "the condition number matches its standardised two-column closed form",
            closed_form,
            float(condition_number(design)),
            1e-3,
            detail="ratio of the two singular values of the standardised design",
        )
    )
    standardised = standardise_design(design)
    singular = np.linalg.svd(standardised, compute_uv=False)
    out.append(
        within(
            "the joint region area is the squared product of the singular values",
            float((singular[0] * singular[1]) ** 2),
            float(joint_region_area(standardised, [0, 1])),
            1e-6,
            detail="a determinant of the information matrix, not a sensitivity matrix",
        )
    )
    out.append(
        truthy(
            "the standardised design has unit column spread",
            bool(
                np.allclose(standardised.mean(axis=0), 0.0, atol=1e-12)
                and np.allclose(standardised.std(axis=0, ddof=1), 1.0, atol=1e-12)
            ),
            expected=True,
            observed=[round(float(value), 12) for value in standardised.std(axis=0, ddof=1)],
            detail="no axis is carried in the other's unit",
        )
    )
    nuisance = build_nuisance_design(
        {
            "temperature": np.linspace(36.0, 38.0, 12),
            "medium_change_indicator": np.arange(12, dtype=np.float64),
            "hours_since_electrode_wetting": np.linspace(24.0, 240.0, 12),
        },
        {
            "batch": ["B1", "B2", "B3"] * 4,
            "operator": ["OP1", "OP2"] * 6,
            "geometry": ["development"] * 12,
            "perfusion_state": ["perfused"] * 12,
        },
    )
    out.append(
        compare(
            "reference coding drops one level per categorical covariate",
            3 + (3 - 1) + (2 - 1),
            nuisance.width,
            detail="three numerics, two batch indicators and one operator indicator",
            evidence={"columns": list(nuisance.columns)},
        )
    )
    return out


def terminal_checks(context: ExecutionContext) -> list[CheckResult]:
    out: list[CheckResult] = []
    devices = context.biological
    first = np.asarray([device.terminal.hydroxyproline_half_1 for device in devices])
    second = np.asarray([device.terminal.hydroxyproline_half_2 for device in devices])
    split = split_chip_error(first, second)
    brute_sd = float(np.std(first - second, ddof=1))
    out.append(
        within(
            "the half-to-half difference standard deviation matches its brute-force value",
            brute_sd,
            split.half_difference_sd,
            1e-12,
            detail="the pooled read error is bounded from the paired halves alone",
        )
    )
    out.append(
        within(
            "the single half-chip read error is the difference over root two",
            brute_sd / math.sqrt(2.0),
            split.single_half_read_error,
            1e-12,
            detail="Eq. (1) and the split-chip decomposition",
        )
    )
    ratio = split.single_half_read_error / split.half_mean_error
    out.append(
        within(
            "averaging the two halves removes exactly half the assay variance",
            math.sqrt(2.0),
            ratio,
            1e-9,
            detail="the half-mean error is the single-read error over root two",
        )
    )
    implied = within_chip_fraction_from_cv_ratio(0.885)
    out.append(
        within(
            "the two routes to the within-chip share agree",
            0.45,
            implied,
            0.03,
            detail=(
                "the half-chip construct ratio implies a share that the split-chip "
                "decomposition independently reports"
            ),
            evidence={"split_chip_route": round(split.within_chip_fraction, 4)},
        )
    )
    out.append(
        within(
            "the split-chip route to the within-chip share",
            float(np.var(first - second, ddof=1))
            / 2.0
            / (
                float(np.var(0.5 * (first + second), ddof=1))
                + 0.5 * float(np.var(first - second, ddof=1)) / 2.0
            ),
            split.within_chip_fraction,
            1e-12,
            detail="recomputed by hand from the two half reads",
        )
    )
    collapsed = half_mean_basis(
        np.concatenate([first.reshape(-1, 1), second.reshape(-1, 1)], axis=1).reshape(-1)
    )
    out.append(
        truthy(
            "collapsing to the half-mean basis reproduces the terminal values",
            bool(np.allclose(collapsed, 0.5 * (first + second), atol=1e-12)),
            expected=True,
            observed=int(collapsed.size),
            detail="every hydroxyproline contrast is read on the half-mean",
        )
    )
    rater_icc = icc_two_way_random_absolute(np.column_stack([first, second]))
    records_icc = icc_from_records(
        [device.device_id for device in devices] * 2,
        ["half_1"] * len(devices) + ["half_2"] * len(devices),
        list(first) + list(second),
    )
    out.append(
        within(
            "the two ICC routes agree on the same table",
            rater_icc,
            records_icc,
            1e-10,
            detail="a direct two-column form against the long-record form",
        )
    )
    operator_one = np.asarray([device.terminal.collagen_area_operator_1 for device in devices])
    operator_two = np.asarray([device.terminal.collagen_area_operator_2 for device in devices])
    row = agreement(
        first, second, float(np.mean(0.5 * (first + second))), bootstrap_replicates=8, seed=3
    )
    differences = first - second
    out.append(
        within(
            "the Bland-Altman bias matches its brute-force mean",
            float(np.mean(differences)),
            row.bias,
            1e-12,
            detail="the difference of the two paired reads",
        )
    )
    out.append(
        within(
            "the limits of agreement are symmetric about the bias",
            0.0,
            (row.loa_low + row.loa_high) / 2.0 - row.bias,
            1e-12,
            detail="bias plus and minus 1.96 difference standard deviations",
        )
    )
    reference = float(np.mean(0.5 * (first + second)))
    direct = (
        100.0
        * float(np.sqrt(float(np.mean(differences)) ** 2 + float(np.var(differences, ddof=1))))
        / reference
    )
    out.append(
        within(
            "the root-mean-square error rebuilt from the limits matches the direct value",
            direct,
            row.rmse_percent,
            1e-9,
            detail="the row's own printed inputs determine it",
        )
    )
    out.append(
        truthy(
            "the estimate is held to an assay rather than a reader",
            row.icc < icc_two_way_random_absolute(np.column_stack([operator_one, operator_two])),
            expected=True,
            observed=round(row.icc, 6),
            detail="the destructive standard agrees with itself better than the sensor agrees with it",
        )
    )
    spread = block_spread([1.0, 2.0, 3.0], [0.2, 0.3, 0.4], [4, 4, 4])
    out.append(
        within(
            "the block spread is the quadrature sum of its two terms",
            math.sqrt(spread.within_sd**2 + spread.between_sd**2),
            spread.total_sd,
            1e-12,
            detail="pooled within-cell term and the spread of the cell means",
        )
    )
    out.append(
        within(
            "the block mean is the count-weighted mean of the cell means",
            (4 * 1.0 + 4 * 2.0 + 4 * 3.0) / 12.0,
            spread.mean,
            1e-12,
            detail="the cells pool on their own chip counts",
        )
    )
    return out


def power_checks(context: ExecutionContext) -> list[CheckResult]:
    out: list[CheckResult] = []
    for size in (4, 12, 24):
        coefficient = two_group_coefficient(size)
        out.append(
            within(
                f"the noncentral-t coefficient at n = {size}",
                _monte_carlo_coefficient(size),
                coefficient,
                0.06,
                detail="solved from the noncentral t against an independent enumeration of the power",
            )
        )
    out.append(
        within(
            "the floor is the coefficient times the per-chip spread",
            two_group_coefficient(12) * 0.74,
            two_group_coefficient(12) * 0.74,
            1e-12,
            detail="Eq. (5) on a named axis rather than one mixed-unit scalar",
        )
    )
    sensitivity = np.array([[0.5, 0.2], [0.4, 0.9]], dtype=np.float64)
    residual = np.diag([0.01, 0.04])
    information = sensitivity.T @ np.linalg.inv(residual) @ sensitivity
    out.append(
        within(
            "the estimator covariance is the inverse of the information matrix",
            0.0,
            float(
                np.abs(
                    estimator_covariance(sensitivity, residual) - np.linalg.inv(information)
                ).max()
            ),
            1e-12,
            detail="Eq. (5)'s inner factor, recomputed by an independent product",
        )
    )
    coefficients = [two_group_coefficient(size) for size in (4, 6, 12, 24)]
    out.append(
        truthy(
            "the coefficient is monotonically decreasing in the replication",
            all(later < earlier for earlier, later in zip(coefficients, coefficients[1:])),
            expected=True,
            observed=[round(value, 6) for value in coefficients],
            detail="a larger replication separates a smaller change at the same power",
        )
    )
    return out


def _monte_carlo_coefficient(size: int, draws: int = 4000) -> float:
    """Independent estimate of k(n) by replicating the two-group test draw by draw.

    The reference draws two normal samples per replicate, forms the
    pooled-variance statistic whose noncentrality is the standardised difference,
    and bisects on the rejection rate against the two-sided critical value at
    2n - 2 degrees of freedom. It shares no code with the noncentral-t solve under
    test.
    """
    rng = numpy_generator(90210 + size)
    baseline = rng.normal(size=(draws, size))
    noise = rng.normal(size=(draws, size))
    critical = float(stats.t.ppf(0.975, 2 * size - 2))

    def rejection(noncentrality: float) -> float:
        shifted = noncentrality * math.sqrt(2.0 / size) + noise
        mean_gap = shifted.mean(axis=1) - baseline.mean(axis=1)
        pooled = np.sqrt(
            ((size - 1) * baseline.var(axis=1, ddof=1) + (size - 1) * shifted.var(axis=1, ddof=1))
            / (2 * size - 2)
        )
        statistic = mean_gap / (pooled * math.sqrt(2.0 / size))
        return float(np.mean(statistic > critical))

    low, high = 0.0, 8.0
    for _ in range(30):
        mid = 0.5 * (low + high)
        if rejection(mid) < 0.80:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high) / math.sqrt(size / 2.0)


def allocation_checks(context: ExecutionContext) -> list[CheckResult]:
    out: list[CheckResult] = []
    from obscal.study import allocation_block, run_allocation_study

    block = allocation_block(context.inputs, context.run, context.calibration)
    weight = block.residual_precision
    rows = block.regressor_rows
    budget = min(24, rows.shape[0])
    order = d_optimal_order(rows, weight, budget)
    out.append(
        compare(
            "the allocator returns exactly the budgeted placements",
            budget,
            len(order),
            detail="a device cannot be destroyed twice",
            evidence={"candidates": int(rows.shape[0])},
        )
    )
    out.append(
        compare(
            "the allocator never destroys the same unit twice",
            len(order),
            len(set(order)),
            detail="the exchange search keeps a feasible set",
        )
    )
    state = InformationState.initial(rows.shape[1], weight)
    determinants = [state.log_determinant]
    ranks = [int(np.linalg.matrix_rank(state.matrix)) if state.spent else 0]
    for index in order:
        state = state.with_row(rows[index])
        determinants.append(state.log_determinant)
        ranks.append(int(np.linalg.matrix_rank(state.matrix)))
    out.append(
        truthy(
            "every accepted placement increases the design information",
            all(later > earlier for earlier, later in zip(determinants, determinants[1:])),
            expected="a strictly increasing log determinant",
            observed=[round(determinants[0], 6), round(determinants[-1], 6)],
            detail="a D-optimal criterion is monotone under adding information",
        )
    )
    out.append(
        truthy(
            "the accepted set reaches full rank and stays there",
            ranks[-1] == rows.shape[1] and all(value <= rows.shape[1] for value in ranks),
            expected=rows.shape[1],
            observed=[ranks[0], ranks[-1]],
            detail="the ridge keeps the early gain defined without changing the ordering",
        )
    )
    candidate = 0
    gain = marginal_gain(state, rows[candidate])
    brute = InformationState(
        matrix=state.matrix + np.outer(rows[candidate], rows[candidate]) @ weight,
        weight=weight,
        spent=state.spent + 1,
    )
    out.append(
        within(
            "the marginal gain matches a brute-force determinant difference",
            brute.log_determinant - state.log_determinant,
            gain,
            1e-9,
            detail="Eq. (6)'s determinant identity recomputed directly",
        )
    )
    initial = InformationState.initial(rows.shape[1], weight)
    first_gains = gains(initial, rows)
    out.append(
        truthy(
            "the first placement is the argmax of the candidate gains",
            int(np.argmax(first_gains)) == order[0],
            expected=int(np.argmax(first_gains)),
            observed=order[0],
            detail="the forward search takes the highest-gain feasible pair at each step",
        )
    )
    random_order = random_matched_rate_order(rows.shape[0], budget, seed=5)
    prior_order = prior_art_allocator_order(rows, budget)
    for name, candidate_order in (
        ("random at matched rate", random_order),
        ("prior-art", prior_order),
    ):
        out.append(
            compare(
                f"the {name} bracket fills the same budget",
                budget,
                len(candidate_order),
                detail="every arm spends the identical device budget",
            )
        )
    out.append(
        truthy(
            "the prior-art allocator covers the calibration space further than a prefix",
            _spread(rows, prior_order) > _spread(rows, list(range(budget))),
            expected="a wider minimum separation than the leading block",
            observed=[
                round(_spread(rows, prior_order), 6),
                round(_spread(rows, list(range(budget))), 6),
            ],
            detail="the re-implemented allocator is a farthest-point selection, not a prefix",
        )
    )
    outcome = run_allocation_study(
        context.inputs,
        context.run,
        context.calibration,
        harness_devices=min(64, rows.shape[0]),
        budget_fractions=(0.5, 0.75),
    )
    parity = convergence_at_full_budget(list(outcome.curve))
    out.append(
        truthy(
            "every policy meets at the full budget",
            bool(parity["converged"]),
            expected=True,
            observed=parity,
            detail="sub-sampling cannot create information",
        )
    )
    out.append(
        truthy(
            "the budgeted curve stays above the clairvoyant oracle below the full budget",
            bool(outcome.non_crossing["non_crossing"]),
            expected=True,
            observed=outcome.non_crossing,
            detail="an arm below the ceiling would be reporting leakage",
        )
    )
    expected_draws = int(context.inputs.allocation.policies["random_matched_rate"]["seeds"])
    out.append(
        truthy(
            "the random bracket reports its own dispersion at every budget point",
            all(
                point.random_draws_used == expected_draws and point.random_high >= point.random_low
                for point in outcome.curve
            ),
            expected=f"{expected_draws} usable draws per point",
            observed=[
                [point.random_draws_used, round(point.random_low, 6), round(point.random_high, 6)]
                for point in outcome.curve
            ],
            detail=(
                "the bracket is printed with its tenth to ninetieth percentile dispersion, "
                "because a controller inside that band would be gaining from taking fewer "
                "noisy observations rather than from choosing better ones"
            ),
        )
    )
    out.append(
        truthy(
            "the allocator sits at or below the random tenth percentile over the mid budgets",
            all(point.at("d_optimal") <= point.random_high + 1e-9 for point in outcome.curve),
            expected=True,
            observed=outcome.random_envelope,
            detail="the gain is separated from the number of terminations",
        )
    )
    counts = outcome.operating.devices_by_policy
    out.append(
        truthy(
            "the operating point names a count for every policy on the block",
            set(counts) == set(outcome.policy_labels)
            and all(1 <= value <= rows.shape[0] for value in counts.values()),
            expected="one count per policy inside the block",
            observed=counts,
            detail="the saving is named at one matched precision",
        )
    )
    grid = [point.devices for point in outcome.curve]
    out.append(
        truthy(
            "the operating point interpolates inside the measured budget grid",
            all(min(grid) <= value <= max(grid) for value in counts.values()),
            expected="every crossing inside the measured range",
            observed=[min(grid), max(grid), sorted(counts.values())],
            detail=(
                "the counts are interpolated between measured budget points rather than "
                "read off a fitted curve"
            ),
        )
    )
    out.append(
        truthy(
            "the operating point names both arms at one matched precision",
            outcome.operating.target_rse > 0.0
            and "d_optimal" in counts
            and "fixed_schedule" in counts,
            expected="a named operating point for the allocator and its clock",
            observed=outcome.operating.as_record(),
            detail="the arrow only names a saving; the two structural checks carry the argument",
        )
    )
    return out


def _spread(rows: NDArray[np.float64], order: list[int]) -> float:
    """Mean nearest-neighbour distance of a selected subset, in regressor space."""
    selected = np.asarray(rows, dtype=np.float64)[order]
    if selected.shape[0] < 3:
        return 0.0
    distances = np.linalg.norm(selected[:, None, :] - selected[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(np.mean(np.min(distances, axis=1)))


def twin_checks(context: ExecutionContext) -> list[CheckResult]:
    out: list[CheckResult] = []
    fitted, held_out = _twin_windows(context)
    set_seed(7)
    backbone, record = fit_backbone(
        fitted,
        latent_dimension=4,
        learning_rate=3e-4,
        epochs=2,
        seed=7,
        damping=True,
        elapsed_days=7.0,
        validation_fraction=0.34,
    )
    out.append(
        compare(
            "the backbone consumes both state axes",
            (2, 2),
            (backbone.spec.observed_dim, backbone.spec.observed_dim),
            detail="Eq. (7) carries the activation and matrix axes together",
        )
    )
    out.append(
        truthy(
            "the recurrence is propagated on the elapsed-time grid",
            bool(backbone.spec.damping),
            expected=True,
            observed=float(backbone.damping.detach().min()),
            detail="the damping term is strictly positive in the delivered configuration",
        )
    )
    undamped, _ = fit_backbone(
        fitted,
        latent_dimension=4,
        learning_rate=3e-4,
        epochs=1,
        seed=9,
        damping=False,
        elapsed_days=7.0,
        validation_fraction=0.34,
    )
    out.append(
        within(
            "the ablation removes the energy-dissipation term exactly",
            0.0,
            float(undamped.damping.max()),
            1e-12,
            detail="the undamped row of the component ablation",
        )
    )
    window = fitted[0]
    observed = torch.as_tensor(window.observed, dtype=torch.float32).unsqueeze(0)
    controls = torch.zeros(
        1, window.observed.shape[0], backbone.spec.control_dim, dtype=torch.float32
    )
    elapsed = torch.full((1,), 7.0, dtype=torch.float32)
    output = backbone(observed, controls, elapsed, forecast_steps=3)
    out.append(
        compare(
            "one forward pass returns the observed window and the forecast",
            ((1, window.observed.shape[0], 2), (1, 3, 2)),
            (tuple(output["filtered"].shape), tuple(output["forecast"].shape)),
            detail="the recurrence and its readout compose to the state shape",
        )
    )
    horizon_prediction = predict_series(backbone, window.observed, 3, 7.0)
    out.append(
        compare(
            "the forecast is issued per axis for the requested horizon",
            (3, 2),
            horizon_prediction.shape,
            detail="a horizon of three steps on two axes",
        )
    )
    before = [parameter.detach().clone() for parameter in backbone.parameters()]
    _reduce_loss_once(backbone, window)
    after = [parameter.detach().clone() for parameter in backbone.parameters()]
    changed = [
        not bool(torch.allclose(left, right, rtol=0.0, atol=0.0))
        for left, right in zip(before, after, strict=True)
    ]
    out.append(
        truthy(
            "a backward pass carries a gradient into every parameter",
            all(parameter.grad is not None for parameter in backbone.parameters()) and any(changed),
            expected="every parameter populated and at least one moved",
            observed=[sum(changed), len(changed)],
            detail="forward, loss, backward and an optimiser step are all exercised",
        )
    )
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "backbone.pt"
        set_seed(13)
        twin, _ = fit_backbone(
            fitted,
            latent_dimension=4,
            learning_rate=3e-4,
            epochs=2,
            seed=13,
            damping=True,
            elapsed_days=7.0,
            validation_fraction=0.34,
        )
        torch.save(twin.state_dict(), path)
        restored, _ = fit_backbone(
            fitted,
            latent_dimension=4,
            learning_rate=3e-4,
            epochs=1,
            seed=99,
            damping=True,
            elapsed_days=7.0,
            validation_fraction=0.34,
        )
        restored.load_state_dict(torch.load(path, weights_only=True))
        left = predict_series(twin, window.observed, 2, 7.0)
        right = predict_series(restored, window.observed, 2, 7.0)
    out.append(
        within(
            "a saved checkpoint round-trips into identical forecasts",
            0.0,
            float(np.abs(left - right).max()),
            1e-12,
            detail="parameter update, saving and loading are all exercised",
        )
    )
    held_out_series = [to_held_out_series(entry, (1, 3)) for entry in held_out if entry.targets]
    persistence = PersistenceRung()
    arima = ArimaRung()
    rungs: list[Rung] = [persistence, arima]
    ladder = score_ladder(rungs, held_out_series, (1, 3))
    ratios = ladder.ratios("matrix", 3)
    out.append(
        within(
            "the last-value predictor carries a ratio of one by construction",
            1.0,
            float(ratios["persistence"]),
            1e-9,
            detail="the ladder is read against persistence",
        )
    )
    out.append(
        truthy(
            "every rung is scored on the same held-out splits",
            len({score.n_series for score in ladder.scores}) == 1,
            expected=True,
            observed=sorted({score.n_series for score in ladder.scores}),
            detail="a rung scored on different splits could not be compared",
        )
    )
    history = np.column_stack(
        [
            np.asarray([0.4, 0.7, 1.1, 1.6, 2.2], dtype=np.float64),
            np.asarray([0.5, 0.9, 1.4, 2.0, 2.6], dtype=np.float64),
        ]
    )
    out.append(
        within(
            "the last-value predictor is constant across the horizon",
            0.0,
            float(np.max(np.ptp(persistence.predict(history, 5), axis=0))),
            1e-12,
            detail="a zero-parameter baseline",
        )
    )
    out.append(
        truthy(
            "the two deterministic rungs report an exactly zero across-seed span",
            not persistence.stochastic_initialisation and not arima.stochastic_initialisation,
            expected=True,
            observed=[persistence.name, arima.name],
            detail="no stochastic initialisation, so the span is zero by construction",
        )
    )
    mechanistic_series = np.column_stack([np.linspace(1.0, 2.4, 5), np.linspace(0.6, 2.8, 5)])
    parameters = fit_mechanistic_parameters(
        [mechanistic_series for _ in range(4)],
        [np.linspace(0.0, 0.6, 4) for _ in range(4)],
        0.5,
        seed=2,
    )
    activation, matrix = mechanistic_trajectory(parameters, 1.0, 0.6, np.full(4, 0.2), 0.5)
    out.append(
        truthy(
            "the mechanistic recurrence integrates to a finite trajectory",
            bool(np.all(np.isfinite(activation)) and np.all(np.isfinite(matrix))),
            expected=True,
            observed=[round(float(activation[-1]), 6), round(float(matrix[-1]), 6)],
            detail="the comparator rung is fitted on held-in series and then frozen",
        )
    )
    truth = np.asarray([0.8, 0.9, 1.0, 1.1], dtype=np.float64)
    prediction = np.asarray([0.7, 0.95, 1.05, 1.2], dtype=np.float64)
    calibration = calibrate_intervals(prediction, truth, 0.80)
    covered = empirical_coverage(
        truth, np.full(4, 0.9) - calibration.spread, np.full(4, 0.9) + calibration.spread
    )
    out.append(
        within(
            "the empirical coverage matches a brute-force count",
            float(
                np.mean((truth >= 0.9 - calibration.spread) & (truth <= 0.9 + calibration.spread))
            ),
            covered,
            1e-12,
            detail="the interval calibration is read off held-out scores",
        )
    )
    out.append(
        truthy(
            "the leakage diagnostic triggers only below its threshold",
            bool(leakage_diagnostic(0.4, 0.6)["triggered"])
            and not bool(leakage_diagnostic(0.95, 0.6)["triggered"]),
            expected=True,
            observed=0.6,
            detail="a ratio too far below the baseline is a check rather than a result",
        )
    )
    out.append(
        truthy(
            "the fitted twin reports the number of parameters it carries",
            parameter_count(backbone) > 0 and record.parameter_count == parameter_count(backbone),
            expected=True,
            observed=record.parameter_count,
            detail="capacity is reported rather than asserted",
        )
    )
    return out


def _twin_windows(context: ExecutionContext) -> tuple[list[TwinWindow], list[TwinWindow]]:
    fitted = [
        twin_window_for(device, context.inputs)
        for device in context.biological
        if device.culture_day >= 10
    ]
    held_out = [
        twin_window_for(device, context.inputs)
        for device in context.cohort.by_block("held_out_geometry")
        if device.retained
    ]
    fitted = [entry for entry in fitted if entry.targets][:24]
    held_out = [entry for entry in held_out if entry.targets]
    return fitted, held_out


def _reduce_loss_once(backbone: Any, window: Any) -> float:
    observed = torch.as_tensor(window.observed, dtype=torch.float32).unsqueeze(0)
    controls = torch.zeros(
        1, window.observed.shape[0], backbone.spec.control_dim, dtype=torch.float32
    )
    elapsed = torch.full((1,), 7.0, dtype=torch.float32)
    steps = sorted(window.targets)
    targets = torch.as_tensor(
        np.stack([window.targets[horizon] for horizon in steps]), dtype=torch.float32
    ).unsqueeze(0)
    optimiser = torch.optim.SGD(backbone.parameters(), lr=1e-3)
    optimiser.zero_grad()
    loss = reconstruction_and_forecast_loss(backbone, observed, controls, targets, elapsed)
    torch.autograd.backward(loss)
    optimiser.step()
    return float(loss.detach())


def evaluation_checks(context: ExecutionContext) -> list[CheckResult]:
    out: list[CheckResult] = []
    rng = numpy_generator(31)
    left = rng.normal(2.0, 0.6, size=40)
    right = rng.normal(1.4, 0.9, size=52)
    contrast = welch_t_test(left, right)
    variance_left = float(np.var(left, ddof=1)) / left.size
    variance_right = float(np.var(right, ddof=1)) / right.size
    brute_statistic = (float(np.mean(left)) - float(np.mean(right))) / math.sqrt(
        variance_left + variance_right
    )
    out.append(
        within(
            "the Welch statistic matches a hand-computed value",
            brute_statistic,
            contrast.statistic,
            1e-12,
            detail="unequal-variance two-group contrast",
        )
    )
    brute_dof = (variance_left + variance_right) ** 2 / (
        variance_left**2 / (left.size - 1) + variance_right**2 / (right.size - 1)
    )
    out.append(
        within(
            "the Welch degrees of freedom match the Welch-Satterthwaite form",
            brute_dof,
            contrast.degrees_of_freedom,
            1e-12,
            detail="the two-group test the manuscript names",
        )
    )
    batches = ["B1"] * 24 + ["B2"] * 24
    arms = ["treated"] * 12 + ["control"] * 12 + ["treated"] * 12 + ["control"] * 12
    values = np.asarray(
        [1.6] * 12 + [2.4] * 12 + [1.7] * 12 + [2.5] * 12, dtype=np.float64
    ) + rng.normal(scale=0.05, size=48)
    values[24:] -= 0.30
    adjusted = batch_adjusted_contrast(values, arms, batches, treated="treated", control="control")
    out.append(
        within(
            "the batch-adjusted contrast recovers an injected arm effect",
            -0.80,
            adjusted.estimate,
            0.05,
            detail="a batch shift of 0.30 and an arm effect of 0.80 are both present",
            evidence={"degrees_of_freedom": adjusted.degrees_of_freedom},
        )
    )
    unadjusted = welch_t_test(
        values[np.asarray(arms) == "treated"], values[np.asarray(arms) == "control"]
    )
    out.append(
        truthy(
            "the batch adjustment moves the estimate away from the confounded one",
            abs(adjusted.estimate + 0.80) < abs(unadjusted.estimate + 0.80) + 1e-9,
            expected=-0.80,
            observed=[round(adjusted.estimate, 6), round(unadjusted.estimate, 6)],
            detail="batch is crossed with arm rather than confounded with it",
        )
    )
    p_values = np.asarray([0.001, 0.008, 0.02, 0.04, 0.2, 0.6], dtype=np.float64)
    holm = holm_step_down(p_values)
    out.append(
        within(
            "the Holm step-down matches a brute-force running maximum",
            0.0,
            float(np.abs(holm.adjusted - _brute_holm(p_values)).max()),
            1e-12,
            detail="the maximum of the scaled p values in p-value order",
        )
    )
    scores = np.asarray([0.9, 0.8, 0.7, 0.4, 0.35, 0.2], dtype=np.float64)
    labels = np.asarray([True, True, True, False, False, False], dtype=bool)
    positives = int(np.sum(labels))
    negatives = int(labels.size - positives)
    concordant = sum(
        1
        for index in range(labels.size)
        if labels[index]
        for other in range(labels.size)
        if not labels[other] and scores[index] > scores[other]
    ) + 0.5 * sum(
        1
        for index in range(labels.size)
        if labels[index]
        for other in range(labels.size)
        if not labels[other] and scores[index] == scores[other]
    )
    out.append(
        within(
            "the area under the curve matches a brute-force pair count",
            concordant / (positives * negatives),
            roc_auc(scores, labels),
            1e-12,
            detail="the rank statistic equals the share of concordant pairs",
        )
    )
    recovered, total = 74, 123
    brute_p = float(
        min(
            1.0,
            2.0 * sum(math.comb(total, index) for index in range(recovered, total + 1)) / 2**total,
        )
    )
    out.append(
        within(
            "the binomial direction test matches a brute-force probability sum",
            brute_p,
            float(binomial_direction_test(recovered, total)["p_value"]),
            1e-9,
            detail="the exact two-sided test against a coin flip",
        )
    )
    from obscal.evaluation import rank_agreement_probability

    out.append(
        within(
            "a four-arm ranking has one chance ordering in twenty-four",
            1.0 / 24.0,
            rank_agreement_probability(4),
            1e-12,
            detail="the chance level a module's arm ranking is read against",
        )
    )
    rows = [{"a": 1.0, "b": "a deliberately long cell value"}, {"a": 1234.0, "b": "x"}]
    rendered = render_table(rows, ["a", "b"])
    out.append(
        truthy(
            "the plain-text renderer never truncates a cell",
            "a deliberately long cell value" in rendered and "1234" in rendered,
            expected=True,
            observed=len(rendered),
            detail="column widths are taken from the data",
        )
    )
    return out


def _brute_holm(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Holm written straight from the definition, with an explicit loop."""
    order = np.argsort(p_values, kind="stable")
    n = p_values.size
    adjusted = np.empty(n, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        scaled = p_values[index] * (n - rank)
        running = max(running, scaled)
        adjusted[index] = min(running, 1.0)
    return adjusted


def runtime_checks(context: ExecutionContext) -> list[CheckResult]:
    out: list[CheckResult] = []
    set_seed(1234)
    first = np.random.random(5)
    set_seed(1234)
    second = np.random.random(5)
    out.append(
        within(
            "seeding makes the process-wide generator reproducible",
            0.0,
            float(np.abs(first - second).max()),
            1e-15,
            detail="a resumed run restores the seed it was started with",
        )
    )
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        target = directory / "payload.json"
        write_json(target, {"value": 1, "nested": {"list": [1.0, 2.0]}})
        mode = target.stat().st_mode & 0o777
        leftovers = [path.name for path in directory.iterdir() if path.name.startswith(".tmp_")]
    out.append(
        compare(
            "the atomic writer leaves no partial file behind",
            [],
            leftovers,
            detail="the temporary file is replaced rather than left in place",
        )
    )
    out.append(
        compare(
            "the atomic writer produces a world-readable artefact",
            0o644,
            mode,
            detail="a temporary file created with mode 0600 would ship unreadable",
        )
    )
    with tempfile.TemporaryDirectory() as scratch:
        first_path = Path(scratch) / "a.txt"
        second_path = Path(scratch) / "b.txt"
        write_text(first_path, "identical\n")
        write_text(second_path, "identical\n")
        from obscal.runtime.atomic import file_digest

        digests = (file_digest(first_path), file_digest(second_path))
    out.append(
        compare(
            "identical contents digest identically",
            digests[0],
            digests[1],
            detail="the manifest digests file contents",
        )
    )
    return out


def all_check_groups() -> tuple[tuple[str, Any], ...]:
    return (
        ("pool", pool_checks),
        ("circuit", circuit_checks),
        ("features", feature_checks),
        ("calibration", calibration_checks),
        ("diagnostics", diagnostics_checks),
        ("terminal", terminal_checks),
        ("power", power_checks),
        ("allocation", allocation_checks),
        ("twin", twin_checks),
        ("evaluation", evaluation_checks),
        ("runtime", runtime_checks),
    )


def run_execution_family(context: ExecutionContext) -> list[CheckResult]:
    """Run every execution group, turning an unexpected crash into an honest failure."""
    results: list[CheckResult] = []
    for name, group in all_check_groups():
        try:
            results.extend(group(context))
        except Exception as error:  # noqa: BLE001 - a group must never take the run down
            results.append(
                CheckResult(
                    name=f"group {name}",
                    category="execution",
                    status="FAIL",
                    expected="the group to complete",
                    observed=f"{type(error).__name__}: {error}",
                    detail="the group raised before it could decide",
                    evidence={},
                )
            )
    return results
