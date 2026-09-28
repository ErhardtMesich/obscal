"""Study layer: joins the device pool to the calibration, allocation and twin.

This is where a run's inputs are assembled from the device records — the sweep
fit, the feature vector, the nuisance block and the terminal panel — so that the
science modules stay free of any file-format concerns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from obscal.allocation import AllocationBlock, AllocationOutcome, run_allocation
from obscal.calibration import (
    CalibrationInputs,
    CalibrationOutcome,
    NuisanceDesign,
    build_nuisance_design,
    fit_pooled,
    run_calibration,
)
from obscal.chip import (
    ChannelGeometry,
    corrected_area_mm2,
    fit_sweep,
    geometric_correction_factor,
    single_frequency_resistance,
)
from obscal.cohort import Cohort, DeviceRecord
from obscal.design import biological_calibration_devices, partition
from obscal.frontend import FEATURE_DIMENSION, FeatureVector, assemble_features, feature_matrix
from obscal.runtime.config import (
    AllocationConfig,
    CalibrationConfig,
    ChipConfig,
    FrontendConfig,
    RunSpec,
    TwinConfig,
)
from obscal.terminal import (
    attenuation_factor,
    block_spread,
    error_covariance,
    split_chip_error,
)
from obscal.twin import TwinWindow

COVERAGE_LADDER_BLOCK = "coverage_ladder"
ACELULAR_BLOCK = "acellular_ladder"
BIOLOGICAL_BLOCK = "biological"
HELD_OUT_GEOMETRY_BLOCK = "held_out_geometry"
MATRIX_READOUT_FEATURE = "log_rb"


@dataclass(frozen=True)
class StudyInputs:
    """Everything a run reads, resolved once."""

    cohort: Cohort
    chip: ChipConfig
    frontend: FrontendConfig
    calibration: CalibrationConfig
    allocation: AllocationConfig
    twin: TwinConfig
    geometry: ChannelGeometry
    correction_factor: float
    area_mm2: float

    @classmethod
    def load(cls, root: Path, cohort: Cohort) -> StudyInputs:
        chip = ChipConfig.load(root)
        frontend = FrontendConfig.load(root)
        geometry = ChannelGeometry.from_config(chip.tissue_channel, chip.perfusion_channel)
        factor = geometric_correction_factor(geometry)
        return cls(
            cohort=cohort,
            chip=chip,
            frontend=frontend,
            calibration=CalibrationConfig.load(root),
            allocation=AllocationConfig.load(root),
            twin=TwinConfig.load(root),
            geometry=geometry,
            correction_factor=factor,
            area_mm2=corrected_area_mm2(geometry, factor),
        )


def terminal_sweep_index(device: DeviceRecord) -> int:
    """The last sweep before harvest, the one the terminal panel pairs with."""
    if device.sweeps.n_sweeps == 0:
        raise ValueError("a device must carry at least one sweep")
    return device.sweeps.n_sweeps - 1


def feature_vector_for(
    device: DeviceRecord, inputs: StudyInputs, *, multi_frequency: bool = True
) -> FeatureVector:
    """Fit the terminal sweep and assemble the Eq. (2) feature vector."""
    index = terminal_sweep_index(device)
    corrected = device.sweeps.at(index) * inputs.correction_factor
    fit = fit_sweep(device.sweeps.frequencies_hz, corrected)
    return assemble_features(
        fit,
        inputs.area_mm2,
        float(device.optical.matrix_texture_x1e3[index]),
        float(device.optical.effluent_pro_collagen_ng_per_ml[index]),
        float(device.optical.coverage_percent[index]),
        device_id=device.device_id,
        culture_day=float(device.culture_day),
        multi_frequency=multi_frequency,
    )


def comparator_resistance(device: DeviceRecord, inputs: StudyInputs) -> float:
    """The single-frequency comparator channel, reported as resistance only."""
    index = terminal_sweep_index(device)
    corrected = device.sweeps.at(index) * inputs.correction_factor
    return single_frequency_resistance(
        float(inputs.frontend.comparator["frequency_hz"]),
        corrected,
        device.sweeps.frequencies_hz,
    )


def calibration_blocks(inputs: StudyInputs, *, include_ladders: bool = True) -> list[DeviceRecord]:
    """The devices the pooled calibration spends, in block order."""
    cohort = inputs.cohort
    blocks = [cohort.by_block(BIOLOGICAL_BLOCK)]
    blocks = [tuple(device for device in block if device.retained) for block in blocks]
    if include_ladders:
        blocks.append(tuple(d for d in cohort.by_block(ACELULAR_BLOCK) if d.retained))
        blocks.append(tuple(d for d in cohort.by_block(COVERAGE_LADDER_BLOCK) if d.retained))
    return [device for block in blocks for device in block]


def nuisance_design(devices: list[DeviceRecord]) -> NuisanceDesign:
    """Artefact covariates as standardised numerics plus reference-coded categories."""
    numeric = {
        "temperature": np.asarray(
            [device.nuisance.temperature_c_mean for device in devices], dtype=np.float64
        ),
        "medium_change_indicator": np.asarray(
            [float(device.nuisance.medium_change_count) for device in devices], dtype=np.float64
        ),
        "hours_since_electrode_wetting": np.asarray(
            [device.nuisance.hours_since_electrode_wetting for device in devices], dtype=np.float64
        ),
    }
    categorical = {
        "batch": [device.nuisance.batch for device in devices],
        "operator": [device.nuisance.operator for device in devices],
        "geometry": [device.nuisance.geometry for device in devices],
        "perfusion_state": ["perfused" if device.perfusion else "static" for device in devices],
    }
    return build_nuisance_design(numeric, categorical)


def nuisance_matrix(devices: list[DeviceRecord]) -> NDArray[np.float64]:
    return nuisance_design(devices).matrix


def measured_state(devices: list[DeviceRecord]) -> NDArray[np.float64]:
    return np.asarray(
        [
            [device.terminal.acta2_fold, device.terminal.hydroxyproline_ug_per_chip]
            for device in devices
        ],
        dtype=np.float64,
    )


def split_chip_error_estimate(devices: list[DeviceRecord]) -> Any:
    """Sigma_u on the matrix axis, identified from the paired half reads."""
    first = np.asarray(
        [device.terminal.hydroxyproline_half_1 for device in devices], dtype=np.float64
    )
    second = np.asarray(
        [device.terminal.hydroxyproline_half_2 for device in devices], dtype=np.float64
    )
    return split_chip_error(first, second)


def activation_design_error(devices: list[DeviceRecord]) -> float:
    """Activation has one value per device, so its Sigma_u entry is a design value."""
    reads = np.asarray([device.terminal.acta2_fold for device in devices], dtype=np.float64)
    return float(np.std(reads, ddof=1))


def calibration_inputs(
    inputs: StudyInputs, run: RunSpec, devices: list[DeviceRecord]
) -> CalibrationInputs:
    """Assemble the calibration arrays from the device records."""
    vectors = [
        feature_vector_for(device, inputs, multi_frequency=run.toggles.multi_frequency_front_end)
        for device in devices
    ]
    split = split_chip_error_estimate(devices)
    state = measured_state(devices)
    matrix_error = (
        split.single_half_read_error
        if run.toggles.split_chip_replication
        else float(np.std(state[:, 1], ddof=1))
    )
    # The activation axis is read once per device, so its Sigma_u entry is the
    # design-time analytical coefficient applied at the spread actually achieved.
    assay_cv = inputs.calibration.attenuation["design_assay_cv"]
    activation_error = float(np.std(state[:, 0], ddof=1)) * float(assay_cv["activation_split"])
    sigma_u = error_covariance(
        matrix_single_read_error=matrix_error,
        activation_design_value=activation_error,
        matrix_design_value=0.0,
    )
    return CalibrationInputs(
        features=feature_matrix(vectors),
        state_measured=measured_state(devices),
        coverage=np.asarray([vector.coverage_percent for vector in vectors], dtype=np.float64),
        times=np.asarray([device.culture_day for device in devices], dtype=np.float64),
        nuisance=nuisance_matrix(devices),
        feature_names=vectors[0].names,
        strata=tuple(device.arm for device in devices),
        error_covariance=sigma_u,
        axis_units=("fold_change", "ug_hydroxyproline_per_chip"),
    )


def run_calibration_study(
    inputs: StudyInputs, run: RunSpec, *, bootstrap_replicates: int | None = None
) -> tuple[CalibrationOutcome, list[DeviceRecord]]:
    devices = calibration_blocks(inputs, include_ladders=run.toggles.decoupling_ladders)
    arrays = calibration_inputs(inputs, run, devices)
    outcome = run_calibration(
        arrays,
        inputs.calibration,
        include_coverage=run.toggles.coverage_covariate,
        include_interaction=run.toggles.two_axis_state,
        has_acellular_rows=run.toggles.decoupling_ladders,
        bootstrap_replicates=bootstrap_replicates,
        seed=run.seed,
    )
    return outcome, devices


def allocation_block(
    inputs: StudyInputs, run: RunSpec, outcome: CalibrationOutcome
) -> AllocationBlock:
    """Build the allocator's candidate pool from the biological block alone."""
    devices = [device for device in inputs.cohort.by_block(BIOLOGICAL_BLOCK) if device.retained]
    arrays = calibration_inputs(inputs, run, devices)
    pooled = fit_pooled(
        features=arrays.features,
        state_measured=arrays.state_measured,
        coverage=arrays.coverage,
        times=arrays.times,
        nuisance=arrays.nuisance,
        feature_names=arrays.feature_names,
        time_basis_columns=3,
        include_coverage=run.toggles.coverage_covariate,
        include_interaction=run.toggles.two_axis_state,
        n_biological=int(arrays.features.shape[0]),
    )
    # The candidate row is the state the device would contribute, weighted by the
    # calibrated front end's own Fisher information for that state, so the
    # criterion acts on the estimand rather than on the whole nuisance design.
    precision = (
        pooled.sensitivity_ols.T
        @ np.linalg.inv(pooled.residual_covariance)
        @ pooled.sensitivity_ols
    )
    centred_state = arrays.state_measured - arrays.state_measured.mean(axis=0, keepdims=True)
    regressor_rows = np.asarray(centred_state, dtype=np.float64)
    return AllocationBlock(
        regressor_rows=regressor_rows,
        features=arrays.features,
        state_measured=arrays.state_measured,
        coverage=arrays.coverage,
        times=arrays.times,
        nuisance=arrays.nuisance,
        strata=arrays.strata,
        arms=tuple(device.arm for device in devices),
        feature_names=arrays.feature_names,
        residual_precision=precision,
        error_covariance=arrays.error_covariance,
        minimum_units=int(pooled.design.width) + 4,
    )


def run_allocation_study(
    inputs: StudyInputs,
    run: RunSpec,
    outcome: CalibrationOutcome,
    *,
    harness_devices: int | None = None,
    budget_fractions: Sequence[float] | None = None,
) -> AllocationOutcome:
    block = allocation_block(inputs, run, outcome)
    truncated = harness_devices is not None and harness_devices < block.regressor_rows.shape[0]
    if harness_devices is not None and harness_devices < block.regressor_rows.shape[0]:
        keep = np.arange(harness_devices, dtype=np.int64)
        block = AllocationBlock(
            regressor_rows=block.regressor_rows[keep],
            features=block.features[keep],
            state_measured=block.state_measured[keep],
            coverage=block.coverage[keep],
            times=block.times[keep],
            nuisance=block.nuisance[keep],
            strata=tuple(block.strata[index] for index in keep),
            arms=tuple(block.arms[index] for index in keep),
            feature_names=block.feature_names,
            residual_precision=block.residual_precision,
            error_covariance=block.error_covariance,
            minimum_units=block.minimum_units,
        )
    lambdas = outcome.lambda_by_axis
    readout_lambda = float(lambdas["matrix"])
    return run_allocation(
        block,
        inputs.allocation,
        readout_feature=MATRIX_READOUT_FEATURE,
        lambda_value=readout_lambda,
        time_basis_columns=3,
        include_coverage=run.toggles.coverage_covariate,
        include_interaction=run.toggles.two_axis_state,
        seed=run.seed,
        delivered_devices=None if truncated else int(outcome.pooled.n_units),
        budget_fractions=budget_fractions,
    )


def twin_windows(inputs: StudyInputs, run: RunSpec) -> tuple[list[TwinWindow], list[TwinWindow]]:
    """Split the biological block into held-in fitting devices and held-out windows."""
    devices = [device for device in inputs.cohort.by_block(BIOLOGICAL_BLOCK) if device.retained]
    windows = [twin_window_for(device, inputs) for device in devices]
    held_out = [device for device in inputs.cohort.by_block("held_out_geometry") if device.retained]
    holdout_windows = [twin_window_for(device, inputs) for device in held_out]
    return windows, holdout_windows


def twin_window_for(device: DeviceRecord, inputs: StudyInputs) -> TwinWindow:
    """The observed front-end series and the destructive values it forecasts to.

    Sensing runs continuously, so the twin sees the curated state estimate on the
    dense day grid up to the moment treatment begins; the held-out values are the
    destructive timepoints the experiment can only buy by consuming a device.
    """
    issued = float(inputs.twin.forecast["issued_from_culture_day"])
    horizons = tuple(int(value) for value in inputs.twin.horizons_days)
    observed_mask = device.series_days <= issued + 1e-9
    observed = np.asarray(device.front_end_state[observed_mask], dtype=np.float64)
    if observed.shape[0] < 2:
        raise ValueError(f"{device.device_id} has too short an observed window")
    control = np.diff(observed[:, 0], prepend=float(observed[0, 0]))
    targets: dict[int, NDArray[np.float64]] = {}
    for horizon in horizons:
        target_day = issued + float(horizon)
        if target_day <= float(device.culture_day) + 1e-9:
            targets[horizon] = np.array(
                [device.terminal.acta2_fold, device.terminal.hydroxyproline_ug_per_chip],
                dtype=np.float64,
            )
    return TwinWindow(
        device_id=device.device_id,
        observed=observed,
        controls=np.asarray(control, dtype=np.float64),
        targets=targets,
    )


def _cell_series(
    devices: list[DeviceRecord], attribute: str
) -> tuple[list[float], list[float], list[int]]:
    groups: dict[tuple[str, int], list[DeviceRecord]] = {}
    for device in devices:
        groups.setdefault((device.arm, device.culture_day), []).append(device)
    means: list[float] = []
    sds: list[float] = []
    counts: list[int] = []
    for key in sorted(groups):
        values = np.asarray(
            [float(getattr(device.terminal, attribute)) for device in groups[key]],
            dtype=np.float64,
        )
        means.append(float(np.mean(values)))
        sds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
        counts.append(int(values.size))
    return means, sds, counts


def spread_summary(inputs: StudyInputs) -> dict[str, float]:
    """Pool the trajectory cells to the block spreads the reproducibility table uses."""
    devices = [device for device in inputs.cohort.by_block(BIOLOGICAL_BLOCK) if device.retained]
    hydroxyproline = block_spread(*_cell_series(devices, "hydroxyproline_ug_per_chip"))
    collagen = block_spread(*_cell_series(devices, "collagen_area_percent"))
    activation = block_spread(*_cell_series(devices, "acta2_fold"))
    return {
        "hydroxyproline_mean": hydroxyproline.mean,
        "hydroxyproline_spread": hydroxyproline.total_sd,
        "hydroxyproline_cv_pct": hydroxyproline.cv_percent,
        "collagen_mean": collagen.mean,
        "collagen_spread": collagen.total_sd,
        "collagen_cv_pct": collagen.cv_percent,
        "activation_mean": activation.mean,
        "activation_spread": activation.total_sd,
        "activation_cv_pct": activation.cv_percent,
        "n_cells": float(len(_cell_series(devices, "acta2_fold")[0])),
        "n_units": float(hydroxyproline.n_units),
    }


def delivered_attenuation(inputs: StudyInputs) -> dict[str, float]:
    """The delivered attenuation factors, computed from the measured spreads.

    Only the matrix axis has paired half reads, so only its factor is identified;
    the other two are design values at the spread achieved and are labelled as
    such rather than presented as measurements.
    """
    spread = spread_summary(inputs)
    devices = [device for device in inputs.cohort.by_block(BIOLOGICAL_BLOCK) if device.retained]
    split = split_chip_error_estimate(devices)
    matrix_factor = attenuation_factor(
        spread["hydroxyproline_spread"], split.single_half_read_error
    )
    return {
        "matrix_lambda_single_half": matrix_factor,
        "matrix_lambda_halves_averaged": attenuation_factor(
            spread["hydroxyproline_spread"], split.single_half_read_error / np.sqrt(2.0)
        ),
        "matrix_single_read_error": split.single_half_read_error,
        "matrix_half_mean_error": split.half_mean_error,
        "matrix_within_chip_fraction": split.within_chip_fraction,
        "activation_design_cv_pct": spread["activation_cv_pct"],
        "collagen_design_cv_pct": spread["collagen_cv_pct"],
    }


def block_census(inputs: StudyInputs) -> dict[str, Any]:
    """The release's own chip census, against the design's nominal partition."""
    retained = inputs.cohort.retained
    blocks: dict[str, int] = {}
    for device in retained:
        blocks[device.block] = blocks.get(device.block, 0) + 1
    return {
        "pool": len(inputs.cohort),
        "retained": len(retained),
        "withdrawn": len(inputs.cohort) - len(retained),
        "withdrawal_share_pct": 100.0 * (len(inputs.cohort) - len(retained)) / len(inputs.cohort),
        "blocks": blocks,
        "design_biological_devices": biological_calibration_devices(),
        "design_full_pool": partition.TOTAL_POOL_DEVICES,
        "feature_dimension": FEATURE_DIMENSION,
    }
