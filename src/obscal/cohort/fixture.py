"""The harness device pool the pipeline executes on.

The article does not deposit per-device rows and carries no executable
implementation, so the release runs its pipeline on a schema-compatible,
harness-supplied device pool built from this specification. Every number the
pool yields is the release's own; no manuscript table value is claimed from it.
The vocabulary throughout is that of a fixture, and the pool is never described
as data collected from the platform.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from obscal.chip.circuit import CircuitParameters, impedance_model
from obscal.cohort.schema import (
    STATE_AXES,
    Cohort,
    DeviceRecord,
    NuisanceRecord,
    OpticalSeries,
    SweepSet,
    TerminalPanel,
    cell_day_grid,
)
from obscal.design.roster import ROSTER, arm as roster_arm
from obscal.design.surface import DOSE_LEVELS, SCHEDULE_LABELS
from obscal.design.withdrawal import classify_device
from obscal.runtime.seed import numpy_generator

FREQUENCIES_HZ: tuple[float, ...] = (100.0, 500.0, 1000.0, 4000.0, 10000.0, 40000.0, 100000.0)
SWEEP_CADENCE_HOURS = 0.5
SWEEP_INTERVAL_HOURS = 0.5
PLAN_BLOCKS: tuple[str, ...] = (
    "biological",
    "acellular_ladder",
    "coverage_ladder",
    "cell_free",
    "held_out_geometry",
    "cross_line",
    "static_counterpart",
)


@dataclass(frozen=True)
class DevicePlan:
    """What one device is going to be, before anything is generated."""

    device_id: str
    block: str
    arm: str
    culture_day: int
    treatment_day: int | None
    batch: str
    operator: str
    geometry: str
    perfusion: bool
    cell_source: str
    hepatocyte_line: str
    coverage_percent: float
    acellular_level: int | None = None
    schedule: str = "single_bolus"
    dose_cells: float = 0.0


@dataclass(frozen=True)
class FixtureSpec:
    """The pool's own generating parameters, read from ``configs/cohort``."""

    payload: Mapping[str, Any]
    seed: int

    @classmethod
    def load(cls, root: Path, seed: int | None = None) -> FixtureSpec:
        with open(root / "configs" / "cohort" / "fixture.yaml", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        cohort_block = cast("Mapping[str, Any]", payload["cohort"])
        resolved = int(seed if seed is not None else cohort_block["seed"])
        return cls(payload=cast("Mapping[str, Any]", payload), seed=resolved)

    def section(self, name: str) -> Mapping[str, Any]:
        return cast("Mapping[str, Any]", self.payload[name])

    def value(self, section: str, key: str) -> float:
        return float(cast("Mapping[str, Any]", self.payload[section])[key])


def _pool_sizes(spec: FixtureSpec) -> dict[str, int]:
    pool = spec.section("pool")
    return {name: int(value) for name, value in pool.items() if name.endswith("_devices")}


def plan_devices(spec: FixtureSpec) -> list[DevicePlan]:
    """Lay out every device of the pool, balanced across batches and operators."""
    sizes = _pool_sizes(spec)
    pool = spec.section("pool")
    batches = int(pool["batches"])
    operators = int(pool["operators"])
    rng = numpy_generator(spec.seed)
    plans: list[DevicePlan] = []
    counter = 0

    def next_slot() -> tuple[str, str]:
        nonlocal counter
        slot = counter
        counter += 1
        return f"B{slot % batches + 1}", f"OP{slot % operators + 1}"

    for cell in [cell for entry in ROSTER for cell in entry.cells]:
        for _ in range(cell.chips):
            batch, operator = next_slot()
            plans.append(
                DevicePlan(
                    device_id="",
                    block="biological",
                    arm=cell.arm,
                    culture_day=cell.culture_day,
                    treatment_day=cell.treatment_day,
                    batch=batch,
                    operator=operator,
                    geometry="development",
                    perfusion=True,
                    cell_source=roster_arm(cell.arm).source,
                    hepatocyte_line="Hep_G2",
                    coverage_percent=float(rng.normal(88.0, 3.4)),
                )
            )
    for level in range(6):
        for _ in range(sizes["acellular_devices"] // 6):
            batch, operator = next_slot()
            plans.append(
                DevicePlan(
                    device_id="",
                    block="acellular_ladder",
                    arm="L0",
                    culture_day=21,
                    treatment_day=None,
                    batch=batch,
                    operator=operator,
                    geometry="development",
                    perfusion=True,
                    cell_source="cell_free",
                    hepatocyte_line="none",
                    coverage_percent=float(rng.normal(88.0, 3.4)),
                    acellular_level=level,
                )
            )
    for level in range(5):
        for _ in range(sizes["coverage_devices"] // 5):
            batch, operator = next_slot()
            plans.append(
                DevicePlan(
                    device_id="",
                    block="coverage_ladder",
                    arm="CL",
                    culture_day=21,
                    treatment_day=None,
                    batch=batch,
                    operator=operator,
                    geometry="development",
                    perfusion=True,
                    cell_source="LX-2 + Hep G2",
                    hepatocyte_line="Hep_G2",
                    coverage_percent=float(
                        spec.section("front_end")["coverage"]["ladder_levels"][level]
                    ),
                )
            )
    for _ in range(sizes["cell_free_devices"]):
        batch, operator = next_slot()
        plans.append(
            DevicePlan(
                device_id="",
                block="cell_free",
                arm="CF",
                culture_day=21,
                treatment_day=None,
                batch=batch,
                operator=operator,
                geometry="development",
                perfusion=True,
                cell_source="cell_free",
                hepatocyte_line="none",
                coverage_percent=0.0,
            )
        )
    for _ in range(sizes["held_out_geometry_devices"]):
        batch, operator = next_slot()
        plans.append(
            DevicePlan(
                device_id="",
                block="held_out_geometry",
                arm="S2",
                culture_day=21,
                treatment_day=14,
                batch=batch,
                operator=operator,
                geometry="held_out_layout",
                perfusion=True,
                cell_source="LX-2 + Hep G2",
                hepatocyte_line="Hep_G2",
                coverage_percent=float(rng.normal(88.0, 3.4)),
            )
        )
    for _ in range(sizes["cross_line_devices"]):
        batch, operator = next_slot()
        plans.append(
            DevicePlan(
                device_id="",
                block="cross_line",
                arm="S4",
                culture_day=21,
                treatment_day=14,
                batch=batch,
                operator=operator,
                geometry="development",
                perfusion=True,
                cell_source="bone_marrow",
                hepatocyte_line="Huh7",
                coverage_percent=float(rng.normal(88.0, 3.4)),
            )
        )
    for index in range(sizes["static_counterpart_devices"]):
        batch, operator = next_slot()
        plans.append(
            DevicePlan(
                device_id="",
                block="static_counterpart",
                arm="S2" if index % 2 == 0 else "S4",
                culture_day=21 if index % 4 < 2 else 14,
                treatment_day=14 if index % 4 < 2 else 7,
                batch=batch,
                operator=operator,
                geometry="development",
                perfusion=False,
                cell_source="LX-2 + Hep G2",
                hepatocyte_line="Hep_G2",
                coverage_percent=float(rng.normal(88.0, 3.4)),
            )
        )
    doses = list(DOSE_LEVELS)
    schedules = list(SCHEDULE_LABELS)
    for index, plan in enumerate(plans):
        schedule = schedules[index % len(schedules)]
        dose = doses[index % len(doses)]
        plans[index] = DevicePlan(**{**plan.__dict__, "schedule": schedule, "dose_cells": dose})
    return [
        DevicePlan(**{**plan.__dict__, "device_id": f"CHIP{index + 1:04d}"})
        for index, plan in enumerate(plans)
    ]


def state_trajectory(
    plan: DevicePlan, spec: FixtureSpec, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate the pool's own activation and deposition dynamics on the day grid."""
    trajectory = spec.section("trajectory")
    series = spec.section("series")
    start = float(series["first_culture_day"])
    step = float(series["step_days"])
    days = cell_day_grid(start, float(plan.culture_day), step)
    treatment_start = float(series["treatment_start_day"])
    ceiling = float(trajectory["activation_injury_ceiling_fold"])
    rise = float(trajectory["activation_injury_rise_per_day"])
    decay = float(trajectory["activation_treated_decay_per_day"])
    baseline = float(trajectory["activation_baseline_fold"])
    deposition = float(trajectory["matrix_deposition_per_day"])
    threshold = float(trajectory["matrix_threshold_fold"])
    turnover = float(trajectory["matrix_turnover_per_day"])
    matrix_zero = float(trajectory["matrix_baseline_ug"])

    activation_factor = 1.0
    if plan.block == "static_counterpart":
        activation_factor = 1.0
    if plan.arm in ("S1", "S6", "CL", "L0", "CF"):
        response = 0.0
    elif plan.arm == "S3":
        response = 0.72
    else:
        response = 1.0
    treated = plan.arm in ("S4", "S5")

    activation = np.empty(days.size, dtype=np.float64)
    matrix = np.empty(days.size, dtype=np.float64)
    current_a = baseline
    current_m = matrix_zero
    for index, day in enumerate(days):
        if day < treatment_start:
            target = baseline + response * (ceiling - baseline) * (
                1.0 - np.exp(-rise * (day - start))
            )
            current_a = target
        else:
            if treated:
                current_a = current_a * float(np.exp(-decay * step))
            else:
                current_a = min(current_a + rise * step * response * (ceiling - current_a), ceiling)
        drive = max(current_a - threshold, 0.0) * activation_factor
        current_m = current_m + step * (deposition * drive - turnover * current_m)
        if plan.arm == "S6":
            current_m += step * float(trajectory["cell_only_matrix_offset_ug"]) / 21.0
        activation[index] = current_a
        matrix[index] = current_m
    if plan.block == "static_counterpart":
        matrix = matrix * float(trajectory["static_matrix_factor"])
    if plan.arm == "S5":
        matrix = matrix * float(trajectory["cross_source_matrix_factor"])
    if plan.hepatocyte_line == "Huh7":
        matrix = matrix * float(trajectory["cross_line_matrix_factor"])
    if plan.block == "acellular_ladder":
        activation = np.zeros_like(activation)
        matrix = np.linspace(0.15, 5.4, 6)[int(plan.acellular_level or 0)] * np.ones_like(matrix)
        matrix = matrix + rng.normal(0.0, 0.05, size=matrix.shape)
    if plan.block == "cell_free":
        activation = np.zeros_like(activation)
        matrix = np.zeros_like(matrix)
    return activation, matrix


def standardised(values: NDArray[np.float64], centre: float, spread: float) -> NDArray[np.float64]:
    if spread <= 0.0:
        raise ValueError("a standardisation needs a positive spread")
    return (values - float(centre)) / float(spread)


def build_features(
    activation: NDArray[np.float64],
    matrix: NDArray[np.float64],
    coverage_percent: NDArray[np.float64],
    spec: FixtureSpec,
    rng: np.random.Generator,
) -> dict[str, NDArray[np.float64]]:
    """Forward-model the five sensor features from the state, in standardised units."""
    front = spec.section("front_end")
    scale = cast("Mapping[str, float]", front["state_scale"])
    a_std = standardised(
        activation, float(scale["activation_centre"]), float(scale["activation_spread"])
    )
    m_std = standardised(matrix, float(scale["matrix_centre"]), float(scale["matrix_spread"]))
    c_std = standardised(
        coverage_percent, float(scale["coverage_centre"]), float(scale["coverage_spread"])
    )
    intercept = cast("Mapping[str, float]", front["intercept"])
    s_a = cast("Mapping[str, float]", front["sensitivity_activation"])
    s_m = cast("Mapping[str, float]", front["sensitivity_matrix"])
    s_c = cast("Mapping[str, float]", front["sensitivity_coverage"])
    noise = cast("Mapping[str, float]", front["feature_noise_sd"])
    features: dict[str, NDArray[np.float64]] = {}
    for name in intercept:
        signal = (
            float(intercept[name])
            + float(s_a[name]) * a_std
            + float(s_m[name]) * m_std
            + float(s_c[name]) * c_std
        )
        features[name] = signal + rng.normal(0.0, float(noise[name]), size=signal.shape)
    return features


def build_sweeps(
    features: Mapping[str, NDArray[np.float64]],
    elapsed_days: NDArray[np.float64],
    spec: FixtureSpec,
    rng: np.random.Generator,
    *,
    geometry_shift: bool = False,
) -> SweepSet:
    """Write the measured sweep block whose circuit fit returns the target features."""
    circuit = spec.section("circuit")
    shift = 1.0 + (
        float(circuit["geometry_baseline_shift_percent"]) / 100.0 if geometry_shift else 0.0
    )
    frequencies = np.asarray(FREQUENCIES_HZ, dtype=np.float64)
    grid = np.arange(elapsed_days.size, dtype=np.int64)
    real = np.empty((elapsed_days.size, frequencies.size), dtype=np.float64)
    imag = np.empty_like(real)
    for row in grid:
        parameters = CircuitParameters(
            r_ext_ohm=float(circuit["series_resistance_ohm"]),
            r_barrier_ohm=float(np.exp(features["log_rb"][row])) * shift,
            c_membrane_f=float(np.exp(features["log_cm"][row])),
            cpe_magnitude=1.0e-5,
            cpe_exponent=float(features["alpha"][row]),
        )
        ideal = impedance_model(parameters, frequencies)
        gain = 1.0 + rng.normal(0.0, 0.004, size=frequencies.size)
        phase = rng.normal(0.0, 0.004, size=frequencies.size)
        measured = ideal * gain * np.exp(1j * phase)
        real[row] = measured.real
        imag[row] = measured.imag
    return SweepSet(
        frequencies_hz=frequencies,
        elapsed_hours=elapsed_days * 24.0,
        real_ohm=real,
        imag_ohm=imag,
    )


def build_optical(
    features: Mapping[str, NDArray[np.float64]],
    elapsed_days: NDArray[np.float64],
    coverage_percent: float,
) -> OpticalSeries:
    return OpticalSeries(
        elapsed_hours=elapsed_days * 24.0,
        coverage_percent=np.full(elapsed_days.size, float(coverage_percent)),
        matrix_texture_x1e3=np.asarray(features["matrix_texture"], dtype=np.float64),
        effluent_pro_collagen_ng_per_ml=np.asarray(
            np.clip(features["effluent_pro_collagen"], 0.0, None), dtype=np.float64
        ),
    )


def build_terminal(
    activation_terminal: float,
    matrix_terminal: float,
    spec: FixtureSpec,
    rng: np.random.Generator,
) -> TerminalPanel:
    """Draw the destructive panel, with the split-chip and operator replicates."""
    terminal = spec.section("terminal")
    hydroxy = cast("Mapping[str, float]", terminal["hydroxyproline"])
    area = cast("Mapping[str, float]", terminal["collagen_area"])
    activation_block = cast("Mapping[str, float]", terminal["activation"])
    col1a1_block = cast("Mapping[str, float]", terminal["col1a1"])
    function = cast("Mapping[str, float]", terminal["function"])

    half_error = float(hydroxy["single_half_read_error_ug"])
    first = float(matrix_terminal) + rng.normal(0.0, half_error)
    second = float(matrix_terminal) + rng.normal(0.0, half_error)
    mean_hyp = 0.5 * (first + second)

    per_ug = float(area["percent_per_ug"])
    area_value = mean_hyp * per_ug
    operator_sd = float(area["operator_sd_percent"])
    operator_one = (
        area_value + float(area["operator_bias_percent"]) / 2.0 + rng.normal(0.0, operator_sd)
    )
    operator_two = (
        area_value - float(area["operator_bias_percent"]) / 2.0 + rng.normal(0.0, operator_sd)
    )

    activation_error = float(activation_block["read_error_fold"])
    alpha = max(float(activation_terminal) + rng.normal(0.0, activation_error), 0.0)
    ratio = float(col1a1_block["ratio_to_acta2"])
    col1 = max(alpha * ratio + rng.normal(0.0, float(col1a1_block["noise_sd_fold"])), 0.0)

    return TerminalPanel(
        hydroxyproline_ug_per_chip=mean_hyp,
        hydroxyproline_half_1=first,
        hydroxyproline_half_2=second,
        collagen_area_percent=0.5 * (operator_one + operator_two),
        collagen_area_operator_1=operator_one,
        collagen_area_operator_2=operator_two,
        acta2_fold=alpha,
        col1a1_fold=col1,
        albumin_ng_per_chip_per_day=max(
            float(function["albumin_base"])
            + float(function["albumin_injury_slope"]) * (alpha - 1.0)
            + rng.normal(0.0, float(function["albumin_noise_sd"])),
            0.0,
        ),
        urea_ug_per_chip_per_day=max(
            float(function["urea_base"])
            + float(function["urea_injury_slope"]) * max(0.0, mean_hyp - 1.0)
            + rng.normal(0.0, float(function["urea_noise_sd"])),
            0.0,
        ),
        pro_collagen_ng_per_ml=float(np.clip(rng.normal(34.0, 7.0), 1.0, None)),
    )


def build_nuisance(plan: DevicePlan, spec: FixtureSpec, rng: np.random.Generator) -> NuisanceRecord:
    nuisance = spec.section("nuisance")
    media_lot = int(rng.integers(1, 4))
    return NuisanceRecord(
        temperature_c_mean=float(
            rng.normal(float(nuisance["temperature_mean_c"]), float(nuisance["temperature_sd_c"]))
        ),
        medium_change_count=int(rng.poisson(float(nuisance["medium_change_count_mean"]))),
        hours_since_electrode_wetting=float(
            max(
                rng.normal(
                    float(nuisance["hours_since_wetting_mean"]),
                    float(nuisance["hours_since_wetting_sd"]),
                ),
                1.0,
            )
        ),
        perfusion_state="perfused" if plan.perfusion else "static",
        geometry=plan.geometry,
        batch=plan.batch,
        operator=plan.operator,
        media_lot=media_lot,
        passage_stellate=int(rng.integers(6, 13)),
        passage_parenchymal=int(rng.integers(8, 17)),
        lot_stromal=int(rng.integers(1, 3)),
    )


def build_inclusion_fields(
    plan: DevicePlan, spec: FixtureSpec, rng: np.random.Generator
) -> str | None:
    """Apply the pre-declared inclusion rules in the order the ledger lists them."""
    nuisance = spec.section("nuisance")
    bubble = bool(rng.random() < 0.052)
    coverage = float(
        plan.coverage_percent if plan.coverage_percent > 0.0 else rng.normal(88.0, 3.4)
    )
    if rng.random() < 0.04:
        drift = float(rng.normal(0.0, 1.0) * 9.0 + 24.0)
    else:
        drift = float(
            rng.normal(
                float(nuisance["drift_ohm_per_day_mean"]),
                float(nuisance["drift_ohm_per_day_sd"]),
            )
        )
    pressure_decay = float(abs(rng.normal(0.0085, 0.0125)))
    half_ratio = float(np.clip(rng.normal(0.5, 0.046), 0.0, 1.0))
    hydroxyproline = float(rng.gamma(6.0, 0.36))
    return classify_device(
        bubble_logged=bubble,
        coverage_percent=coverage,
        pressure_decay_per_min=pressure_decay,
        electrode_drift_ohm_per_day=drift,
        half_mass_ratio=half_ratio,
        hydroxyproline_ug_per_chip=hydroxyproline,
    )


def build_device(plan: DevicePlan, spec: FixtureSpec, rng: np.random.Generator) -> DeviceRecord:
    """Generate one device: trajectory, sensed block, terminal panel and ledger entry."""
    series = spec.section("series")
    trajectory = spec.section("trajectory")
    activation, matrix = state_trajectory(plan, spec, rng)
    # One device-level deviation per chip, carried into both the sensed state and
    # the destructive panel, so the two read the same realised device.
    activation = activation + rng.normal(
        0.0, float(trajectory["device_deviation_activation_fold"]), size=activation.shape
    )
    matrix = matrix + rng.normal(
        0.0, float(trajectory["device_deviation_matrix_ug"]), size=matrix.shape
    )
    days = cell_day_grid(
        float(series["first_culture_day"]),
        float(plan.culture_day),
        float(series["step_days"]),
    )
    coverage = np.full(days.size, float(plan.coverage_percent))
    features = build_features(activation, matrix, coverage, spec, rng)
    sweeps = build_sweeps(features, days, spec, rng, geometry_shift=plan.geometry != "development")
    optical = build_optical(features, days, plan.coverage_percent)
    terminal = build_terminal(float(activation[-1]), float(matrix[-1]), spec, rng)
    nuisance = build_nuisance(plan, spec, rng)
    state = spec.section("state_estimate")
    front_end = np.column_stack(
        [
            activation + rng.normal(0.0, float(state["activation_sd_fold"]), size=activation.size),
            matrix + rng.normal(0.0, float(state["matrix_sd_ug"]), size=matrix.size),
        ]
    )
    return DeviceRecord(
        device_id=plan.device_id,
        arm=plan.arm,
        block=plan.block,
        culture_day=plan.culture_day,
        treatment_day=plan.treatment_day,
        series_days=days,
        front_end_state=front_end,
        sweeps=sweeps,
        optical=optical,
        terminal=terminal,
        nuisance=nuisance,
        withdrawn_class=build_inclusion_fields(plan, spec, rng),
        perfusion=plan.perfusion,
        cell_source=plan.cell_source,
        hepatocyte_line=plan.hepatocyte_line,
    )


def build_cohort_fixture(
    spec: FixtureSpec | None = None, *, root: Path | None = None, seed: int | None = None
) -> Cohort:
    """Build the whole schema-compatible device pool deterministically."""
    if spec is None:
        if root is None:
            raise ValueError("building the pool needs either a spec or a repository root")
        spec = FixtureSpec.load(root, seed)
    rng = numpy_generator(spec.seed)
    plans = plan_devices(spec)
    devices = tuple(build_device(plan, spec, rng) for plan in plans)
    return Cohort(devices=devices, provenance=str(spec.section("cohort")["provenance"]))


def state_axis_names() -> tuple[str, ...]:
    return STATE_AXES
