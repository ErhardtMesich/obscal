"""Record schema for one device's sensing and terminal observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

STATE_AXES: tuple[str, ...] = ("activation", "matrix")
TERMINAL_ASSAYS: tuple[str, ...] = (
    "hydroxyproline_half_1",
    "hydroxyproline_half_2",
    "collagen_area_operator_1",
    "collagen_area_operator_2",
    "acta2",
    "col1a1",
    "albumin",
    "urea",
)


@dataclass(frozen=True)
class SweepSet:
    """Sweeps recorded on one device, on the fixed frequency grid."""

    frequencies_hz: NDArray[np.float64]
    elapsed_hours: NDArray[np.float64]
    real_ohm: NDArray[np.float64]
    imag_ohm: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.frequencies_hz.ndim != 1:
            raise ValueError("the frequency grid must be one-dimensional")
        if self.real_ohm.shape != self.imag_ohm.shape:
            raise ValueError("real and imaginary parts must share a shape")
        if self.real_ohm.shape != (self.elapsed_hours.size, self.frequencies_hz.size):
            raise ValueError("the sweep block must be sweeps by frequencies")

    @property
    def n_sweeps(self) -> int:
        return int(self.elapsed_hours.size)

    def at(self, sweep_index: int) -> NDArray[np.complex128]:
        return np.asarray(
            self.real_ohm[sweep_index] + 1j * self.imag_ohm[sweep_index], dtype=np.complex128
        )


@dataclass(frozen=True)
class OpticalSeries:
    """Same-clock non-electrical readings: coverage, texture and the effluent marker."""

    elapsed_hours: NDArray[np.float64]
    coverage_percent: NDArray[np.float64]
    matrix_texture_x1e3: NDArray[np.float64]
    effluent_pro_collagen_ng_per_ml: NDArray[np.float64]

    def __post_init__(self) -> None:
        shapes = {
            self.elapsed_hours.shape,
            self.coverage_percent.shape,
            self.matrix_texture_x1e3.shape,
            self.effluent_pro_collagen_ng_per_ml.shape,
        }
        if len(shapes) != 1:
            raise ValueError("every optical channel must share the sample grid")

    @property
    def n_samples(self) -> int:
        return int(self.elapsed_hours.size)


@dataclass(frozen=True)
class TerminalPanel:
    """The destructive panel read once per device, with its replicate split."""

    hydroxyproline_ug_per_chip: float
    hydroxyproline_half_1: float
    hydroxyproline_half_2: float
    collagen_area_percent: float
    collagen_area_operator_1: float
    collagen_area_operator_2: float
    acta2_fold: float
    col1a1_fold: float
    albumin_ng_per_chip_per_day: float
    urea_ug_per_chip_per_day: float
    pro_collagen_ng_per_ml: float

    def as_record(self) -> dict[str, float]:
        return {
            "hydroxyproline_ug_per_chip": self.hydroxyproline_ug_per_chip,
            "hydroxyproline_half_1": self.hydroxyproline_half_1,
            "hydroxyproline_half_2": self.hydroxyproline_half_2,
            "collagen_area_percent": self.collagen_area_percent,
            "collagen_area_operator_1": self.collagen_area_operator_1,
            "collagen_area_operator_2": self.collagen_area_operator_2,
            "acta2_fold": self.acta2_fold,
            "col1a1_fold": self.col1a1_fold,
            "albumin_ng_per_chip_per_day": self.albumin_ng_per_chip_per_day,
            "urea_ug_per_chip_per_day": self.urea_ug_per_chip_per_day,
            "pro_collagen_ng_per_ml": self.pro_collagen_ng_per_ml,
        }


@dataclass(frozen=True)
class NuisanceRecord:
    """The artefact covariates the regression carries as z."""

    temperature_c_mean: float
    medium_change_count: int
    hours_since_electrode_wetting: float
    perfusion_state: str
    geometry: str
    batch: str
    operator: str
    media_lot: int
    passage_stellate: int
    passage_parenchymal: int
    lot_stromal: int


@dataclass(frozen=True)
class DeviceRecord:
    """One device: how it was run, what it sensed and what it yielded at harvest."""

    device_id: str
    arm: str
    block: str
    culture_day: int
    treatment_day: int | None
    series_days: NDArray[np.float64]
    front_end_state: NDArray[np.float64]
    sweeps: SweepSet
    optical: OpticalSeries
    terminal: TerminalPanel
    nuisance: NuisanceRecord
    withdrawn_class: str | None
    perfusion: bool
    cell_source: str
    hepatocyte_line: str

    def __post_init__(self) -> None:
        if self.front_end_state.shape != (self.series_days.size, len(STATE_AXES)):
            raise ValueError("the front-end state series must be days by axes")

    @property
    def retained(self) -> bool:
        return self.withdrawn_class is None

    def state_at_day(self, culture_day: float) -> NDArray[np.float64]:
        index = int(np.argmin(np.abs(self.series_days - float(culture_day))))
        state: NDArray[np.float64] = np.asarray(self.front_end_state[index], dtype=np.float64)
        return state

    def labelled(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "arm": self.arm,
            "block": self.block,
            "batch": self.nuisance.batch,
            "operator": self.nuisance.operator,
            "culture_day": self.culture_day,
            "treatment_day": self.treatment_day,
            "perfusion": self.perfusion,
            "cell_source": self.cell_source,
            "hepatocyte_line": self.hepatocyte_line,
            "withdrawn_class": self.withdrawn_class,
        }


@dataclass(frozen=True)
class Cohort:
    """A whole device pool with the provenance label of how it was produced."""

    devices: tuple[DeviceRecord, ...]
    provenance: str

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("a cohort needs at least one device")
        identifiers = [device.device_id for device in self.devices]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("device identifiers must be unique")

    def __len__(self) -> int:
        return len(self.devices)

    @property
    def retained(self) -> tuple[DeviceRecord, ...]:
        return tuple(device for device in self.devices if device.retained)

    def by_block(self, block: str) -> tuple[DeviceRecord, ...]:
        return tuple(device for device in self.devices if device.block == block)

    def by_arm(self, arm: str) -> tuple[DeviceRecord, ...]:
        return tuple(device for device in self.devices if device.arm == arm)

    def by_batch(self, batch: str) -> tuple[DeviceRecord, ...]:
        return tuple(device for device in self.devices if device.nuisance.batch == batch)

    def by_operator(self, operator: str) -> tuple[DeviceRecord, ...]:
        return tuple(device for device in self.devices if device.nuisance.operator == operator)

    def at_day(self, block: str, culture_day: int) -> tuple[DeviceRecord, ...]:
        return tuple(
            device
            for device in self.by_block(block)
            if device.culture_day == culture_day and device.retained
        )

    def summary(self) -> dict[str, Any]:
        blocks: dict[str, int] = {}
        for device in self.devices:
            blocks[device.block] = blocks.get(device.block, 0) + 1
        arms: dict[str, int] = {}
        for device in self.devices:
            arms[device.arm] = arms.get(device.arm, 0) + 1
        return {
            "provenance": self.provenance,
            "devices": len(self.devices),
            "retained": len(self.retained),
            "withdrawn": len(self.devices) - len(self.retained),
            "blocks": blocks,
            "arms": arms,
        }


def cell_day_grid(start_day: float, end_day: float, step_days: float) -> NDArray[np.float64]:
    """A dense culture-day grid, used for the front-end series and the sweep clock."""
    if step_days <= 0.0:
        raise ValueError("the day step must be positive")
    if end_day < start_day:
        raise ValueError("the series cannot end before it starts")
    count = int(round((end_day - start_day) / step_days)) + 1
    return np.linspace(start_day, end_day, count)


def stack_terminal(panels: Sequence[TerminalPanel]) -> NDArray[np.float64]:
    keys = tuple(panels[0].as_record())
    return np.array(
        [[panel.as_record()[key] for key in keys] for panel in panels], dtype=np.float64
    )
