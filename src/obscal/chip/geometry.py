"""Chip geometry, the study-constant geometric correction and pre-seeding gates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class ChannelGeometry:
    """Physical layout of one device, in micrometres where a length is implied."""

    unit_face_width_um: float
    central_lumen_um: float
    microneedle_inlet_um: float
    channel_height_um: float
    channel_length_mm: float
    flow_rate_uL_per_h: float
    devices_per_reservoir: int

    @classmethod
    def from_config(
        cls, tissue: Mapping[str, Any], perfusion: Mapping[str, Any]
    ) -> ChannelGeometry:
        return cls(
            unit_face_width_um=float(tissue["unit_face_width_um"]),
            central_lumen_um=float(tissue["central_lumen_um"]),
            microneedle_inlet_um=float(tissue["microneedle_inlet_um"]),
            channel_height_um=float(tissue["channel_height_um"]),
            channel_length_mm=float(tissue["channel_length_mm"]),
            flow_rate_uL_per_h=float(perfusion["flow_rate_uL_per_h"]),
            devices_per_reservoir=int(perfusion["devices_per_reservoir"]),
        )

    @property
    def hexagon_side_um(self) -> float:
        return self.unit_face_width_um / math.sqrt(3.0)

    @property
    def floor_area_um2(self) -> float:
        hexagon = 1.5 * math.sqrt(3.0) * self.hexagon_side_um**2
        lumen = math.pi * (0.5 * self.central_lumen_um) ** 2
        return hexagon - lumen

    @property
    def wetted_perimeter_um(self) -> float:
        return 6.0 * self.hexagon_side_um + math.pi * self.central_lumen_um

    @property
    def floor_area_mm2(self) -> float:
        return self.floor_area_um2 * 1e-6

    @property
    def reservoir_residence_h(self) -> float:
        return 1.0 / self.turnovers_per_hour

    @property
    def turnovers_per_hour(self) -> float:
        return self.flow_rate_uL_per_h / max(self.reservoir_volume_uL, 1e-9)

    @property
    def reservoir_volume_uL(self) -> float:
        return float(self.devices_per_reservoir) * 500.0


def geometric_correction_factor(geometry: ChannelGeometry) -> float:
    """Study-constant factor applied to every sweep.

    Ref: Sec. Device, cultures and injury model; the correction is taken from the
    printed geometry [29] and is never refitted per device, timepoint or arm,
    because a front end allowed to absorb geometry per device would absorb the
    artefact the cell-free arm exists to expose.

    Engineering default: the manuscript states that the correction is computed
    from the geometry but does not print its functional form. The form used here
    is the floor area divided by the floor area plus the perimeter term that the
    current path adds through the channel walls and the central lumen.
    """
    floor = geometry.floor_area_um2
    spreading = geometry.wetted_perimeter_um * geometry.channel_height_um / math.pi
    return floor / (floor + spreading)


def corrected_area_mm2(geometry: ChannelGeometry, factor: float | None = None) -> float:
    """Area denominator that turns a fitted resistance into an ohmic area product."""
    used = geometric_correction_factor(geometry) if factor is None else float(factor)
    return geometry.floor_area_mm2 / used


def baseline_gate(
    cell_free_spectrum: Sequence[float],
    batch_median: Sequence[float],
    batch_sd: Sequence[float],
    threshold_sd: float,
) -> tuple[bool, int]:
    """Pre-seeding check: a device departing from the batch median is withdrawn."""
    if not (len(cell_free_spectrum) == len(batch_median) == len(batch_sd)):
        raise ValueError("spectrum, median and sd must share the frequency grid")
    breaches = 0
    for value, centre, spread in zip(cell_free_spectrum, batch_median, batch_sd, strict=True):
        if spread <= 0.0:
            continue
        if abs(value - centre) > threshold_sd * spread:
            breaches += 1
    return breaches == 0, breaches


def leak_check(pressure_decay_per_min: float, threshold: float) -> bool:
    """Channel-leak gate: relative pressure decay per minute must stay inside band."""
    return pressure_decay_per_min <= threshold


def seeding_coverage_in_band(coverage_percent: float, band: Sequence[float]) -> bool:
    low, high = float(band[0]) * 100.0, float(band[1]) * 100.0
    return low <= coverage_percent <= high


def half_mass_ratio_ok(ratio: float, band: Sequence[float]) -> bool:
    return float(band[0]) <= ratio <= float(band[1])


def electrode_drift_ok(drift_ohm_per_day: float, threshold: float) -> bool:
    return abs(drift_ohm_per_day) <= threshold


def geometry_summary(geometry: ChannelGeometry) -> dict[str, float | str]:
    factor = geometric_correction_factor(geometry)
    payload: dict[str, Any] = {
        "unit_shape": "hexagon",
        "hexagon_side_um": round(geometry.hexagon_side_um, 4),
        "floor_area_mm2": round(geometry.floor_area_mm2, 6),
        "wetted_perimeter_um": round(geometry.wetted_perimeter_um, 4),
        "channel_height_um": geometry.channel_height_um,
        "geometric_correction_factor": round(factor, 8),
        "corrected_area_mm2": round(corrected_area_mm2(geometry, factor), 6),
        "flow_rate_uL_per_h": geometry.flow_rate_uL_per_h,
        "devices_per_reservoir": geometry.devices_per_reservoir,
    }
    return cast("dict[str, float | str]", payload)
