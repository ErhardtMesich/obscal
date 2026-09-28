"""Cell-free blank channel: measured drift, medium steps and resolvable change.

Ref: Sec. Continuous sensing front end and feature extraction; SI Group C.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BlankDriftPoint:
    """One frequency of the cell-free channel, carried through a full run."""

    frequency_hz: float
    day0_magnitude_ohm: float
    drift_ohm_per_day: float
    medium_step_ohm: float
    temperature_coefficient_pct_per_c: float

    @property
    def relative_drift_pct_per_day(self) -> float:
        return 100.0 * self.drift_ohm_per_day / max(self.day0_magnitude_ohm, 1e-12)

    @property
    def drift_absolute(self) -> float:
        return abs(self.drift_ohm_per_day)


@dataclass(frozen=True)
class BlankChannel:
    """The whole frequency grid of one cell-free channel."""

    points: tuple[BlankDriftPoint, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("a blank channel needs at least one frequency")
        order = [point.frequency_hz for point in self.points]
        if order != sorted(order):
            raise ValueError("blank-channel points must be ordered by frequency")

    @property
    def frequencies_hz(self) -> NDArray[np.float64]:
        return np.array([point.frequency_hz for point in self.points], dtype=np.float64)

    @property
    def day0_magnitude(self) -> NDArray[np.float64]:
        return np.array([point.day0_magnitude_ohm for point in self.points], dtype=np.float64)

    @property
    def drift_per_day(self) -> NDArray[np.float64]:
        return np.array([point.drift_ohm_per_day for point in self.points], dtype=np.float64)

    @property
    def relative_drift_pct_per_day(self) -> NDArray[np.float64]:
        return np.array(
            [point.relative_drift_pct_per_day for point in self.points], dtype=np.float64
        )

    def blank_trace(self, days: NDArray[np.float64]) -> NDArray[np.float64]:
        """Day-0 magnitude plus the continuous drift term at each frequency."""
        elapsed = days.reshape(-1, 1)
        return self.day0_magnitude.reshape(1, -1) + elapsed * self.drift_per_day.reshape(1, -1)

    def medium_exchange_step(self) -> NDArray[np.float64]:
        return np.array([point.medium_step_ohm for point in self.points], dtype=np.float64)

    def temperature_coefficient(self) -> NDArray[np.float64]:
        return np.array(
            [point.temperature_coefficient_pct_per_c for point in self.points], dtype=np.float64
        )

    def at(self, frequency_hz: float) -> BlankDriftPoint:
        return min(self.points, key=lambda point: abs(point.frequency_hz - frequency_hz))


def blank_channel_from_records(
    frequency_hz: Sequence[float],
    day0_magnitude_ohm: Sequence[float],
    drift_ohm_per_day: Sequence[float],
    medium_step_ohm: Sequence[float],
    temperature_coefficient_pct_per_c: Sequence[float],
) -> BlankChannel:
    points = tuple(
        BlankDriftPoint(
            frequency_hz=float(freq),
            day0_magnitude_ohm=float(day0),
            drift_ohm_per_day=float(drift),
            medium_step_ohm=float(step),
            temperature_coefficient_pct_per_c=float(coeff),
        )
        for freq, day0, drift, step, coeff in zip(
            frequency_hz,
            day0_magnitude_ohm,
            drift_ohm_per_day,
            medium_step_ohm,
            temperature_coefficient_pct_per_c,
            strict=True,
        )
    )
    return BlankChannel(points=points)


def resolvable_change(drift_ohm_per_day: float, interval_days: float, multiple: float) -> float:
    """Smallest resolvable impedance change: a multiple of the measured blank drift.

    No published drift figure is adopted, because the two independent sources
    report in incompatible units and one of them is medium evaporation rather
    than electrode drift.
    """
    return float(multiple) * abs(drift_ohm_per_day) * float(interval_days)


def resolvable_change_table(
    blank: BlankChannel, interval_days: float, multiple: float
) -> dict[str, list[float]]:
    drift = blank.drift_per_day
    resolvable = drift * float(interval_days) * float(multiple)
    return {
        "frequency_hz": blank.frequencies_hz.tolist(),
        "day0_magnitude_ohm": blank.day0_magnitude.tolist(),
        "drift_ohm_per_day": drift.tolist(),
        "drift_pct_per_day": blank.relative_drift_pct_per_day.tolist(),
        "medium_step_ohm": blank.medium_exchange_step().tolist(),
        "temperature_coefficient_pct_per_c": blank.temperature_coefficient().tolist(),
        "resolvable_change_ohm": abs(resolvable).tolist(),
    }


def temperature_corrected(
    measured_ohm: NDArray[np.float64],
    temperature_c: NDArray[np.float64],
    reference_c: float,
    coefficients_pct_per_c: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Remove the measured temperature term from a recorded magnitude series."""
    delta = temperature_c - float(reference_c)
    factors = 1.0 + coefficients_pct_per_c * delta / 100.0
    return measured_ohm / factors


def detect_medium_exchange_steps(
    magnitude_ohm: NDArray[np.float64], times_h: NDArray[np.float64], jump_threshold: float
) -> NDArray[np.int64]:
    """Discrete steps at medium exchange, kept apart from the continuous drift."""
    if magnitude_ohm.shape != times_h.shape:
        raise ValueError("magnitude and time must align")
    if magnitude_ohm.size < 2:
        return np.zeros(magnitude_ohm.shape, dtype=np.int64)
    flags = np.zeros(magnitude_ohm.shape, dtype=np.int64)
    for index in range(1, magnitude_ohm.size):
        step = abs(float(magnitude_ohm[index]) - float(magnitude_ohm[index - 1]))
        if step >= jump_threshold:
            flags[index] = 1
    return flags


def bubble_events(
    magnitude_ohm: NDArray[np.float64], baseline_ohm: NDArray[np.float64], excursion_fraction: float
) -> NDArray[np.int64]:
    """Flag sweeps whose excursion over the smoothed baseline marks an occlusion."""
    if magnitude_ohm.shape != baseline_ohm.shape:
        raise ValueError("magnitude and baseline must align")
    ratio = np.abs(magnitude_ohm - baseline_ohm) / np.maximum(np.abs(baseline_ohm), 1e-12)
    return (ratio >= excursion_fraction).astype(np.int64)


@dataclass(frozen=True)
class ExcitationBias:
    """Residual bias from the excitation shape, carried as a bounded systematic."""

    resistance_pct: float
    capacitance_pct: float
    published_band_pct: tuple[float, float]

    @property
    def within_published_band(self) -> bool:
        low, high = self.published_band_pct
        return low <= abs(self.resistance_pct) <= high

    def report(self) -> dict[str, float | bool]:
        return {
            "resistance_bias_pct": self.resistance_pct,
            "capacitance_bias_pct": self.capacitance_pct,
            "band_low_pct": self.published_band_pct[0],
            "band_high_pct": self.published_band_pct[1],
            "within_published_band": self.within_published_band,
        }


def excitation_bias_from_square_wave(
    sinusoid_ohm: NDArray[np.float64],
    square_ohm: NDArray[np.float64],
    published_band_pct: tuple[float, float],
) -> ExcitationBias:
    """Relative offset a square-wave stimulus introduces on the resistive channel."""
    if sinusoid_ohm.shape != square_ohm.shape:
        raise ValueError("the two waveforms must share a sample grid")
    reference = float(np.mean(sinusoid_ohm))
    shifted = float(np.mean(square_ohm))
    resistance = 100.0 * (shifted - reference) / max(reference, 1e-12)
    return ExcitationBias(
        resistance_pct=resistance,
        capacitance_pct=math.nan,
        published_band_pct=(float(published_band_pct[0]), float(published_band_pct[1])),
    )


def band_information_share(information: Sequence[float]) -> list[float]:
    """Share of the total design information carried by each excitation band."""
    total = float(sum(information))
    if total <= 0.0:
        raise ValueError("information shares need a positive total")
    return [100.0 * float(value) / total for value in information]
