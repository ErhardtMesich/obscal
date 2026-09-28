"""Frequency-resolved sensitivity, information share and the band asymmetry.

Ref: Sec. Measured sensitivity of the electrical front end; SI Table S4.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BandSensitivity:
    """One excitation band, standardised so the three axes are commensurable."""

    frequency_hz: float
    activation: float
    matrix: float
    coverage: float
    floor_ug_hyp: float | None = None

    @property
    def coverage_to_matrix_ratio(self) -> float:
        if self.matrix == 0.0:
            raise ValueError("matrix sensitivity must be non-zero for a ratio")
        return self.coverage / self.matrix


@dataclass(frozen=True)
class BandTable:
    """The band table with its information shares and the resistive-band summary."""

    bands: tuple[BandSensitivity, ...]
    matrix_information_share: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.bands) != len(self.matrix_information_share):
            raise ValueError("every band needs an information share")

    @property
    def total_matrix_information(self) -> float:
        return float(sum(band.matrix**2 for band in self.bands))

    def resistive_band_share(self, low_hz: float, high_hz: float) -> float:
        share = 0.0
        for band, value in zip(self.bands, self.matrix_information_share, strict=True):
            if low_hz <= band.frequency_hz <= high_hz:
                share += value
        return share

    def peak_frequency(self, axis: str) -> float:
        key = _axis_getter(axis)
        return float(max(self.bands, key=key).frequency_hz)

    def as_records(self) -> list[dict[str, float]]:
        records: list[dict[str, float]] = []
        for band, share in zip(self.bands, self.matrix_information_share, strict=True):
            record = {
                "frequency_hz": band.frequency_hz,
                "activation_sensitivity": band.activation,
                "matrix_sensitivity": band.matrix,
                "coverage_sensitivity": band.coverage,
                "matrix_information_share_pct": share,
                "coverage_to_matrix_ratio": band.coverage_to_matrix_ratio,
            }
            if band.floor_ug_hyp is not None:
                record["floor_ug_hyp"] = band.floor_ug_hyp
            records.append(record)
        return records


def _axis_getter(axis: str) -> Callable[[BandSensitivity], float]:
    if axis == "activation":
        return lambda band: band.activation
    if axis == "matrix":
        return lambda band: band.matrix
    if axis == "coverage":
        return lambda band: band.coverage
    raise ValueError(f"unknown axis {axis!r}")


def matrix_information_shares(matrix: Sequence[float]) -> tuple[float, ...]:
    """Proportion of the total matrix-axis design information a band carries.

    A band's standardised sensitivity already divides by the sensor residual
    scale, so the information it contributes is the square of that sensitivity.
    """
    values = np.asarray(matrix, dtype=np.float64)
    total = float(np.sum(values**2))
    if total <= 0.0:
        raise ValueError("at least one band must carry matrix information")
    return tuple(100.0 * float(value) / total for value in values**2)


def band_table(bands: Sequence[BandSensitivity]) -> BandTable:
    ordered = tuple(sorted(bands, key=lambda band: band.frequency_hz))
    shares = matrix_information_shares([band.matrix for band in ordered])
    return BandTable(bands=ordered, matrix_information_share=shares)


def band_floor(band_standard_sensitivity: float, residual_sd: float, coefficient: float) -> float:
    """Detectability floor a single band would deliver on its own."""
    if band_standard_sensitivity <= 0.0:
        raise ValueError("a band floor needs a positive standardised sensitivity")
    return float(coefficient) * residual_sd / band_standard_sensitivity


def fused_front_end_sensitivity(
    band_sensitivities: Sequence[float], extra_features: Sequence[float]
) -> float:
    """Standardised sensitivity of the fused front end.

    Fusing bands and adding features that a single band cannot see raises the
    fused sensitivity above every constituent band; the combination is taken in
    quadrature because each feature is on a common standardised scale.
    """
    combined = list(band_sensitivities) + list(extra_features)
    return float(np.sqrt(np.sum(np.square(np.asarray(combined, dtype=np.float64)))))


def coverage_matrix_gap(fused_coverage: float, fused_matrix: float) -> dict[str, float]:
    """The platform's central asymmetry, with the interval arithmetic attached."""
    ratio = fused_coverage / fused_matrix
    return {
        "fused_coverage_sensitivity": fused_coverage,
        "fused_matrix_sensitivity": fused_matrix,
        "coverage_to_matrix_ratio": ratio,
        "matrix_share_of_coverage": fused_matrix / fused_coverage,
    }


def band_ordering(table: BandTable, axis: str) -> list[float]:
    """Frequencies ordered by descending standardised sensitivity on one axis."""
    key = _axis_getter(axis)
    return [band.frequency_hz for band in sorted(table.bands, key=key, reverse=True)]


def frequency_grid(values: Sequence[float]) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)
