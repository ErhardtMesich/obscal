"""Assembly of the p = 5 sensor feature vector.

Ref: Sec. Continuous sensing front end and feature extraction, Eq. (2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from obscal.chip.circuit import CircuitFit

ELECTRICAL_FEATURES: tuple[str, ...] = ("log_rb", "log_cm", "alpha")
OPTICAL_FEATURES: tuple[str, ...] = ("matrix_texture",)
EFFLUENT_FEATURES: tuple[str, ...] = ("effluent_pro_collagen",)
FULL_FRONT_END: tuple[str, ...] = ELECTRICAL_FEATURES + OPTICAL_FEATURES + EFFLUENT_FEATURES
SINGLE_FREQUENCY_FRONT_END: tuple[str, ...] = (
    "teer_single_frequency",
    "matrix_texture",
    "effluent_pro_collagen",
)
FEATURE_DIMENSION = len(FULL_FRONT_END)


@dataclass(frozen=True)
class FeatureVector:
    """One device-time observation: the sensor features plus the coverage covariate."""

    device_id: str
    culture_day: float
    values: tuple[float, ...]
    coverage_percent: float
    names: tuple[str, ...] = FULL_FRONT_END

    def __post_init__(self) -> None:
        if len(self.values) != len(self.names):
            raise ValueError("feature values and names must align")

    @property
    def dimension(self) -> int:
        return len(self.values)

    def as_array(self) -> NDArray[np.float64]:
        return np.array(self.values, dtype=np.float64)

    def get(self, name: str) -> float:
        index = self.names.index(name)
        return float(self.values[index])


def feature_names(*, multi_frequency: bool = True) -> tuple[str, ...]:
    return FULL_FRONT_END if multi_frequency else SINGLE_FREQUENCY_FRONT_END


def assemble_features(
    fit: CircuitFit,
    area_mm2: float,
    matrix_texture_x1e3: float,
    effluent_pro_collagen_ng_per_ml: float,
    coverage_percent: float,
    *,
    device_id: str,
    culture_day: float,
    multi_frequency: bool = True,
) -> FeatureVector:
    """Build the sensor vector from one fitted sweep and its same-clock readings."""
    electrical = fit.features(area_mm2)
    values: tuple[float, ...]
    if multi_frequency:
        values = (
            electrical["log_rb"],
            electrical["log_cm"],
            electrical["alpha"],
            float(matrix_texture_x1e3),
            float(effluent_pro_collagen_ng_per_ml),
        )
        names = FULL_FRONT_END
    else:
        values = (
            electrical["barrier_resistance_area_product"],
            float(matrix_texture_x1e3),
            float(effluent_pro_collagen_ng_per_ml),
        )
        names = SINGLE_FREQUENCY_FRONT_END
    return FeatureVector(
        device_id=device_id,
        culture_day=float(culture_day),
        values=values,
        coverage_percent=float(coverage_percent),
        names=names,
    )


def feature_matrix(vectors: Sequence[FeatureVector]) -> NDArray[np.float64]:
    if not vectors:
        raise ValueError("at least one feature vector is required")
    names = vectors[0].names
    for vector in vectors:
        if vector.names != names:
            raise ValueError("all feature vectors in a block must share a feature set")
    return np.vstack([vector.as_array() for vector in vectors])


def coverage_vector(vectors: Sequence[FeatureVector]) -> NDArray[np.float64]:
    return np.array([vector.coverage_percent for vector in vectors], dtype=np.float64)


def culture_day_vector(vectors: Sequence[FeatureVector]) -> NDArray[np.float64]:
    return np.array([vector.culture_day for vector in vectors], dtype=np.float64)


def standardised_sensitivity(slope: float, interquartile_range: float, residual_sd: float) -> float:
    """Slope in units of the covariate's own spread per residual standard deviation."""
    if residual_sd <= 0.0:
        raise ValueError("residual standard deviation must be positive")
    return float(slope) * float(interquartile_range) / float(residual_sd)


def sensitivity_ratio(numerator: float, denominator: float) -> float:
    """Ratio of two standardised sensitivities on a common scale."""
    if denominator == 0.0:
        raise ValueError("the denominator sensitivity must be non-zero")
    return float(numerator) / float(denominator)


def centre_on_block_mean(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values - float(np.mean(values))


def unit_scale(values: NDArray[np.float64]) -> NDArray[np.float64]:
    scale = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if scale <= 0.0:
        raise ValueError("cannot standardise a constant vector")
    return values / scale


def interquartile_range(values: NDArray[np.float64]) -> float:
    low, high = np.percentile(values, [25.0, 75.0])
    return float(high) - float(low)
