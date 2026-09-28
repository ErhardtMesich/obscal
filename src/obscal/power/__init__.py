"""Power layer: the exact noncentral-t coefficient and the detectability floors."""

from obscal.power.noncentral import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    DetectabilityFloor,
    coefficient_grid,
    direction_vector,
    directional_floor,
    estimator_covariance,
    floor_from_estimator,
    floor_from_sigma,
    floor_grid,
    floor_ratio,
    minimum_detectable_difference,
    noncentral_t_critical,
    noncentrality_for_power,
    replication_requirement,
    two_group_coefficient,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_POWER",
    "DetectabilityFloor",
    "coefficient_grid",
    "direction_vector",
    "directional_floor",
    "estimator_covariance",
    "floor_from_estimator",
    "floor_from_sigma",
    "floor_grid",
    "floor_ratio",
    "minimum_detectable_difference",
    "noncentral_t_critical",
    "noncentrality_for_power",
    "replication_requirement",
    "two_group_coefficient",
]
