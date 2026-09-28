"""Identifiability diagnostics run before any slope is reported.

Ref: Sec. Calibration design and identifiability; SI Table S9 Panel B.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def standardise_design(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Centre and scale every column, so no axis is carried in the other's unit."""
    if matrix.ndim != 2:
        raise ValueError("the design must be two-dimensional")
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    spread = np.std(centred, axis=0, ddof=1)
    if np.any(spread <= 0.0):
        raise ValueError("a design column is invariant and cannot be standardised")
    return np.asarray(centred / spread.reshape(1, -1), dtype=np.float64)


def variance_inflation_factors(design: NDArray[np.float64]) -> NDArray[np.float64]:
    """One factor per column: how much its variance is inflated by the others."""
    standardised = standardise_design(design)
    correlation = np.corrcoef(standardised, rowvar=False)
    correlation = np.atleast_2d(correlation)
    inflation = np.empty(correlation.shape[0], dtype=np.float64)
    for index in range(correlation.shape[0]):
        others = [other for other in range(correlation.shape[0]) if other != index]
        if not others:
            inflation[index] = 1.0
            continue
        block = correlation[np.ix_(others, others)]
        cross = correlation[np.ix_(others, [index])]
        squared_multiple = float((cross.T @ np.linalg.solve(block, cross)).reshape(-1)[0])
        inflation[index] = 1.0 / max(1.0 - squared_multiple, 1e-12)
    return inflation


def pairwise_variance_inflation(correlation: float) -> float:
    """Two-column case, written out so the diagnostic has a hand check."""
    squared = float(correlation) ** 2
    if squared >= 1.0:
        raise ValueError("a perfect correlation leaves the pair unidentified")
    return 1.0 / (1.0 - squared)


def condition_number(design: NDArray[np.float64]) -> float:
    """Ratio of the largest to the smallest singular value of the standardised design."""
    standardised = standardise_design(design)
    singular = np.linalg.svd(standardised, compute_uv=False)
    if singular.min() <= 0.0:
        raise ValueError("a singular design has no finite condition number")
    return float(singular.max() / singular.min())


def joint_region_area(design: NDArray[np.float64], columns: Sequence[int]) -> float:
    """Determinant of the information matrix restricted to a parameter block.

    The joint confidence region of a parameter block is an ellipse whose area
    scales with the inverse square root of this determinant, so a design with a
    smaller value leaves a larger region.
    """
    indices = list(columns)
    if len(indices) < 2:
        raise ValueError("a joint region needs at least two parameters")
    block = design[:, indices]
    information = block.T @ block
    determinant = float(np.linalg.det(information))
    if determinant <= 0.0:
        raise ValueError("the parameter block is singular")
    return determinant


def relative_region_area(
    design: NDArray[np.float64], columns: Sequence[int], reference_design: NDArray[np.float64]
) -> float:
    """Joint region area against a reference design, normalised so the reference is one."""
    reference = joint_region_area(reference_design, columns)
    return float(np.sqrt(reference / joint_region_area(design, columns)))


@dataclass(frozen=True)
class IdentifiabilityAudit:
    """What the design leaves identified, priced in inflation and in correlation."""

    vif_state_block: NDArray[np.float64]
    condition_number: float
    state_correlation: float
    jointly_identified: bool
    verdict: str

    def as_record(self) -> dict[str, object]:
        return {
            "vif_activation": float(self.vif_state_block[0]),
            "vif_matrix": float(self.vif_state_block[1]),
            "condition_number": self.condition_number,
            "state_correlation": self.state_correlation,
            "jointly_identified": self.jointly_identified,
            "verdict": self.verdict,
        }


def identifiability_audit(
    design: NDArray[np.float64],
    state_columns: Sequence[int],
    *,
    rank_one_tolerance: float = 0.999,
    vif_threshold: float = 10.0,
) -> IdentifiabilityAudit:
    """Audit the two state axes before any slope is reported.

    Without at least one arm of each decoupling kind the design matrix has rank
    one in the plane the two axes span, and a small fitted matrix sensitivity is
    then indistinguishable from a design artefact.
    """
    indices = list(state_columns)
    if len(indices) != 2:
        raise ValueError("the audit prices exactly two state axes")
    block = design[:, indices]
    standardised = standardise_design(block)
    correlation = float(standardised[:, 0] @ standardised[:, 1] / (standardised.shape[0] - 1))
    vifs = variance_inflation_factors(block)
    rank_one = abs(correlation) >= rank_one_tolerance
    identified = (not rank_one) and bool(np.all(vifs < 1e9))
    verdict = "both_axes_identified" if identified else "rank_deficient_in_practice"
    return IdentifiabilityAudit(
        vif_state_block=vifs,
        condition_number=condition_number(block),
        state_correlation=correlation,
        jointly_identified=identified,
        verdict=verdict,
    )


def cascade_directional_derivative(
    ols_sensitivity: NDArray[np.float64], axis_weights: Sequence[float]
) -> NDArray[np.float64]:
    """Directional derivative along the activation-to-deposition cascade.

    Ref: Algorithm 1 step C. Where the acellular ladder contributes no rows the
    two slopes are declared jointly unidentified and this derivative is the only
    quantity the design still supports.
    """
    weights = np.asarray(axis_weights, dtype=np.float64)
    if weights.shape[0] != ols_sensitivity.shape[0]:
        raise ValueError("the cascade weights must match the axis count")
    normaliser = float(np.linalg.norm(weights))
    if normaliser == 0.0:
        raise ValueError("the cascade direction must be non-zero")
    unit = weights / normaliser
    return np.asarray(np.asarray(ols_sensitivity, dtype=np.float64).T @ unit, dtype=np.float64)


def identifiability_from_arm_mix(
    has_acellular_rows: bool, has_coverage_rows: bool
) -> dict[str, object]:
    """The structural conclusion the ladder mix licenses."""
    identified: list[str] = []
    if has_acellular_rows:
        identified.append("matrix")
    if has_coverage_rows:
        identified.append("coverage")
    both_state_axes = has_acellular_rows
    return {
        "matrix_axis_identified": has_acellular_rows,
        "coverage_identified": has_coverage_rows,
        "both_state_axes_identified": both_state_axes,
        "identified_quantities": tuple(identified),
        "verdict": "both_axes_identified" if both_state_axes else "rank_deficient_in_practice",
    }


def variance_inflation_from_arm_mix(
    vif_state_block: tuple[float, float],
    vif_biological_only: tuple[float, float],
) -> dict[str, float]:
    """How much the ladders beat fitting on biological arms alone."""
    with_ladders = float(max(vif_state_block))
    without = float(max(vif_biological_only))
    return {
        "vif_with_ladders": with_ladders,
        "vif_biological_arms_only": without,
        "inflation_added": without - with_ladders,
        "ratio": without / with_ladders,
    }
