"""The four comparator policies the allocator is priced against.

Ref: Sec. Component ablation; SI Table S11 footnote.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from obscal.runtime.seed import numpy_generator


def fixed_schedule_order(
    times: NDArray[np.float64], arms: Sequence[str], budget: int, timepoints: int
) -> list[int]:
    """The pre-declared clock: terminate the declared timepoints in order.

    The clock is fixed before the run and is never tuned, so it needs no model.
    """
    ordered_timepoints = sorted({float(value) for value in np.unique(times)})[:timepoints]
    label_array = np.asarray(list(arms), dtype=object)
    order: list[int] = []
    for timepoint in ordered_timepoints:
        at_time = np.flatnonzero(times == timepoint)
        for arm in dict.fromkeys(label_array.tolist()):
            same_arm = [index for index in at_time if label_array[index] == arm]
            order.extend(int(index) for index in same_arm)
    remaining = [index for index in range(times.size) if index not in set(order)]
    order.extend(remaining)
    if len(order) < budget:
        raise ValueError("the fixed clock cannot fill the requested budget")
    return order[:budget]


def random_matched_rate_order(n_candidates: int, budget: int, seed: int) -> list[int]:
    """A uniform draw at exactly the matched rate: sub-sampling at random."""
    rng = numpy_generator(seed)
    drawn = rng.permutation(n_candidates)[:budget]
    return [int(index) for index in drawn]


def random_bracket(
    n_candidates: int, budget: int, seeds: int, *, base_seed: int = 0
) -> list[list[int]]:
    """One draw per seed, so the bracket's dispersion can be printed."""
    return [
        random_matched_rate_order(n_candidates, budget, base_seed + offset)
        for offset in range(seeds)
    ]


def prior_art_allocator_order(rows: NDArray[np.float64], budget: int) -> list[int]:
    """A re-implemented subset-selection allocator from the bioprocess literature.

    Ref: SI Table S11, "the prior-art bracket by re-implementing a published
    allocator". The published allocator selects the samples that reach the
    offline reference by covering the calibration space rather than by any
    information criterion, so the allocator implemented here is the maximin
    (farthest-point) selection on the standardised regressor rows. The reference
    is repeatable per batch there and consumes its unit here, which is the
    difference the manuscript names.
    """
    standardised = (rows - rows.mean(axis=0, keepdims=True)) / np.where(
        rows.std(axis=0, ddof=1, keepdims=True) > 0.0, rows.std(axis=0, ddof=1, keepdims=True), 1.0
    )
    centroid = standardised.mean(axis=0)
    distances = np.linalg.norm(standardised - centroid, axis=1)
    first = int(np.argmax(distances))
    selected = [first]
    remaining = set(range(rows.shape[0]))
    remaining.discard(first)
    while len(selected) < budget and remaining:
        current = standardised[selected]
        best_index = -1
        best_distance = -np.inf
        for candidate in sorted(remaining):
            distance = float(np.min(np.linalg.norm(current - standardised[candidate], axis=1)))
            if distance > best_distance:
                best_distance = distance
                best_index = candidate
        selected.append(best_index)
        remaining.discard(best_index)
    if len(selected) < budget:
        raise ValueError("the prior-art allocator cannot fill the requested budget")
    return selected


def clairvoyant_oracle_order(
    rows: NDArray[np.float64],
    realised: NDArray[np.float64],
    precision: NDArray[np.float64],
    budget: int,
) -> list[int]:
    """The oracle sees the full future trajectory and is a ceiling, not a competitor."""
    scores = np.empty(rows.shape[0], dtype=np.float64)
    for index in range(rows.shape[0]):
        scores[index] = abs(float(realised[index])) * float(rows[index] @ precision @ rows[index])
    order = list(np.argsort(-scores, kind="stable"))
    return [int(index) for index in order[:budget]]


def policy_orders(
    *,
    rows: NDArray[np.float64],
    times: NDArray[np.float64],
    arms: Sequence[str],
    realised: NDArray[np.float64],
    precision: NDArray[np.float64],
    budget: int,
    fixed_timepoints: int,
    random_seeds: int,
    base_seed: int,
) -> dict[str, list[list[int]]]:
    """Every comparator's placement order, with the random bracket kept whole."""
    return {
        "fixed_schedule": [fixed_schedule_order(times, arms, budget, fixed_timepoints)],
        "random_matched_rate": random_bracket(
            rows.shape[0], budget, random_seeds, base_seed=base_seed
        ),
        "prior_art_allocator": [prior_art_allocator_order(rows, budget)],
        "clairvoyant_oracle": [clairvoyant_oracle_order(rows, realised, precision, budget)],
    }


def percentile_bracket(values: Sequence[float], low: float, high: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("a bracket needs at least one value")
    return float(np.percentile(array, low)), float(np.percentile(array, high))


def label_policies(names: Sequence[str]) -> dict[str, str]:
    """Human-readable labels, kept beside the machine names used in the report."""
    mapping = {
        "d_optimal": "ObsCal, D-optimal",
        "fixed_schedule": "Fixed schedule",
        "random_matched_rate": "Random at matched rate",
        "prior_art_allocator": "Prior-art allocator",
        "clairvoyant_oracle": "Clairvoyant oracle",
    }
    return {name: mapping[name] for name in names if name in mapping}


def is_constrained_against_unconstrained(left: str, right: str) -> bool:
    """Whether a pair may legitimately be asked for parity at full budget.

    Closed-loop and open-loop acquisition consume the identical device budget and
    differ only in the schedule spending it, so parity between them is not the
    expected result; the budgeted-against-oracle comparison is the one where
    parity at full budget is required.
    """
    budgeted = {"d_optimal", "fixed_schedule", "random_matched_rate", "prior_art_allocator"}
    return left in budgeted and right in budgeted
