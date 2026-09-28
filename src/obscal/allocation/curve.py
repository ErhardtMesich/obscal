"""The budget-against-precision curve and its two structural checks.

Ref: Sec. Component ablation; Fig. 3; SI Table S11.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

RseEvaluator = Callable[[Sequence[int]], float]

POLICY_COLUMNS: tuple[str, ...] = (
    "d_optimal",
    "fixed_schedule",
    "random_matched_rate",
    "prior_art_allocator",
    "clairvoyant_oracle",
)

UNIT_PERCENT = 100.0


@dataclass(frozen=True)
class BudgetPoint:
    """One row of the allocation curve, every policy read at the same spent count."""

    fraction: float
    devices: int
    rse: dict[str, float]
    random_low: float
    random_high: float
    random_draws_used: int = 0

    def at(self, policy: str) -> float:
        if policy not in self.rse:
            raise KeyError(f"policy {policy!r} is not on this point")
        return self.rse[policy]

    def as_record(self) -> dict[str, float | int]:
        record: dict[str, float | int] = {
            "fraction": self.fraction,
            "devices": self.devices,
            "random_p10": self.random_low,
            "random_p90": self.random_high,
            "random_draws_used": self.random_draws_used,
        }
        for policy, value in self.rse.items():
            record[f"rse::{policy}"] = value
        return record


def budget_devices(fraction: float, block_devices: int) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("a budget fraction must lie in (0, 1]")
    return int(round(fraction * block_devices))


def subset_rse(
    indices: Sequence[int],
    evaluator: RseEvaluator,
) -> float:
    if len(indices) < 4:
        raise ValueError("a budget point below four units carries no slope precision")
    return float(evaluator(indices))


def build_curve(
    *,
    orders: dict[str, list[list[int]]],
    budget_fractions: Sequence[float],
    block_devices: int,
    evaluator: RseEvaluator,
) -> list[BudgetPoint]:
    """Read every policy at each budget point on the same block.

    A closed-loop and an open-loop arm spend the identical device budget and
    differ only in the schedule spending it, so their curves are read at the same
    device counts.
    """
    points: list[BudgetPoint] = []
    for fraction in budget_fractions:
        devices = budget_devices(fraction, block_devices)
        values: dict[str, float] = {}
        low = float("nan")
        high = float("nan")
        used = 0
        for policy, orderings in orders.items():
            if policy == "random_matched_rate":
                all_draws = [subset_rse(list(draw)[:devices], evaluator) for draw in orderings]
                draws = [value for value in all_draws if np.isfinite(value)]
                if not draws:
                    values[policy] = float("inf")
                    low, high = float("inf"), float("inf")
                    used = 0
                else:
                    values[policy] = float(np.median(draws))
                    low = float(np.percentile(draws, 10.0))
                    high = float(np.percentile(draws, 90.0))
                    used = len(draws)
                continue
            values[policy] = subset_rse(list(orderings[0])[:devices], evaluator)
        points.append(
            BudgetPoint(
                fraction=float(fraction),
                devices=devices,
                rse=values,
                random_low=low,
                random_high=high,
                random_draws_used=used,
            )
        )
    return points


def d_optimal_order_curve(
    order: Sequence[int],
    budget_fractions: Sequence[float],
    block_devices: int,
    evaluator: RseEvaluator,
) -> list[BudgetPoint]:
    """The allocator's own curve, read on its own placement order."""
    return build_curve(
        orders={"d_optimal": [list(order)]},
        budget_fractions=budget_fractions,
        block_devices=block_devices,
        evaluator=evaluator,
    )


def oracle_curve(
    *,
    n_candidates: int,
    budget_fractions: Sequence[float],
    block_devices: int,
    evaluator: RseEvaluator,
    minimum_units: int,
) -> list[BudgetPoint]:
    """The clairvoyant ceiling: at every budget the best subset the future allows.

    The oracle sees the realised trajectory, so it re-optimises at each budget
    instead of committing to one order. A budgeted curve cannot beat it, which is
    the non-crossing property the allocation table asserts.
    """
    selected: list[int] = []
    points: list[BudgetPoint] = []
    for fraction in budget_fractions:
        devices = budget_devices(fraction, block_devices)
        target = max(devices, minimum_units)
        while len(selected) < target:
            remaining = [index for index in range(n_candidates) if index not in set(selected)]
            if not remaining:
                break
            best_index = remaining[0]
            best_value = float("inf")
            for candidate in remaining:
                value = float(evaluator(selected + [candidate]))
                if value < best_value:
                    best_value = value
                    best_index = candidate
            selected.append(best_index)
        value = float(evaluator(selected))
        points.append(
            BudgetPoint(
                fraction=float(fraction),
                devices=devices,
                rse={"clairvoyant_oracle": value},
                random_low=value,
                random_high=value,
                random_draws_used=1,
            )
        )
    return points


def convergence_at_full_budget(
    curve: Sequence[BudgetPoint],
    policy_columns: Sequence[str] = POLICY_COLUMNS,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """All policies must meet once the same devices have been consumed."""
    full = curve[-1]
    values = {policy: full.at(policy) for policy in policy_columns if policy in full.rse}
    reference = float(np.mean(list(values.values())))
    spread = max(abs(value - reference) for value in values.values())
    return {
        "budget": full.devices,
        "values": values,
        "spread": spread,
        "converged": spread <= tolerance * max(reference, 1.0),
    }


def non_crossing_above_oracle(
    curve: Sequence[BudgetPoint], policy: str = "d_optimal", oracle: str = "clairvoyant_oracle"
) -> dict[str, object]:
    """The budgeted curve must stay above the oracle below the full budget.

    Sub-sampling cannot create information, so a budgeted estimator owes an
    unbudgeted one that parity; an arm falling below the oracle would be
    reporting leakage.
    """
    gaps: list[float] = []
    violations: list[int] = []
    for point in curve[:-1]:
        gap = point.at(policy) - point.at(oracle)
        gaps.append(gap)
        if gap < -1e-9:
            violations.append(point.devices)
    return {
        "policy": policy,
        "oracle": oracle,
        "min_gap": min(gaps) if gaps else float("nan"),
        "max_gap": max(gaps) if gaps else float("nan"),
        "violations_at_devices": violations,
        "non_crossing": not violations,
    }


def random_envelope_advantage(
    curve: Sequence[BudgetPoint], policy: str = "d_optimal"
) -> dict[str, object]:
    """Between which budget points the allocator sits under the random tenth percentile."""
    below: list[int] = []
    above: list[int] = []
    for point in curve[:-1]:
        (below if point.at(policy) < point.random_low else above).append(point.devices)
    return {
        "below_p10_at_devices": below,
        "at_or_above_p10_at_devices": above,
    }


@dataclass(frozen=True)
class OperatingPoint:
    """The only place a saving is named."""

    target_rse: float
    devices_by_policy: dict[str, int]
    fraction_by_policy: dict[str, float]
    interpolated: bool

    def saving(
        self, reference: str = "fixed_schedule", policy: str = "d_optimal"
    ) -> dict[str, float | int]:
        return {
            "policy_devices": self.devices_by_policy[policy],
            "reference_devices": self.devices_by_policy[reference],
            "devices_saved": self.devices_by_policy[reference] - self.devices_by_policy[policy],
            "fraction_saved": self.fraction_by_policy[reference] - self.fraction_by_policy[policy],
        }

    def as_record(self) -> dict[str, object]:
        return {
            "target_corrected_rse_pct": self.target_rse,
            "devices": self.devices_by_policy,
            "fractions": self.fraction_by_policy,
            "interpolated_between_measured_points": self.interpolated,
        }


def operating_point(
    curve: Sequence[BudgetPoint],
    policy_columns: Sequence[str],
    target_rse_percent: float,
    block_devices: int,
) -> OperatingPoint:
    """Interpolate the device count each policy needs to reach a matched precision."""
    devices = np.array([point.devices for point in curve], dtype=np.float64)
    devices_by_policy: dict[str, int] = {}
    fraction_by_policy: dict[str, float] = {}
    any_interpolated = False
    for policy in policy_columns:
        values = np.array([point.at(policy) for point in curve], dtype=np.float64)
        crossed = _crossing_devices(devices, values, target_rse_percent)
        devices_by_policy[policy] = int(crossed)
        fraction_by_policy[policy] = float(crossed) / float(block_devices)
        if crossed not in set(devices.tolist()):
            any_interpolated = True
    return OperatingPoint(
        target_rse=float(target_rse_percent),
        devices_by_policy=devices_by_policy,
        fraction_by_policy=fraction_by_policy,
        interpolated=any_interpolated,
    )


def _crossing_devices(
    devices: NDArray[np.float64], values: NDArray[np.float64], target: float
) -> float:
    """First device count at which the curve reaches the target precision."""
    order = np.argsort(devices)
    devices = devices[order]
    values = values[order]
    for index, value in enumerate(values):
        if value <= target:
            if index == 0:
                return float(devices[0])
            previous_value = values[index - 1]
            previous_devices = devices[index - 1]
            if previous_value == value:
                return float(devices[index])
            weight = (previous_value - target) / (previous_value - value)
            return float(previous_devices + weight * (devices[index] - previous_devices))
    raise ValueError("no budget point reaches the requested precision")


def curve_records(curve: Sequence[BudgetPoint]) -> list[dict[str, float | int]]:
    return [point.as_record() for point in curve]


def table_footnote_checkpoint(
    curve: Sequence[BudgetPoint], delivered_devices: int, delivered_rse: float
) -> dict[str, float | bool]:
    """Whether the delivered fit's precision sits between the allocated columns.

    Ref: SI Table S11 footnote. The delivered pooled fit is not an allocated
    subset, so its precision is checked for lying between the two bracketing
    columns interpolated to its own device count.
    """
    devices = np.array([point.devices for point in curve], dtype=np.float64)
    if delivered_devices < devices.min() or delivered_devices > devices.max():
        raise ValueError("the delivered device count lies outside the curve's range")
    allocated = float(np.interp(delivered_devices, devices, [p.at("d_optimal") for p in curve]))
    clocked = float(np.interp(delivered_devices, devices, [p.at("fixed_schedule") for p in curve]))
    lower, upper = min(allocated, clocked), max(allocated, clocked)
    return {
        "allocated_interpolated": allocated,
        "clocked_interpolated": clocked,
        "delivered_rse": delivered_rse,
        "between_bracketing_columns": bool(lower - 1e-9 <= delivered_rse <= upper + 1e-9),
    }
