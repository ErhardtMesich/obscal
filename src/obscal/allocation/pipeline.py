"""The allocation step: placement, comparator brackets and the precision curve."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from obscal.allocation.curve import (
    BudgetPoint,
    OperatingPoint,
    budget_devices,
    build_curve,
    convergence_at_full_budget,
    d_optimal_order_curve,
    non_crossing_above_oracle,
    operating_point,
    oracle_curve,
    random_envelope_advantage,
    table_footnote_checkpoint,
)
from obscal.allocation.doptimal import (
    InformationState,
    d_optimal_order,
    exchange_step_count,
    gains,
)
from obscal.allocation.policies import (
    is_constrained_against_unconstrained,
    label_policies,
    percentile_bracket,
    policy_orders,
)
from obscal.calibration.attenuation import moment_correction, sample_state_covariance
from obscal.calibration.regression import fit_pooled
from obscal.runtime.config import AllocationConfig

MATRIX_AXIS_INDEX = 1
MINIMUM_FIT_UNITS = 16
"""Below this many units the pooled design is under-determined and the slope has no precision."""


@dataclass(frozen=True)
class AllocationBlock:
    """The biological calibration block the allocator spends, as arrays."""

    regressor_rows: NDArray[np.float64]
    features: NDArray[np.float64]
    state_measured: NDArray[np.float64]
    coverage: NDArray[np.float64]
    times: NDArray[np.float64]
    nuisance: NDArray[np.float64]
    strata: tuple[str, ...]
    arms: tuple[str, ...]
    feature_names: tuple[str, ...]
    residual_precision: NDArray[np.float64]
    error_covariance: NDArray[np.float64]
    minimum_units: int = 16

    def __post_init__(self) -> None:
        n = self.regressor_rows.shape[0]
        if self.features.shape[0] != n:
            raise ValueError("regressor rows and features must share the block axis")
        if len(self.strata) != n or len(self.arms) != n:
            raise ValueError("every unit needs a stratum and an arm label")


@dataclass(frozen=True)
class AllocationOutcome:
    """The curve, the two structural checks and the named operating point."""

    curve: list[BudgetPoint]
    d_optimal_curve: list[BudgetPoint]
    order: list[int]
    exchange_steps: int
    convergence: dict[str, Any]
    non_crossing: dict[str, Any]
    random_envelope: dict[str, Any]
    operating: OperatingPoint
    saving: dict[str, float | int]
    delivered_checkpoint: dict[str, float | bool] | None
    policy_labels: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "exchange_steps": self.exchange_steps,
            "structural_checks": {
                "convergence": self.convergence,
                "non_crossing": self.non_crossing,
                "random_envelope": self.random_envelope,
            },
            "operating_point": self.operating.as_record(),
            "saving": self.saving,
            "delivered_checkpoint": self.delivered_checkpoint,
        }


def make_rse_evaluator(
    block: AllocationBlock,
    *,
    readout_feature: str,
    lambda_value: float,
    time_basis_columns: int,
    include_coverage: bool,
    include_interaction: bool,
) -> Any:
    """Refit the pooled calibration on a subset and return the corrected slope's RSE.

    Ref: SI Table S11 convention. The entry is the relative standard error of the
    errors-in-variables-corrected matrix-axis calibration slope on the designated
    readout at the stated attenuation factor; the correction widens the interval
    by the reciprocal of that factor while the reported slope keeps its
    attenuated value, which is why the corrected figure exceeds the naive one by
    exactly that reciprocal.
    """
    feature_index = block.feature_names.index(readout_feature)

    def evaluate(indices: Any) -> float:
        selection = np.asarray(list(indices), dtype=np.int64)
        if selection.size < 4:
            return float("inf")
        subset = block_subset(block, selection)
        n_bio = int(selection.size)
        if n_bio < block.minimum_units:
            return float("inf")
        pooled = fit_pooled(
            features=subset["features"],
            state_measured=subset["state_measured"],
            coverage=subset["coverage"],
            times=subset["times"],
            nuisance=subset["nuisance"],
            feature_names=block.feature_names,
            time_basis_columns=time_basis_columns,
            include_coverage=include_coverage,
            include_interaction=include_interaction,
            n_biological=n_bio,
        )
        sample = sample_state_covariance(subset["state_measured"])
        corrected = moment_correction(pooled.sensitivity_ols, sample, block.error_covariance)
        slope = float(corrected.corrected_or_ols[feature_index, MATRIX_AXIS_INDEX])
        standard_error = float(pooled.coefficient_se("state::")[MATRIX_AXIS_INDEX, feature_index])
        if slope == 0.0:
            return float("inf")
        return 100.0 * standard_error / abs(slope) / float(lambda_value)

    return evaluate


def block_subset(
    block: AllocationBlock, indices: NDArray[np.int64]
) -> dict[str, NDArray[np.float64]]:
    return {
        "features": block.features[indices],
        "state_measured": block.state_measured[indices],
        "coverage": block.coverage[indices],
        "times": block.times[indices],
        "nuisance": block.nuisance[indices],
        "regressor_rows": block.regressor_rows[indices],
    }


def run_allocation(
    block: AllocationBlock,
    config: AllocationConfig,
    *,
    readout_feature: str,
    lambda_value: float,
    time_basis_columns: int,
    include_coverage: bool = True,
    include_interaction: bool = True,
    seed: int = 0,
    delivered_devices: int | None = None,
    budget_fractions: Sequence[float] | None = None,
) -> AllocationOutcome:
    """Search, price and check the placement of the next destructive assay."""
    block_devices = int(block.regressor_rows.shape[0])
    fractions = (
        list(config.budget_fractions) if budget_fractions is None else list(budget_fractions)
    )
    fractions = [
        value for value in fractions if budget_devices(value, block_devices) >= block.minimum_units
    ]
    if not fractions:
        raise ValueError("no budget point on this block carries enough units for a slope")
    fractions.append(1.0)
    fractions = sorted(set(fractions))
    full_budget = block_devices
    evaluator = make_rse_evaluator(
        block,
        readout_feature=readout_feature,
        lambda_value=lambda_value,
        time_basis_columns=time_basis_columns,
        include_coverage=include_coverage,
        include_interaction=include_interaction,
    )
    order = d_optimal_order(
        block.regressor_rows,
        block.residual_precision,
        full_budget,
        max_exchanges=int(config.policies["prior_art_allocator"].get("exchange_steps", 0)),
    )
    steps = int(config.policies["prior_art_allocator"].get("exchange_steps", 0))
    realised = block.state_measured[:, MATRIX_AXIS_INDEX]
    orders = policy_orders(
        rows=block.regressor_rows,
        times=block.times,
        arms=block.arms,
        realised=realised,
        precision=block.residual_precision,
        budget=full_budget,
        fixed_timepoints=int(config.policies["fixed_schedule"]["timepoints"]),
        random_seeds=int(config.policies["random_matched_rate"]["seeds"]),
        base_seed=seed,
    )
    orders["d_optimal"] = [order]
    orders.pop("clairvoyant_oracle", None)
    curve = build_curve(
        orders=orders,
        budget_fractions=fractions,
        block_devices=block_devices,
        evaluator=evaluator,
    )
    oracle_points = oracle_curve(
        n_candidates=int(block.regressor_rows.shape[0]),
        budget_fractions=fractions,
        block_devices=block_devices,
        evaluator=evaluator,
        minimum_units=MINIMUM_FIT_UNITS,
    )
    curve = [
        BudgetPoint(
            fraction=point.fraction,
            devices=point.devices,
            rse={**point.rse, "clairvoyant_oracle": oracle_points[index].rse["clairvoyant_oracle"]},
            random_low=point.random_low,
            random_high=point.random_high,
            random_draws_used=point.random_draws_used,
        )
        for index, point in enumerate(curve)
    ]
    own_curve = d_optimal_order_curve(order, fractions, block_devices, evaluator)
    target = float(config.operating_point["matched_corrected_rse"]) * 100.0
    columns = tuple(orders.keys())
    operating = operating_point(curve, columns, target, block_devices)
    checkpoint = None
    if delivered_devices is not None:
        delivered = float(
            np.interp(
                delivered_devices,
                [point.devices for point in curve],
                [point.at("d_optimal") for point in curve],
            )
        )
        checkpoint = table_footnote_checkpoint(curve, delivered_devices, delivered)
    return AllocationOutcome(
        curve=curve,
        d_optimal_curve=own_curve,
        order=order,
        exchange_steps=steps,
        convergence=convergence_at_full_budget(curve, columns),
        non_crossing=non_crossing_above_oracle(curve),
        random_envelope=random_envelope_advantage(curve),
        operating=operating,
        saving=operating.saving(),
        delivered_checkpoint=checkpoint,
        policy_labels=label_policies(columns),
    )


def bracket_analysis(curve: list[BudgetPoint], policy: str) -> dict[str, float]:
    """Where a policy sits against the random bracket, printed not summarised."""
    inside = [point for point in curve if point.random_low <= point.at(policy) <= point.random_high]
    below = [point for point in curve if point.at(policy) < point.random_low]
    return {
        "n_points": float(len(curve)),
        "n_inside_random_bracket": float(len(inside)),
        "n_below_random_bracket": float(len(below)),
        "random_p10_median": float(np.median([point.random_low for point in curve])),
        "random_p90_median": float(np.median([point.random_high for point in curve])),
    }


def information_diagnostics(
    rows: NDArray[np.float64], weight: NDArray[np.float64], budget: int
) -> dict[str, float]:
    state = InformationState.initial(rows.shape[1], weight)
    first_gains = gains(state, rows)
    return {
        "n_candidates": float(rows.shape[0]),
        "budget": float(budget),
        "max_first_gain": float(np.max(first_gains)),
        "min_first_gain": float(np.min(first_gains)),
        "condition_number": float(
            np.linalg.cond(rows.T @ rows @ weight) if rows.shape[0] else float("nan")
        ),
    }


def exchange_step_median(
    rows: NDArray[np.float64], weight: NDArray[np.float64], budgets: list[int]
) -> float:
    """Median exchange count across budgets, the figure the per-stage budget quotes."""
    counts = [exchange_step_count(rows, weight, budget) for budget in budgets]
    return float(np.median(counts))


def random_bracket_dispersion(values: list[float], low: float, high: float) -> dict[str, float]:
    p10, p90 = percentile_bracket(values, low, high)
    return {
        "p10": p10,
        "p90": p90,
        "median": float(np.median(values)),
        "n_draws": float(len(values)),
    }


def incompatible_parity_claim(left: str, right: str) -> bool:
    """True when a pair may not be held to parity at full budget."""
    return not is_constrained_against_unconstrained(left, right)
