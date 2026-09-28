"""D-optimal placement, the comparator brackets and the precision curve."""

from __future__ import annotations

import numpy as np
import pytest

from obscal.allocation import (
    InformationState,
    budget_devices,
    build_curve,
    convergence_at_full_budget,
    d_optimal_order,
    gains,
    marginal_gain,
    non_crossing_above_oracle,
    operating_point,
    policy_orders,
    prior_art_allocator_order,
    random_bracket,
    random_matched_rate_order,
    select_by_policy,
    table_footnote_checkpoint,
)
from obscal.allocation.curve import oracle_curve


def _pool(n: int = 30, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(n, 2))
    weight = np.linalg.inv(np.array([[0.04, 0.0], [0.0, 0.09]]))
    return rows, weight


def test_information_state_log_determinant_grows_with_information() -> None:
    rows, weight = _pool()
    state = InformationState.initial(2, weight)
    assert state.spent == 0
    assert state.as_record()["spent"] == 0
    for index in range(2):
        before = state.log_determinant
        state = state.with_row(rows[index])
        assert state.spent == index + 1
        assert state.log_determinant > before
    # A rank-one information matrix is singular, so the criterion carries a
    # numerical ridge rather than collapsing to a determinant of zero.
    assert np.isfinite(state.log_determinant)


def test_marginal_gain_matches_a_direct_determinant_difference() -> None:
    rows, weight = _pool()
    state = InformationState.initial(2, weight).with_row(rows[0])
    gain = marginal_gain(state, rows[1])
    candidate = InformationState(
        matrix=state.matrix + np.outer(rows[1], rows[1]) @ weight, weight=weight, spent=2
    )
    assert gain == pytest.approx(candidate.log_determinant - state.log_determinant, rel=1e-9)


def test_greedy_search_is_monotone_and_takes_the_first_argmax() -> None:
    rows, weight = _pool()
    order = d_optimal_order(rows, weight, 6)
    assert len(order) == len(set(order)) == 6
    initial = InformationState.initial(2, weight)
    assert int(np.argmax(gains(initial, rows))) == order[0]
    state = initial
    previous = state.log_determinant
    for index in order:
        state = state.with_row(rows[index])
        assert state.log_determinant >= previous - 1e-9
        previous = state.log_determinant


def test_exchange_pass_never_lowers_the_determinant() -> None:
    rows, weight = _pool(seed=7)
    greedy = d_optimal_order(rows, weight, 8)
    refined = d_optimal_order(rows, weight, 8, max_exchanges=20)
    assert len(refined) == 8
    greedy_value = _determinant(rows, weight, greedy)
    refined_value = _determinant(rows, weight, refined)
    assert refined_value >= greedy_value - 1e-9


def _determinant(rows: np.ndarray, weight: np.ndarray, selected: list[int]) -> float:
    state = InformationState.initial(rows.shape[1], weight)
    for index in selected:
        state = state.with_row(rows[index])
    return state.log_determinant


def test_budget_rejects_an_overspend() -> None:
    rows, weight = _pool(6)
    with pytest.raises(ValueError):
        d_optimal_order(rows, weight, 40)


def test_policies_fill_the_budget_without_repeats() -> None:
    rows, weight = _pool(20)
    random_order = random_matched_rate_order(20, 8, seed=1)
    prior_order = prior_art_allocator_order(rows, 8)
    assert len(random_order) == len(set(random_order)) == 8
    assert len(prior_order) == len(set(prior_order)) == 8
    bracket = random_bracket(20, 8, seeds=6, base_seed=3)
    assert len(bracket) == 6
    assert bracket[0] != bracket[1]
    with pytest.raises(ValueError):
        select_by_policy(lambda rows, weight, budget: [0, 0], rows, weight, 2)


def test_prior_art_allocator_spreads_further_than_a_prefix() -> None:
    rows, weight = _pool(24, seed=11)
    prior = prior_art_allocator_order(rows, 12)
    prefix = list(range(12))
    assert _mean_nearest(rows, prior) > _mean_nearest(rows, prefix)


def _mean_nearest(rows: np.ndarray, order: list[int]) -> float:
    selected = rows[order]
    distances = np.linalg.norm(selected[:, None, :] - selected[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(np.mean(np.min(distances, axis=1)))


def test_budget_devices_and_curve_reading() -> None:
    assert budget_devices(0.5, 148) == 74
    with pytest.raises(ValueError):
        budget_devices(0.0, 148)
    rows, weight = _pool(24, seed=13)

    def evaluator(indices: list[int]) -> float:
        return 100.0 / max(len(indices), 1)

    orders = policy_orders(
        rows=rows,
        times=np.linspace(3.0, 21.0, 24),
        arms=["S1"] * 12 + ["S2"] * 12,
        realised=rows[:, 1],
        precision=weight,
        budget=24,
        fixed_timepoints=4,
        random_seeds=8,
        base_seed=1,
    )
    orders["d_optimal"] = [list(range(24))]
    curve = build_curve(
        orders=orders, budget_fractions=[0.5, 1.0], block_devices=24, evaluator=evaluator
    )
    assert len(curve) == 2
    assert curve[0].devices == 12
    assert curve[0].random_draws_used == 8
    parity = convergence_at_full_budget(curve, ("d_optimal", "fixed_schedule"))
    assert "converged" in parity
    crossing = non_crossing_above_oracle(curve)
    assert "non_crossing" in crossing


def test_oracle_curve_is_a_ceiling_on_a_planted_objective() -> None:
    rows, weight = _pool(20, seed=17)

    def evaluator(indices: list[int]) -> float:
        return 100.0 / max(len(indices), 1)

    orders = {"d_optimal": [list(range(20))]}
    d_curve = build_curve(
        orders=orders, budget_fractions=[0.5], block_devices=20, evaluator=evaluator
    )
    oracle = oracle_curve(
        n_candidates=20,
        budget_fractions=[0.5],
        block_devices=20,
        evaluator=evaluator,
        minimum_units=5,
    )
    assert oracle[0].rse["clairvoyant_oracle"] <= d_curve[0].at("d_optimal") + 1e-9


def test_operating_point_interpolates_between_measured_points() -> None:
    rows, weight = _pool(24)

    def evaluator(indices: list[int]) -> float:
        return 100.0 - 2.0 * len(indices)

    orders = {"d_optimal": [list(range(24))], "fixed_schedule": [list(range(24))]}
    curve = build_curve(
        orders=orders,
        budget_fractions=[0.25, 0.5, 0.75, 1.0],
        block_devices=24,
        evaluator=evaluator,
    )
    point = operating_point(curve, ("d_optimal", "fixed_schedule"), 70.0, 24)
    assert 12 <= point.devices_by_policy["d_optimal"] <= 15
    assert "interpolated_between_measured_points" in point.as_record()
    checkpoint = table_footnote_checkpoint(curve, 18, 64.0)
    assert 1.0e-9 <= checkpoint["allocated_interpolated"] <= 100.0
