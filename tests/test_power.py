"""The exact noncentral-t coefficient and the detectability floor."""

from __future__ import annotations

import numpy as np
import pytest

from obscal.power import (
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

PRINTED = {3: 3.0709, 4: 2.3808, 6: 1.7955, 12: 1.1968, 16: 1.0237, 24: 0.8261}


@pytest.mark.parametrize("size,printed", sorted(PRINTED.items()))
def test_coefficient_reproduces_the_printed_value(size: int, printed: float) -> None:
    assert two_group_coefficient(size) == pytest.approx(printed, abs=5e-5)


def test_coefficient_is_monotone_and_refuses_a_singleton_group() -> None:
    values = [two_group_coefficient(size) for size in (3, 4, 6, 12, 24)]
    assert all(later < earlier for earlier, later in zip(values, values[1:]))
    with pytest.raises(ValueError):
        two_group_coefficient(1)


def test_noncentral_helpers_agree_with_the_coefficient() -> None:
    df = 2 * 12 - 2
    ncp = noncentrality_for_power(df, 0.05, 0.80)
    assert two_group_coefficient(12) == pytest.approx(ncp / np.sqrt(12 / 2))
    assert noncentral_t_critical(df, 0.05) == pytest.approx(2.0739, abs=1e-4)
    with pytest.raises(ValueError):
        noncentral_t_critical(0, 0.05)


def test_coefficient_grid_and_floor_follow_the_formula() -> None:
    grid = coefficient_grid([12, 4])
    assert grid[4] == pytest.approx(PRINTED[4], abs=5e-5)
    floor = floor_from_sigma("matrix", "ug_hyp", 0.74, 12)
    assert floor.floor == pytest.approx(minimum_detectable_difference(grid[12], 0.74))
    assert floor.coefficient == pytest.approx(PRINTED[12], abs=5e-5)
    floors = floor_grid("matrix", "ug_hyp", 0.74, [12, 24])
    assert len(floors) == 2
    with pytest.raises(ValueError):
        floor_from_sigma("matrix", "ug_hyp", 0.0, 12)


def test_replication_requirement_inverts_the_floor() -> None:
    requirement = replication_requirement(50.0, 41.4, [3, 4, 6, 12, 16, 24])
    assert requirement["n"] == 12
    assert requirement["detectable_percent"] == pytest.approx(49.55, abs=0.05)
    unreachable = replication_requirement(1.0, 41.4, [3, 4, 6])
    assert not unreachable["reached"]


def test_estimator_covariance_and_directional_floor() -> None:
    sensitivity = np.array([[0.5, 0.2], [0.4, 0.9]])
    residual = np.diag([0.01, 0.04])
    information = sensitivity.T @ np.linalg.inv(residual) @ sensitivity
    covariance = estimator_covariance(sensitivity, residual)
    assert np.allclose(covariance, np.linalg.inv(information))
    floor = directional_floor(covariance, direction_vector("matrix", ("activation", "matrix")), 12)
    assert floor > 0.0
    named = floor_from_estimator(
        sensitivity, residual, ("activation", "matrix"), "activation", 12, "fold"
    )
    assert named.unit == "fold"
    assert named.floor == pytest.approx(named.coefficient * named.sigma)
    with pytest.raises(KeyError):
        direction_vector("other", ("activation", "matrix"))


def test_floor_ratio_and_axis_independence() -> None:
    assert floor_ratio(0.89, 1.76) == pytest.approx(0.89 / 1.76)
    with pytest.raises(ValueError):
        floor_ratio(1.0, 0.0)
