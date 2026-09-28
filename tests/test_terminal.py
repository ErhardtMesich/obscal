"""Split-chip error, agreement against the standard and channel reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from obscal.terminal import (
    agreement,
    agreement_ordering_is_licensed,
    attenuation_factor,
    attenuation_from_design,
    block_spread,
    channel_reproducibility,
    channels_excellent_on_both_axes,
    common_mass_basis,
    cv_percent,
    cv_spread_ratio,
    error_covariance,
    grade_cv,
    grade_icc,
    half_mean_basis,
    icc_absolute_agreement,
    icc_from_records,
    icc_two_way_random_absolute,
    per_axis_attenuation,
    pooled_within_cv,
    rmse_direct,
    rmse_from_loa,
    single_half_counterpart,
    split_chip_error,
    variance_split,
    within_chip_fraction_from_cv_ratio,
    within_chip_fraction_from_halves,
)

Z_95 = 1.959963984540054


def _halves(seed: int = 3, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = rng.normal(2.0, 1.0, size=n)
    error = rng.normal(0.0, 0.45, size=(n, 2))
    return truth + error[:, 0], truth + error[:, 1]


def test_split_chip_error_follows_the_printed_chain() -> None:
    first, second = _halves()
    error = split_chip_error(first, second)
    width = error.limits_of_agreement_width
    assert error.half_difference_sd == pytest.approx(width / (2 * Z_95), rel=1e-9)
    assert error.single_half_read_error == pytest.approx(error.half_difference_sd / np.sqrt(2))
    assert error.half_mean_error == pytest.approx(error.single_half_read_error / np.sqrt(2))
    assert error.n_devices == first.size


def test_split_chip_error_refuses_a_thin_block() -> None:
    with pytest.raises(ValueError):
        split_chip_error(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        split_chip_error(np.array([1.0, 2.0]), np.array([1.0]))


def test_within_chip_fraction_agrees_between_its_two_routes() -> None:
    first, second = _halves()
    from_halves = within_chip_fraction_from_halves(first, second)
    error_variance = float(np.var(first - second, ddof=1)) / 2.0
    mean_variance = float(np.var(0.5 * (first + second), ddof=1))
    assert from_halves == pytest.approx(error_variance / (mean_variance + error_variance / 2.0))
    assert 0.0 < from_halves < 1.0
    assert within_chip_fraction_from_cv_ratio(0.885) == pytest.approx(0.4335, abs=5e-4)
    with pytest.raises(ValueError):
        within_chip_fraction_from_cv_ratio(1.5)


def test_attenuation_factor_is_one_at_zero_assay_error() -> None:
    assert attenuation_factor(1.0, 0.0) == pytest.approx(1.0)
    assert attenuation_factor(1.0, 1.0) == pytest.approx(0.5)
    assert attenuation_from_design(0.55, 0.19) == pytest.approx(0.8934, abs=5e-4)
    with pytest.raises(ValueError):
        attenuation_factor(0.0, 1.0)


def test_per_axis_attenuation_reproduces_the_scalar_form_on_a_diagonal_covariance() -> None:
    spread = np.array([1.046, 0.820])
    sigma_u = error_covariance(0.303, 0.19, 0.0)
    factors = per_axis_attenuation(spread, sigma_u)
    for index, axis in enumerate(spread):
        assert factors[index] == pytest.approx(
            attenuation_factor(axis, np.sqrt(sigma_u[index, index]))
        )


def test_icc_routes_agree_and_are_bounded() -> None:
    first, second = _halves()
    direct = icc_two_way_random_absolute(np.column_stack([first, second]))
    records = icc_from_records(
        [f"d{index}" for index in range(first.size)] * 2,
        ["a"] * first.size + ["b"] * first.size,
        list(first) + list(second),
    )
    assert direct == pytest.approx(records)
    assert direct == pytest.approx(icc_absolute_agreement(first, second))
    assert -1.0 <= direct <= 1.0


def test_icc_rejects_an_incomplete_table() -> None:
    with pytest.raises(ValueError):
        icc_from_records(["a", "b"], ["r1", "r2"], [1.0, 2.0])
    with pytest.raises(ValueError):
        icc_two_way_random_absolute(np.ones((1, 2)))


def test_agreement_reports_limits_and_an_interval() -> None:
    first, second = _halves()
    row = agreement(first, second, reference_mean=2.0, bootstrap_replicates=10, seed=1)
    assert row.icc_low <= row.icc <= row.icc_high
    assert row.loa_low < row.bias < row.loa_high
    assert row.n == first.size
    assert row.rmse_percent == pytest.approx(
        rmse_from_loa(row.bias, row.loa_low, row.loa_high, 2.0)
    )
    assert rmse_direct(first, second, 2.0) == pytest.approx(row.rmse_percent, rel=1e-9)


def test_agreement_ordering_licenses_only_the_right_direction() -> None:
    licensed = agreement_ordering_is_licensed(0.79, 0.89)
    assert licensed["ordering_licensed"]
    assert not agreement_ordering_is_licensed(0.95, 0.80)["ordering_licensed"]


def test_common_mass_basis_centres_the_difference_on_zero() -> None:
    values = np.array([1.0, 2.0, 3.0])
    placed = common_mass_basis(values, source_mean=2.0, target_mean=1.0)
    assert np.mean(placed) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        common_mass_basis(values, 0.0, 1.0)


def test_block_spread_is_the_quadrature_sum_and_the_weighted_mean() -> None:
    spread = block_spread([1.0, 2.0, 3.0], [0.2, 0.3, 0.4], [4, 4, 4])
    assert spread.total_sd == pytest.approx(np.sqrt(spread.within_sd**2 + spread.between_sd**2))
    assert spread.mean == pytest.approx(2.0)
    assert spread.n_units == 12
    assert spread.cv_percent == pytest.approx(100 * spread.total_sd / spread.mean)


def test_half_mean_basis_is_the_mean_of_pairs() -> None:
    values = np.array([1.0, 3.0, 10.0, 14.0])
    assert half_mean_basis(values).tolist() == [2.0, 12.0]
    with pytest.raises(ValueError):
        half_mean_basis(np.array([1.0, 2.0, 3.0]))


def test_channel_reproducibility_grades_on_the_between_run_axis() -> None:
    groups = [np.array([1.0, 1.02, 0.98]), np.array([1.1, 1.12, 1.08])]
    record = channel_reproducibility(
        channel="terminal/example",
        within_run_groups=groups,
        between_run_means=np.array([1.0, 1.1]),
        between_operator_means=np.array([1.0, 1.05]),
        icc_within_run=0.9,
        icc_between_runs=0.7,
    )
    assert record.grade == grade_cv(record.between_run_cv)
    assert record.as_record()["channel"] == "terminal/example"
    assert grade_cv(3.0) == "excellent"
    assert grade_cv(10.0) == "acceptable"
    assert grade_cv(30.0) == "poor"
    assert grade_icc(0.9) == "excellent"
    assert grade_icc(0.5) == "acceptable"
    assert grade_icc(0.1) == "poor"


def test_pooled_within_cv_and_spread_ratio() -> None:
    cv = pooled_within_cv([np.array([1.0, 1.1, 0.9]), np.array([2.0, 2.2, 1.8])])
    assert cv > 0.0
    assert cv_percent(np.array([1.0, 1.1])) == pytest.approx(
        100 * np.std([1.0, 1.1], ddof=1) / 1.05
    )
    ratio = cv_spread_ratio({"a": 2.8, "b": 18.4})
    assert ratio["worst_channel"] == "b"
    assert ratio["ratio"] == pytest.approx(18.4 / 2.8)
    assert single_half_counterpart(21.5) == pytest.approx(21.5 / 0.885)


def test_channels_excellent_on_both_axes_needs_both_thresholds() -> None:
    good = channel_reproducibility(
        "impedance",
        [np.array([1.0, 1.0])],
        np.array([1.0, 1.04]),
        np.array([1.0, 1.02]),
        0.94,
        0.88,
    )
    poor = channel_reproducibility(
        "acta2", [np.array([1.0, 1.3])], np.array([1.0, 1.4]), np.array([1.0, 1.2]), 0.74, 0.21
    )
    assert channels_excellent_on_both_axes([good, poor]) == ["impedance"]


def test_variance_split_shares_sum_to_one() -> None:
    split = variance_split(0.303, 0.335)
    assert split["within_share"] + split["between_share"] == pytest.approx(1.0)
    assert split["total_sd"] == pytest.approx(np.sqrt(0.303**2 + 0.335**2))
