"""Sensor feature assembly, band sensitivity and the cell-free channel."""

from __future__ import annotations

import numpy as np
import pytest

from obscal.chip import fit_sweep, impedance_model
from obscal.chip.circuit import CircuitParameters
from obscal.frontend import (
    FEATURE_DIMENSION,
    FULL_FRONT_END,
    BandSensitivity,
    assemble_features,
    band_floor,
    band_ordering,
    band_table,
    centre_on_block_mean,
    coefficient_of_determination,
    coverage_matrix_gap,
    fused_front_end_sensitivity,
    interquartile_range,
    matrix_information_shares,
    rank_perturbations,
    residual_contribution,
    schedule_prediction,
    sensitivity_ratio,
    standardised_sensitivity,
)

FREQUENCIES = np.array([100.0, 500.0, 1e3, 4e3, 1e4, 4e4, 1e5])
TRUTH = CircuitParameters(120.0, 60.0, 3.0e-7, 1e-5, 0.86)


def _fit() -> object:
    return fit_sweep(FREQUENCIES, impedance_model(TRUTH, FREQUENCIES))


def test_fused_front_end_holds_the_printed_dimension() -> None:
    vector = assemble_features(
        _fit(),
        area_mm2=0.5,
        matrix_texture_x1e3=5.0,
        effluent_pro_collagen_ng_per_ml=30.0,
        coverage_percent=88.0,
        device_id="CHIP0001",
        culture_day=21.0,
    )
    assert vector.dimension == FEATURE_DIMENSION
    assert vector.names == FULL_FRONT_END
    assert "coverage" not in vector.names
    assert vector.get("matrix_texture") == pytest.approx(5.0)


def test_single_frequency_front_end_collapses_the_electrical_block() -> None:
    vector = assemble_features(
        _fit(),
        area_mm2=0.5,
        matrix_texture_x1e3=5.0,
        effluent_pro_collagen_ng_per_ml=30.0,
        coverage_percent=88.0,
        device_id="CHIP0001",
        culture_day=21.0,
        multi_frequency=False,
    )
    assert vector.dimension == 3


def test_feature_vector_rejects_mismatched_names() -> None:
    vector = assemble_features(_fit(), 0.5, 5.0, 30.0, 88.0, device_id="c", culture_day=21.0)
    with pytest.raises(ValueError):
        type(vector)(device_id="c", culture_day=21.0, values=(1.0, 2.0), coverage_percent=1.0)


def test_standardised_sensitivity_is_scale_free() -> None:
    assert standardised_sensitivity(0.5, 2.0, 1.0) == pytest.approx(1.0)
    assert sensitivity_ratio(8.12, 1.09) == pytest.approx(7.4495, rel=1e-4)
    with pytest.raises(ValueError):
        standardised_sensitivity(0.5, 2.0, 0.0)


def test_centring_and_unit_scaling_are_idempotent_in_the_right_sense() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    centred = centre_on_block_mean(values)
    assert np.mean(centred) == pytest.approx(0.0)
    assert interquartile_range(values) == pytest.approx(1.5)


def test_band_table_shares_partition_the_matrix_information() -> None:
    table = band_table(
        [
            BandSensitivity(frequency_hz=100.0, activation=0.21, matrix=0.47, coverage=1.84),
            BandSensitivity(frequency_hz=500.0, activation=0.34, matrix=0.71, coverage=2.36),
            BandSensitivity(frequency_hz=1000.0, activation=0.46, matrix=0.66, coverage=3.12),
        ]
    )
    assert sum(table.matrix_information_share) == pytest.approx(100.0)
    assert table.peak_frequency("matrix") == 500.0
    assert table.peak_frequency("coverage") == 1000.0
    assert table.resistive_band_share(500.0, 1000.0) == pytest.approx(
        sum(table.matrix_information_share[1:])
    )


def test_band_ordering_ranks_by_descending_sensitivity() -> None:
    table = band_table(
        [
            BandSensitivity(frequency_hz=100.0, activation=0.1, matrix=0.2, coverage=1.0),
            BandSensitivity(frequency_hz=500.0, activation=0.4, matrix=0.9, coverage=0.5),
        ]
    )
    assert band_ordering(table, "matrix") == [500.0, 100.0]
    assert band_ordering(table, "activation") == [500.0, 100.0]
    with pytest.raises(ValueError):
        band_ordering(table, "unknown")


def test_matrix_information_shares_are_the_squared_share_of_the_total() -> None:
    shares = matrix_information_shares([0.47, 0.71])
    total = 0.47**2 + 0.71**2
    assert shares[0] == pytest.approx(100 * 0.47**2 / total)


def test_fused_sensitivity_exceeds_every_constituent_band() -> None:
    fused = fused_front_end_sensitivity([0.71, 0.66], [0.5, 0.4])
    assert fused > 0.71
    assert fused == pytest.approx(np.sqrt(0.71**2 + 0.66**2 + 0.5**2 + 0.4**2))


def test_coverage_matrix_gap_reports_both_ways_round() -> None:
    gap = coverage_matrix_gap(8.12, 1.09)
    assert gap["coverage_to_matrix_ratio"] == pytest.approx(7.4495, rel=1e-4)
    assert gap["matrix_share_of_coverage"] == pytest.approx(1.09 / 8.12)


def test_band_floor_falls_with_a_stronger_band() -> None:
    assert band_floor(0.71, 0.74, 1.1968) < band_floor(0.47, 0.74, 1.1968)


def test_schedule_prediction_recovers_a_planted_trend() -> None:
    days = np.linspace(1.0, 21.0, 40)
    channel = 0.8 * days + 0.4
    prediction = schedule_prediction(days, channel)
    assert prediction.r_squared > 0.99
    assert prediction.n_observations == 40


def test_schedule_prediction_reports_the_residual_contribution_with_covariates() -> None:
    rng = np.random.default_rng(4)
    days = np.linspace(1.0, 21.0, 60)
    covariate = rng.normal(size=(60, 2))
    channel = 0.5 * days + covariate[:, 0] * 3.0
    prediction = schedule_prediction(days, channel, modelled_covariates=covariate)
    assert 0.0 <= prediction.residual_contribution_after_covariates <= 1.0


def test_coefficient_of_determination_and_residual_contribution() -> None:
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    assert coefficient_of_determination(observed, observed) == pytest.approx(1.0)
    assert residual_contribution(1.0, 3.0, 8.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        coefficient_of_determination(observed, observed[:-1])
    with pytest.raises(ValueError):
        residual_contribution(1.0, 2.0, 0.0)


def test_perturbation_ranking_orders_by_added_error() -> None:
    ranked = rank_perturbations([("medium_swap", 0.07), ("no_covariates", 0.57)])
    assert [name for name, _ in ranked] == ["no_covariates", "medium_swap"]
