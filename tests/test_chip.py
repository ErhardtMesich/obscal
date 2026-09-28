"""Device geometry, equivalent-circuit fitting and blank-channel instrumentation."""

from __future__ import annotations

import numpy as np
import pytest

from obscal.chip import (
    ChannelGeometry,
    band_information_share,
    baseline_gate,
    blank_channel_from_records,
    bubble_events,
    corrected_area_mm2,
    detect_medium_exchange_steps,
    electrode_drift_ok,
    excitation_bias_from_square_wave,
    fit_sweep,
    geometric_correction_factor,
    half_mass_ratio_ok,
    impedance_model,
    leak_check,
    residual_degrees_of_freedom,
    resolvable_change,
    seeding_coverage_in_band,
    single_frequency_resistance,
    temperature_corrected,
)
from obscal.chip.circuit import CircuitParameters

FREQUENCIES = np.array([100.0, 500.0, 1e3, 4e3, 1e4, 4e4, 1e5])
TRUTH = CircuitParameters(
    r_ext_ohm=120.0, r_barrier_ohm=60.0, c_membrane_f=3.0e-7, cpe_magnitude=1e-5, cpe_exponent=0.86
)


def test_residual_degrees_of_freedom_is_the_over_determination() -> None:
    assert residual_degrees_of_freedom(7) == 9
    assert residual_degrees_of_freedom(2) == -1


def test_impedance_model_is_the_series_sum_it_claims() -> None:
    omega = 2.0 * np.pi * FREQUENCIES.astype(np.complex128)
    barrier = TRUTH.r_barrier_ohm / (1.0 + 1j * omega * TRUTH.r_barrier_ohm * TRUTH.c_membrane_f)
    interface = 1.0 / (TRUTH.cpe_magnitude * (1j * omega) ** TRUTH.cpe_exponent)
    expected = TRUTH.r_ext_ohm + barrier + interface
    assert np.allclose(impedance_model(TRUTH, FREQUENCIES), expected)


def test_fit_recovers_an_exactly_modelled_sweep() -> None:
    fit = fit_sweep(FREQUENCIES, impedance_model(TRUTH, FREQUENCIES))
    assert fit.converged
    assert fit.residual_chi_square < 1e-12
    assert fit.parameters.r_barrier_ohm == pytest.approx(TRUTH.r_barrier_ohm, rel=1e-4)
    assert fit.parameters.cpe_exponent == pytest.approx(TRUTH.cpe_exponent, abs=1e-4)


def test_fit_is_wrong_when_the_sweep_is_generated_from_other_parameters() -> None:
    other = CircuitParameters(120.0, 140.0, 3.0e-7, 1e-5, 0.80)
    fit = fit_sweep(FREQUENCIES, impedance_model(other, FREQUENCIES))
    assert fit.parameters.r_barrier_ohm != pytest.approx(TRUTH.r_barrier_ohm, rel=1e-3)
    assert abs(fit.parameters.cpe_exponent - TRUTH.cpe_exponent) > 1e-3


def test_fit_rejects_misaligned_input() -> None:
    with pytest.raises(ValueError):
        fit_sweep(FREQUENCIES, impedance_model(TRUTH, FREQUENCIES)[:-1])


def test_feature_block_reports_the_three_electrical_entries() -> None:
    fit = fit_sweep(FREQUENCIES, impedance_model(TRUTH, FREQUENCIES))
    features = fit.features(area_mm2=0.5)
    assert set(features) >= {"log_rb", "log_cm", "alpha"}
    assert features["barrier_resistance_area_product"] == pytest.approx(TRUTH.r_barrier_ohm * 0.5)


def test_single_frequency_comparator_reads_a_resistance() -> None:
    sweep = impedance_model(TRUTH, FREQUENCIES)
    value = single_frequency_resistance(1000.0, sweep, FREQUENCIES)
    assert value == pytest.approx(float(sweep[2].real))


def test_geometric_correction_is_one_study_constant() -> None:
    geometry = ChannelGeometry(
        unit_face_width_um=1000.0,
        central_lumen_um=300.0,
        microneedle_inlet_um=200.0,
        channel_height_um=250.0,
        channel_length_mm=12.0,
        flow_rate_uL_per_h=30.0,
        devices_per_reservoir=12,
    )
    factor = geometric_correction_factor(geometry)
    assert 0.0 < factor < 1.0
    assert factor == geometric_correction_factor(geometry)
    assert corrected_area_mm2(geometry, factor) > geometry.floor_area_mm2


def test_geometric_correction_falls_as_the_channel_deepens() -> None:
    shallow = ChannelGeometry(1000.0, 300.0, 200.0, 250.0, 12.0, 30.0, 12)
    deep = ChannelGeometry(1000.0, 300.0, 200.0, 500.0, 12.0, 30.0, 12)
    assert geometric_correction_factor(deep) < geometric_correction_factor(shallow)


def test_pre_seeding_gates_reject_what_they_should() -> None:
    ok, breaches = baseline_gate([1.0, 1.0], [1.0, 1.0], [0.1, 0.1], 3.0)
    assert ok and breaches == 0
    ok, breaches = baseline_gate([1.5, 1.0], [1.0, 1.0], [0.1, 0.1], 3.0)
    assert not ok and breaches == 1
    assert leak_check(0.04, 0.05)
    assert not leak_check(0.06, 0.05)
    assert seeding_coverage_in_band(90.0, [0.82, 0.96])
    assert not seeding_coverage_in_band(70.0, [0.82, 0.96])
    assert half_mass_ratio_ok(0.5, [0.4, 0.6])
    assert not half_mass_ratio_ok(0.9, [0.4, 0.6])
    assert electrode_drift_ok(10.0, 15.0)
    assert not electrode_drift_ok(20.0, 15.0)


def test_resolvable_change_is_a_multiple_of_the_measured_drift() -> None:
    assert resolvable_change(5.2, 14.0, 3.0) == pytest.approx(218.4)


def test_blank_channel_reports_its_own_drift_and_step() -> None:
    channel = blank_channel_from_records(
        frequency_hz=[100.0, 1000.0],
        day0_magnitude_ohm=[4820.0, 1930.0],
        drift_ohm_per_day=[11.4, 5.2],
        medium_step_ohm=[38.6, 21.3],
        temperature_coefficient_pct_per_c=[-1.74, -1.61],
    )
    assert channel.relative_drift_pct_per_day[0] == pytest.approx(100 * 11.4 / 4820.0)
    assert channel.at(900.0).frequency_hz == 1000.0
    trace = channel.blank_trace(np.array([0.0, 14.0]))
    assert trace[1, 0] == pytest.approx(4820.0 + 14.0 * 11.4)
    with pytest.raises(ValueError):
        blank_channel_from_records([1000.0, 100.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])


def test_temperature_correction_inverts_its_own_coefficient() -> None:
    measured = np.array([110.0, 90.0])
    temperature = np.array([38.0, 36.0])
    coefficients = np.array([-1.5, -1.5])
    corrected = temperature_corrected(measured, temperature, 37.0, coefficients)
    recovered = temperature_corrected(corrected, 37.0 - temperature, 0.0, coefficients)
    assert np.allclose(corrected, measured / (1.0 + coefficients * (temperature - 37.0) / 100.0))
    assert np.all(np.isfinite(recovered))


def test_medium_exchange_steps_are_discrete_and_separate_from_drift() -> None:
    magnitude = np.array([100.0, 100.1, 125.0, 125.2])
    flags = detect_medium_exchange_steps(magnitude, np.arange(4.0), jump_threshold=5.0)
    assert flags.tolist() == [0, 0, 1, 0]


def test_bubble_events_flag_excursions() -> None:
    magnitude = np.array([100.0, 130.0, 101.0])
    baseline = np.array([100.0, 100.0, 100.0])
    assert bubble_events(magnitude, baseline, 0.25).tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        bubble_events(magnitude, baseline[:-1], 0.25)


def test_excitation_bias_is_carried_inside_its_published_band() -> None:
    bias = excitation_bias_from_square_wave(
        np.array([100.0] * 3), np.array([94.0] * 3), (0.17, 8.96)
    )
    assert bias.resistance_pct == pytest.approx(-6.0)
    assert bias.within_published_band
    assert not excitation_bias_from_square_wave(
        np.array([100.0]), np.array([50.0]), (0.17, 8.96)
    ).within_published_band


def test_band_information_shares_sum_to_one_hundred() -> None:
    shares = band_information_share([0.47, 0.71, 0.66])
    assert sum(shares) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        band_information_share([0.0, 0.0])
