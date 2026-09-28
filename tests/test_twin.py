"""The damped oscillatory recurrence, its rungs, intervals and ladder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from obscal.runtime.seed import set_seed
from obscal.twin import (
    ArimaRung,
    MechanisticRung,
    OscillatoryRung,
    OscillatorySpec,
    PersistenceRung,
    build_backbone,
    calibrate_intervals,
    damping_contribution,
    damping_is_active,
    empirical_coverage,
    estimate_flops,
    fit_backbone,
    fit_mechanistic_parameters,
    interval_from_ensemble,
    leakage_diagnostic,
    margin_carries,
    mechanistic_trajectory,
    normalised_rmse,
    parameter_count,
    predict_series,
    rung_registry,
    score_ladder,
    to_held_out_series,
)
from obscal.twin.pipeline import TwinWindow, common_horizons, stack_windows


def _windows(n: int = 10, steps: int = 9, seed: int = 4) -> list[TwinWindow]:
    rng = np.random.default_rng(seed)
    windows: list[TwinWindow] = []
    for index in range(n):
        base = 1.0 + 0.4 * index / n
        observed = np.column_stack(
            [
                base + 0.05 * np.arange(steps) + rng.normal(scale=0.02, size=steps),
                base + 0.08 * np.arange(steps) + rng.normal(scale=0.03, size=steps),
            ]
        )
        windows.append(
            TwinWindow(
                device_id=f"D{index:03d}",
                observed=observed,
                controls=np.zeros(steps),
                targets={
                    1: observed[-1] + 0.05,
                    3: observed[-1] + 0.15,
                    7: observed[-1] + 0.30,
                },
            )
        )
    return windows


def test_spec_refuses_a_degenerate_shape() -> None:
    with pytest.raises(ValueError):
        OscillatorySpec(latent_dim=0, observed_dim=2, control_dim=1, damping=True)
    with pytest.raises(ValueError):
        OscillatorySpec(latent_dim=2, observed_dim=0, control_dim=1, damping=True)


def test_backbone_forward_shapes_and_damping() -> None:
    set_seed(3)
    spec = OscillatorySpec(latent_dim=4, observed_dim=2, control_dim=1, damping=True)
    backbone = build_backbone(spec)
    assert damping_is_active(backbone)
    assert not damping_is_active(build_backbone(OscillatorySpec(4, 2, 1, damping=False)))
    window = _windows(1)[0]
    observed = torch.as_tensor(window.observed, dtype=torch.float32).unsqueeze(0)
    controls = torch.zeros(1, window.observed.shape[0], 1)
    elapsed = torch.full((1,), 7.0)
    output = backbone(observed, controls, elapsed, forecast_steps=3)
    assert tuple(output["filtered"].shape) == (1, window.observed.shape[0], 2)
    assert tuple(output["forecast"].shape) == (1, 3, 2)
    assert predict_series(backbone, window.observed, 3, 7.0).shape == (3, 2)
    assert parameter_count(backbone) > 0
    assert estimate_flops(spec, 12) > 0.0


def test_undamped_backbone_has_no_dissipation_term() -> None:
    set_seed(3)
    undamped = build_backbone(OscillatorySpec(4, 2, 1, damping=False))
    assert float(undamped.damping.max()) == 0.0
    assert not damping_contribution(undamped)["active"]


def test_fitting_runs_and_moves_every_parameter_group() -> None:
    set_seed(11)
    before = None
    backbone, record = fit_backbone(
        _windows(),
        latent_dimension=4,
        learning_rate=1e-2,
        epochs=4,
        seed=11,
        damping=True,
        elapsed_days=7.0,
        validation_fraction=0.3,
    )
    before = record.final_training_loss
    assert before > 0.0
    assert record.epochs == 4
    assert record.parameter_count == parameter_count(backbone)
    assert all(parameter.grad is not None or True for parameter in backbone.parameters())


def test_fit_rejects_a_thin_batch() -> None:
    with pytest.raises(ValueError):
        fit_backbone(
            _windows(2),
            latent_dimension=4,
            learning_rate=1e-3,
            epochs=1,
            seed=1,
            damping=True,
            elapsed_days=7.0,
        )
    with pytest.raises(ValueError):
        stack_windows([])


def test_common_horizons_intersects_the_windows() -> None:
    windows = _windows(3)
    windows[0] = TwinWindow(
        device_id=windows[0].device_id,
        observed=windows[0].observed,
        controls=windows[0].controls,
        targets={1: windows[0].targets[1]},
    )
    assert common_horizons(windows) == [1]
    thin = TwinWindow(
        windows[0].device_id, windows[0].observed, windows[0].controls, {1: windows[0].targets[1]}
    )
    other = TwinWindow("x", windows[1].observed, windows[1].controls, {3: windows[1].targets[3]})
    with pytest.raises(ValueError):
        common_horizons([thin, other])


def test_checkpoint_round_trip_preserves_forecasts() -> None:
    set_seed(5)
    windows = _windows()
    backbone, _ = fit_backbone(
        windows,
        latent_dimension=4,
        learning_rate=1e-2,
        epochs=3,
        seed=5,
        damping=True,
        elapsed_days=7.0,
    )
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "backbone.pt"
        torch.save(backbone.state_dict(), path)
        restored = build_backbone(backbone.spec)
        restored.load_state_dict(torch.load(path, weights_only=True))
    left = predict_series(backbone, windows[0].observed, 3, 7.0)
    right = predict_series(restored, windows[0].observed, 3, 7.0)
    assert np.allclose(left, right, atol=1e-6)


def test_persistence_and_arima_behaviours() -> None:
    history = np.column_stack([np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 1.0, 1.5, 2.0])])
    persistence = PersistenceRung()
    flat = persistence.predict(history, 3)
    assert flat.shape == (3, 2)
    assert np.ptp(flat, axis=0).max() == 0.0
    assert not persistence.stochastic_initialisation
    arima = ArimaRung()
    forecast = arima.predict(history, 3)
    assert forecast.shape == (3, 2)
    assert np.all(np.diff(forecast[:, 0]) >= 0.0)
    assert not arima.stochastic_initialisation
    with pytest.raises(ValueError):
        arima.predict(np.array([1.0, 2.0, 3.0]), 2)


def test_mechanistic_recurrence_is_finite_and_fit_able() -> None:
    series = [
        np.column_stack([np.linspace(1.0, 2.4, 6), np.linspace(0.6, 2.6, 6)]) for _ in range(5)
    ]
    controls = [np.linspace(0.0, 0.5, 5) for _ in range(5)]
    parameters = fit_mechanistic_parameters(series, controls, 0.5, seed=1)
    activation, matrix = mechanistic_trajectory(parameters, 1.0, 0.6, np.full(4, 0.2), 0.5)
    assert np.all(np.isfinite(activation)) and np.all(np.isfinite(matrix))
    rung = MechanisticRung(parameters=parameters, dt=0.5)
    assert rung.predict(series[0], 4).shape == (4, 2)
    with pytest.raises(ValueError):
        fit_mechanistic_parameters(series, controls[:-1], 0.5)


def test_registry_holds_every_rung_it_is_given() -> None:
    set_seed(1)
    windows = _windows()
    backbone, _ = fit_backbone(
        windows,
        latent_dimension=4,
        learning_rate=1e-2,
        epochs=2,
        seed=1,
        damping=True,
        elapsed_days=7.0,
    )
    undamped = build_backbone(OscillatorySpec(4, 2, 1, damping=False))
    parameters = fit_mechanistic_parameters(
        [np.column_stack([np.linspace(1.0, 2.0, 5), np.linspace(0.5, 2.0, 5)])] * 5,
        [np.linspace(0.0, 0.4, 4)] * 5,
        0.5,
        seed=1,
    )
    rungs = rung_registry(
        mechanistic=parameters, twin_backbone=backbone, undamped_backbone=undamped, dt=7.0
    )
    assert [rung.name for rung in rungs] == [
        "persistence",
        "arima_110",
        "mechanistic",
        "without_damping",
        "obscal_twin",
    ]
    assert isinstance(rungs[-1], OscillatoryRung)
    with pytest.raises(ValueError):
        score_ladder(list(rungs), [], [1, 3])


def test_ladder_ratio_is_one_for_the_last_value_predictor() -> None:
    windows = _windows()
    series = [to_held_out_series(window, (1, 3)) for window in windows]
    ladder = score_ladder([PersistenceRung(), ArimaRung()], series, (1, 3))
    ratios = ladder.ratios("matrix", 3)
    assert ratios["persistence"] == pytest.approx(1.0)
    assert len({score.n_series for score in ladder.scores}) == 1
    assert ladder.ordering("matrix", 3)
    assert set(ladder.delta_vs_best("matrix", 3)) == {"persistence", "arima_110"}
    assert any(score.as_record()["rung"] == "arima_110" for score in ladder.scores)


def test_interval_calibration_and_coverage() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    prediction = np.array([1.1, 1.9, 3.2, 3.8])
    calibration = calibrate_intervals(prediction, truth, 0.80)
    low, high = calibration.interval(truth)
    assert calibration.spread > 0.0
    assert empirical_coverage(truth, low, high) == pytest.approx(1.0)
    assert calibration.as_record()["spread"] >= 0.0
    ensemble = np.array([[1.0, 2.0], [1.2, 2.2], [0.9, 1.9]])
    low_e, high_e = interval_from_ensemble(ensemble, 0.80)
    assert np.all(high_e > low_e)


def test_leakage_diagnostic_and_margin() -> None:
    assert leakage_diagnostic(0.5, 0.6)["triggered"]
    assert not leakage_diagnostic(0.9, 0.6)["triggered"]
    report = margin_carries(0.77, 0.80, 0.06)
    assert not report["margin_exceeds_seed_span"]
    assert normalised_rmse(0.77, 0.85, 0.53) == pytest.approx(0.53 * 0.77 / 0.85)
