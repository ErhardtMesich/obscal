"""A minimal training loop on the harness configuration, plus a single-batch overfit.

These exercise data reading, forward, loss, backward, parameter update and
checkpoint save/load on the installed stack, at a scale a test can afford.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from obscal.cohort import Cohort, DeviceRecord
from obscal.runtime.config import RunSpec, load_run_spec
from obscal.runtime.seed import set_seed
from obscal.study import StudyInputs, terminal_sweep_index, twin_window_for
from obscal.twin import (
    OscillatorySpec,
    build_backbone,
    predict_series,
    reconstruction_and_forecast_loss,
)
from obscal.twin.pipeline import TwinWindow, stack_windows


def _windows() -> list[TwinWindow]:
    rng = np.random.default_rng(21)
    windows: list[TwinWindow] = []
    for index in range(12):
        steps = 9
        drift = 0.06 + 0.01 * index
        observed = np.column_stack(
            [
                1.0
                + 0.1 * index / 12
                + drift * np.arange(steps)
                + rng.normal(scale=0.01, size=steps),
                0.6
                + 0.15 * index / 12
                + 1.6 * drift * np.arange(steps)
                + rng.normal(scale=0.02, size=steps),
            ]
        )
        windows.append(
            TwinWindow(
                device_id=f"H{index:03d}",
                observed=observed,
                controls=np.zeros(steps),
                targets={1: observed[-1] + drift, 3: observed[-1] + 3 * drift},
            )
        )
    return windows


def test_smoke_config_is_small_and_labeled(project_root: Path) -> None:
    harness = load_run_spec(project_root / "configs/run/_harness.yaml")
    assert harness.is_harness
    assert harness.option("harness")["training_steps"] == 2


def test_two_step_loop_reduces_the_loss() -> None:
    set_seed(31)
    windows = _windows()
    observed, controls, targets = stack_windows(windows)
    backbone = build_backbone(OscillatorySpec(4, 2, 1, damping=True))
    elapsed = torch.full((1,), 7.0)
    optimiser = torch.optim.Adam(backbone.parameters(), lr=5e-3)
    losses: list[float] = []
    for _ in range(4):
        optimiser.zero_grad()
        loss = reconstruction_and_forecast_loss(backbone, observed, controls, targets, elapsed)
        torch.autograd.backward(loss)
        optimiser.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert np.all(np.isfinite(losses))


def test_single_batch_overfit_drives_the_reconstruction_down() -> None:
    set_seed(37)
    windows = _windows()[:1]
    observed, controls, targets = stack_windows(windows)
    backbone = build_backbone(OscillatorySpec(8, 2, 1, damping=True))
    elapsed = torch.full((1,), 7.0)
    optimiser = torch.optim.Adam(backbone.parameters(), lr=5e-2)
    first = None
    for _ in range(120):
        optimiser.zero_grad()
        loss = reconstruction_and_forecast_loss(backbone, observed, controls, targets, elapsed)
        torch.autograd.backward(loss)
        optimiser.step()
        if first is None:
            first = float(loss.detach())
    final = float(loss.detach())
    assert final < first * 0.5
    prediction = predict_series(backbone, windows[0].observed, 3, 7.0)
    assert prediction.shape == (3, 2)
    assert np.all(np.isfinite(prediction))


def test_forward_loss_backward_update_save_and_load() -> None:
    set_seed(41)
    windows = _windows()
    observed, controls, targets = stack_windows(windows)
    backbone = build_backbone(OscillatorySpec(4, 2, 1, damping=True))
    elapsed = torch.full((1,), 7.0)
    optimiser = torch.optim.SGD(backbone.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in backbone.parameters()]
    optimiser.zero_grad()
    loss = reconstruction_and_forecast_loss(backbone, observed, controls, targets, elapsed)
    torch.autograd.backward(loss)
    assert all(parameter.grad is not None for parameter in backbone.parameters())
    optimiser.step()
    after = [parameter.detach().clone() for parameter in backbone.parameters()]
    assert any(
        not bool(torch.allclose(left, right, rtol=0.0, atol=0.0))
        for left, right in zip(before, after, strict=True)
    )
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "smoke.pt"
        torch.save(backbone.state_dict(), path)
        restored = build_backbone(backbone.spec)
        restored.load_state_dict(torch.load(path, weights_only=True))
    assert np.allclose(
        predict_series(backbone, windows[0].observed, 2, 7.0),
        predict_series(restored, windows[0].observed, 2, 7.0),
        atol=1e-6,
    )


def test_windows_are_drawn_from_the_pool(
    study_inputs: StudyInputs, harness_cohort: Cohort, run_spec: RunSpec
) -> None:
    device: DeviceRecord = next(
        entry
        for entry in harness_cohort.retained
        if entry.block == "biological" and entry.culture_day >= 10
    )
    window = twin_window_for(device, study_inputs)
    assert window.observed.shape[1] == 2
    assert window.observed.shape[0] >= 2
    assert window.controls.shape[0] == window.observed.shape[0]
    assert window.targets
    index = terminal_sweep_index(device)
    assert device.sweeps.n_sweeps == index + 1
    assert run_spec.toggles.damping
