"""Fitting the damped oscillatory twin and consuming the calibrated front end.

Ref: Sec. Forecasting and terminal-assay allocation, Eq. (7); SI Table S1 Panel A.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from obscal.runtime.config import TwinConfig
from obscal.runtime.seed import set_seed
from obscal.twin.ladder import HeldOutSeries
from obscal.twin.oscillatory import (
    DampedOscillatoryBackbone,
    OscillatorySpec,
    build_backbone,
    damping_contribution,
    estimate_flops,
    parameter_count,
    stiffness_spectrum,
)
from obscal.twin.rungs import (
    MechanisticParameters,
    Rung,
    fit_mechanistic_parameters,
    rung_registry,
)

CONTROL_DIM = 1


@dataclass(frozen=True)
class TwinWindow:
    """One device's observed front-end series and the destructive values it forecasts to."""

    device_id: str
    observed: NDArray[np.float64]
    controls: NDArray[np.float64]
    targets: dict[int, NDArray[np.float64]]

    def __post_init__(self) -> None:
        if self.observed.ndim != 2 or self.observed.shape[1] != 2:
            raise ValueError("the observed window must be steps by two axes")
        if self.controls.shape != (self.observed.shape[0],):
            raise ValueError("every observed step needs one control value")


@dataclass(frozen=True)
class TrainingRecord:
    """What the fitting pass actually produced, reported rather than asserted."""

    epochs: int
    best_validation_loss: float
    best_epoch: int
    final_training_loss: float
    gradient_norm_sd_last_window: float
    learning_rate: float
    latent_dimension: int
    parameter_count: int
    flops_millions: float
    damping: dict[str, float]
    stiffness: dict[str, float]

    def as_record(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "best_validation_loss": self.best_validation_loss,
            "best_epoch": self.best_epoch,
            "final_training_loss": self.final_training_loss,
            "gradient_norm_sd_last_window": self.gradient_norm_sd_last_window,
            "learning_rate": self.learning_rate,
            "latent_dimension": self.latent_dimension,
            "parameter_count": self.parameter_count,
            "flops_millions": self.flops_millions,
            "damping": self.damping,
            "stiffness": self.stiffness,
        }


def backpropagate(loss: Tensor) -> None:
    """Typed wrapper, so no call site needs an untyped-call suppression."""
    torch.autograd.backward(loss)


def common_horizons(windows: Sequence[TwinWindow]) -> list[int]:
    """The horizons every window in a batch can be scored on."""
    if not windows:
        raise ValueError("a batch needs at least one window")
    shared = set(windows[0].targets)
    for window in windows[1:]:
        shared &= set(window.targets)
    if not shared:
        raise ValueError("the windows share no forecast horizon")
    return sorted(shared)


def stack_windows(windows: Sequence[TwinWindow]) -> tuple[Tensor, Tensor, Tensor]:
    observed = torch.as_tensor(
        np.stack([window.observed for window in windows]), dtype=torch.float32
    )
    controls = torch.as_tensor(
        np.stack([window.controls for window in windows]), dtype=torch.float32
    ).unsqueeze(-1)
    horizons = common_horizons(windows)
    targets = torch.as_tensor(
        np.stack(
            [np.stack([window.targets[horizon] for horizon in horizons]) for window in windows]
        ),
        dtype=torch.float32,
    )
    return observed, controls, targets


def reconstruction_and_forecast_loss(
    backbone: DampedOscillatoryBackbone,
    observed: Tensor,
    controls: Tensor,
    targets: Tensor,
    elapsed: Tensor,
) -> Tensor:
    """One-pass objective: reproduce the observed window, then reach the held-out values."""
    steps = targets.shape[1]
    output = backbone(observed, controls, elapsed, forecast_steps=steps)
    filtered: Tensor = output["filtered"]
    forecast: Tensor = output["forecast"]
    reconstruction: Tensor = torch.mean((filtered - observed) ** 2)
    prediction: Tensor = torch.mean((forecast - targets) ** 2)
    return reconstruction + prediction


def fit_backbone(
    windows: Sequence[TwinWindow],
    *,
    latent_dimension: int,
    learning_rate: float,
    epochs: int,
    seed: int,
    damping: bool,
    elapsed_days: float,
    validation_fraction: float = 0.25,
    readout_hidden: int = 32,
    record_gradient_window: int = 20,
) -> tuple[DampedOscillatoryBackbone, TrainingRecord]:
    """Fit Theta on held-in devices and freeze it before any held-out device is scored."""
    if len(windows) < 4:
        raise ValueError("fitting the backbone needs at least four held-in devices")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("the validation fraction must lie in [0, 1)")
    set_seed(seed)
    spec = OscillatorySpec(
        latent_dim=latent_dimension,
        observed_dim=2,
        control_dim=CONTROL_DIM,
        damping=damping,
        readout_hidden=readout_hidden,
    )
    backbone = build_backbone(spec)
    observed, controls, targets = stack_windows(windows)
    n_series = observed.shape[0]
    n_validation = int(round(validation_fraction * n_series))
    n_train = n_series - n_validation
    if n_train < 2:
        raise ValueError("the training split is too small to fit")
    elapsed = torch.full((1,), float(elapsed_days), dtype=torch.float32)
    optimiser = torch.optim.Adam(backbone.parameters(), lr=learning_rate)
    gradient_norms: list[float] = []
    best_loss = float("inf")
    best_epoch = 0
    final_loss = float("inf")
    for epoch in range(1, epochs + 1):
        backbone.train()
        optimiser.zero_grad()
        train_loss = reconstruction_and_forecast_loss(
            backbone, observed[:n_train], controls[:n_train], targets[:n_train], elapsed
        )
        backpropagate(train_loss)
        norm = math.sqrt(
            sum(
                float((parameter.grad**2).sum())
                for parameter in backbone.parameters()
                if parameter.grad is not None
            )
        )
        gradient_norms.append(norm)
        optimiser.step()
        final_loss = float(train_loss.detach())
        if n_validation > 0:
            backbone.eval()
            with torch.no_grad():
                validation_loss = reconstruction_and_forecast_loss(
                    backbone,
                    observed[n_train:],
                    controls[n_train:],
                    targets[n_train:],
                    elapsed,
                )
            candidate = float(validation_loss.detach())
        else:
            candidate = final_loss
        if candidate < best_loss:
            best_loss = candidate
            best_epoch = epoch
    window = gradient_norms[-record_gradient_window:] if gradient_norms else [0.0]
    record = TrainingRecord(
        epochs=int(epochs),
        best_validation_loss=best_loss,
        best_epoch=int(best_epoch),
        final_training_loss=final_loss,
        gradient_norm_sd_last_window=float(np.std(window, ddof=1)) if len(window) > 1 else 0.0,
        learning_rate=float(learning_rate),
        latent_dimension=int(latent_dimension),
        parameter_count=parameter_count(backbone),
        flops_millions=estimate_flops(spec, int(observed.shape[1]) + int(targets.shape[1])),
        damping=damping_contribution(backbone),
        stiffness=stiffness_spectrum(backbone),
    )
    return backbone, record


def predict_series(
    backbone: DampedOscillatoryBackbone,
    observed: NDArray[np.float64],
    horizon: int,
    elapsed_days: float,
) -> NDArray[np.float64]:
    """Propagate with the treatment input only, no sensor input."""
    backbone.eval()
    observations: Tensor = torch.as_tensor(observed, dtype=torch.float32).unsqueeze(0)
    controls: Tensor = torch.zeros(
        1, observed.shape[0], backbone.spec.control_dim, dtype=torch.float32
    )
    elapsed: Tensor = torch.full((1,), float(elapsed_days), dtype=torch.float32)
    with torch.no_grad():
        output = backbone(observations, controls, elapsed, forecast_steps=horizon)
    forecast: Tensor = output["forecast"]
    return np.asarray(forecast.squeeze(0).numpy(), dtype=np.float64)


def to_held_out_series(window: TwinWindow, horizons: Sequence[int]) -> HeldOutSeries:
    """Project a twin window onto the ladder's held-out-split container."""
    targets = {
        horizon: window.targets[horizon] for horizon in horizons if horizon in window.targets
    }
    return HeldOutSeries(device_id=window.device_id, observed=window.observed, targets=targets)


def assemble_rungs(
    twin_config: TwinConfig,
    *,
    twin_backbone: DampedOscillatoryBackbone | None,
    undamped_backbone: DampedOscillatoryBackbone | None,
    mechanistic: MechanisticParameters | None,
    elapsed_days: float,
) -> list[Rung]:
    return rung_registry(
        mechanistic=mechanistic,
        twin_backbone=twin_backbone,
        undamped_backbone=undamped_backbone,
        dt=elapsed_days,
    )


def fit_mechanistic(
    windows: Sequence[TwinWindow], elapsed_days: float, seed: int
) -> MechanisticParameters:
    """Fit the mechanistic comparator on the same held-in devices as the backbone."""
    histories: list[NDArray[np.float64]] = []
    controls: list[NDArray[np.float64]] = []
    for window in windows:
        horizon = max(window.targets)
        series = np.vstack([window.observed, window.targets[horizon].reshape(1, -1)])
        histories.append(series)
        controls.append(np.diff(window.observed[:, 0]))
    return fit_mechanistic_parameters(histories, controls, elapsed_days, seed=seed)
