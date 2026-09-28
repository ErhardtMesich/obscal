"""The rungs of the forecasting ladder.

Ref: SI Table S2. Every rung meets the same held-out splits, per axis, because
the two fibrosis readouts are independently regulated and a pooled error would
average across axes the literature reports moving differently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import optimize

from obscal.twin.oscillatory import DampedOscillatoryBackbone

PERSISTENCE_RATIO = 1.0


class Rung(Protocol):
    """A forecasting strategy that can be scored on held-out series.

    The two labels are read-only, so a frozen dataclass with the same members
    satisfies the protocol without declaring them as mutable attributes.
    """

    @property
    def name(self) -> str: ...

    @property
    def stochastic_initialisation(self) -> bool: ...

    def predict(self, history: NDArray[np.float64], steps: int) -> NDArray[np.float64]:
        """Forecast `steps` values ahead, per axis, from the observed history."""


@dataclass(frozen=True)
class PersistenceRung:
    """The zero-parameter last-value predictor every other rung is priced against."""

    name: str = "persistence"
    stochastic_initialisation: bool = False

    def predict(self, history: NDArray[np.float64], steps: int) -> NDArray[np.float64]:
        if history.size == 0:
            raise ValueError("a last-value forecast needs at least one observation")
        return np.asarray(np.tile(history[-1], (steps, 1)), dtype=np.float64)


@dataclass(frozen=True)
class ArimaRung:
    """ARIMA(1,1,0): differencing plus a single autoregressive coefficient."""

    name: str = "arima_110"
    stochastic_initialisation: bool = False
    order: tuple[int, int, int] = (1, 1, 0)

    def predict(self, history: NDArray[np.float64], steps: int) -> NDArray[np.float64]:
        if history.ndim != 2:
            raise ValueError("the ladder carries both axes, so history is steps by axes")
        return np.column_stack(
            [self._one_axis(history[:, column], steps) for column in range(history.shape[1])]
        )

    @staticmethod
    def _one_axis(series: NDArray[np.float64], steps: int) -> NDArray[np.float64]:
        if series.size < 3:
            return np.full(steps, float(series[-1]), dtype=np.float64)
        differences = np.diff(series)
        response = differences[1:]
        predictor = differences[:-1]
        denominator = float(predictor @ predictor)
        coefficient = float(response @ predictor) / denominator if denominator > 0.0 else 0.0
        forecast = np.empty(steps, dtype=np.float64)
        level = float(series[-1])
        increment = float(differences[-1])
        for index in range(steps):
            increment = coefficient * increment
            level = level + increment
            forecast[index] = level
        return forecast


@dataclass(frozen=True)
class MechanisticParameters:
    """A first-order mechanistically parameterised deposition recurrence.

    Engineering default: the manuscript names a re-fitted mechanistic comparator
    and reports its performance, but does not print its equations, so the form
    here is an explicit choice rather than a transcription.
    """

    activation_relaxation: float
    activation_ceiling: float
    deposition_rate: float
    deposition_threshold: float
    matrix_decay: float

    def as_vector(self) -> NDArray[np.float64]:
        return np.array(
            [
                self.activation_relaxation,
                self.activation_ceiling,
                self.deposition_rate,
                self.deposition_threshold,
                self.matrix_decay,
            ],
            dtype=np.float64,
        )


def mechanistic_trajectory(
    parameters: MechanisticParameters,
    activation0: float,
    matrix0: float,
    control: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate the recurrence by explicit Euler on the elapsed-time grid."""
    activation = np.empty(control.size + 1, dtype=np.float64)
    matrix = np.empty(control.size + 1, dtype=np.float64)
    activation[0] = activation0
    matrix[0] = matrix0
    for index in range(control.size):
        current_a = activation[index]
        current_m = matrix[index]
        drive = float(control[index])
        equilibrium = parameters.activation_ceiling - drive
        acceleration = parameters.activation_relaxation * (equilibrium - current_a)
        deposition = parameters.deposition_rate * max(
            current_a - parameters.deposition_threshold, 0.0
        )
        activation[index + 1] = current_a + dt * acceleration
        matrix[index + 1] = current_m + dt * (deposition - parameters.matrix_decay * current_m)
    return activation, matrix


def fit_mechanistic_parameters(
    histories: Sequence[NDArray[np.float64]],
    controls: Sequence[NDArray[np.float64]],
    dt: float,
    *,
    seed: int = 0,
) -> MechanisticParameters:
    """Fit the recurrence on held-in series by least squares, then freeze it."""
    targets = [
        (
            np.asarray(series, dtype=np.float64)[:, 0],
            np.asarray(series, dtype=np.float64)[:, 1],
        )
        for series in histories
    ]
    drives = [np.asarray(c, dtype=np.float64) for c in controls]
    if len(targets) != len(drives):
        raise ValueError("every series needs its own control input")
    for (activation_target, _), drive in zip(targets, drives, strict=True):
        if activation_target.size != drive.size + 1:
            raise ValueError("a series needs one control value per interval")

    def residual(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        parameters = MechanisticParameters(
            activation_relaxation=float(np.exp(vector[0])),
            activation_ceiling=float(vector[1]),
            deposition_rate=float(np.exp(vector[2])),
            deposition_threshold=float(vector[3]),
            matrix_decay=float(np.exp(vector[4])),
        )
        pieces: list[NDArray[np.float64]] = []
        for (activation_target, matrix_target), drive in zip(targets, drives, strict=True):
            activation, matrix = mechanistic_trajectory(
                parameters,
                float(activation_target[0]),
                float(matrix_target[0]),
                drive,
                dt,
            )
            pieces.append(activation - activation_target)
            pieces.append((matrix - matrix_target) / max(float(np.mean(matrix_target)), 1e-9))
        return np.concatenate(pieces)

    guess = np.array([np.log(0.4), 2.0, np.log(0.3), 0.5, np.log(0.05)], dtype=np.float64)
    solution = optimize.least_squares(residual, guess, max_nfev=400)
    return MechanisticParameters(
        activation_relaxation=float(np.exp(solution.x[0])),
        activation_ceiling=float(solution.x[1]),
        deposition_rate=float(np.exp(solution.x[2])),
        deposition_threshold=float(solution.x[3]),
        matrix_decay=float(np.exp(solution.x[4])),
    )


@dataclass(frozen=True)
class MechanisticRung:
    """The re-fitted mechanistic comparator, integrated forward under treatment."""

    parameters: MechanisticParameters
    dt: float
    name: str = "mechanistic"
    stochastic_initialisation: bool = True

    def predict(self, history: NDArray[np.float64], steps: int) -> NDArray[np.float64]:
        activation = np.asarray(history[:, 0], dtype=np.float64)
        matrix = np.asarray(history[:, 1], dtype=np.float64)
        controls = np.full(
            steps, float(np.mean(np.abs(np.diff(activation)))) if activation.size > 1 else 0.0
        )
        forecast_activation, forecast_matrix = mechanistic_trajectory(
            self.parameters,
            float(activation[-1]),
            float(matrix[-1]),
            controls,
            self.dt,
        )
        return np.column_stack([forecast_activation[1:], forecast_matrix[1:]])


@dataclass(frozen=True)
class OscillatoryRung:
    """A fitted backbone used as a forecasting strategy on one held-out series."""

    backbone: DampedOscillatoryBackbone
    dt: float
    name: str = "obscal_twin"
    stochastic_initialisation: bool = True

    def predict(self, history: NDArray[np.float64], steps: int) -> NDArray[np.float64]:
        backbone = self.backbone
        backbone.eval()
        observations: torch.Tensor = torch.as_tensor(history, dtype=torch.float32).unsqueeze(0)
        controls: torch.Tensor = torch.zeros(
            1, history.shape[0], backbone.spec.control_dim, dtype=torch.float32
        )
        elapsed: torch.Tensor = torch.full((1,), float(self.dt), dtype=torch.float32)
        with torch.no_grad():
            output = backbone(observations, controls, elapsed, forecast_steps=steps)
        forecast: torch.Tensor = output["forecast"]
        return np.asarray(forecast.squeeze(0).numpy(), dtype=np.float64)


def rung_registry(
    *,
    mechanistic: MechanisticParameters | None,
    twin_backbone: DampedOscillatoryBackbone | None,
    undamped_backbone: DampedOscillatoryBackbone | None,
    dt: float,
) -> list[Rung]:
    """The five rungs, in the order the ladder table lists them."""
    rungs: list[Rung] = [PersistenceRung(), ArimaRung()]
    if mechanistic is not None:
        rungs.append(MechanisticRung(parameters=mechanistic, dt=dt))
    if undamped_backbone is not None:
        rungs.append(OscillatoryRung(backbone=undamped_backbone, dt=dt, name="without_damping"))
    if twin_backbone is not None:
        rungs.append(OscillatoryRung(backbone=twin_backbone, dt=dt, name="obscal_twin"))
    return rungs


def rung_by_name(rungs: Sequence[Rung], name: str) -> Rung:
    for rung in rungs:
        if rung.name == name:
            return rung
    raise KeyError(f"no rung named {name!r}")


def stochastic_rungs(rungs: Sequence[Rung]) -> list[str]:
    return [rung.name for rung in rungs if rung.stochastic_initialisation]


def deterministic_rungs(rungs: Sequence[Rung]) -> list[str]:
    return [rung.name for rung in rungs if not rung.stochastic_initialisation]
