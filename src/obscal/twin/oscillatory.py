"""The damped oscillatory state-space recurrence.

Ref: Sec. Forecasting and terminal-assay allocation, Eq. (7). The second-order
oscillatory form and its implicit-explicit discretisation follow the damped
oscillatory state-space family the manuscript cites; the layer structure is an
engineering default because the manuscript prints fitted quantities rather than
the backbone's parameterisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

STIFFNESS_FLOOR = 1e-4
DAMPING_FLOOR = 1e-6


@dataclass(frozen=True)
class OscillatorySpec:
    """Shape of one backbone: latent state, observation and control widths."""

    latent_dim: int
    observed_dim: int
    control_dim: int
    damping: bool
    readout_hidden: int = 32

    def __post_init__(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("the latent state dimension must be positive")
        if self.observed_dim < 1:
            raise ValueError("the observed dimension must be positive")


class DampedOscillatoryBackbone(nn.Module):
    """xi(t) = D_Theta(xi(t-), x_obs(t), u(t), dt); x_hat(t|t) = C_Theta xi(t).

    Treatment enters the state equation as an input rather than being appended to
    the observation vector, because only a state equation separates what was done
    to a culture from what was measured on it.
    """

    stiffness_raw: Tensor
    damping_raw: Tensor

    def __init__(self, spec: OscillatorySpec) -> None:
        super().__init__()
        self.spec = spec
        self.stiffness_raw = nn.Parameter(torch.zeros(spec.latent_dim))
        self.damping_raw = nn.Parameter(torch.full((spec.latent_dim,), -2.0))
        self.control_map = nn.Linear(spec.control_dim, spec.latent_dim, bias=False)
        self.observation_map = nn.Linear(spec.observed_dim, spec.latent_dim, bias=False)
        self.readout = nn.Sequential(
            nn.Linear(2 * spec.latent_dim, spec.readout_hidden),
            nn.Tanh(),
            nn.Linear(spec.readout_hidden, spec.observed_dim),
        )

    @property
    def stiffness(self) -> Tensor:
        return torch.nn.functional.softplus(self.stiffness_raw) + STIFFNESS_FLOOR

    @property
    def damping(self) -> Tensor:
        if not self.spec.damping:
            return torch.zeros_like(self.damping_raw)
        return torch.nn.functional.softplus(self.damping_raw) + DAMPING_FLOOR

    def initial_state(self, batch: int, device: torch.device) -> Tensor:
        zeros: Tensor = torch.zeros(batch, 2 * self.spec.latent_dim, device=device)
        return zeros

    def step(
        self,
        state: Tensor,
        observation: Tensor,
        control: Tensor,
        elapsed: Tensor,
    ) -> Tensor:
        """One implicit-explicit update of the second-order system."""
        latent = self.spec.latent_dim
        position = state[:, :latent]
        velocity = state[:, latent:]
        injection = self.control_map(control) + self.observation_map(observation)
        restored: Tensor = injection - self.stiffness * position
        damped: Tensor = (velocity + elapsed * restored) / (1.0 + elapsed * self.damping)
        advanced: Tensor = position + elapsed * damped
        updated: Tensor = torch.cat([advanced, damped], dim=-1)
        return updated

    def read(self, state: Tensor) -> Tensor:
        projected: Tensor = self.readout(state)
        return projected

    def forward(
        self,
        observations: Tensor,
        controls: Tensor,
        elapsed: Tensor,
        *,
        forecast_steps: int = 0,
    ) -> dict[str, Tensor]:
        """Roll the recurrence over the observed window, then forecast with control only."""
        batch, steps, _ = observations.shape
        state = self.initial_state(batch, observations.device)
        observed_outputs: list[Tensor] = []
        for index in range(steps):
            state = self.step(state, observations[:, index], controls[:, index], elapsed)
            observed_outputs.append(self.read(state))
        filtered: Tensor = torch.stack(observed_outputs, dim=1)
        forecasts: list[Tensor] = []
        if forecast_steps > 0:
            zero_observation = torch.zeros_like(observations[:, -1])
            last_control = (
                controls[:, -1]
                if controls.shape[1] > 0
                else torch.zeros(batch, self.spec.control_dim, device=observations.device)
            )
            for _ in range(forecast_steps):
                state = self.step(state, zero_observation, last_control, elapsed)
                forecasts.append(self.read(state))
        predicted: Tensor = torch.stack(forecasts, dim=1) if forecasts else filtered[:, -1:, :]
        return {"filtered": filtered, "forecast": predicted, "final_state": state}


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def estimate_flops(spec: OscillatorySpec, steps: int) -> float:
    """Multiply-accumulate estimate for one forward pass, in millions."""
    latent = spec.latent_dim
    per_step = (
        2 * latent
        + latent * spec.control_dim
        + latent * spec.observed_dim
        + 2 * latent * spec.readout_hidden
        + spec.readout_hidden * spec.observed_dim
    )
    return float(2 * per_step * steps) / 1e6


def build_backbone(spec: OscillatorySpec) -> DampedOscillatoryBackbone:
    return DampedOscillatoryBackbone(spec)


def damping_is_active(module: DampedOscillatoryBackbone) -> bool:
    if not module.spec.damping:
        return False
    return float(module.damping.detach().min()) > 0.0


def damping_contribution(module: DampedOscillatoryBackbone) -> dict[str, float]:
    """The energy-dissipation term the ablation removes."""
    values = module.damping.detach()
    return {
        "damping_min": float(values.min()),
        "damping_mean": float(values.mean()),
        "damping_max": float(values.max()),
        "active": damping_is_active(module),
    }


def stiffness_spectrum(module: DampedOscillatoryBackbone) -> dict[str, float]:
    values = module.stiffness.detach()
    return {
        "stiffness_min": float(values.min()),
        "stiffness_mean": float(values.mean()),
        "stiffness_max": float(values.max()),
    }
