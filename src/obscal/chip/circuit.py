"""Equivalent-circuit fit of a multi-frequency sweep.

Ref: Sec. Continuous sensing front end and feature extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

FREE_PARAMETERS = 5
"""R_ext, R_barrier, C_membrane, CPE magnitude, CPE exponent."""

REAL_QUANTITIES_PER_FREQUENCY = 2
"""Each sweep point supplies a real part and an imaginary part."""


@dataclass(frozen=True)
class CircuitParameters:
    """The five free parameters of the delivered equivalent circuit."""

    r_ext_ohm: float
    r_barrier_ohm: float
    c_membrane_f: float
    cpe_magnitude: float
    cpe_exponent: float

    def as_vector(self) -> NDArray[np.float64]:
        return np.array(
            [
                self.r_ext_ohm,
                self.r_barrier_ohm,
                self.c_membrane_f,
                self.cpe_magnitude,
                self.cpe_exponent,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CircuitFit:
    """One fitted sweep, with the residual statistic a lack-of-fit test reads."""

    parameters: CircuitParameters
    residual_chi_square: float
    residual_dof: int
    lack_of_fit_p: float
    converged: bool
    n_frequencies: int

    def features(self, area_mm2: float) -> dict[str, float]:
        """The three electrical entries of the Eq. (2) feature vector."""
        barrier_area_product = self.parameters.r_barrier_ohm * area_mm2
        return {
            "log_rb": math.log(max(self.parameters.r_barrier_ohm, 1e-9)),
            "log_cm": math.log(max(self.parameters.c_membrane_f, 1e-30)),
            "alpha": self.parameters.cpe_exponent,
            "barrier_resistance_area": self.parameters.r_barrier_ohm,
            "barrier_resistance_area_product": barrier_area_product,
        }


def residual_degrees_of_freedom(n_frequencies: int) -> int:
    """Over-determination of the fit: 2 real readings per frequency less 5 unknowns."""
    return REAL_QUANTITIES_PER_FREQUENCY * n_frequencies - FREE_PARAMETERS


def impedance_model(
    parameters: CircuitParameters, frequencies_hz: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Complex impedance of the series R_ext + (R_b || C_m) + interface CPE."""
    omega = 2.0 * math.pi * frequencies_hz.astype(np.complex128)
    angular = 1j * omega
    barrier_branch = parameters.r_barrier_ohm / (
        1.0 + angular * parameters.r_barrier_ohm * parameters.c_membrane_f
    )
    interface = 1.0 / (parameters.cpe_magnitude * angular**parameters.cpe_exponent)
    return (parameters.r_ext_ohm + barrier_branch + interface).astype(np.complex128)


def lack_of_fit(residuals: NDArray[np.float64], n_frequencies: int) -> tuple[float, int, float]:
    """Residual chi-square on the over-determination the fit leaves."""
    chi_square = float(np.sum(residuals**2))
    dof = residual_degrees_of_freedom(n_frequencies)
    if dof <= 0:
        return chi_square, dof, float("nan")
    return chi_square, dof, float(stats.chi2.sf(chi_square, dof))


def fit_sweep(
    frequencies_hz: NDArray[np.float64],
    measured_ohm: NDArray[np.complex128],
    *,
    alpha_upper: float = 1.0,
    alpha_lower: float = 0.05,
    max_iterations: int = 400,
) -> CircuitFit:
    """Least-squares fit of one sweep, weighted on the sweep's own unit scale."""
    if frequencies_hz.shape != measured_ohm.shape:
        raise ValueError("frequency grid and measured sweep must align")
    scale = float(np.median(np.abs(measured_ohm)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("sweep magnitude is not usable")
    measured = measured_ohm / scale
    lower = np.array(
        [np.log(1e-6), np.log(1e-6), np.log(1e-15), np.log(1e-12), alpha_lower], dtype=np.float64
    )
    upper = np.array(
        [np.log(1e4), np.log(1e4), np.log(1e-3), np.log(1e2), alpha_upper], dtype=np.float64
    )

    def expand(vector: NDArray[np.float64]) -> CircuitParameters:
        return CircuitParameters(
            r_ext_ohm=float(np.exp(vector[0])),
            r_barrier_ohm=float(np.exp(vector[1])),
            c_membrane_f=float(np.exp(vector[2])),
            cpe_magnitude=float(np.exp(vector[3])),
            cpe_exponent=float(vector[4]),
        )

    def residual(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        model = impedance_model(expand(vector), frequencies_hz) / scale
        gap = model - measured
        return np.concatenate([gap.real, gap.imag])

    best: optimize.OptimizeResult | None = None
    for guess in _initial_guesses(measured, frequencies_hz):
        candidate = optimize.least_squares(
            residual,
            guess,
            bounds=(lower, upper),
            method="trf",
            max_nfev=max_iterations,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        if best is None or float(candidate.cost) < float(best.cost):
            best = candidate
    if best is None:
        raise RuntimeError("the circuit fit produced no candidate")
    solution = best
    fitted = expand(np.asarray(solution.x, dtype=np.float64))
    chi_square, dof, p_value = lack_of_fit(
        np.asarray(solution.fun, dtype=np.float64), frequencies_hz.size
    )
    return CircuitFit(
        parameters=fitted,
        residual_chi_square=chi_square,
        residual_dof=dof,
        lack_of_fit_p=p_value,
        converged=bool(solution.success),
        n_frequencies=int(frequencies_hz.size),
    )


def _initial_guesses(
    measured: NDArray[np.complex128], frequencies_hz: NDArray[np.float64]
) -> list[NDArray[np.float64]]:
    """A small deterministic set of seeds taken from the sweep's own shape.

    A single seed sits in a local minimum for some corners of the parameter
    space, so the fit is run from a handful and the best residual is kept.
    """
    order = np.argsort(frequencies_hz)
    lowest = int(order[0])
    highest = int(order[-1])
    magnitude = np.abs(measured)
    series = max(float(measured[highest].real), 1e-3)
    barrier = max(float(magnitude[lowest]) - series, 1e-3)
    knee = int(np.argmin(measured.imag))
    omega = 2.0 * math.pi * max(float(frequencies_hz[knee]), 1.0)
    capacitance = 1.0 / max(omega * barrier, 1e-12)
    interface = 1.0 / max(float(magnitude[highest]) * omega, 1e-12)
    guesses: list[NDArray[np.float64]] = []
    for series_scale in (0.5, 1.0):
        for barrier_scale in (0.6, 1.0):
            for capacitance_scale in (1.0 / 16.0, 1.0, 16.0):
                guesses.append(
                    np.array(
                        [
                            np.log(min(max(series * series_scale, 1e-6), 1e4)),
                            np.log(min(max(barrier * barrier_scale, 1e-6), 1e4)),
                            np.log(min(max(capacitance * capacitance_scale, 1e-15), 1e-3)),
                            np.log(min(max(interface, 1e-12), 1e2)),
                            0.85,
                        ],
                        dtype=np.float64,
                    )
                )
    return guesses


def sweep_summary(fit: CircuitFit, area_mm2: float) -> dict[str, Any]:
    """Report-ready view of one fitted sweep."""
    features = fit.features(area_mm2)
    return {
        "r_ext_ohm": fit.parameters.r_ext_ohm,
        "r_barrier_ohm": fit.parameters.r_barrier_ohm,
        "c_membrane_f": fit.parameters.c_membrane_f,
        "cpe_magnitude": fit.parameters.cpe_magnitude,
        "cpe_exponent": fit.parameters.cpe_exponent,
        "log_rb": features["log_rb"],
        "log_cm": features["log_cm"],
        "alpha": features["alpha"],
        "barrier_resistance_area_product_ohm_mm2": features["barrier_resistance_area_product"],
        "residual_chi_square": fit.residual_chi_square,
        "residual_dof": fit.residual_dof,
        "lack_of_fit_p": fit.lack_of_fit_p,
        "converged": fit.converged,
    }


def series_artefact_check(
    offset_ohm: float,
    standard_error_ohm: float,
    arithmetic_standard_error_ohm: float,
    *,
    slope: float = 1.0,
) -> dict[str, float]:
    """Series-offset test used by the electrode-siting decision and its ablation."""
    statistic = offset_ohm / max(standard_error_ohm, 1e-12)
    dof = 2
    p_value = float(2.0 * stats.t.sf(abs(statistic), dof))
    return {
        "offset_ohm": offset_ohm,
        "standard_error_ohm": standard_error_ohm,
        "t_statistic": statistic,
        "p_value": p_value,
        "arithmetic_standard_error_ohm": arithmetic_standard_error_ohm,
        "slope": slope,
    }


def single_frequency_resistance(
    frequency_hz: float, measured_ohm: NDArray[np.complex128], grid_hz: NDArray[np.float64]
) -> float:
    """Comparator channel: one frequency read as transepithelial resistance."""
    index = int(np.argmin(np.abs(grid_hz - frequency_hz)))
    value = complex(measured_ohm[index])
    return float(value.real)
