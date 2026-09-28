"""Predictive intervals from a held-out-calibrated ensemble spread.

Ref: Sec. Forecasting and terminal-assay allocation, SI Table S2 coverage column.
The backbone supplies no native uncertainty quantification, so the intervals are
calibrated after the fact on held-out series and that provenance is stated rather
than implied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def absolute_scores(
    prediction: NDArray[np.float64], truth: NDArray[np.float64]
) -> NDArray[np.float64]:
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must align")
    return np.abs(prediction - truth)


def conformal_quantile(scores: NDArray[np.float64], nominal: float) -> float:
    """Finite-sample upper quantile of the held-out absolute scores."""
    if scores.size == 0:
        raise ValueError("interval calibration needs held-out scores")
    if not 0.0 < nominal < 1.0:
        raise ValueError("a nominal coverage must lie strictly inside (0, 1)")
    ordered = np.sort(scores)
    rank = int(np.ceil((scores.size + 1) * nominal))
    index = min(max(rank, 1), scores.size) - 1
    return float(ordered[index])


@dataclass(frozen=True)
class IntervalCalibration:
    """The spread that a nominal level buys on the held-out split it was fitted on."""

    nominal: float
    spread: float
    n_scores: int
    source: str = "held_out_calibrated_ensemble_spread"

    def interval(
        self, prediction: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            np.asarray(prediction - self.spread, dtype=np.float64),
            np.asarray(prediction + self.spread, dtype=np.float64),
        )

    def as_record(self) -> dict[str, float | int | str]:
        return {
            "nominal": self.nominal,
            "spread": self.spread,
            "n_scores": self.n_scores,
            "source": self.source,
        }


def calibrate_intervals(
    held_out_prediction: NDArray[np.float64],
    held_out_truth: NDArray[np.float64],
    nominal: float,
) -> IntervalCalibration:
    scores = absolute_scores(held_out_prediction, held_out_truth).reshape(-1)
    return IntervalCalibration(
        nominal=float(nominal),
        spread=conformal_quantile(scores, nominal),
        n_scores=int(scores.size),
    )


def empirical_coverage(
    truth: NDArray[np.float64], low: NDArray[np.float64], high: NDArray[np.float64]
) -> float:
    if not (truth.shape == low.shape == high.shape):
        raise ValueError("truth and both interval ends must align")
    inside = (truth >= low) & (truth <= high)
    return float(np.mean(inside))


def coverage_report(
    truth: NDArray[np.float64],
    low: NDArray[np.float64],
    high: NDArray[np.float64],
    nominal: float,
) -> dict[str, float]:
    empirical = empirical_coverage(truth, low, high)
    return {
        "nominal": float(nominal),
        "empirical": empirical,
        "under_covering": bool(empirical < nominal),
        "gap": empirical - float(nominal),
        "n_intervals": float(truth.size),
    }


def ensemble_spread(replicate_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Spread across an ensemble of fitted backbones, one per training seed."""
    if replicate_predictions.ndim != 2:
        raise ValueError("an ensemble is a replicate-by-time matrix")
    if replicate_predictions.shape[0] < 2:
        raise ValueError("an ensemble spread needs at least two replicates")
    return np.asarray(np.std(replicate_predictions, axis=0, ddof=1), dtype=np.float64)


def interval_from_ensemble(
    replicate_predictions: NDArray[np.float64], nominal: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interval from the ensemble spread, widened to the requested nominal level."""
    spread = ensemble_spread(replicate_predictions)
    factor = float(stats.norm.ppf(0.5 + nominal / 2.0))
    centre = np.asarray(np.mean(replicate_predictions, axis=0), dtype=np.float64)
    return centre - factor * spread, centre + factor * spread


def ranking_by_interval_honesty(coverage_by_rung: dict[str, float], nominal: float) -> list[str]:
    """Rungs ordered by how close their empirical coverage sits to nominal."""
    return sorted(coverage_by_rung, key=lambda name: abs(coverage_by_rung[name] - nominal))


def coverage_of_all_rungs(
    truth: NDArray[np.float64],
    intervals_by_rung: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    nominal: float,
) -> dict[str, float]:
    return {
        name: empirical_coverage(truth, low, high)
        for name, (low, high) in intervals_by_rung.items()
    }
