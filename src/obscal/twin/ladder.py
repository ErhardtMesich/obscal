"""Scoring the forecasting ladder against a last-value predictor.

Ref: Sec. Component ablation; Fig. 4c-d; SI Table S2, Table S17.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from obscal.twin.intervals import IntervalCalibration
from obscal.twin.rungs import PERSISTENCE_RATIO, Rung

AXES: tuple[str, ...] = ("activation", "matrix")
AXIS_COLUMN: dict[str, int] = {"activation": 0, "matrix": 1}


@dataclass(frozen=True)
class HeldOutSeries:
    """One device's observed window and the held-out values the rungs are scored on."""

    device_id: str
    observed: NDArray[np.float64]
    targets: dict[int, NDArray[np.float64]]

    def __post_init__(self) -> None:
        if self.observed.ndim != 2:
            raise ValueError("the observed window must be steps by axis")
        for horizon, target in self.targets.items():
            if target.shape != (len(AXES),):
                raise ValueError(f"the target at horizon {horizon} must carry both axes")


@dataclass(frozen=True)
class HorizonScore:
    """One rung, one axis, one horizon, with the ratio the ladder is read on."""

    rung: str
    axis: str
    horizon_days: int
    rmse: float
    persistence_rmse: float
    ratio_to_persistence: float
    n_series: int
    coverage: float | None = None

    @property
    def beats_persistence(self) -> bool:
        return self.ratio_to_persistence < PERSISTENCE_RATIO

    def as_record(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "axis": self.axis,
            "horizon_days": self.horizon_days,
            "rmse": self.rmse,
            "persistence_rmse": self.persistence_rmse,
            "ratio_to_persistence": self.ratio_to_persistence,
            "n_series": self.n_series,
            "coverage": self.coverage,
            "beats_persistence": self.beats_persistence,
        }


@dataclass(frozen=True)
class LadderResult:
    """Every rung scored on the identical held-out splits."""

    scores: tuple[HorizonScore, ...]
    horizon_days: tuple[int, ...]
    axes_scored: tuple[str, ...] = AXES

    def at(self, rung: str, axis: str, horizon: int) -> HorizonScore:
        for score in self.scores:
            if score.rung == rung and score.axis == axis and score.horizon_days == horizon:
                return score
        raise KeyError(f"no score for {rung!r} on {axis!r} at horizon {horizon}")

    def ordering(self, axis: str, horizon: int) -> list[str]:
        """Rungs ordered by point error, best first, on one axis and horizon."""
        rows = [
            score for score in self.scores if score.axis == axis and score.horizon_days == horizon
        ]
        return [score.rung for score in sorted(rows, key=lambda item: item.rmse)]

    def delta_vs_best(self, axis: str, horizon: int) -> dict[str, float]:
        rows = [
            score for score in self.scores if score.axis == axis and score.horizon_days == horizon
        ]
        best = min(row.rmse for row in rows)
        return {row.rung: row.rmse - best for row in rows}

    def ratios(self, axis: str, horizon: int) -> dict[str, float]:
        rows = [
            score for score in self.scores if score.axis == axis and score.horizon_days == horizon
        ]
        return {row.rung: row.ratio_to_persistence for row in rows}

    def records(self) -> list[dict[str, Any]]:
        return [score.as_record() for score in self.scores]


@dataclass
class _Accumulator:
    squared_error: dict[tuple[str, int], list[float]] = field(default_factory=dict)
    persistence_squared: dict[int, list[float]] = field(default_factory=dict)
    intervals_seen: dict[tuple[str, int], list[bool]] = field(default_factory=dict)


def score_ladder(
    rungs: Sequence[Rung],
    series: Sequence[HeldOutSeries],
    horizons: Sequence[int],
    *,
    interval_calibrations: dict[str, dict[int, IntervalCalibration]] | None = None,
) -> LadderResult:
    """Every rung forecasts every held-out series, per axis, on the same splits."""
    if not series:
        raise ValueError("the ladder needs at least one held-out series")
    accumulators = {axis: _Accumulator() for axis in AXES}
    for axis in AXES:
        column = AXIS_COLUMN[axis]
        for rung in rungs:
            for entry in series:
                history = np.asarray(entry.observed, dtype=np.float64)
                for horizon in horizons:
                    if horizon not in entry.targets:
                        continue
                    prediction = np.asarray(rung.predict(history, horizon), dtype=np.float64)
                    if prediction.shape != (horizon, history.shape[1]):
                        raise ValueError(
                            f"rung {rung.name!r} must forecast every axis for every step"
                        )
                    truth = float(entry.targets[horizon][column])
                    predicted = float(prediction[-1, column])
                    accumulators[axis].squared_error.setdefault((rung.name, horizon), []).append(
                        (predicted - truth) ** 2
                    )
                    if rung.name == "persistence":
                        accumulators[axis].persistence_squared.setdefault(horizon, []).append(
                            (predicted - truth) ** 2
                        )
                    if interval_calibrations is not None:
                        calibration = interval_calibrations.get(rung.name, {}).get(horizon)
                        if calibration is not None:
                            inside = abs(predicted - truth) <= calibration.spread
                            accumulators[axis].intervals_seen.setdefault(
                                (rung.name, horizon), []
                            ).append(inside)

    scores: list[HorizonScore] = []
    for axis in AXES:
        accumulator = accumulators[axis]
        for (rung_name, horizon), squares in sorted(
            accumulator.squared_error.items(), key=lambda item: (item[0][1], item[0][0])
        ):
            rmse = float(np.sqrt(np.mean(squares)))
            persistence_squares = accumulator.persistence_squared[horizon]
            persistence_rmse = float(np.sqrt(np.mean(persistence_squares)))
            coverage = None
            seen = accumulator.intervals_seen.get((rung_name, horizon))
            if seen:
                coverage = float(np.mean(seen))
            scores.append(
                HorizonScore(
                    rung=rung_name,
                    axis=axis,
                    horizon_days=int(horizon),
                    rmse=rmse,
                    persistence_rmse=persistence_rmse,
                    ratio_to_persistence=(
                        rmse / persistence_rmse if persistence_rmse > 0.0 else float("nan")
                    ),
                    n_series=len(squares),
                    coverage=coverage,
                )
            )
    return LadderResult(scores=tuple(scores), horizon_days=tuple(sorted(set(horizons))))


def leakage_diagnostic(ratio: float, threshold: float) -> dict[str, object]:
    """A ratio below the plausibility floor is a leakage diagnostic, not a result.

    Ref: Algorithm 2 step 28. The check re-examines the held-out split and reports
    the check rather than the number.
    """
    triggered = bool(ratio < threshold)
    return {
        "ratio": ratio,
        "threshold": threshold,
        "triggered": triggered,
        "action": (
            "re-check the held-out split and report the check" if triggered else "report the ratio"
        ),
    }


def normalised_rmse(rmse: float, reference_rmse: float, reference_normalised: float) -> float:
    """Put a rung's error on the normalised scale another table uses.

    The two scales carry one ordering: the reference rung's own error and its
    normalised value define the conversion.
    """
    if reference_rmse <= 0.0:
        raise ValueError("the reference error must be positive")
    return float(rmse) * float(reference_normalised) / float(reference_rmse)


def across_seed_range(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("an across-seed range needs at least one value")
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "span": float(array.max() - array.min()),
        "pooled": float(array.mean()),
        "n_seeds": float(array.size),
    }


def seed_dominance(per_seed: dict[str, dict[int, float]], direction: str) -> dict[str, int]:
    """Which seed is best on each criterion, and whether any seed wins twice."""

    def better(left: float, right: float) -> bool:
        return left < right if direction == "lower_is_better" else left > right

    winners: dict[str, int] = {}
    for criterion, values in per_seed.items():
        seeds = list(values)
        best = seeds[0]
        for seed in seeds[1:]:
            if better(values[seed], values[best]):
                best = seed
        winners[criterion] = best
    return winners


def no_seed_dominates(winners: dict[str, int]) -> bool:
    return len(set(winners.values())) == len(winners)


def margin_carries(
    rung_rmse: float, comparator_rmse: float, across_seed_span: float
) -> dict[str, object]:
    """Whether a rung's margin over a comparator survives its own seed spread."""
    margin = comparator_rmse - rung_rmse
    return {
        "margin": margin,
        "across_seed_span": across_seed_span,
        "margin_exceeds_seed_span": bool(margin > across_seed_span),
        "read_as": "resolved" if margin > across_seed_span else "not_resolved_by_this_experiment",
    }


def interval_honesty_order(coverage_by_rung: dict[str, float], nominal: float) -> list[str]:
    return sorted(coverage_by_rung, key=lambda name: abs(coverage_by_rung[name] - nominal))


def rung_failing_persistence(result: LadderResult, axis: str, horizon: int) -> list[str]:
    return [
        rung for rung, ratio in result.ratios(axis, horizon).items() if ratio > PERSISTENCE_RATIO
    ]


def sustained_advantage(result: LadderResult, rung: str, axis: str) -> dict[str, float | int]:
    """The highest horizon at which a rung still clears the last-value predictor."""
    ratios = [
        (score.horizon_days, score.ratio_to_persistence)
        for score in result.scores
        if score.rung == rung and score.axis == axis
    ]
    ratios.sort()
    clearing = [horizon for horizon, ratio in ratios if ratio < PERSISTENCE_RATIO]
    return {
        "max_clearing_horizon": max(clearing) if clearing else 0,
        "ratio_at_max_horizon": (
            dict(ratios).get(max(clearing), float("nan")) if clearing else float("nan")
        ),
        "n_rung_horizons": len(ratios),
    }


def coverage_at_nominal(
    coverage: dict[str, float], nominal: float, tolerance: float = 0.0
) -> list[str]:
    return [rung for rung, value in coverage.items() if value >= nominal - tolerance]
