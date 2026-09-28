"""Device-level bootstrap that keeps the treatment strata intact.

Ref: Algorithm 1 step B; SI per-stage budget (2000 replicates).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from obscal.runtime.seed import numpy_generator

DEFAULT_REPLICATES = 2000


@dataclass(frozen=True)
class BootstrapResult:
    """A percentile interval and the replicate distribution behind it."""

    point: NDArray[np.float64]
    replicates: NDArray[np.float64]
    low: NDArray[np.float64]
    high: NDArray[np.float64]
    n_replicates: int
    level: float

    @property
    def standard_error(self) -> NDArray[np.float64]:
        return np.asarray(np.std(self.replicates, axis=0, ddof=1), dtype=np.float64)

    def as_record(self, names: Sequence[str] | None = None) -> dict[str, object]:
        labels = (
            list(names)
            if names is not None
            else [f"stat_{index}" for index in range(self.point.size)]
        )
        return {
            "level": self.level,
            "n_replicates": self.n_replicates,
            "point": self.point.tolist(),
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "standard_error": self.standard_error.tolist(),
            "names": labels,
        }


def stratified_indices(
    strata: Sequence[str], rng: np.random.Generator, *, preserve_strata: bool = True
) -> NDArray[np.int64]:
    """Resample unit indices, optionally drawing within each stratum.

    Preserving the strata keeps every replicate's treatment mix identical to the
    delivered design's, which is what the pooled fit is read on.
    """
    labels = np.asarray(list(strata), dtype=object)
    n = labels.shape[0]
    if not preserve_strata:
        return np.asarray(rng.integers(0, n, size=n), dtype=np.int64)
    order = list(dict.fromkeys(labels.tolist()))
    drawn: list[NDArray[np.int64]] = []
    for label in order:
        members = np.flatnonzero(labels == label).astype(np.int64)
        if members.size == 0:
            continue
        picks = rng.integers(0, members.size, size=members.size)
        drawn.append(members[picks])
    return np.concatenate(drawn).astype(np.int64)


def bootstrap(
    statistic: Callable[[NDArray[np.int64]], NDArray[np.float64]],
    strata: Sequence[str],
    *,
    n_units: int,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = 0,
    level: float = 0.95,
    preserve_strata: bool = True,
) -> BootstrapResult:
    """Percentile bootstrap of a statistic that reads unit-indexed arrays."""
    if replicates < 2:
        raise ValueError("a bootstrap needs at least two replicates")
    rng = numpy_generator(seed)
    full = np.arange(n_units, dtype=np.int64)
    point = np.asarray(statistic(full), dtype=np.float64)
    draws = np.empty((replicates, point.size), dtype=np.float64)
    for index in range(replicates):
        draws[index] = statistic(stratified_indices(strata, rng, preserve_strata=preserve_strata))
    tail = 50.0 * (1.0 - level)
    low, high = np.percentile(draws, [tail, 100.0 - tail], axis=0)
    return BootstrapResult(
        point=point,
        replicates=draws,
        low=np.asarray(low, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        n_replicates=int(replicates),
        level=float(level),
    )


def covers_truth(low: float, high: float, truth: float) -> bool:
    return bool(low <= truth <= high)


def coverage_of_interval(intervals: NDArray[np.float64], truths: NDArray[np.float64]) -> float:
    """Empirical share of intervals that contain their own truth."""
    if intervals.shape[0] != truths.shape[0]:
        raise ValueError("every interval needs one truth")
    inside = (intervals[:, 0] <= truths) & (truths <= intervals[:, 1])
    return float(np.mean(inside))


def bootstrap_excluding(
    statistic: Callable[[NDArray[np.int64]], NDArray[np.float64]],
    strata: Sequence[str],
    *,
    n_units: int,
    exclude: int,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap restricted to a held-out index, used by leave-one-device-out checks."""
    if not 0 <= exclude < n_units:
        raise ValueError("the excluded index must lie inside the unit range")
    kept = np.array([index for index in range(n_units) if index != exclude], dtype=np.int64)
    subset = [strata[index] for index in kept]
    return bootstrap(statistic, subset, n_units=int(kept.size), replicates=replicates, seed=seed)
