"""The pooled, locally linear calibration regression.

Ref: Sec. Calibration design and identifiability, Eq. (3); Algorithm 1 step B.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

STATE_AXES: tuple[str, ...] = ("activation", "matrix")
NUISANCE_CATEGORICAL: tuple[str, ...] = ("batch", "operator", "geometry", "perfusion_state")
NUISANCE_NUMERIC: tuple[str, ...] = (
    "temperature",
    "medium_change_indicator",
    "hours_since_electrode_wetting",
)


@dataclass(frozen=True)
class TimeBasis:
    """Smooth basis in time, centred at the block mean.

    Ref: Algorithm 1 step B; time enters as a covariate and as an interaction
    rather than by slicing, so the fit is never assembled slice by slice.
    """

    columns: int
    reference: float
    scale: float

    @classmethod
    def fit(cls, times: NDArray[np.float64], columns: int) -> TimeBasis:
        if columns < 0:
            raise ValueError("the time basis cannot carry a negative column count")
        reference = float(np.mean(times))
        scale = float(np.std(times, ddof=1)) if times.size > 1 else 0.0
        if scale <= 0.0:
            # A subset drawn at a single termination time carries no time
            # variation, so the basis collapses and the design drops it.
            return cls(columns=0, reference=reference, scale=1.0)
        return cls(columns=columns, reference=reference, scale=scale)

    def evaluate(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.columns == 0:
            return np.zeros((times.size, 0), dtype=np.float64)
        scaled = (times - self.reference) / self.scale
        return np.column_stack([scaled**power for power in range(1, self.columns + 1)])

    @property
    def varies(self) -> bool:
        return self.columns > 0


@dataclass(frozen=True)
class NuisanceDesign:
    """Artefact covariates as a numeric design block with named columns."""

    matrix: NDArray[np.float64]
    columns: tuple[str, ...]

    @property
    def width(self) -> int:
        return int(self.matrix.shape[1])

    def drop(self, names: Sequence[str]) -> NuisanceDesign:
        keep = [index for index, name in enumerate(self.columns) if name not in set(names)]
        return NuisanceDesign(
            matrix=self.matrix[:, keep],
            columns=tuple(self.columns[index] for index in keep),
        )


def build_nuisance_design(
    numeric: Mapping[str, NDArray[np.float64]],
    categorical: Mapping[str, Sequence[str]],
    reference_levels: Mapping[str, str] | None = None,
) -> NuisanceDesign:
    """Standardised numerics plus reference-coded categoricals, intercept-free."""
    n_rows = _row_count(numeric, categorical)
    blocks: list[NDArray[np.float64]] = []
    names: list[str] = []
    for key in NUISANCE_NUMERIC:
        if key not in numeric:
            continue
        column = np.asarray(numeric[key], dtype=np.float64)
        spread = float(np.std(column, ddof=1)) if column.size > 1 else 0.0
        blocks.append(
            ((column - float(np.mean(column))) / spread).reshape(-1, 1)
            if spread > 0
            else column.reshape(-1, 1)
        )
        names.append(f"z::{key}")
    for key in NUISANCE_CATEGORICAL:
        if key not in categorical:
            continue
        levels = list(dict.fromkeys(categorical[key]))
        reference = (reference_levels or {}).get(key, levels[0])
        for level in levels:
            if level == reference:
                continue
            indicator = np.array([1.0 if value == level else 0.0 for value in categorical[key]])
            blocks.append(indicator.reshape(-1, 1))
            names.append(f"z::{key}={level}")
    matrix = np.hstack(blocks) if blocks else np.zeros((n_rows, 0), dtype=np.float64)
    return NuisanceDesign(matrix=matrix, columns=tuple(names))


def _row_count(
    numeric: Mapping[str, NDArray[np.float64]], categorical: Mapping[str, Sequence[str]]
) -> int:
    for column in numeric.values():
        return int(np.asarray(column).shape[0])
    for labels in categorical.values():
        return len(labels)
    raise ValueError("a nuisance design needs at least one covariate")


@dataclass(frozen=True)
class DesignAssembly:
    """The Eq. (3) design matrix with the column blocks named."""

    matrix: NDArray[np.float64]
    columns: tuple[str, ...]
    state_columns: tuple[int, int]
    interaction_columns: tuple[int, int]
    coverage_column: int
    nuisance_columns: tuple[int, ...]

    @property
    def width(self) -> int:
        return int(self.matrix.shape[1])

    def block(self, names: Sequence[str]) -> NDArray[np.float64]:
        indices = [index for index, name in enumerate(self.columns) if name in set(names)]
        return self.matrix[:, indices]


def assemble_design(
    *,
    times: NDArray[np.float64],
    time_basis: TimeBasis,
    state_measured: NDArray[np.float64],
    coverage: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    include_coverage: bool = True,
    include_interaction: bool = True,
) -> DesignAssembly:
    """Columns [1, basis(t), (w - mean), (c - mean), z, (w - mean)(t - mean)]."""
    if state_measured.ndim != 2 or state_measured.shape[1] != len(STATE_AXES):
        raise ValueError("the measured state must be a units-by-two matrix")
    n_rows = state_measured.shape[0]
    centred_state = state_measured - state_measured.mean(axis=0, keepdims=True)
    centred_time = times - float(np.mean(times))
    centred_coverage = coverage - float(np.mean(coverage))

    blocks: list[NDArray[np.float64]] = [np.ones((n_rows, 1), dtype=np.float64)]
    names: list[str] = ["intercept"]
    basis = time_basis.evaluate(times)
    if basis.shape[1] > 0:
        blocks.append(basis)
        names.extend(f"basis::{index}" for index in range(basis.shape[1]))
    blocks.append(centred_state)
    state_columns = (len(names), len(names) + len(STATE_AXES) - 1)
    names.extend(f"state::{axis}" for axis in STATE_AXES)
    coverage_column = -1
    if include_coverage:
        blocks.append(centred_coverage.reshape(-1, 1))
        coverage_column = len(names)
        names.append("coverage")
    nuisance_columns: tuple[int, ...] = ()
    if nuisance.shape[1] > 0:
        start = len(names)
        blocks.append(nuisance)
        nuisance_columns = tuple(range(start, start + nuisance.shape[1]))
        names.extend(f"nuisance::{index}" for index in range(nuisance.shape[1]))
    interaction_columns = (-1, -1)
    if include_interaction and time_basis.varies:
        interaction = centred_state * centred_time.reshape(-1, 1)
        blocks.append(interaction)
        interaction_columns = (len(names), len(names) + len(STATE_AXES) - 1)
        names.extend(f"state_by_time::{axis}" for axis in STATE_AXES)

    raw = np.hstack(blocks).astype(np.float64)
    keep, names = _independent_columns(raw, names)
    if keep is None:
        raise ValueError("the calibration design is rank deficient")
    index_of = {name: place for place, name in enumerate(names)}
    matrix = raw[:, keep]
    state_columns = (index_of["state::activation"], index_of["state::matrix"])
    coverage_column = index_of.get("coverage", -1)
    nuisance_columns = tuple(
        sorted(index_of[name] for name in names if name.startswith("nuisance::"))
    )
    interaction_present = [
        index_of[name]
        for name in ("state_by_time::activation", "state_by_time::matrix")
        if name in index_of
    ]
    interaction_columns = (
        (interaction_present[0], interaction_present[-1]) if interaction_present else (-1, -1)
    )
    return DesignAssembly(
        matrix=matrix,
        columns=tuple(names),
        state_columns=state_columns,
        interaction_columns=interaction_columns,
        coverage_column=coverage_column,
        nuisance_columns=nuisance_columns,
    )


def _independent_columns(
    matrix: NDArray[np.float64], names: list[str]
) -> tuple[list[int] | None, list[str]]:
    """Keep the estimand first, then the columns that add rank, in a fixed order.

    A subset of devices drawn from a single batch or a single operator leaves the
    corresponding nuisance indicator constant, so the design would otherwise lose
    rank. The estimand columns keep priority so the slopes stay identified.
    """
    priority = (
        "intercept",
        "state::",
        "coverage",
        "basis::",
        "state_by_time::",
        "nuisance::",
    )
    ranked: list[int] = []
    for prefix in priority:
        ranked.extend(
            index
            for index, name in enumerate(names)
            if (name == prefix if prefix == "intercept" else name.startswith(prefix))
        )
    if len(ranked) != matrix.shape[1]:
        raise ValueError("the design column names do not match the design blocks")
    kept: list[int] = []
    for index in ranked:
        candidate = kept + [index]
        if np.linalg.matrix_rank(matrix[:, candidate]) == len(candidate):
            kept.append(index)
    if not kept or matrix[:, kept].shape[1] < 2:
        return None, []
    return kept, [names[index] for index in kept]


@dataclass(frozen=True)
class PooledFit:
    """One pooled fit of the p-feature front end on the measured state."""

    coefficients: NDArray[np.float64]
    design: DesignAssembly
    feature_names: tuple[str, ...]
    residual_covariance: NDArray[np.float64]
    n_units: int
    n_parameters: int
    inflation: float

    @property
    def residual_dof(self) -> int:
        return self.n_units - self.n_parameters

    @property
    def sensitivity_ols(self) -> NDArray[np.float64]:
        """J_OLS, features by axes, as the notation registry defines it."""
        start, stop = self.design.state_columns
        return np.asarray(self.coefficients[start : stop + 1, :].T, dtype=np.float64)

    @property
    def coverage_sensitivity(self) -> NDArray[np.float64]:
        if self.design.coverage_column < 0:
            raise ValueError("this fit carries no coverage regressor")
        return self.coefficients[self.design.coverage_column, :]

    @property
    def interaction(self) -> NDArray[np.float64]:
        start, stop = self.design.interaction_columns
        if start < 0:
            raise ValueError("this fit carries no state-by-time interaction")
        return np.asarray(self.coefficients[start : stop + 1, :].T, dtype=np.float64)

    def coefficient_se(self, term: str) -> NDArray[np.float64]:
        """Standard errors of one design block, laid out term by feature.

        The term must carry its separator so that the interaction block's
        ``state_by_time::`` columns are not picked up by the ``state::`` prefix.
        """
        indices = [index for index, name in enumerate(self.design.columns) if name.startswith(term)]
        if not indices:
            raise KeyError(f"no design column starts with {term!r}")
        precision = np.linalg.inv(self.design.matrix.T @ self.design.matrix)
        feature_variance = np.diag(self.residual_covariance)
        out = np.empty((len(indices), len(self.feature_names)), dtype=np.float64)
        for row, column_index in enumerate(indices):
            weight = precision[column_index, column_index]
            for feature in range(len(self.feature_names)):
                out[row, feature] = np.sqrt(feature_variance[feature] * weight)
        return out

    def feature_residual_sd(self) -> NDArray[np.float64]:
        return np.sqrt(np.diag(self.residual_covariance))


def plugin_inflation(n_units: int, feature_dimension: int) -> float:
    """The plug-in covariance inflation the manuscript prints for this design.

    Ref: Sec. Calibration design and identifiability. At p = 5 and 116 biological
    devices the factor is 1.064.
    """
    denominator = n_units - feature_dimension - 2
    if denominator <= 0:
        raise ValueError("the plug-in inflation is undefined for this block size")
    return n_units / denominator


def residual_covariance(
    residuals: NDArray[np.float64], n_units: int, feature_dimension: int
) -> NDArray[np.float64]:
    """Residual covariance over the biological rows, on the inflated denominator."""
    denominator = n_units - feature_dimension - 2
    if denominator <= 0:
        raise ValueError("the residual covariance is undefined for this block size")
    return np.asarray(residuals.T @ residuals / denominator, dtype=np.float64)


def fit_pooled(
    *,
    features: NDArray[np.float64],
    state_measured: NDArray[np.float64],
    coverage: NDArray[np.float64],
    times: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    feature_names: Sequence[str],
    time_basis_columns: int = 3,
    include_coverage: bool = True,
    include_interaction: bool = True,
    n_biological: int | None = None,
    inflation: str = "plugin",
) -> PooledFit:
    """Least squares of the feature block on the Eq. (3) design."""
    if features.shape[0] != state_measured.shape[0]:
        raise ValueError("features and measured state must share the unit axis")
    if features.shape[1] != len(feature_names):
        raise ValueError("feature names must match the feature block width")
    basis = TimeBasis.fit(times, time_basis_columns)
    design = assemble_design(
        times=times,
        time_basis=basis,
        state_measured=state_measured,
        coverage=coverage,
        nuisance=nuisance,
        include_coverage=include_coverage,
        include_interaction=include_interaction,
    )
    solution, *_ = np.linalg.lstsq(design.matrix, features, rcond=None)
    residual = features - design.matrix @ solution
    n_units = int(features.shape[0]) if n_biological is None else int(n_biological)
    dimension = len(feature_names)
    factor = plugin_inflation(n_units, dimension) if inflation == "plugin" else 1.0
    covariance = residual_covariance(residual, n_units, dimension)
    return PooledFit(
        coefficients=np.asarray(solution, dtype=np.float64),
        design=design,
        feature_names=tuple(feature_names),
        residual_covariance=covariance,
        n_units=n_units,
        n_parameters=design.width,
        inflation=factor,
    )


def wald_test(estimate: float, standard_error: float, residual_dof: int) -> dict[str, float]:
    """Wald statistic and two-sided P for one slope."""
    if standard_error <= 0.0:
        return {
            "estimate": estimate,
            "standard_error": standard_error,
            "t": float("nan"),
            "p": float("nan"),
        }
    statistic = float(estimate) / float(standard_error)
    p_value = float(2.0 * stats.t.sf(abs(statistic), residual_dof))
    return {"estimate": estimate, "standard_error": standard_error, "t": statistic, "p": p_value}


def confidence_interval(
    estimate: float, standard_error: float, residual_dof: int, level: float = 0.95
) -> tuple[float, float]:
    critical = float(stats.t.ppf(0.5 + level / 2.0, residual_dof))
    half_width = critical * float(standard_error)
    return float(estimate) - half_width, float(estimate) + half_width
