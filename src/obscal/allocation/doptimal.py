"""D-optimal placement of the next destructive assay.

Ref: Sec. Forecasting and terminal-assay allocation, Eq. (6); Algorithm 2.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

RIDGE_SCALE = 1.0e-9
"""Numerical ridge, relative to the weight's own scale.

The information matrix is singular until as many units have been placed as
there are parameters, so the bare determinant is identically zero there and the
marginal gain is undefined. A ridge of numerical size keeps the criterion
well-defined and leaves the ordering of full-rank states untouched.
"""


@dataclass(frozen=True)
class InformationState:
    """The design information matrix accrued so far."""

    matrix: NDArray[np.float64]
    weight: NDArray[np.float64]
    spent: int
    ridge: float = 0.0

    @classmethod
    def initial(cls, parameter_count: int, weight: NDArray[np.float64]) -> InformationState:
        scale = float(np.mean(np.diag(weight)))
        return cls(
            matrix=np.zeros((parameter_count, parameter_count), dtype=np.float64),
            weight=weight,
            spent=0,
            ridge=RIDGE_SCALE * scale,
        )

    @property
    def log_determinant(self) -> float:
        sign, value = np.linalg.slogdet(self.matrix + self.ridge * np.eye(self.matrix.shape[0]))
        if sign <= 0.0:
            return float("-inf")
        return float(value)

    def with_row(self, row: NDArray[np.float64]) -> InformationState:
        updated = self.matrix + np.outer(row, row) @ self.weight
        return InformationState(
            matrix=updated, weight=self.weight, spent=self.spent + 1, ridge=self.ridge
        )

    def as_record(self) -> dict[str, float | int]:
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return {
            "spent": self.spent,
            "log_determinant": self.log_determinant,
            "min_eigenvalue": float(eigenvalues.min()) if self.spent > 0 else 0.0,
        }


def marginal_gain(information: InformationState, row: NDArray[np.float64]) -> float:
    """Eq. (5) of the allocation step: log det(M + f W f^T) - log det M.

    D-optimality is chosen over criteria built on the extreme eigenvalues of the
    same matrix because a determinant is invariant under linear reparameterisation
    of the state and an eigenvalue is not.
    """
    return information.with_row(row).log_determinant - information.log_determinant


def gains(information: InformationState, rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Marginal gain of every feasible candidate at the current state."""
    return np.array([marginal_gain(information, row) for row in rows], dtype=np.float64)


def d_optimal_order(
    rows: NDArray[np.float64],
    weight: NDArray[np.float64],
    budget: int,
    *,
    initial: InformationState | None = None,
    max_exchanges: int | None = None,
) -> list[int]:
    """Greedy forward selection, then an exchange pass that can only improve.

    Ref: Algorithm 2, "the D-optimal exchange search over the calibration grid
    converges in a median of 11 exchange steps".
    """
    n_candidates = rows.shape[0]
    if budget > n_candidates:
        raise ValueError("the budget cannot exceed the candidate pool")
    state = initial if initial is not None else InformationState.initial(rows.shape[1], weight)
    order: list[int] = []
    available = set(range(n_candidates))
    while len(order) < budget:
        candidates = sorted(available)
        scores = gains(state, rows[candidates])
        best = candidates[int(np.argmax(scores))]
        order.append(best)
        available.discard(best)
        state = state.with_row(rows[best])
    if max_exchanges is None or max_exchanges <= 0:
        return order
    return _exchange_pass(rows, weight, order, budget, max_exchanges)


def _exchange_pass(
    rows: NDArray[np.float64],
    weight: NDArray[np.float64],
    order: list[int],
    budget: int,
    max_exchanges: int,
) -> list[int]:
    """Drop one selected unit and re-add the best available, while it helps."""
    selected = list(order)
    available = [index for index in range(rows.shape[0]) if index not in set(selected)]
    for _ in range(max_exchanges):
        base = _information_of(rows, weight, selected)
        baseline = base.log_determinant
        best_value = baseline
        best_swap: tuple[int, int] | None = None
        for position in range(budget):
            reduced = selected[:position] + selected[position + 1 :]
            state = _information_of(rows, weight, reduced)
            for candidate in available:
                value = state.with_row(rows[candidate]).log_determinant
                if value > best_value + 1e-12:
                    best_value = value
                    best_swap = (position, candidate)
        if best_swap is None:
            break
        position, candidate = best_swap
        available.append(selected[position])
        available.remove(candidate)
        selected[position] = candidate
    return selected


def _information_of(
    rows: NDArray[np.float64], weight: NDArray[np.float64], selected: Sequence[int]
) -> InformationState:
    state = InformationState.initial(rows.shape[1], weight)
    for index in selected:
        state = state.with_row(rows[index])
    return state


def leverage(row: NDArray[np.float64], precision: NDArray[np.float64]) -> float:
    """Scalar information a row contributes along a single fitted direction."""
    return float(row @ precision @ row)


def oracle_order(
    rows: NDArray[np.float64],
    realised: NDArray[np.float64],
    weight: NDArray[np.float64],
    budget: int,
) -> list[int]:
    """Clairvoyant placement: the oracle sees the realised future trajectory.

    The oracle is a ceiling, not a competitor; it is the pair the realised
    trajectory makes most informative.
    """
    precision = np.asarray(weight, dtype=np.float64)
    scores = np.array(
        [
            abs(float(realised[index])) * leverage(rows[index], precision)
            for index in range(rows.shape[0])
        ],
        dtype=np.float64,
    )
    order = list(np.argsort(-scores, kind="stable"))
    return [int(index) for index in order[:budget]]


def exchange_step_count(rows: NDArray[np.float64], weight: NDArray[np.float64], budget: int) -> int:
    """How many exchange steps the search takes to stop improving."""
    greedy = d_optimal_order(rows, weight, budget)
    refined = _exchange_pass(rows, weight, greedy, budget, max_exchanges=1000)
    return sum(1 for left, right in zip(greedy, refined, strict=True) if left != right)


def select_by_policy(
    policy: Callable[[NDArray[np.float64], NDArray[np.float64], int], list[int]],
    rows: NDArray[np.float64],
    weight: NDArray[np.float64],
    budget: int,
) -> list[int]:
    """Apply a policy callable and validate the order it returns."""
    order = policy(rows, weight, budget)
    if len(order) != budget:
        raise ValueError("a policy must return exactly the budgeted number of placements")
    if len(set(order)) != len(order):
        raise ValueError("a policy cannot destroy the same unit twice")
    return order
