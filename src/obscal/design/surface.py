"""The dose-by-schedule treatment surface and its matched-dose constraint.

Ref: Sec. Device, cultures and injury model; SI Table S10, Table S21.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

DOSE_LEVELS: tuple[float, ...] = (1.25e4, 2.5e4, 5.0e4, 1.0e5)
"""Total cells delivered per chip over the 14 treatment days."""

SCHEDULE_LABELS: tuple[str, ...] = (
    "single_bolus",
    "two_fractions",
    "three_fractions",
    "continuous",
)

SCHEDULE_TREATMENT_DAYS: dict[str, tuple[int, ...]] = {
    "single_bolus": (0,),
    "two_fractions": (0, 5),
    "three_fractions": (0, 3, 7),
    "continuous": tuple(range(14)),
}

FORECAST_DRIVEN_DOSE = 5.0e4
FORECAST_DRIVEN_SCHEDULE = "three_fractions"

STATE_AXES: tuple[str, ...] = ("activation", "matrix")


@dataclass(frozen=True)
class SurfaceCell:
    """One cell of the dose-by-schedule surface, measured or forecast."""

    dose_cells: float
    schedule: str
    activation_fold: float
    matrix_ug_hyp: float
    pro_collagen_ng_per_ml: float
    measured: bool
    chips: int

    @property
    def key(self) -> tuple[float, str]:
        return (self.dose_cells, self.schedule)


@dataclass(frozen=True)
class TreatmentSurface:
    """The whole 4x4 surface at matched cumulative dose."""

    cells: tuple[SurfaceCell, ...]
    untreated: SurfaceCell

    def __post_init__(self) -> None:
        keys = {cell.key for cell in self.cells}
        if len(keys) != len(self.cells):
            raise ValueError("surface cells must be unique")
        if len(self.cells) != len(DOSE_LEVELS) * len(SCHEDULE_LABELS):
            raise ValueError("the surface must carry every dose-by-schedule cell")

    @property
    def measured_cells(self) -> tuple[SurfaceCell, ...]:
        return tuple(cell for cell in self.cells if cell.measured)

    @property
    def forecast_cells(self) -> tuple[SurfaceCell, ...]:
        return tuple(cell for cell in self.cells if not cell.measured)

    def at(self, dose_cells: float, schedule: str) -> SurfaceCell:
        for cell in self.cells:
            if cell.key == (dose_cells, schedule):
                return cell
        raise KeyError(f"no surface cell at dose {dose_cells} schedule {schedule!r}")

    def column(self, schedule: str) -> tuple[SurfaceCell, ...]:
        return tuple(
            sorted(
                (cell for cell in self.cells if cell.schedule == schedule),
                key=lambda cell: cell.dose_cells,
            )
        )

    def forecast_driven(self) -> SurfaceCell:
        return self.at(FORECAST_DRIVEN_DOSE, FORECAST_DRIVEN_SCHEDULE)

    def max_dose_cell(self, schedule: str) -> SurfaceCell:
        return self.column(schedule)[-1]


def per_fraction_dose(schedule: str, dose_cells: float) -> float:
    """Cells delivered per intervention under a schedule at a matched total dose."""
    days = SCHEDULE_TREATMENT_DAYS[schedule]
    return float(dose_cells) / float(len(days))


def delivery_days(schedule: str) -> tuple[int, ...]:
    return SCHEDULE_TREATMENT_DAYS[schedule]


def matched_cumulative_dose_report(surface: TreatmentSurface) -> dict[str, float]:
    """Confirm the four schedules of a dose row carry the same total dose."""
    report: dict[str, float] = {}
    for dose in DOSE_LEVELS:
        row = [cell for cell in surface.cells if cell.dose_cells == dose]
        totals = {
            per_fraction_dose(cell.schedule, cell.dose_cells) * len(delivery_days(cell.schedule))
            for cell in row
        }
        if len(totals) != 1:
            raise ValueError(f"dose row {dose} does not carry a matched total dose")
        report[f"{dose:.2e}"] = totals.pop()
    return report


def monotonic_with_dose(cells: Sequence[SurfaceCell], endpoint: str) -> bool:
    """Whether an endpoint falls monotonically with total dose at one schedule."""
    if endpoint not in ("activation", "matrix"):
        raise ValueError(f"unknown endpoint {endpoint!r}")
    ordered = sorted(cells, key=lambda cell: cell.dose_cells)
    values = [
        cell.activation_fold if endpoint == "activation" else cell.matrix_ug_hyp for cell in ordered
    ]
    return all(later < earlier for earlier, later in zip(values, values[1:]))


def monotonicity_verdict(surface: TreatmentSurface) -> dict[str, object]:
    """State the verdict in words and per cell, rather than arguing it later."""
    per_cell: dict[str, dict[str, bool]] = {}
    for schedule in SCHEDULE_LABELS:
        column = surface.column(schedule)
        per_cell[schedule] = {
            "activation": monotonic_with_dose(column, "activation"),
            "matrix": monotonic_with_dose(column, "matrix"),
        }
    both = all(flags["activation"] and flags["matrix"] for flags in per_cell.values())
    return {
        "per_schedule": per_cell,
        "cells_with_both_axes_monotone": sum(
            2 for flags in per_cell.values() if flags["activation"] and flags["matrix"]
        ),
        "cells_total_axes": 2 * len(SCHEDULE_LABELS),
        "both_axes_monotone_everywhere": both,
        "dose_optimum_claimed": False if both else None,
    }


def constraint_axis_report(
    surface: TreatmentSurface, floor_ug_hyp: float
) -> dict[str, float | bool | str]:
    """Report the dosing result on the constraint axis when no optimum exists."""
    forecast_driven = surface.forecast_driven()
    continuous_max = surface.max_dose_cell("continuous")
    bolus_max = surface.max_dose_cell("single_bolus")
    three_fraction_max = surface.max_dose_cell("three_fractions")
    conceded = forecast_driven.matrix_ug_hyp - continuous_max.matrix_ug_hyp
    cells_saved = 1.0 - forecast_driven.dose_cells / continuous_max.dose_cells
    return {
        "forecast_driven_matrix_ug_hyp": forecast_driven.matrix_ug_hyp,
        "continuous_max_matrix_ug_hyp": continuous_max.matrix_ug_hyp,
        "single_bolus_max_matrix_ug_hyp": bolus_max.matrix_ug_hyp,
        "three_fraction_max_matrix_ug_hyp": three_fraction_max.matrix_ug_hyp,
        "conceded_ug_hyp_against_continuous": conceded,
        "conceded_ug_hyp_against_single_bolus": bolus_max.matrix_ug_hyp
        - forecast_driven.matrix_ug_hyp,
        "dose_saved_fraction": cells_saved,
        "concession_below_floor": bool(conceded < floor_ug_hyp),
        "criterion_adjudicable": bool(conceded >= floor_ug_hyp),
        "status": (
            "forecast_not_measured"
            if forecast_driven.matrix_ug_hyp != continuous_max.matrix_ug_hyp
            else "measured"
        ),
    }


def surface_records(surface: TreatmentSurface) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "dose_cells": surface.untreated.dose_cells,
            "schedule": "none",
            "activation_fold": surface.untreated.activation_fold,
            "matrix_ug_hyp": surface.untreated.matrix_ug_hyp,
            "pro_collagen_ng_per_ml": surface.untreated.pro_collagen_ng_per_ml,
            "status": "measured" if surface.untreated.measured else "forecast",
            "chips": surface.untreated.chips,
        }
    ]
    for cell in sorted(surface.cells, key=lambda item: (item.schedule, item.dose_cells)):
        records.append(
            {
                "dose_cells": cell.dose_cells,
                "schedule": cell.schedule,
                "activation_fold": cell.activation_fold,
                "matrix_ug_hyp": cell.matrix_ug_hyp,
                "pro_collagen_ng_per_ml": cell.pro_collagen_ng_per_ml,
                "status": "measured" if cell.measured else "forecast",
                "chips": cell.chips,
            }
        )
    return records


def treated_against_untreated(surface: TreatmentSurface, endpoint: str) -> dict[str, float]:
    """Reduction of an endpoint in the forecast-driven arm against untreated injury."""
    if endpoint not in STATE_AXES:
        raise ValueError(f"unknown endpoint {endpoint!r}")
    treated = surface.forecast_driven()
    reference = surface.untreated
    treated_value = treated.activation_fold if endpoint == "activation" else treated.matrix_ug_hyp
    reference_value = (
        reference.activation_fold if endpoint == "activation" else reference.matrix_ug_hyp
    )
    return {
        "treated": treated_value,
        "untreated": reference_value,
        "absolute_reduction": reference_value - treated_value,
        "relative_reduction_fraction": (reference_value - treated_value) / reference_value,
    }


def source_comparison(
    surface_records_by_arm: Mapping[str, float], source_arms: Sequence[str]
) -> dict[str, float]:
    """Pairwise difference between two treatment sources at the primary endpoint."""
    if len(source_arms) != 2:
        raise ValueError("a source comparison needs exactly two arms")
    first, second = source_arms
    return {
        f"{first}_matrix_ug_hyp": surface_records_by_arm[first],
        f"{second}_matrix_ug_hyp": surface_records_by_arm[second],
        "difference_ug_hyp": surface_records_by_arm[first] - surface_records_by_arm[second],
    }
