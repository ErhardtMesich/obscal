"""Arm roster, chip partition and the timepoint grid.

Ref: Sec. Device, cultures and injury model; SI Table S7, Table S9 Panel B.
"""

from __future__ import annotations

from dataclasses import dataclass

ARM_LABELS: dict[str, str] = {
    "S1": "uninjured control",
    "S2": "ethanol injury, untreated",
    "S3": "ethanol injury, washout",
    "S4": "ethanol injury, bone marrow-derived stromal treatment",
    "S5": "ethanol injury, umbilical cord-derived stromal treatment",
    "S6": "stromal cells alone",
}

ARM_ROLES: dict[str, str] = {
    "S1": "reference for the untreated injury arm",
    "S2": "untreated injury reference and the matrix axis carrier",
    "S3": "injury withdrawn without treatment",
    "S4": "primary treatment arm and the powered contrast",
    "S5": "cross-source treatment arm",
    "S6": "measures the therapeutic's own matrix contribution",
}

TREATED_ARMS: tuple[str, ...] = ("S4", "S5")
INJURY_ARMS: tuple[str, ...] = ("S2", "S3", "S4", "S5")


@dataclass(frozen=True)
class TimepointCell:
    """One arm-by-timepoint cell, as the trajectory table reports it."""

    arm: str
    culture_day: int
    treatment_day: int | None
    chips: int

    @property
    def label(self) -> str:
        treatment = "pre" if self.treatment_day is None else str(self.treatment_day)
        return f"{self.arm} day {self.culture_day} (treatment {treatment})"


@dataclass(frozen=True)
class ArmSpec:
    """An arm with its role, its source biology and its timepoint cells."""

    arm: str
    label: str
    role: str
    source: str
    cells: tuple[TimepointCell, ...]

    @property
    def chips(self) -> int:
        return sum(cell.chips for cell in self.cells)

    @property
    def culture_days(self) -> tuple[int, ...]:
        return tuple(cell.culture_day for cell in self.cells)

    @property
    def on_21_day_schedule(self) -> int:
        return sum(cell.chips for cell in self.cells if cell.culture_day == 21)


def _cell(arm: str, day: int, treatment: int | None, chips: int) -> TimepointCell:
    return TimepointCell(arm=arm, culture_day=day, treatment_day=treatment, chips=chips)


ROSTER: tuple[ArmSpec, ...] = (
    ArmSpec(
        arm="S1",
        label=ARM_LABELS["S1"],
        role=ARM_ROLES["S1"],
        source="LX-2 + Hep G2",
        cells=(
            _cell("S1", 3, None, 4),
            _cell("S1", 7, None, 4),
            _cell("S1", 21, 14, 4),
        ),
    ),
    ArmSpec(
        arm="S2",
        label=ARM_LABELS["S2"],
        role=ARM_ROLES["S2"],
        source="LX-2 + Hep G2, ethanol and acetaldehyde",
        cells=(
            _cell("S2", 3, None, 4),
            _cell("S2", 7, None, 4),
            _cell("S2", 10, 3, 4),
            _cell("S2", 14, 7, 4),
            _cell("S2", 21, 14, 12),
            _cell("S2", 28, 21, 4),
        ),
    ),
    ArmSpec(
        arm="S3",
        label=ARM_LABELS["S3"],
        role=ARM_ROLES["S3"],
        source="LX-2 + Hep G2, ethanol and acetaldehyde",
        cells=(
            _cell("S3", 3, None, 4),
            _cell("S3", 7, None, 4),
            _cell("S3", 14, 7, 4),
            _cell("S3", 21, 14, 12),
            _cell("S3", 28, 21, 4),
        ),
    ),
    ArmSpec(
        arm="S4",
        label=ARM_LABELS["S4"],
        role=ARM_ROLES["S4"],
        source="LX-2 + Hep G2, bone marrow-derived stromal cells",
        cells=(
            _cell("S4", 3, None, 4),
            _cell("S4", 7, None, 4),
            _cell("S4", 10, 3, 4),
            _cell("S4", 14, 7, 4),
            _cell("S4", 21, 14, 12),
            _cell("S4", 28, 21, 4),
        ),
    ),
    ArmSpec(
        arm="S5",
        label=ARM_LABELS["S5"],
        role=ARM_ROLES["S5"],
        source="LX-2 + Hep G2, umbilical cord-derived stromal cells",
        cells=(
            _cell("S5", 3, None, 4),
            _cell("S5", 7, None, 4),
            _cell("S5", 10, 3, 4),
            _cell("S5", 14, 7, 4),
            _cell("S5", 21, 14, 12),
            _cell("S5", 28, 21, 4),
        ),
    ),
    ArmSpec(
        arm="S6",
        label=ARM_LABELS["S6"],
        role=ARM_ROLES["S6"],
        source="bone marrow-derived stromal cells alone",
        cells=(
            _cell("S6", 3, None, 4),
            _cell("S6", 7, None, 4),
            _cell("S6", 21, 14, 4),
        ),
    ),
)

CELL_ONLY_ARM = "S6"
INJURY_REFERENCE_ARM = "S2"
CONTROL_ARM = "S1"
PRIMARY_TREATMENT_ARM = "S4"

PRIMARY_ENDPOINT_CULTURE_DAY = 21
PRIMARY_ENDPOINT_TREATMENT_DAY = 14
EXPLORATORY_CULTURE_DAY = 28
TRAJECTORY_TIMEPOINTS: tuple[int, ...] = (3, 7, 10, 14, 21, 28)


def arm(arm_id: str) -> ArmSpec:
    for spec in ROSTER:
        if spec.arm == arm_id:
            return spec
    raise KeyError(f"unknown arm {arm_id!r}")


def trajectory_cells() -> tuple[TimepointCell, ...]:
    return tuple(cell for spec in ROSTER for cell in spec.cells)


def arm_totals() -> dict[str, int]:
    return {spec.arm: spec.chips for spec in ROSTER}


def biological_calibration_devices() -> int:
    """The 148-device biological block the calibration is fitted inside."""
    return sum(spec.chips for spec in ROSTER)


def on_21_day_schedule() -> int:
    return sum(spec.on_21_day_schedule for spec in ROSTER)


def timepoint_counts() -> dict[int, int]:
    counts: dict[int, int] = {}
    for cell in trajectory_cells():
        counts[cell.culture_day] = counts.get(cell.culture_day, 0) + cell.chips
    return counts


def arm_timepoint_means(
    arm_id: str, endpoint: str, table: dict[str, dict[int, float]]
) -> dict[int, float]:
    """Read one endpoint's per-timepoint mean for an arm from a trajectory table."""
    if endpoint not in table:
        raise KeyError(f"endpoint {endpoint!r} is not in the trajectory table")
    series = table[endpoint]
    days = {cell.culture_day for cell in arm(arm_id).cells}
    return {day: series[day] for day in sorted(days) if day in series}


def primary_contrast_cells() -> tuple[TimepointCell, ...]:
    """Cells carrying the pre-specified primary contrast at the primary endpoint."""
    return tuple(
        cell for cell in trajectory_cells() if cell.culture_day == PRIMARY_ENDPOINT_CULTURE_DAY
    )


def supporting_timepoint_cells() -> tuple[TimepointCell, ...]:
    return tuple(
        cell
        for cell in trajectory_cells()
        if cell.culture_day not in (PRIMARY_ENDPOINT_CULTURE_DAY, EXPLORATORY_CULTURE_DAY)
    )
