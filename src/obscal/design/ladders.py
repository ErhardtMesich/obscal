"""The non-biological decoupling ladders.

Ref: Sec. Decoupling ladders: acellular matrix, coverage and early activation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LadderLevel:
    """One level of a decoupling ladder with its nominal covariate value."""

    index: int
    nominal: float
    unit: str
    devices: int


@dataclass(frozen=True)
class Ladder:
    """A ladder that holds the state axes fixed while one covariate varies."""

    name: str
    role: str
    levels: tuple[LadderLevel, ...]

    @property
    def devices(self) -> int:
        return sum(level.devices for level in self.levels)

    @property
    def level_count(self) -> int:
        return len(self.levels)

    @property
    def is_balanced(self) -> bool:
        return len({level.devices for level in self.levels}) == 1

    def values(self) -> tuple[float, ...]:
        return tuple(level.nominal for level in self.levels)


ACELULAR_LEVELS: tuple[float, ...] = (0.25, 0.50, 1.00, 2.00, 4.00, 8.00)
SEEDING_DENSITY_LEVELS: tuple[float, ...] = (0.40, 0.70, 1.00, 1.40, 1.80)


def acellular_ladder() -> Ladder:
    """Cell-free collagen levels: matrix varies while activation and coverage are zero."""
    return Ladder(
        name="acellular_matrix",
        role="attenuation_free_matrix_sensitivity",
        levels=tuple(
            LadderLevel(index=index, nominal=value, unit="ug_collagen_per_device", devices=4)
            for index, value in enumerate(ACELULAR_LEVELS)
        ),
    )


def seeding_density_ladder() -> Ladder:
    """Seeding density at fixed state: identifies coverage separately from the axes."""
    return Ladder(
        name="seeding_density",
        role="coverage_sensitivity_at_fixed_state",
        levels=tuple(
            LadderLevel(index=index, nominal=value, unit="relative_seeding_density", devices=4)
            for index, value in enumerate(SEEDING_DENSITY_LEVELS)
        ),
    )


def ladders() -> tuple[Ladder, ...]:
    return (acellular_ladder(), seeding_density_ladder())


def ladder_devices() -> int:
    return sum(ladder.devices for ladder in ladders())


@dataclass(frozen=True)
class EarlyActivationWindow:
    """The window where activation has risen and deposition has not yet followed."""

    start_culture_day: int
    end_culture_day: int
    devices: int
    arms: tuple[str, ...]

    def contains(self, culture_day: int) -> bool:
        return self.start_culture_day <= culture_day <= self.end_culture_day


def early_activation_window() -> EarlyActivationWindow:
    return EarlyActivationWindow(
        start_culture_day=7,
        end_culture_day=10,
        devices=8,
        arms=("S2", "S4", "S5"),
    )


def decoupling_coverage() -> dict[str, object]:
    """What the ladders buy: which axes each arm leaves identifiable."""
    return {
        "acellular_ladder_devices": acellular_ladder().devices,
        "acellular_ladder_levels": acellular_ladder().level_count,
        "seeding_density_devices": seeding_density_ladder().devices,
        "seeding_density_levels": seeding_density_ladder().level_count,
        "early_activation_devices": early_activation_window().devices,
        "ladder_devices_total": ladder_devices(),
        "full_design_devices": 192,
        "ladder_share_of_full_design_pct": 100.0 * ladder_devices() / 192.0,
        "axes_identified_by_acellular": ("matrix",),
        "axes_identified_by_seeding_density": ("coverage",),
        "both_axes_identified_when": "at least one ladder arm of each kind is present",
    }


def coverage_design_values(design_spread_cv: float, assay_cv: float) -> dict[str, float]:
    """Design-time attenuation of the optically measured coverage covariate."""
    spread = float(design_spread_cv) ** 2
    assay = float(assay_cv) ** 2
    return {
        "design_spread_cv": float(design_spread_cv),
        "assay_cv": float(assay_cv),
        "lambda": spread / (spread + assay),
    }
