"""Pre-declared inclusion rules and the withdrawal ledger.

Ref: Sec. Platform characterisation and culture longevity; SI Table S14.
"""

from __future__ import annotations

from dataclasses import dataclass

from obscal.design import partition


@dataclass(frozen=True)
class WithdrawalClass:
    """One mutually exclusive reason a device left the study."""

    name: str
    count: int
    criterion: str
    associated_magnitude: str
    associated_input_property: str

    @property
    def share_of_pool_pct(self) -> float:
        return 100.0 * self.count / partition.TOTAL_POOL_DEVICES


WITHDRAWAL_CLASSES: tuple[WithdrawalClass, ...] = (
    WithdrawalClass(
        name="bubble_occlusion",
        count=14,
        criterion="occlusion logged during perfusion",
        associated_magnitude="occlusion logged",
        associated_input_property="manifold pressure transient",
    ),
    WithdrawalClass(
        name="seeding_confluence_outside_band",
        count=9,
        criterion="coverage below 82% or above 96%",
        associated_magnitude="coverage < 82% or > 96%",
        associated_input_property="cell lot and passage",
    ),
    WithdrawalClass(
        name="channel_leak",
        count=7,
        criterion="pressure decay above 5% per minute at the pre-perfusion check",
        associated_magnitude="pressure decay > 5%/min",
        associated_input_property="bonding batch",
    ),
    WithdrawalClass(
        name="electrode_drift_above_threshold",
        count=5,
        criterion="drift above 15 ohm per day at 1 kHz",
        associated_magnitude="> 15 ohm/day at 1 kHz",
        associated_input_property="electrode wetting time",
    ),
    WithdrawalClass(
        name="unequal_terminal_split",
        count=6,
        criterion="half-mass ratio outside 0.4 to 0.6",
        associated_magnitude="half-mass ratio outside 0.4 to 0.6",
        associated_input_property="section plane placement",
    ),
    WithdrawalClass(
        name="hydroxyproline_below_quantification",
        count=4,
        criterion="hydroxyproline below 0.18 ug per chip",
        associated_magnitude="< 0.18 ug per chip",
        associated_input_property="early termination time",
    ),
)


def withdrawal_class(name: str) -> WithdrawalClass:
    for entry in WITHDRAWAL_CLASSES:
        if entry.name == name:
            return entry
    raise KeyError(f"unknown withdrawal class {name!r}")


def total_withdrawn() -> int:
    return sum(entry.count for entry in WITHDRAWAL_CLASSES)


def withdrawal_share_pct() -> float:
    return 100.0 * total_withdrawn() / partition.TOTAL_POOL_DEVICES


def ledger() -> list[dict[str, object]]:
    return [
        {
            "class": entry.name,
            "count": entry.count,
            "share_of_pool_pct": entry.share_of_pool_pct,
            "criterion": entry.criterion,
            "associated_magnitude": entry.associated_magnitude,
            "associated_input_property": entry.associated_input_property,
        }
        for entry in WITHDRAWAL_CLASSES
    ]


def ledger_closure() -> dict[str, tuple[float, float]]:
    """Counts and shares must sum to the pool's withdrawn total."""
    return {
        "class_counts_sum_to_total": (
            float(total_withdrawn()),
            float(partition.WITHDRAWN_DEVICES),
        ),
        "remaining_pool": (
            float(partition.TOTAL_POOL_DEVICES - total_withdrawn()),
            float(partition.analysed_devices()),
        ),
        "shares_sum_to_total": (
            float(sum(entry.share_of_pool_pct for entry in WITHDRAWAL_CLASSES)),
            float(withdrawal_share_pct()),
        ),
    }


def classify_device(
    *,
    bubble_logged: bool,
    coverage_percent: float,
    pressure_decay_per_min: float,
    electrode_drift_ohm_per_day: float,
    half_mass_ratio: float,
    hydroxyproline_ug_per_chip: float,
) -> str | None:
    """Apply the pre-declared rules in the order the ledger lists them."""
    if bubble_logged:
        return "bubble_occlusion"
    if not 82.0 <= coverage_percent <= 96.0:
        return "seeding_confluence_outside_band"
    if pressure_decay_per_min > 0.05:
        return "channel_leak"
    if abs(electrode_drift_ohm_per_day) > 15.0:
        return "electrode_drift_above_threshold"
    if not 0.4 <= half_mass_ratio <= 0.6:
        return "unequal_terminal_split"
    if hydroxyproline_ug_per_chip < 0.18:
        return "hydroxyproline_below_quantification"
    return None


def class_counts(decisions: list[str | None]) -> dict[str, int]:
    """Tally a sequence of classification decisions against the ledger classes."""
    counts = {entry.name: 0 for entry in WITHDRAWAL_CLASSES}
    retained = 0
    for decision in decisions:
        if decision is None:
            retained += 1
        elif decision in counts:
            counts[decision] += 1
        else:
            raise ValueError(f"unledgered withdrawal reason {decision!r}")
    counts["retained"] = retained
    return counts


def exclusion_partition_is_exhaustive(decisions: list[str | None]) -> bool:
    """Every device is either withdrawn into exactly one class or retained."""
    counts = class_counts(decisions)
    withdrawn = sum(counts[entry.name] for entry in WITHDRAWAL_CLASSES)
    return withdrawn + counts["retained"] == len(decisions)
