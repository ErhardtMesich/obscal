"""The chip budget and its partition into blocks.

Ref: Sec. Experimental design, batching and power; Table 1; SI Table S9 Panel B.
"""

from __future__ import annotations

from dataclasses import dataclass

from obscal.design.roster import biological_calibration_devices, on_21_day_schedule

CALIBRATION_FIT_DEVICES = 116
HELD_OUT_DEVICES = 32
FULL_DESIGN_DEVICES = 192
WITHDRAWN_DEVICES = 45


@dataclass(frozen=True)
class ChipBlock:
    """One block of the device budget, with the role it plays and its fit status."""

    name: str
    devices: int
    role: str
    in_calibration_fit: bool
    in_full_design: bool

    @property
    def share_of_pool(self) -> float:
        return 100.0 * self.devices / TOTAL_POOL_DEVICES


BLOCKS: tuple[ChipBlock, ...] = (
    ChipBlock(
        name="biological",
        devices=148,
        role="six arm-by-timepoint biological arms carrying both state axes",
        in_calibration_fit=True,
        in_full_design=True,
    ),
    ChipBlock(
        name="acellular_ladder",
        devices=24,
        role="attenuation-free matrix sensitivity at zero activation and zero coverage",
        in_calibration_fit=True,
        in_full_design=True,
    ),
    ChipBlock(
        name="coverage_ladder",
        devices=20,
        role="coverage sensitivity at fixed state",
        in_calibration_fit=True,
        in_full_design=True,
    ),
    ChipBlock(
        name="cell_free",
        devices=8,
        role="artefact-confound control on the identical schedule",
        in_calibration_fit=False,
        in_full_design=False,
    ),
    ChipBlock(
        name="held_out_geometry",
        devices=12,
        role="instrument-level external test on a second channel layout",
        in_calibration_fit=False,
        in_full_design=False,
    ),
    ChipBlock(
        name="cross_line",
        devices=12,
        role="cross-hepatocyte-line generalization arm",
        in_calibration_fit=False,
        in_full_design=False,
    ),
    ChipBlock(
        name="static_counterpart",
        devices=36,
        role="perfused against static platform validation",
        in_calibration_fit=False,
        in_full_design=False,
    ),
)

TOTAL_POOL_DEVICES = 260
ON_21_DAY_SCHEDULE = 216


def block(name: str) -> ChipBlock:
    for entry in BLOCKS:
        if entry.name == name:
            return entry
    raise KeyError(f"unknown chip block {name!r}")


def total_devices() -> int:
    return sum(entry.devices for entry in BLOCKS)


def full_design_devices() -> int:
    return sum(entry.devices for entry in BLOCKS if entry.in_full_design)


def analysed_devices() -> int:
    return TOTAL_POOL_DEVICES - WITHDRAWN_DEVICES


def held_out_devices() -> int:
    return biological_calibration_devices() - CALIBRATION_FIT_DEVICES


def off_21_day_schedule() -> int:
    return TOTAL_POOL_DEVICES - ON_21_DAY_SCHEDULE


def fit_and_held_out_agree() -> bool:
    return held_out_devices() == HELD_OUT_DEVICES


def budget_partition() -> dict[str, object]:
    """Report-ready view of the whole budget, with every identity stated."""
    return {
        "blocks": [
            {
                "name": entry.name,
                "devices": entry.devices,
                "role": entry.role,
                "in_calibration_fit": entry.in_calibration_fit,
                "in_full_design": entry.in_full_design,
                "share_of_pool_pct": entry.share_of_pool,
            }
            for entry in BLOCKS
        ],
        "total_devices": total_devices(),
        "biological_devices": biological_calibration_devices(),
        "full_design_devices": full_design_devices(),
        "calibration_fit_devices": CALIBRATION_FIT_DEVICES,
        "held_out_devices": held_out_devices(),
        "withdrawn_devices": WITHDRAWN_DEVICES,
        "analysed_devices": analysed_devices(),
        "on_21_day_schedule": on_21_day_schedule(),
        "off_21_day_schedule": off_21_day_schedule(),
        "withdrawal_share_pct": 100.0 * WITHDRAWN_DEVICES / TOTAL_POOL_DEVICES,
    }


def block_closure_identities() -> dict[str, tuple[float, float]]:
    """Pairs that must be equal for the published budget to be self-consistent."""
    return {
        "arm_totals_sum_to_biological_block": (
            float(biological_calibration_devices()),
            float(block("biological").devices),
        ),
        "biological_plus_ladders_is_full_design": (
            float(
                biological_calibration_devices()
                + block("acellular_ladder").devices
                + block("coverage_ladder").devices
            ),
            float(full_design_devices()),
        ),
        "blocks_sum_to_pool": (float(total_devices()), float(TOTAL_POOL_DEVICES)),
        "withdrawn_leaves_analysed": (
            float(TOTAL_POOL_DEVICES - WITHDRAWN_DEVICES),
            float(analysed_devices()),
        ),
        "schedule_split_fills_pool": (
            float(ON_21_DAY_SCHEDULE + off_21_day_schedule()),
            float(TOTAL_POOL_DEVICES),
        ),
        "fit_plus_held_out_is_block": (
            float(CALIBRATION_FIT_DEVICES + held_out_devices()),
            float(biological_calibration_devices()),
        ),
    }
