"""On-disk form of the harness device pool.

Arrays are stored per device under a namespaced key, and a JSON index carries the
labels, so a reader reconstructs the same records without any pickled object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from obscal.cohort.schema import (
    Cohort,
    DeviceRecord,
    NuisanceRecord,
    OpticalSeries,
    SweepSet,
    TerminalPanel,
)
from obscal.runtime.atomic import read_json, write_json

ARRAY_FILE = "devices.npz"
INDEX_FILE = "index.json"
SWEEP_FIELDS: tuple[str, ...] = ("frequencies_hz", "elapsed_hours", "real_ohm", "imag_ohm")
OPTICAL_FIELDS: tuple[str, ...] = (
    "elapsed_hours",
    "coverage_percent",
    "matrix_texture_x1e3",
    "effluent_pro_collagen_ng_per_ml",
)
TERMINAL_FIELDS: tuple[str, ...] = (
    "hydroxyproline_ug_per_chip",
    "hydroxyproline_half_1",
    "hydroxyproline_half_2",
    "collagen_area_percent",
    "collagen_area_operator_1",
    "collagen_area_operator_2",
    "acta2_fold",
    "col1a1_fold",
    "albumin_ng_per_chip_per_day",
    "urea_ug_per_chip_per_day",
    "pro_collagen_ng_per_ml",
)
NUISANCE_FIELDS: tuple[str, ...] = (
    "temperature_c_mean",
    "medium_change_count",
    "hours_since_electrode_wetting",
    "perfusion_state",
    "geometry",
    "batch",
    "operator",
    "media_lot",
    "passage_stellate",
    "passage_parenchymal",
    "lot_stromal",
)


def _key(device_id: str, field: str) -> str:
    return f"{device_id}::{field}"


def write_cohort(directory: Path, cohort: Cohort) -> Path:
    """Write the pool and its index atomically enough for a rerun to be identical."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    index: list[dict[str, Any]] = []
    for device in cohort.devices:
        arrays[_key(device.device_id, "series_days")] = device.series_days
        arrays[_key(device.device_id, "front_end_state")] = device.front_end_state
        for field in SWEEP_FIELDS:
            arrays[_key(device.device_id, f"sweep::{field}")] = getattr(device.sweeps, field)
        for field in OPTICAL_FIELDS:
            arrays[_key(device.device_id, f"optical::{field}")] = getattr(device.optical, field)
        terminal_record = device.terminal.as_record()
        for field in TERMINAL_FIELDS:
            arrays[_key(device.device_id, f"terminal::{field}")] = np.asarray(
                terminal_record[field], dtype=np.float64
            )
        nuisance_record = {field: getattr(device.nuisance, field) for field in NUISANCE_FIELDS}
        index.append(
            {
                **device.labelled(),
                "series_steps": int(device.series_days.size),
                "n_sweeps": device.sweeps.n_sweeps,
                "front_end_state_width": int(device.front_end_state.shape[1]),
                "nuisance": nuisance_record,
            }
        )
    np.savez_compressed(directory / ARRAY_FILE, **arrays)
    write_json(
        directory / INDEX_FILE,
        {"provenance": cohort.provenance, "devices": index, "array_file": ARRAY_FILE},
    )
    return directory


def read_cohort(directory: Path) -> Cohort:
    """Rebuild the pool from the array file and the index."""
    payload = cast("dict[str, Any]", read_json(directory / INDEX_FILE))
    arrays = np.load(directory / ARRAY_FILE, allow_pickle=False)
    devices: list[DeviceRecord] = []
    for entry in cast("list[dict[str, Any]]", payload["devices"]):
        device_id = str(entry["device_id"])
        nuisance = NuisanceRecord(
            temperature_c_mean=float(entry["nuisance"]["temperature_c_mean"]),
            medium_change_count=int(entry["nuisance"]["medium_change_count"]),
            hours_since_electrode_wetting=float(entry["nuisance"]["hours_since_electrode_wetting"]),
            perfusion_state=str(entry["nuisance"]["perfusion_state"]),
            geometry=str(entry["nuisance"]["geometry"]),
            batch=str(entry["nuisance"]["batch"]),
            operator=str(entry["nuisance"]["operator"]),
            media_lot=int(entry["nuisance"]["media_lot"]),
            passage_stellate=int(entry["nuisance"]["passage_stellate"]),
            passage_parenchymal=int(entry["nuisance"]["passage_parenchymal"]),
            lot_stromal=int(entry["nuisance"]["lot_stromal"]),
        )
        sweeps = SweepSet(
            frequencies_hz=arrays[_key(device_id, "sweep::frequencies_hz")],
            elapsed_hours=arrays[_key(device_id, "sweep::elapsed_hours")],
            real_ohm=arrays[_key(device_id, "sweep::real_ohm")],
            imag_ohm=arrays[_key(device_id, "sweep::imag_ohm")],
        )
        optical = OpticalSeries(
            elapsed_hours=arrays[_key(device_id, "optical::elapsed_hours")],
            coverage_percent=arrays[_key(device_id, "optical::coverage_percent")],
            matrix_texture_x1e3=arrays[_key(device_id, "optical::matrix_texture_x1e3")],
            effluent_pro_collagen_ng_per_ml=arrays[
                _key(device_id, "optical::effluent_pro_collagen_ng_per_ml")
            ],
        )
        terminal = TerminalPanel(
            hydroxyproline_ug_per_chip=float(
                arrays[_key(device_id, "terminal::hydroxyproline_ug_per_chip")]
            ),
            hydroxyproline_half_1=float(arrays[_key(device_id, "terminal::hydroxyproline_half_1")]),
            hydroxyproline_half_2=float(arrays[_key(device_id, "terminal::hydroxyproline_half_2")]),
            collagen_area_percent=float(arrays[_key(device_id, "terminal::collagen_area_percent")]),
            collagen_area_operator_1=float(
                arrays[_key(device_id, "terminal::collagen_area_operator_1")]
            ),
            collagen_area_operator_2=float(
                arrays[_key(device_id, "terminal::collagen_area_operator_2")]
            ),
            acta2_fold=float(arrays[_key(device_id, "terminal::acta2_fold")]),
            col1a1_fold=float(arrays[_key(device_id, "terminal::col1a1_fold")]),
            albumin_ng_per_chip_per_day=float(
                arrays[_key(device_id, "terminal::albumin_ng_per_chip_per_day")]
            ),
            urea_ug_per_chip_per_day=float(
                arrays[_key(device_id, "terminal::urea_ug_per_chip_per_day")]
            ),
            pro_collagen_ng_per_ml=float(
                arrays[_key(device_id, "terminal::pro_collagen_ng_per_ml")]
            ),
        )
        devices.append(
            DeviceRecord(
                device_id=device_id,
                arm=str(entry["arm"]),
                block=str(entry["block"]),
                culture_day=int(entry["culture_day"]),
                treatment_day=(
                    None if entry["treatment_day"] is None else int(entry["treatment_day"])
                ),
                series_days=arrays[_key(device_id, "series_days")],
                front_end_state=arrays[_key(device_id, "front_end_state")],
                sweeps=sweeps,
                optical=optical,
                terminal=terminal,
                nuisance=nuisance,
                withdrawn_class=cast("str | None", entry["withdrawn_class"]),
                perfusion=bool(entry["perfusion"]),
                cell_source=str(entry["cell_source"]),
                hepatocyte_line=str(entry["hepatocyte_line"]),
            )
        )
    return Cohort(devices=tuple(devices), provenance=str(payload["provenance"]))


def cohort_is_present(directory: Path) -> bool:
    return (directory / ARRAY_FILE).is_file() and (directory / INDEX_FILE).is_file()


def summarise(directory: Path) -> dict[str, Any]:
    payload = cast("dict[str, Any]", read_json(directory / INDEX_FILE))
    devices = cast("list[dict[str, Any]]", payload["devices"])
    blocks: dict[str, int] = {}
    for entry in devices:
        blocks[str(entry["block"])] = blocks.get(str(entry["block"]), 0) + 1
    return {
        "provenance": payload["provenance"],
        "devices": len(devices),
        "blocks": blocks,
    }
