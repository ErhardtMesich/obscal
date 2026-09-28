"""The harness device pool: shape, partition, determinism and on-disk form."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from obscal.cohort import (
    Cohort,
    FixtureSpec,
    build_cohort_fixture,
    cohort_is_present,
    read_cohort,
    summarise,
    write_cohort,
)
from obscal.cohort.schema import STATE_AXES, cell_day_grid
from obscal.design import TOTAL_POOL_DEVICES, biological_calibration_devices
from obscal.study import StudyInputs, block_census, calibration_blocks


def test_pool_is_complete_and_partitioned(harness_cohort: Cohort) -> None:
    assert len(harness_cohort) == TOTAL_POOL_DEVICES
    blocks = {
        name: len(harness_cohort.by_block(name))
        for name in (
            "biological",
            "acellular_ladder",
            "coverage_ladder",
            "cell_free",
            "held_out_geometry",
            "cross_line",
            "static_counterpart",
        )
    }
    assert blocks["biological"] == biological_calibration_devices()
    assert blocks["cell_free"] == 8
    assert blocks["static_counterpart"] == 36
    assert sum(blocks.values()) == TOTAL_POOL_DEVICES


def test_device_records_carry_both_axes_and_a_terminal_panel(harness_cohort: Cohort) -> None:
    device = harness_cohort.by_block("biological")[0]
    assert device.front_end_state.shape[1] == len(STATE_AXES)
    assert device.series_days.size == device.front_end_state.shape[0]
    assert device.sweeps.real_ohm.shape[1] == device.sweeps.frequencies_hz.size
    assert device.sweeps.imag_ohm.shape == device.sweeps.real_ohm.shape
    assert device.optical.n_samples == device.series_days.size
    assert device.terminal.hydroxyproline_ug_per_chip == pytest.approx(
        0.5 * (device.terminal.hydroxyproline_half_1 + device.terminal.hydroxyproline_half_2)
    )
    assert device.labelled()["batch"].startswith("B")


def test_builder_is_a_pure_function_of_its_specification(project_root: Path) -> None:
    spec = FixtureSpec.load(project_root)
    first = build_cohort_fixture(spec)
    second = build_cohort_fixture(spec)
    assert all(
        np.array_equal(left.front_end_state, right.front_end_state)
        for left, right in zip(first.devices, second.devices, strict=True)
    )
    assert np.array_equal(
        np.array([d.terminal.hydroxyproline_ug_per_chip for d in first.devices]),
        np.array([d.terminal.hydroxyproline_ug_per_chip for d in second.devices]),
    )


def test_round_trip_through_the_on_disk_form(harness_cohort: Cohort, tmp_path: Path) -> None:
    directory = write_cohort(tmp_path / "cohort", harness_cohort)
    assert cohort_is_present(directory)
    reloaded = read_cohort(directory)
    assert len(reloaded) == len(harness_cohort)
    for left, right in zip(harness_cohort.devices, reloaded.devices, strict=True):
        assert left.device_id == right.device_id
        assert left.terminal == right.terminal
        assert left.nuisance == right.nuisance
        assert np.array_equal(left.front_end_state, right.front_end_state)
        assert np.array_equal(left.sweeps.real_ohm, right.sweeps.real_ohm)
    assert summarise(directory)["devices"] == len(harness_cohort)


def test_census_and_calibration_blocks(study_inputs: StudyInputs) -> None:
    census = block_census(study_inputs)
    assert census["pool"] == TOTAL_POOL_DEVICES
    assert census["retained"] + census["withdrawn"] == TOTAL_POOL_DEVICES
    assert census["feature_dimension"] == 5
    with_ladders = calibration_blocks(study_inputs, include_ladders=True)
    without = calibration_blocks(study_inputs, include_ladders=False)
    assert len(with_ladders) > len(without)
    assert all(device.retained for device in with_ladders)


def test_cohort_rejects_duplicate_identifiers(harness_cohort: Cohort) -> None:
    duplicated = harness_cohort.devices[:2]
    with pytest.raises(ValueError):
        Cohort(devices=(duplicated[0], duplicated[0]), provenance="x")


def test_cell_day_grid_is_closed_at_both_ends() -> None:
    grid = cell_day_grid(3.0, 7.0, 0.5)
    assert grid[0] == 3.0 and grid[-1] == 7.0
    assert grid.size == 9
    with pytest.raises(ValueError):
        cell_day_grid(7.0, 3.0, 0.5)
    with pytest.raises(ValueError):
        cell_day_grid(0.0, 1.0, 0.0)
