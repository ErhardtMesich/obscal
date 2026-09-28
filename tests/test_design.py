"""Arm roster, chip budget, decoupling ladders, treatment surface and withdrawals."""

from __future__ import annotations

import pytest

from obscal.design import (
    DOSE_LEVELS,
    ROSTER,
    SCHEDULE_LABELS,
    STATE_AXES,
    SurfaceCell,
    TreatmentSurface,
    acellular_ladder,
    analysed_devices,
    arm,
    arm_totals,
    biological_calibration_devices,
    block,
    block_closure_identities,
    budget_partition,
    class_counts,
    classify_device,
    constraint_axis_report,
    coverage_design_values,
    decoupling_coverage,
    delivery_days,
    early_activation_window,
    exclusion_partition_is_exhaustive,
    fit_and_held_out_agree,
    full_design_devices,
    held_out_devices,
    ladder_devices,
    ledger,
    ledger_closure,
    matched_cumulative_dose_report,
    monotonicity_verdict,
    off_21_day_schedule,
    on_21_day_schedule,
    per_fraction_dose,
    primary_contrast_cells,
    seeding_density_ladder,
    timepoint_counts,
    total_devices,
    total_withdrawn,
    trajectory_cells,
    withdrawal_share_pct,
)


def test_roster_reproduces_the_published_arm_totals() -> None:
    assert arm_totals() == {"S1": 12, "S2": 32, "S3": 28, "S4": 32, "S5": 32, "S6": 12}
    assert biological_calibration_devices() == 148
    assert len(trajectory_cells()) == 29
    assert arm("S2").chips == 32
    with pytest.raises(KeyError):
        arm("S9")


def test_timepoint_grid_matches_the_trajectory_table() -> None:
    counts = timepoint_counts()
    assert counts[21] == 56
    assert counts[3] == 24
    assert arm("S1").on_21_day_schedule == 4
    assert len(primary_contrast_cells()) == 6
    assert arm("S6").culture_days == (3, 7, 21)


def test_chip_blocks_partition_the_pool() -> None:
    assert total_devices() == 260
    assert full_design_devices() == 192
    assert analysed_devices() == 215
    assert held_out_devices() == 32
    assert off_21_day_schedule() == 44
    assert on_21_day_schedule() == 56
    assert fit_and_held_out_agree()
    assert block("biological").devices == 148
    with pytest.raises(KeyError):
        block("nothing")


def test_every_budget_identity_closes() -> None:
    for left, right in block_closure_identities().values():
        assert left == right
    partition = budget_partition()
    assert partition["total_devices"] == 260
    assert partition["withdrawal_share_pct"] == pytest.approx(100 * 45 / 260)


def test_withdrawal_ledger_is_exhaustive_and_closes() -> None:
    assert total_withdrawn() == 45
    assert withdrawal_share_pct() == pytest.approx(100 * 45 / 260)
    for left, right in ledger_closure().values():
        assert abs(left - right) < 1e-9
    assert len(ledger()) == 6
    assert sum(entry["count"] for entry in ledger()) == 45


def test_classification_applies_the_rules_in_the_ledger_order() -> None:
    assert (
        classify_device(
            bubble_logged=True,
            coverage_percent=90.0,
            pressure_decay_per_min=0.001,
            electrode_drift_ohm_per_day=1.0,
            half_mass_ratio=0.5,
            hydroxyproline_ug_per_chip=1.0,
        )
        == "bubble_occlusion"
    )
    assert (
        classify_device(
            bubble_logged=False,
            coverage_percent=70.0,
            pressure_decay_per_min=0.001,
            electrode_drift_ohm_per_day=1.0,
            half_mass_ratio=0.5,
            hydroxyproline_ug_per_chip=1.0,
        )
        == "seeding_confluence_outside_band"
    )
    assert (
        classify_device(
            bubble_logged=False,
            coverage_percent=90.0,
            pressure_decay_per_min=0.001,
            electrode_drift_ohm_per_day=1.0,
            half_mass_ratio=0.5,
            hydroxyproline_ug_per_chip=1.0,
        )
        is None
    )
    decisions = ["bubble_occlusion", None, "channel_leak"]
    assert class_counts(decisions)["retained"] == 1
    assert exclusion_partition_is_exhaustive(decisions)
    with pytest.raises(ValueError):
        class_counts(["unledgered"])


def test_ladders_carry_the_printed_level_counts() -> None:
    assert acellular_ladder().level_count == 6
    assert acellular_ladder().devices == 24
    assert seeding_density_ladder().level_count == 5
    assert seeding_density_ladder().devices == 20
    assert ladder_devices() == 44
    assert acellular_ladder().is_balanced
    assert len(acellular_ladder().values()) == 6
    coverage = decoupling_coverage()
    assert coverage["axes_identified_by_acellular"] == ("matrix",)
    assert coverage["axes_identified_by_seeding_density"] == ("coverage",)
    assert coverage["ladder_devices_total"] == 44


def test_early_activation_window_is_a_three_day_window() -> None:
    window = early_activation_window()
    assert window.contains(8)
    assert not window.contains(14)


def test_coverage_design_values_reproduce_the_split_half_routes() -> None:
    design = coverage_design_values(0.55, 0.1900)
    assert design["lambda"] == pytest.approx(0.8934, abs=5e-4)


def _surface() -> TreatmentSurface:
    cells = [
        SurfaceCell(
            dose_cells=dose,
            schedule=schedule,
            activation_fold=3.0 - 0.4 * dose_index,
            matrix_ug_hyp=3.4 - 0.3 * dose_index,
            pro_collagen_ng_per_ml=40.0,
            measured=dose_index == 2,
            chips=4,
        )
        for dose_index, dose in enumerate(DOSE_LEVELS)
        for schedule in SCHEDULE_LABELS
    ]
    untreated = SurfaceCell(
        dose_cells=0.0,
        schedule="none",
        activation_fold=3.1,
        matrix_ug_hyp=3.42,
        pro_collagen_ng_per_ml=41.6,
        measured=True,
        chips=12,
    )
    return TreatmentSurface(cells=tuple(cells), untreated=untreated)


def test_surface_is_complete_and_matched_at_equal_total_dose() -> None:
    surface = _surface()
    assert len(surface.cells) == 16
    assert len(surface.measured_cells) == 4
    assert len(surface.forecast_cells) == 12
    totals = matched_cumulative_dose_report(surface)
    assert set(totals.values()) == set(DOSE_LEVELS)
    assert len(totals) == len(DOSE_LEVELS)
    with pytest.raises(KeyError):
        surface.at(1.0, "single_bolus")


def test_delivery_days_and_per_fraction_dose_agree() -> None:
    assert delivery_days("three_fractions") == (0, 3, 7)
    assert per_fraction_dose("three_fractions", 5.0e4) == pytest.approx(5.0e4 / 3)
    assert len(delivery_days("continuous")) == 14


def test_monotonicity_verdict_is_stated_per_cell() -> None:
    verdict = monotonicity_verdict(_surface())
    assert verdict["both_axes_monotone_everywhere"]
    assert verdict["cells_with_both_axes_monotone"] == verdict["cells_total_axes"]
    assert verdict["dose_optimum_claimed"] is False


def test_constraint_axis_report_names_the_concession() -> None:
    surface = _surface()
    cells = list(surface.cells)
    target = next(
        index
        for index, cell in enumerate(cells)
        if cell.dose_cells == 5.0e4 and cell.schedule == "three_fractions"
    )
    cells[target] = SurfaceCell(5.0e4, "three_fractions", 1.9, 2.26, 36.8, True, 12)
    continuous = next(
        index
        for index, cell in enumerate(cells)
        if cell.dose_cells == 1.0e5 and cell.schedule == "continuous"
    )
    cells[continuous] = SurfaceCell(1.0e5, "continuous", 1.66, 2.01, 36.6, False, 0)
    report = constraint_axis_report(
        TreatmentSurface(cells=tuple(cells), untreated=surface.untreated), 0.89
    )
    assert report["conceded_ug_hyp_against_continuous"] == pytest.approx(0.25)
    assert report["concession_below_floor"]
    assert not report["criterion_adjudicable"]


def test_state_axes_are_named_and_not_pooled() -> None:
    assert STATE_AXES == ("activation", "matrix")
    assert len({spec.arm for spec in ROSTER}) == 6
