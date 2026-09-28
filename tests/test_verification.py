"""The verification layer itself: primitives, the manuscript family and the artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from obscal.runtime.config import RELEASE_SLUG
from obscal.verification import (
    FAIL,
    NOT_RUN,
    PASS,
    CheckResult,
    all_checks,
    build_claim_map,
    build_manifest,
    claimed_values as cv,
    code_status,
    compare,
    failures,
    inventory,
    manifest_digest,
    manuscript_discrepancies,
    not_run,
    overall_status,
    summarise,
    table_not_deposited_checks,
    truthy,
    verify_manifest,
    within,
    write_integrity_manifest,
)
from obscal.verification.claims import CLAIMS, DEVIATIONS
from obscal.verification.manuscript import (
    allocation_and_forecast_arithmetic,
    band_and_agreement_arithmetic,
    calibration_arithmetic,
    capacity_and_cost_arithmetic,
    design_and_transfer_arithmetic,
    dose_surface_arithmetic,
    internal_inconsistencies,
    power_and_floor_arithmetic,
    replication_arithmetic,
    reproducibility_arithmetic,
    roster_and_budget,
    slopes_and_correction,
    split_chip_and_spreads,
    treatment_and_external_arithmetic,
)


def test_check_primitives_hold_their_contract() -> None:
    assert compare("a", 1, 1).status == PASS
    assert compare("a", 1, 2).status == FAIL
    assert within("a", 1.0, 1.05, 0.1).status == PASS
    assert within("a", 1.0, 1.2, 0.1).status == FAIL
    assert truthy("a", True).status == PASS
    assert not_run("a", "reason").status == NOT_RUN
    with pytest.raises(ValueError):
        CheckResult(name="a", category="unknown", status=PASS)
    with pytest.raises(ValueError):
        CheckResult(name="a", category="execution", status="MAYBE")


def test_summaries_and_overall_status() -> None:
    results = [truthy("a", True), truthy("b", False), not_run("c", "reason")]
    counts = summarise(results)
    assert counts[PASS] == 1
    assert counts[FAIL] == 1
    assert counts[NOT_RUN] == 1
    assert counts["total"] == 3
    assert overall_status(results) == "PARTIALLY_VERIFIED"
    assert code_status(results) == "PARTIALLY_VERIFIED"
    assert len(failures(results)) == 1
    assert overall_status([truthy("a", True)]) == "VERIFIED"


def test_manuscript_family_is_almost_entirely_closed() -> None:
    results = all_checks()
    counts = summarise(results)
    assert counts[PASS] > 240
    discrepancies = manuscript_discrepancies(results)
    assert counts[FAIL] == len(discrepancies)
    assert all(entry["category"] == "manuscript" for entry in discrepancies)
    assert len(table_not_deposited_checks()) == 2


@pytest.mark.parametrize(
    "group",
    [
        roster_and_budget,
        calibration_arithmetic,
        slopes_and_correction,
        band_and_agreement_arithmetic,
        split_chip_and_spreads,
        power_and_floor_arithmetic,
        replication_arithmetic,
        allocation_and_forecast_arithmetic,
        design_and_transfer_arithmetic,
        capacity_and_cost_arithmetic,
        reproducibility_arithmetic,
        treatment_and_external_arithmetic,
        dose_surface_arithmetic,
        internal_inconsistencies,
    ],
)
def test_every_manuscript_group_only_fails_on_a_documented_row(group: object) -> None:
    produced = group()  # type: ignore[operator]
    assert produced
    for result in produced:
        if result.status == FAIL:
            assert (
                "discrepan" in result.detail.lower()
                or "tense" in result.detail.lower()
                or result.name
                in {
                    "acellular ladder column is reproducible from one design coefficient",
                    "activation attenuation is a value at the spread achieved",
                }
            )


def test_the_documented_discrepancies_are_exactly_the_two_named_rows() -> None:
    failing = {entry["check"] for entry in manuscript_discrepancies(all_checks())}
    assert failing == {
        "acellular ladder column is reproducible from one design coefficient",
        "activation attenuation is a value at the spread achieved",
    }


def test_the_activation_discrepancy_recomputes_from_the_printed_tables() -> None:
    """The design ladder at the delivered spread does not give the printed factor."""
    delivered = cv.BLOCK_SPREADS_PRINTED["acta2"]["cv_pct"] / 100.0
    row = cv.DESIGN_ATTENUATION_LADDER["activation_split"]
    point = cv.DESIGN_POINT_SPREAD
    assay = point * ((1.0 / row[point]) - 1.0) ** 0.5
    at_achieved = delivered**2 / (delivered**2 + assay**2)
    assert round(at_achieved, 2) == 0.82
    assert round(at_achieved, 2) != float(cv.BLOCK_SPREADS_PRINTED["acta2"]["lambda"])


def test_the_acellular_column_needs_a_different_coefficient_per_row() -> None:
    column = cv.DESIGN_ATTENUATION_LADDER["acellular_ladder"]
    coefficients = {
        spread: (spread * ((1.0 / factor) - 1.0) ** 0.5 if factor < 1.0 else 0.0)
        for spread, factor in column.items()
    }
    assert coefficients[0.40] > 0.0
    assert coefficients[0.55] == 0.0


def test_claim_map_resolves_against_the_shipped_check_names(project_root: Path) -> None:
    payload = json.loads((project_root / "verification_report.json").read_text(encoding="utf-8"))
    results = [
        CheckResult(
            name=entry["check"],
            category=entry["category"],
            status=entry["status"],
            expected=entry["expected"],
            observed=entry["observed"],
            detail=entry["detail"],
        )
        for entry in payload["checks"]
    ]
    claim_map = build_claim_map(results)
    assert claim_map["n_claims"] == len(CLAIMS)
    assert claim_map["n_deviations"] == len(DEVIATIONS)
    assert not claim_map["unmatched_check_names"]
    assert all(entry["code_anchors"] for entry in claim_map["claims"])
    assert all(entry["paper_location"] for entry in claim_map["deviations"])


def test_manifest_round_trips_inside_a_scratch_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("two\n", encoding="utf-8")
    target = write_integrity_manifest(tmp_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["n_files"] == 2
    assert payload["manifest_digest"] == manifest_digest(payload["files"])
    assert verify_manifest(tmp_path)["intact"]
    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    verification = verify_manifest(tmp_path)
    assert not verification["intact"]
    assert verification["stale"] == ["a.txt"]


def test_manifest_excludes_itself_and_skips_caches(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_bytes(b"\x00")
    (tmp_path / "integrity_manifest.json").write_text("{}", encoding="utf-8")
    names = {
        path.name for path in inventory(tmp_path, exclude=tmp_path / "integrity_manifest.json")
    }
    assert names == {"keep.txt"}
    payload = build_manifest(tmp_path)
    # The recorded name is the release's own slug, not the checkout directory,
    # so the shipped bytes do not depend on what the clone is called.
    assert payload["root"] == RELEASE_SLUG
    assert payload["excluded"]


def test_shipped_report_and_claim_map_are_complete(project_root: Path) -> None:
    report = json.loads((project_root / "verification_report.json").read_text(encoding="utf-8"))
    claim_map = json.loads((project_root / "claim_to_code.json").read_text(encoding="utf-8"))
    assert report["summary"]["total"] == len(report["checks"])
    assert report["summary"]["total"] > 340
    assert report["summary"]["overall_status"] == "PARTIALLY_VERIFIED"
    assert report["summary"]["manuscript_discrepancies"] == 2
    assert report["integrity_manifest"]["written_last"]
    assert claim_map["n_claims"] == len(CLAIMS)
    assert claim_map["deviations"]
    assert "harness" in json.dumps(claim_map)
