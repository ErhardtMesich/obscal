"""The verification driver: two check families, the gates, and the shipped artefacts."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obscal.cohort import build_cohort_fixture
from obscal.cohort.fixture import FixtureSpec
from obscal.evaluation import render_key_values, render_table, write_report
from obscal.runtime.atomic import write_json
from obscal.runtime.config import RELEASE_SLUG, load_run_spec, repo_root
from obscal.verification import claimed_values as cv
from obscal.verification.checks import (
    CheckResult,
    code_status,
    failures,
    overall_status,
    summarise,
)
from obscal.verification.claims import build_claim_map
from obscal.verification.execution import build_context, run_execution_family
from obscal.verification.hygiene import (
    config_hygiene_checks,
    dataset_link_checks,
    phrase_checks,
    privacy_checks,
    structure_checks,
)
from obscal.verification.manifest import MANIFEST_NAME, verify_manifest, write_integrity_manifest
from obscal.verification.manuscript import all_checks, table_not_deposited_checks

LOGGER = logging.getLogger("obscal.verify")
CLAIM_FILE = "claim_to_code.json"
REPORT_FILE = "verification_report.json"
SUMMARY_FILE = "verification_summary.txt"

GATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ruff", (sys.executable, "-m", "ruff", "check", ".")),
    ("black", (sys.executable, "-m", "black", "--check", ".")),
    ("isort", (sys.executable, "-m", "isort", "--check-only", ".")),
    ("mypy", (sys.executable, "-m", "mypy", "--strict", "src/obscal")),
    ("pytest", (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider")),
)

GATE_ENVIRONMENT = {
    "COLUMNS": "120",
    "LINES": "40",
    "NO_COLOR": "1",
    "PY_COLORS": "0",
    "PYTHONHASHSEED": "0",
}


@dataclass(frozen=True)
class DriverOptions:
    root: Path
    run: Path
    probe_live: bool
    skip_gates: bool
    seed: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run the release's verification driver")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run", type=Path, default=Path("configs/run/primary.yaml"))
    parser.add_argument("--probe-live", action="store_true")
    parser.add_argument("--skip-gates", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def run_gates(root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    environment = {**os.environ, **GATE_ENVIRONMENT}
    for label, command in GATES:
        try:
            completed = subprocess.run(
                list(command),
                cwd=root,
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
            code = completed.returncode
        except OSError as error:
            results.append(
                CheckResult(
                    name=f"gate {label}",
                    category="gate",
                    status="BLOCKED",
                    expected="the tool to run",
                    observed=type(error).__name__,
                    detail="the tool is not available in this environment",
                    evidence={"label": label},
                )
            )
            continue
        results.append(
            CheckResult(
                name=f"gate {label}",
                category="gate",
                status="PASS" if code == 0 else "FAIL",
                expected=0,
                observed=code,
                detail="the tool's exit code decides the gate",
                evidence={"label": label},
            )
        )
    return results


def printed_result_strings() -> set[str]:
    """The manuscript's measured outcomes, as they would appear in a configuration.

    Only the article's *measured* tables are read, and only their non-integer
    entries: a whole number in a table is as likely to be a count the configs
    legitimately share as it is an outcome, and a design-ladder entry is a
    method constant by the article's own definition. What remains is the set of
    decimal results a config must not be pinned to.
    """
    values: set[str] = set()

    def add(value: float | int | None) -> None:
        if value is None:
            return
        number = float(value)
        if number == int(number):
            return
        values.add(f"{number}")

    for row in cv.SLOPE_TABLE:
        add(row["corrected"])
        add(row["naive"])
    for floors in cv.DETECTABILITY_FLOORS.values():
        for value in floors.values():
            add(value)
    for row in cv.AGREEMENT_ROWS:
        add(row["icc"])
        add(row["rmse_pct"])
    for key, value in cv.ALLOCATION_PRINTED.items():
        if key in {"fractions", "devices"}:
            continue
        if isinstance(value, (tuple, list)):
            for entry in value:
                add(entry)
        else:
            add(value)
    for row in cv.FORECAST_LADDER_PRINTED:
        add(row["ratio"])
        add(row["matrix_7day"])
        add(row["coverage"])
    for block in cv.BLOCK_SPREADS_PRINTED.values():
        add(block["spread"])
        add(block["cv_pct"])
    for row in cv.REPRODUCIBILITY_PRINTED:
        add(row["between_runs"])
        add(row["between_operators"])
    return {value for value in values if value not in {"1.0", "0.0", "0.5", "2.0"}}


def gather(options: DriverOptions) -> tuple[list[CheckResult], dict[str, Any], dict[str, Any]]:
    root = options.root
    resolved_run = root / options.run if not options.run.is_absolute() else options.run
    run = load_run_spec(resolved_run)
    cohort = build_cohort_fixture(FixtureSpec.load(root, options.seed or run.seed))
    context = build_context(root, run, cohort)

    results: list[CheckResult] = []
    results.extend(all_checks())
    results.extend(table_not_deposited_checks())
    results.extend(run_execution_family(context))
    results.extend(phrase_checks(root))
    results.extend(privacy_checks(root))
    results.extend(structure_checks(root))
    results.extend(config_hygiene_checks(root, printed_result_strings()))
    results.extend(dataset_link_checks(root, probe_live=options.probe_live))
    if not options.skip_gates:
        results.extend(run_gates(root))
    else:
        results.append(
            CheckResult(
                name="gate family",
                category="gate",
                status="NOT_RUN",
                expected="the tool gates to run",
                observed=None,
                detail="the gates were skipped for this invocation and are not shipped evidence",
                evidence={},
            )
        )

    claim_map = build_claim_map(results)
    claim_results = [
        CheckResult(
            name=f"claim {entry['id']} maps to executed code",
            category="claim",
            status=(
                "PASS"
                if not entry["check_statuses"] or entry["verification"] != "NOT_RUN"
                else "NOT_RUN"
            ),
            expected="PASS",
            observed=entry["verification"],
            detail=entry["statement"][:160],
            evidence={"anchors": entry["code_anchors"]},
        )
        for entry in claim_map["claims"]
    ]
    results.extend(claim_results)
    claim_map = build_claim_map(results)

    context_payload = {
        "cohort": cohort.summary(),
        "run": {"name": run.name, "kind": run.kind, "title": run.title, "seed": run.seed},
        "toggles_removed": list(run.toggles.removed()),
    }
    return results, claim_map, context_payload


def render_summary(
    results: Sequence[CheckResult],
    claim_map: dict[str, Any],
    context_payload: dict[str, Any],
    root_label: str,
) -> str:
    counts = summarise(results)
    discrepancies = [result for result in failures(results) if result.category == "manuscript"]
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_category.setdefault(result.category, {})
        bucket[result.status] = bucket.get(result.status, 0) + 1
    category_rows = [
        {
            "family": name,
            **{status: values.get(status, 0) for status in ("PASS", "FAIL", "NOT_RUN", "BLOCKED")},
        }
        for name, values in sorted(by_category.items())
    ]
    failing_rows = [
        {
            "check": result.name,
            "family": result.category,
            "expected": result.expected,
            "observed": result.observed,
        }
        for result in failures(results)
    ]
    return "\n".join(
        [
            f"release root           : {root_label}",
            f"run                    : {context_payload['run']['name']} ({context_payload['run']['kind']})",
            "",
            "checks by family",
            render_table(category_rows, ["family", "PASS", "FAIL", "NOT_RUN", "BLOCKED"]),
            render_key_values(
                {
                    "overall status": overall_status(results),
                    "code status (manuscript family excluded)": code_status(results),
                    "checks run": counts["total"],
                    "manuscript discrepancies": len(discrepancies),
                    "claims mapped": claim_map["n_claims"],
                    "deviations registered": claim_map["n_deviations"],
                }
            ),
            (
                render_table(
                    failing_rows,
                    ["check", "family", "expected", "observed"],
                    title="failures",
                )
                if failing_rows
                else "no failures"
            ),
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    root = repo_root(args.root if args.root is not None else Path("configs/run/primary.yaml"))
    options = DriverOptions(
        root=root,
        run=args.run,
        probe_live=args.probe_live,
        skip_gates=args.skip_gates,
        seed=args.seed,
    )
    results, claim_map, context_payload = gather(options)

    counts = summarise(results)
    report = {
        "release_root": RELEASE_SLUG,
        "run": context_payload["run"],
        "cohort": context_payload["cohort"],
        "toggles_removed": context_payload["toggles_removed"],
        "summary": {
            **counts,
            "overall_status": overall_status(results),
            "code_status": code_status(results),
            "manuscript_discrepancies": sum(
                1 for result in failures(results) if result.category == "manuscript"
            ),
        },
        "interpretation": (
            "A failing check in the manuscript family is a discrepancy inside the article's own "
            "printed tables, not a defect in this release. Every cohort-level quantity below is "
            "the release's own on its harness device pool, because the article deposits no "
            "per-device rows and no executable implementation."
        ),
        "checks": [result.as_record() for result in results],
        "manuscript_discrepancies": [
            result.as_record() for result in failures(results) if result.category == "manuscript"
        ],
        "integrity_manifest": {"name": MANIFEST_NAME, "written_last": True},
    }
    claim_payload = {
        "release_root": RELEASE_SLUG,
        "run": context_payload["run"],
        "n_claims": claim_map["n_claims"],
        "n_deviations": claim_map["n_deviations"],
        "claims": claim_map["claims"],
        "deviations": claim_map["deviations"],
        "unmatched_check_names": claim_map["unmatched_check_names"],
    }
    summary_text = render_summary(results, claim_map, context_payload, RELEASE_SLUG)

    write_json(root / CLAIM_FILE, claim_payload)
    write_json(root / REPORT_FILE, report)
    write_report(root / SUMMARY_FILE, summary_text)
    manifest_path = write_integrity_manifest(root)

    verification = verify_manifest(root)
    LOGGER.info(
        "wrote %s, %s, %s and %s (manifest intact: %s)",
        CLAIM_FILE,
        REPORT_FILE,
        SUMMARY_FILE,
        manifest_path.name,
        verification["intact"],
    )
    if not verification["intact"]:
        LOGGER.error("the manifest does not describe the live tree: %s", verification)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
