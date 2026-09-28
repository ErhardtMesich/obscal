"""Run the budgeted allocation and write its curve.

Ref: Sec. Forecasting and terminal-assay allocation, Eq. (6); Algorithm 2.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from obscal.allocation import curve_records
from obscal.cohort import build_cohort_fixture, cohort_is_present, read_cohort
from obscal.cohort.fixture import FixtureSpec
from obscal.evaluation import render_key_values, render_table, write_report
from obscal.runtime.atomic import write_json
from obscal.runtime.config import load_run_spec, repo_root
from obscal.study import StudyInputs, run_allocation_study, run_calibration_study

LOGGER = logging.getLogger("obscal.allocate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run the terminal-assay allocation")
    parser.add_argument("--run", type=Path, default=Path("configs/run/primary.yaml"))
    parser.add_argument("--cohort", type=Path, default=Path("runs/cohort"))
    parser.add_argument("--out", type=Path, default=Path("runs/primary"))
    parser.add_argument("--harness-devices", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    root = repo_root(args.run)
    run = load_run_spec(root / args.run if not args.run.is_absolute() else args.run)
    if cohort_is_present(Path(args.cohort)):
        cohort = read_cohort(Path(args.cohort))
    else:
        cohort = build_cohort_fixture(FixtureSpec.load(root, run.seed))
    inputs = StudyInputs.load(root, cohort)
    calibration, _ = run_calibration_study(inputs, run, bootstrap_replicates=0)
    outcome = run_allocation_study(inputs, run, calibration, harness_devices=args.harness_devices)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": run.name,
        "budget_block_devices": int(outcome.curve[-1].devices),
        "curve": curve_records(outcome.curve),
        **outcome.summary(),
    }
    write_json(out_dir / "allocation.json", payload)
    write_report(out_dir / "allocation.txt", _render(payload))
    LOGGER.info("allocation written to %s", out_dir / "allocation.json")
    return 0


def _render(payload: dict[str, object]) -> str:
    curve = payload["curve"]
    assert isinstance(curve, list)
    columns = ["fraction", "devices", "random_p10", "random_p90"] + sorted(
        key for key in curve[0] if str(key).startswith("rse::")
    )
    operating = payload["operating_point"]
    assert isinstance(operating, dict)
    checks = payload["structural_checks"]
    assert isinstance(checks, dict)
    return "\n".join(
        [
            f"run: {payload['run']}",
            "",
            render_table(curve, columns, title="budget against corrected slope precision"),
            "",
            render_key_values(
                {
                    "target corrected RSE (%)": operating["target_corrected_rse_pct"],
                    "converged at full budget": checks["convergence"]["converged"],
                    "non-crossing above oracle": checks["non_crossing"]["non_crossing"],
                }
            ),
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
