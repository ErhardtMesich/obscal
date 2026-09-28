"""Run the pooled calibration and write its report.

Ref: Sec. Calibration design and identifiability, Eq. (3)-(4); Algorithm 1.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from obscal.cohort import build_cohort_fixture, cohort_is_present, read_cohort
from obscal.cohort.fixture import FixtureSpec
from obscal.evaluation import render_key_values, render_table, write_report
from obscal.runtime.atomic import write_json
from obscal.runtime.config import load_run_spec, repo_root
from obscal.study import (
    StudyInputs,
    block_census,
    delivered_attenuation,
    run_calibration_study,
    spread_summary,
)

LOGGER = logging.getLogger("obscal.calibrate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run the pooled calibration")
    parser.add_argument("--run", type=Path, default=Path("configs/run/primary.yaml"))
    parser.add_argument("--cohort", type=Path, default=Path("runs/cohort"))
    parser.add_argument("--out", type=Path, default=Path("runs/primary"))
    parser.add_argument("--bootstrap", type=int, default=None)
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
    outcome, devices = run_calibration_study(inputs, run, bootstrap_replicates=args.bootstrap)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": run.name,
        "summary": outcome.summary(),
        "census": block_census(inputs),
        "block_spreads": spread_summary(inputs),
        "delivered_attenuation": delivered_attenuation(inputs),
        "n_devices_in_fit": len(devices),
    }
    write_json(out_dir / "calibration.json", payload)
    write_report(out_dir / "calibration.txt", _render(payload))
    LOGGER.info("calibration written to %s", out_dir / "calibration.json")
    return 0


def _render(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    slope_rows = summary["slopes"]
    assert isinstance(slope_rows, list)
    attenuation = payload["delivered_attenuation"]
    assert isinstance(attenuation, dict)
    lambdas = summary["lambda"]
    assert isinstance(lambdas, dict)
    parts = [
        f"run: {payload['run']}",
        "",
        render_key_values(
            {
                "devices in fit": summary["n_units"],
                "feature dimension": summary["feature_dimension"],
                "design columns": summary["design_columns"],
                "covariance inflation": summary["covariance_inflation"],
                "attenuation status": summary["attenuation_status"],
            }
        ),
        "",
        render_table(
            slope_rows,
            ["feature", "axis", "naive_slope", "corrected_slope", "lambda", "status"],
            title="pooled calibration, naive and moment-corrected",
        ),
        "",
        render_table(
            [{"key": key, "value": value} for key, value in {**lambdas, **attenuation}.items()],
            ["key", "value"],
            title="delivered attenuation factors",
        ),
    ]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
