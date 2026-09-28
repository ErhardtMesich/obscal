"""Fit the damped oscillatory twin and score the forecasting ladder.

Ref: Sec. Forecasting and terminal-assay allocation, Eq. (7); SI Table S2, S17.
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
from obscal.study import StudyInputs, twin_windows
from obscal.twin import (
    assemble_rungs,
    fit_backbone,
    fit_mechanistic,
    leakage_diagnostic,
    score_ladder,
    sustained_advantage,
    to_held_out_series,
)

LOGGER = logging.getLogger("obscal.forecast")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fit the twin and score the ladder")
    parser.add_argument("--run", type=Path, default=Path("configs/run/primary.yaml"))
    parser.add_argument("--cohort", type=Path, default=Path("runs/cohort"))
    parser.add_argument("--out", type=Path, default=Path("runs/primary"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dimension", type=int, default=None)
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
    fitted, held_out = twin_windows(inputs, run)
    dimension = int(args.dimension or inputs.twin.latent_dimension)
    epochs = int(args.epochs or 120)
    learning_rate = float(inputs.twin.training["learning_rate_deployed"])
    twin_backbone, record = fit_backbone(
        fitted,
        latent_dimension=dimension,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=run.seed,
        damping=run.toggles.damping,
        elapsed_days=float(inputs.twin.forecast["issued_from_culture_day"]),
    )
    undamped, undamped_record = fit_backbone(
        fitted,
        latent_dimension=dimension,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=run.seed + 1,
        damping=False,
        elapsed_days=float(inputs.twin.forecast["issued_from_culture_day"]),
    )
    mechanistic = fit_mechanistic(
        fitted, float(inputs.twin.forecast["issued_from_culture_day"]), run.seed
    )
    rungs = assemble_rungs(
        inputs.twin,
        twin_backbone=twin_backbone,
        undamped_backbone=undamped,
        mechanistic=mechanistic,
        elapsed_days=float(inputs.twin.forecast["issued_from_culture_day"]),
    )
    horizons = tuple(int(value) for value in inputs.twin.horizons_days)
    series = [to_held_out_series(window, horizons) for window in held_out if window.targets]
    ladder = score_ladder(rungs, series, horizons)
    ratios = ladder.ratios("matrix", max(horizons))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": run.name,
        "latent_dimension": dimension,
        "epochs": epochs,
        "training": record.as_record(),
        "undamped_training": undamped_record.as_record(),
        "n_held_out_series": len(series),
        "ladder": ladder.records(),
        "ordering_matrix_longest_horizon": ladder.ordering("matrix", max(horizons)),
        "ordering_activation_longest_horizon": ladder.ordering("activation", max(horizons)),
        "leakage_diagnostic": leakage_diagnostic(
            float(ratios.get("obscal_twin", float("nan"))),
            float(inputs.twin.forecast["leakage_diagnostic_ratio_threshold"]),
        ),
        "sustained_advantage_matrix": sustained_advantage(ladder, "obscal_twin", "matrix"),
        "sustained_advantage_activation": sustained_advantage(ladder, "obscal_twin", "activation"),
        "interpretation": (
            "these are the release's own numbers on the harness device pool; no manuscript "
            "table value is claimed"
        ),
    }
    write_json(out_dir / "forecast.json", payload)
    write_report(out_dir / "forecast.txt", _render(payload))
    LOGGER.info("forecast written to %s", out_dir / "forecast.json")
    return 0


def _render(payload: dict[str, object]) -> str:
    ladder = payload["ladder"]
    assert isinstance(ladder, list)
    training = payload["training"]
    assert isinstance(training, dict)
    diagnostic = payload["leakage_diagnostic"]
    assert isinstance(diagnostic, dict)
    return (
        "\n".join(
            [
                f"run: {payload['run']}",
                "",
                render_key_values(
                    {
                        "latent dimension": payload["latent_dimension"],
                        "epochs": payload["epochs"],
                        "held-out series": payload["n_held_out_series"],
                        "parameter count": training["parameter_count"],
                        "best validation loss": training["best_validation_loss"],
                        "leakage diagnostic": diagnostic["action"],
                    }
                ),
                "",
                render_table(
                    ladder,
                    [
                        "rung",
                        "axis",
                        "horizon_days",
                        "rmse",
                        "persistence_rmse",
                        "ratio_to_persistence",
                        "coverage",
                    ],
                    title="forecasting ladder, per axis and horizon",
                ),
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
