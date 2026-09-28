"""Build the harness device pool on disk.

Ref: Sec. Device, cultures and injury model; Table 1.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from obscal.cohort import FixtureSpec, build_cohort_fixture, write_cohort
from obscal.runtime.config import repo_root

LOGGER = logging.getLogger("obscal.build_cohort")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build the harness device pool")
    parser.add_argument("--out", type=Path, default=Path("runs/cohort"))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    root = repo_root(args.root if args.root is not None else Path("configs/run/primary.yaml"))
    spec = FixtureSpec.load(root, args.seed)
    cohort = build_cohort_fixture(spec)
    directory = write_cohort(Path(args.out), cohort)
    summary = cohort.summary()
    LOGGER.info(
        "wrote %d devices to %s (%d retained, %d withdrawn)",
        summary["devices"],
        directory,
        summary["retained"],
        summary["withdrawn"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
