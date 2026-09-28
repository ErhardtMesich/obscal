#!/usr/bin/env bash
# Build the device-level cohort fixture the calibration and allocation entry
# points consume, then run the pooled calibration over it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${HERE}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0

RUN_CONFIG="${1:-configs/run/primary.yaml}"
OUT_DIR="${2:-runs/primary}"

"${PYTHON_BIN}" -m obscal.cli.build_cohort --out "${OUT_DIR}/cohort"
"${PYTHON_BIN}" -m obscal.cli.calibrate --run "${RUN_CONFIG}" --cohort "${OUT_DIR}/cohort" --out "${OUT_DIR}"
"${PYTHON_BIN}" -m obscal.cli.allocate --run "${RUN_CONFIG}" --cohort "${OUT_DIR}/cohort" --out "${OUT_DIR}"
