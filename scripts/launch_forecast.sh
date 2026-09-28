#!/usr/bin/env bash
# Fit the damped oscillatory twin on held-in devices, then score the five rungs
# of the forecasting ladder on the identical held-out splits.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${HERE}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0

RUN_CONFIG="${1:-configs/run/primary.yaml}"
OUT_DIR="${2:-runs/primary}"

"${PYTHON_BIN}" -m obscal.cli.forecast \
  --run "${RUN_CONFIG}" \
  --cohort "${OUT_DIR}/cohort" \
  --calibration "${OUT_DIR}/calibration.json" \
  --out "${OUT_DIR}"
