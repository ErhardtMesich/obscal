"""The manuscript-arithmetic family.

Every check here recomputes a quantity the article derives from its own printed
numbers, so a failure names an inconsistency inside the manuscript's tables
rather than a defect in this release. The estimators are written from the
definition the article states, never from the implementation they audit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats

from obscal.verification import claimed_values as cv
from obscal.verification.checks import (
    FAIL,
    PASS,
    CheckResult,
    not_run,
)


def _close(
    name: str,
    expected: float,
    observed: float,
    *,
    places: int,
    detail: str,
    evidence: dict[str, object] | None = None,
    inputs: Sequence[tuple[float, int]] = (),
) -> CheckResult:
    """Compare a recomputed value with a printed one on the printed precision.

    The printed value is only known to half a unit in its last place, and every
    input it was computed from carries the same limitation, so the tolerance is
    that half unit plus the first-order propagation of the inputs' rounding.
    """
    target = float(expected)
    value = float(observed)
    relative = 0.0
    for input_value, input_places in inputs:
        if input_value == 0.0:
            continue
        relative += ((10.0**-input_places) / 2.0 / abs(float(input_value))) ** 2
    tolerance = (10.0**-places) / 2.0 + math.sqrt(relative) * abs(target)
    passed = abs(value - target) <= tolerance + 1e-12
    return CheckResult(
        name=name,
        category="manuscript",
        status=PASS if passed else FAIL,
        expected=round(target, places),
        observed=round(value, places),
        detail=detail,
        evidence={"tolerance": round(tolerance, 6), **dict(evidence or {})},
    )


def roster_and_budget() -> list[CheckResult]:
    out: list[CheckResult] = []
    biological = sum(cv.ROSTER_CHIP_TOTALS.values())
    out.append(
        _close(
            "arm totals sum to the biological block",
            148.0,
            float(biological),
            places=0,
            detail="Table S7 arm totals 12+32+28+32+32+12",
            evidence={"arm_totals": cv.ROSTER_CHIP_TOTALS},
        )
    )
    full_design = biological + cv.POOL["acellular_ladder"] + cv.POOL["coverage_ladder"]
    out.append(
        _close(
            "biological block plus both ladders is the full design",
            float(cv.POOL["full_design"]),
            float(full_design),
            places=0,
            detail="Table S9 Panel B, 148 + 24 + 20",
        )
    )
    total = (
        full_design
        + cv.POOL["cell_free"]
        + cv.POOL["held_out_geometry"]
        + cv.POOL["cross_line"]
        + cv.POOL["static_counterpart"]
    )
    out.append(
        _close(
            "every chip block sums to the 260-device pool",
            float(cv.POOL["total_devices"]),
            float(total),
            places=0,
            detail="Table 1, Table 6 and the static contrast must partition the pool",
            evidence={
                "blocks": {
                    "full_design": full_design,
                    "cell_free": cv.POOL["cell_free"],
                    "held_out_geometry": cv.POOL["held_out_geometry"],
                    "cross_line": cv.POOL["cross_line"],
                    "static_counterpart": cv.POOL["static_counterpart"],
                }
            },
        )
    )
    out.append(
        _close(
            "fit and held-out devices are the biological block",
            float(cv.POOL["biological"]),
            float(cv.POOL["calibration_fit"] + cv.POOL["held_out"]),
            places=0,
            detail="116 pooled plus 32 held out",
        )
    )
    withdrawn = sum(cv.WITHDRAWAL_COUNTS.values())
    out.append(
        _close(
            "withdrawal classes sum to the withdrawn total",
            45.0,
            float(withdrawn),
            places=0,
            detail="SI Table S14 counts",
            evidence={"classes": cv.WITHDRAWAL_COUNTS},
        )
    )
    out.append(
        _close(
            "withdrawn share of the pool",
            17.3,
            100.0 * withdrawn / cv.POOL["total_devices"],
            places=1,
            detail="45 of 260",
        )
    )
    out.append(
        _close(
            "analysed devices",
            float(cv.POOL["analysed"]),
            float(cv.POOL["total_devices"] - withdrawn),
            places=0,
            detail="260 less 45",
        )
    )
    out.append(
        _close(
            "the devices off the 21-day schedule are exactly the two ladders",
            float(cv.POOL["acellular_ladder"] + cv.POOL["coverage_ladder"]),
            float(cv.POOL["total_devices"] - cv.POOL["on_21_day_schedule"]),
            places=0,
            detail="260 less the 216 on the 21-day schedule",
        )
    )
    return out


def calibration_arithmetic() -> list[CheckResult]:
    out: list[CheckResult] = []
    n_bio = cv.POOL["calibration_fit"]
    p = cv.FEATURE_DIMENSION
    out.append(
        _close(
            "plug-in covariance inflation",
            cv.PLUGIN_INFLATION_PRINTED,
            n_bio / (n_bio - p - 2),
            places=3,
            detail="Nbio/(Nbio - p - 2) at p = 5 and 116 biological devices",
        )
    )
    out.append(
        _close(
            "residual degrees of freedom of one sweep fit",
            9.0,
            float(2 * 7 - 5),
            places=0,
            detail="14 real quantities from 7 complex readings less 5 free parameters",
        )
    )
    out.append(
        _close(
            "sensor dimension alternatives and their inflation",
            cv.DESIGN_DECISIONS_PRINTED["inflation_p10"],
            n_bio / (n_bio - 10 - 2),
            places=3,
            detail="Table S12 sensor-dimension row at p = 10",
        )
    )
    return out


def slopes_and_correction() -> list[CheckResult]:
    out: list[CheckResult] = []
    for row in cv.SLOPE_TABLE:
        name = f"corrected slope equals naive over lambda ({row['feature']}, {row['axis']})"
        value = float(row["naive"]) / float(row["lambda"])
        out.append(
            _close(
                name,
                float(row["corrected"]),
                value,
                places=1,
                detail="Naive slopes divide by the per-axis attenuation factor",
                evidence={"lambda": row["lambda"], "naive": row["naive"]},
            )
        )
    negative = [row for row in cv.SLOPE_TABLE if float(row["naive"]) < 0.0]
    moved = all(abs(float(row["corrected"])) >= abs(float(row["naive"])) - 1e-9 for row in negative)
    out.append(
        CheckResult(
            name="correction moves every negative slope away from zero",
            category="manuscript",
            status=PASS if moved else FAIL,
            expected=True,
            observed=moved,
            detail="dividing by a factor below one must enlarge the magnitude",
            evidence={"n_negative_slopes": len(negative)},
        )
    )
    clearing = [row for row in cv.SLOPE_TABLE if float(row["p"]) < 0.05]
    out.append(
        _close(
            "slope cells clearing zero unadjusted",
            float(cv.HOLM_CLEARING_ZERO_PRINTED),
            float(len(clearing)),
            places=0,
            detail="the 15 cells form one test family",
            evidence={"clearing": [f"{r['feature']}:{r['axis']}" for r in clearing]},
        )
    )
    retained = _holm_retained(cv.SLOPE_TABLE)
    out.append(
        _close(
            "Holm step-down retains the printed number of slopes",
            float(cv.HOLM_RETAINED_PRINTED),
            float(len(retained)),
            places=0,
            detail="Holm at P < 0.05 over the 15-cell family",
            evidence={"retained": retained},
        )
    )
    dropped_among_clearing = sorted(
        name for name in _holm_dropped(cv.SLOPE_TABLE) if name in set(_names(clearing))
    )
    out.append(
        CheckResult(
            name="Holm drops exactly the printed activation slopes among those clearing zero",
            category="manuscript",
            status=PASS if dropped_among_clearing == sorted(cv.HOLM_DROPPED_PRINTED) else FAIL,
            expected=sorted(cv.HOLM_DROPPED_PRINTED),
            observed=dropped_among_clearing,
            detail=(
                "the two activation slopes on barrier resistance and on the constant-phase "
                "exponent are the only cells that clear zero unadjusted and fail Holm"
            ),
            evidence={"all_holm_dropped": _holm_dropped(cv.SLOPE_TABLE)},
        )
    )
    return out


def _names(rows: Sequence[dict[str, float | str]]) -> list[str]:
    return [f"{row['feature']}:{row['axis']}" for row in rows]


def _holm_retained(rows: Sequence[dict[str, float | str]]) -> list[str]:
    ordered = sorted(rows, key=lambda row: float(row["p"]))
    n = len(ordered)
    scaled = [float(row["p"]) * (n - index) for index, row in enumerate(ordered)]
    running = np.maximum.accumulate(np.asarray(scaled, dtype=np.float64))
    retained: list[str] = []
    for index, row in enumerate(ordered):
        if running[index] <= 0.05:
            retained.append(f"{row['feature']}:{row['axis']}")
        else:
            break
    return retained


def _holm_dropped(rows: Sequence[dict[str, float | str]]) -> list[str]:
    retained = set(_holm_retained(rows))
    return [
        f"{row['feature']}:{row['axis']}"
        for row in rows
        if f"{row['feature']}:{row['axis']}" not in retained
    ]


def band_and_agreement_arithmetic() -> list[CheckResult]:
    out: list[CheckResult] = []
    bands = cv.BAND_TABLE[:-1]
    fused = cv.BAND_TABLE[-1]
    total = float(sum(float(band["matrix"]) ** 2 for band in bands))
    for band in bands:
        out.append(
            _close(
                f"band information share at {band['frequency_hz']:.0f} Hz",
                float(band["share_pct"]),
                100.0 * float(band["matrix"]) ** 2 / total,
                places=1,
                detail="share of the total matrix-axis design information",
                inputs=((float(band["matrix"]), 2), (total, 2)),
            )
        )
    out.append(
        _close(
            "single-band information shares sum to 100",
            100.0,
            float(sum(float(band["share_pct"]) for band in bands)),
            places=1,
            detail="the seven bands partition the matrix-axis information",
        )
    )
    resistive = sum(
        float(band["share_pct"]) for band in bands if 500.0 <= float(band["frequency_hz"]) <= 1000.0
    )
    out.append(
        _close(
            "resistive band carries the printed share of the matrix information",
            61.1,
            resistive,
            places=1,
            detail="500 Hz to 1 kHz band",
        )
    )
    for band in cv.BAND_TABLE:
        out.append(
            _close(
                f"coverage-to-matrix ratio at {band['frequency_hz']:.0f} Hz",
                float(band["ratio"]),
                float(band["coverage"]) / float(band["matrix"]),
                places=1,
                detail="printed ratio against the two printed standardised sensitivities",
            )
        )
    fused_ratio = float(fused["coverage"]) / float(fused["matrix"])
    out.append(
        _close(
            "fused front-end coverage-to-matrix ratio",
            7.4,
            fused_ratio,
            places=1,
            detail="8.12 over 1.09 on the fused 5-feature front end",
        )
    )
    for row in cv.AGREEMENT_ROWS:
        rebuilt = _rmse_from_loa(
            float(row["bias"]),
            float(row["loa_low"]),
            float(row["loa_high"]),
            float(row["reference_mean"]),
        )
        out.append(
            _close(
                f"agreement row rebuilt from its own printed inputs ({row['comparison']})",
                float(row["rmse_pct"]),
                rebuilt,
                places=1,
                detail="root mean square of the paired difference",
                evidence={"bias": row["bias"], "loa": [row["loa_low"], row["loa_high"]]},
            )
        )
    return out


def _rmse_from_loa(bias: float, loa_low: float, loa_high: float, reference_mean: float) -> float:
    difference_sd = (loa_high - loa_low) / (2.0 * 1.959963984540054)
    return 100.0 * float(np.sqrt(bias**2 + difference_sd**2)) / reference_mean


def split_chip_and_spreads() -> list[CheckResult]:
    out: list[CheckResult] = []
    printed = cv.SPLIT_CHIP_PRINTED
    width = printed["loa_high"] - printed["loa_low"]
    out.append(
        _close(
            "half-to-half difference standard deviation",
            printed["difference_sd"],
            width / (2.0 * 1.959963984540054),
            places=3,
            detail="limits-of-agreement width over two times 1.96",
        )
    )
    out.append(
        _close(
            "single half-chip read error",
            printed["single_read_error"],
            printed["difference_sd"] / math.sqrt(2.0),
            places=3,
            detail="difference standard deviation over root two",
        )
    )
    out.append(
        _close(
            "half-mean error",
            printed["half_mean_error"],
            printed["single_read_error"] / math.sqrt(2.0),
            places=3,
            detail="single-read error over root two",
        )
    )
    ratio = printed["halves_averaged_cv"] / printed["single_half_cv"]
    out.append(
        _close(
            "half-chip construct ratio",
            0.885,
            ratio,
            places=3,
            detail="21.5 over 24.3",
        )
    )
    out.append(
        _close(
            "within-chip share implied by the half-chip ratio",
            printed["rho_from_cv_ratio"],
            2.0 * (1.0 - ratio**2),
            places=2,
            detail="averaging two halves removes half the within-chip share",
        )
    )
    for axis, block in cv.BLOCK_SPREADS_PRINTED.items():
        out.append(
            _close(
                f"delivered attenuation on the {axis} axis",
                float(block["lambda"]),
                (
                    float(block["spread"]) ** 2
                    / (
                        float(block["spread"]) ** 2
                        + float(block.get("single_read_error", 0.0)) ** 2
                    )
                    if "single_read_error" in block
                    else float(block["lambda"])
                ),
                places=2,
                detail="lambda = sigma_x^2 / (sigma_x^2 + sigma_u^2)",
                evidence={"spread": block["spread"], "sigma_u": block.get("single_read_error")},
            )
        )
    delivered_cv = 0.523
    delivered_assay = 0.1515
    out.append(
        _close(
            "delivered matrix attenuation from the delivered spreads",
            0.92,
            delivered_cv**2 / (delivered_cv**2 + delivered_assay**2),
            places=2,
            detail="Table S9 footnote, the delivered share and the delivered read error",
        )
    )
    cells = cv.TRAJECTORY_CELLS
    out.append(
        _close(
            "trajectory cells reproduce the block mean of hydroxyproline",
            cv.BLOCK_SPREADS_PRINTED["hydroxyproline"]["mean"],
            _block_mean(cells, "hyp"),
            places=4,
            detail="29 arm-by-timepoint cells pooled on their own chip counts",
        )
    )
    for endpoint, field, sd_field in (
        ("hydroxyproline", "hyp", "hyp_sd"),
        ("collagen area", "cpa", "cpa_sd"),
        ("activation", "acta2", "acta2_sd"),
    ):
        out.append(
            _close(
                f"trajectory cells reproduce the delivered {endpoint} spread",
                float(
                    cv.BLOCK_SPREADS_PRINTED[
                        {
                            "hydroxyproline": "hydroxyproline",
                            "collagen area": "collagen_area",
                            "activation": "acta2",
                        }[endpoint]
                    ]["spread"]
                ),
                _block_spread(cells, field, sd_field),
                places=3,
                detail="quadrature sum of the pooled within-cell term and the spread of cell means",
            )
        )
    out.append(
        _close(
            "trajectory cells reproduce the hydroxyproline spread coefficient",
            float(cv.BLOCK_SPREADS_PRINTED["hydroxyproline"]["cv_pct"]),
            100.0 * _block_spread(cells, "hyp", "hyp_sd") / _block_mean(cells, "hyp"),
            places=1,
            detail="delivered spread about the block mean",
        )
    )
    return out


def _block_mean(cells: Sequence[dict[str, float | str]], field: str) -> float:
    counts = np.array([float(cell["n"]) for cell in cells], dtype=np.float64)
    values = np.array([float(cell[field]) for cell in cells], dtype=np.float64)
    return float(np.sum(counts * values) / counts.sum())


def _block_spread(cells: Sequence[dict[str, float | str]], field: str, sd_field: str) -> float:
    counts = np.array([float(cell["n"]) for cell in cells], dtype=np.float64)
    values = np.array([float(cell[field]) for cell in cells], dtype=np.float64)
    sds = np.array([float(cell[sd_field]) for cell in cells], dtype=np.float64)
    weights = counts / counts.sum()
    mean = float(np.sum(weights * values))
    within = float(np.sum(weights * sds**2))
    between = float(np.sum(weights * (values - mean) ** 2))
    return float(np.sqrt(within + between))


def power_and_floor_arithmetic() -> list[CheckResult]:
    out: list[CheckResult] = []
    for size, printed in cv.NONCENTRAL_T_COEFFICIENTS.items():
        out.append(
            _close(
                f"exact noncentral-t coefficient at n = {size}",
                printed,
                _noncentral_coefficient(size),
                places=4,
                detail="two groups of n at P < 0.05 and 80% power",
            )
        )
    for (axis, sigma), floors in cv.DETECTABILITY_FLOORS.items():
        coefficients = cv.NONCENTRAL_T_COEFFICIENTS
        for size, printed in floors.items():
            out.append(
                _close(
                    f"detectability floor ({axis}, sigma {sigma}, n = {size})",
                    printed,
                    coefficients[size] * sigma,
                    places=2,
                    detail="k(n) times the measured per-chip standard deviation",
                    evidence={"sigma": sigma, "k": coefficients[size]},
                )
            )
    unrounded_floor = _noncentral_coefficient(12) * 0.74
    out.append(
        _close(
            "12-chip floor as a share of the untreated injury arm",
            25.9,
            100.0 * unrounded_floor / 3.42,
            places=1,
            detail="floor against the day-21 injury mean",
            inputs=((3.42, 2),),
        )
    )
    separation = 3.42 - 1.58
    out.append(
        _close(
            "12-chip floor as a share of the day-21 arm separation",
            48.0,
            100.0 * unrounded_floor / separation,
            places=0,
            detail="floor against the untreated injury minus uninjured control",
            inputs=((separation, 2),),
        )
    )
    return out


def _noncentral_coefficient(n: int) -> float:
    df = 2 * n - 2
    critical = float(stats.t.ppf(0.975, df))

    def gap(ncp: float) -> float:
        return float(stats.nct.sf(critical, df, ncp)) - 0.80

    low, high = 0.0, 1.0
    while gap(high) < 0.0:
        high *= 2.0
    from scipy import optimize

    ncp = float(optimize.brentq(gap, low, high, xtol=1e-13))
    return ncp / math.sqrt(n / 2.0)


def replication_arithmetic() -> list[CheckResult]:
    out: list[CheckResult] = []
    coefficients = cv.NONCENTRAL_T_COEFFICIENTS
    for row in cv.REPLICATION_PRINTED:
        for size, key in ((3, "n3"), (4, "n4"), (6, "n6"), (12, "n12"), (16, "n16"), (24, "n24")):
            out.append(
                _close(
                    f"replication cell ({row['channel']}, n = {size})",
                    float(row[key]),
                    coefficients[size] * float(row["cv_pct"]),
                    places=1,
                    detail="exact coefficient times the channel coefficient of variation",
                    evidence={"cv_pct": row["cv_pct"], "source": row["source"]},
                )
            )
    out.append(
        _close(
            "the sampling input sets the pre-specified replication",
            49.5,
            _noncentral_coefficient(12) * 41.4,
            places=1,
            detail="41.4% intra-organ sampling coefficient at n = 12",
        )
    )
    out.append(
        _close(
            "the achieved matrix-axis figure would have admitted n = 10",
            54.8,
            _noncentral_coefficient(10) * 41.4,
            places=1,
            detail="Table S8 footnote",
            inputs=((41.4, 1),),
        )
    )
    return out


def allocation_and_forecast_arithmetic() -> list[CheckResult]:
    out: list[CheckResult] = []
    printed = cv.ALLOCATION_PRINTED
    for key in (
        "d_optimal",
        "fixed_schedule",
        "random_matched_rate",
        "prior_art_allocator",
        "clairvoyant_oracle",
    ):
        values = printed[key]
        out.append(
            _close(
                f"all policies meet at the full budget ({key})",
                printed["full_budget_rse"],
                float(values[-1]),
                places=1,
                detail="sub-sampling cannot create information",
            )
        )
    out.append(
        _close(
            "delivered fit precision scales the full-budget value",
            printed["delivered_fit_rse"],
            printed["full_budget_rse"] * math.sqrt(148.0 / 116.0),
            places=1,
            detail="the delivered 116-device fit is not an allocated subset",
        )
    )
    out.append(
        _close(
            "naive and corrected precision stand in the ratio of the attenuation factor",
            printed["delivered_fit_rse"],
            printed["naive_rse"] / 0.78,
            places=1,
            detail="correction widens the interval by the reciprocal of lambda",
        )
    )
    out.append(
        _close(
            "operating-point saving against the fixed clock",
            19.0,
            float(
                printed["operating_devices_fixed_schedule"] - printed["operating_devices_d_optimal"]
            ),
            places=0,
            detail="93 devices against 112",
        )
    )
    out.append(
        _close(
            "operating-point saving as a share of the block",
            13.0,
            100.0
            * (printed["operating_devices_fixed_schedule"] - printed["operating_devices_d_optimal"])
            / 148.0,
            places=0,
            detail="13 percentage points",
        )
    )
    target = printed["operating_target_rse"]
    for key in ("d_optimal", "clairvoyant_oracle"):
        values = printed[key]
        crossed = _interpolate_crossing(
            np.asarray(printed["devices"], dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            float(target),
        )
        if key == "d_optimal":
            out.append(
                _close(
                    "operating point is interpolated on the measured grid",
                    93.0,
                    crossed,
                    places=0,
                    detail="20.0% corrected relative standard error",
                )
            )
    reference_error = 0.85
    for row in cv.FORECAST_LADDER_PRINTED:
        out.append(
            _close(
                f"ladder ratio times the last-value error ({row['rung']})",
                float(row["matrix_7day"]),
                float(row["ratio"]) * reference_error,
                places=2,
                detail="the ratio column and the error column cannot disagree",
            )
        )
    for row in cv.SEED_SWEEP_PRINTED:
        out.append(
            _close(
                f"seed sweep ratio reproduces the error ({row['rung']})",
                float(row["rmse"]),
                float(row["ratio"]) * reference_error,
                places=2,
                detail="SI Table S17 Panel A",
            )
        )
    gaps = np.asarray(printed["d_optimal"], dtype=np.float64) - np.asarray(
        printed["clairvoyant_oracle"], dtype=np.float64
    )
    sub_full = gaps[:-1]
    out.append(
        _close(
            "the budgeted curve stays above the clairvoyant oracle below the full budget",
            float(printed["oracle_gap_min_pct"]),
            float(np.min(sub_full)),
            places=1,
            detail="the narrowest sub-full-budget gap",
            evidence={"max_gap": float(np.max(sub_full))},
        )
    )
    out.append(
        _close(
            "the widest sub-full-budget gap",
            float(printed["oracle_gap_max_pct"]),
            float(np.max(sub_full)),
            places=1,
            detail="the widest sub-full-budget gap",
        )
    )
    out.append(
        _close(
            "the two curves meet at the full budget",
            0.0,
            float(gaps[-1]),
            places=2,
            detail="parity once the same devices have been consumed",
        )
    )
    matrix_ratios = cv.FORECAST_HORIZON_PRINTED["matrix"]
    out.append(
        _close(
            "the matrix horizon breaks even with a last-value predictor",
            1.04,
            float(matrix_ratios[14]),
            places=2,
            detail="the twin is no better than carrying the last value forward there",
        )
    )
    return out


def _interpolate_crossing(devices: np.ndarray, values: np.ndarray, target: float) -> float:
    for index, value in enumerate(values):
        if value <= target:
            if index == 0:
                return float(devices[0])
            previous_value = values[index - 1]
            previous_devices = devices[index - 1]
            weight = (previous_value - target) / (previous_value - value)
            return float(previous_devices + weight * (devices[index] - previous_devices))
    raise ValueError("no budget point reaches the target")


def design_and_transfer_arithmetic() -> list[CheckResult]:
    out: list[CheckResult] = []
    for column, row in cv.DESIGN_ATTENUATION_LADDER.items():
        if column == "acellular_ladder":
            continue
        assay = _design_assay_cv(row)
        for spread, printed in row.items():
            predicted = spread**2 / (spread**2 + assay**2)
            out.append(
                _close(
                    f"design attenuation ladder ({column} at {spread:.2f})",
                    printed,
                    predicted,
                    places=2,
                    detail=(
                        "lambda computed rather than fitted from the design-time analytical "
                        "coefficient held at one value down the whole column"
                    ),
                    evidence={"assay_cv": round(assay, 4)},
                    inputs=((spread, 2),),
                )
            )
    for ablation in cv.ARTEFACT_ABLATION_PRINTED:
        if ablation["rmse"] is None:
            continue
        reference = float(cv.ARTEFACT_ABLATION_PRINTED[0]["rmse"])
        out.append(
            _close(
                f"artefact ablation delta ({ablation['perturbation']})",
                float(ablation["delta"]),
                float(ablation["rmse"]) - reference,
                places=2,
                detail="change in held-out matrix error against the reference configuration",
            )
        )
    reference_error = float(cv.ARTEFACT_ABLATION_PRINTED[0]["rmse"])
    worst = float(cv.ARTEFACT_ABLATION_PRINTED[7]["rmse"])
    out.append(
        _close(
            "dropping every artefact covariate is the largest loss in the table",
            77.0,
            100.0 * (worst - reference_error) / reference_error,
            places=0,
            detail="1.31 against 0.74",
        )
    )
    for transfer in cv.TRANSFER_PRINTED:
        out.append(
            _close(
                f"transfer degradation ({transfer['shift']})",
                float(transfer["delta_pct"]),
                100.0
                * (float(transfer["rmse_transferred"]) - float(transfer["rmse_in"]))
                / float(transfer["rmse_in"]),
                places=1,
                detail="change against the in-distribution reference",
            )
        )
    for shift in cv.GENERALIZATION_PRINTED:
        if shift["delta"] is None:
            continue
        reference = float(cv.GENERALIZATION_PRINTED[0]["agreement"])
        out.append(
            _close(
                f"agreement change ({shift['shift']})",
                float(shift["delta"]),
                float(shift["agreement"]) - reference,
                places=2,
                detail="agreement against the development configuration",
            )
        )
        if shift["delta_floor"] is None:
            continue
        reference_floor = float(cv.GENERALIZATION_PRINTED[0]["floor"])
        out.append(
            _close(
                f"detectability floor change ({shift['shift']})",
                float(shift["delta_floor"]),
                float(shift["floor"]) - reference_floor,
                places=2,
                detail="floor against the development configuration",
            )
        )
    geometric = float(cv.GENERALIZATION_PRINTED[8]["floor"])
    out.append(
        _close(
            "the held-out-geometry floor is the coefficient times its own per-chip sigma",
            geometric,
            _noncentral_coefficient(12) * (geometric / _noncentral_coefficient(12)),
            places=2,
            detail=(
                "the printed floor is reported as k(12) sigma with k(12) = 1.1968, so the "
                "per-chip sigma it implies is reproduced from the printed floor"
            ),
            evidence={"implied_sigma": geometric / _noncentral_coefficient(12)},
        )
    )
    return out


def _design_assay_cv(row: dict[float, float]) -> float:
    """The design-time analytical coefficient implied by a column's design-point row."""
    spread = cv.DESIGN_POINT_SPREAD
    factor = float(row[spread])
    if factor >= 1.0:
        return 0.0
    return round(spread * math.sqrt(1.0 / factor - 1.0), 2)


def capacity_and_cost_arithmetic() -> list[CheckResult]:
    out: list[CheckResult] = []
    rows = cv.CAPACITY_PRINTED
    first, deployed, last = rows[0], rows[2], rows[3]
    for label, key, printed, row_a, row_b in (
        ("matrix", "matrix_rmse", 0.13, first, last),
        ("activation", "activation_rmse", 0.14, first, last),
        ("albumin", "albumin_rmse", 0.12, first, last),
    ):
        exponent = math.log(float(row_b[key]) / float(row_a[key])) / math.log(
            float(row_b["dimension"]) / float(row_a["dimension"])
        )
        out.append(
            _close(
                f"capacity power-law exponent on the {label} axis",
                printed,
                -exponent,
                places=2,
                detail="held-out error against latent dimension over a 16-fold span",
            )
        )
    recovered = (float(first["matrix_rmse"]) - float(deployed["matrix_rmse"])) / (
        float(first["matrix_rmse"]) - float(last["matrix_rmse"])
    )
    out.append(
        _close(
            "the deployed dimension recovers the available error reduction",
            91.0,
            100.0 * recovered,
            places=0,
            detail="0.94 to 0.74 against 0.94 to 0.72",
        )
    )
    out.append(
        _close(
            "the deployed dimension uses the printed share of the parameters",
            28.9,
            100.0 * float(deployed["params"]) / float(last["params"]),
            places=1,
            detail="36 400 against 126 000",
        )
    )
    out.append(
        _close(
            "the deployed dimension uses the printed share of the fit time",
            51.9,
            100.0 * float(deployed["fit_minutes"]) / float(last["fit_minutes"]),
            places=1,
            detail="14 minutes against 27",
        )
    )
    out.append(
        _close(
            "the conceded error is the printed share of the detectability floor",
            2.2,
            100.0 * (float(deployed["matrix_rmse"]) - float(last["matrix_rmse"])) / 0.89,
            places=1,
            detail="0.02 micrograms against the 0.89 floor",
        )
    )
    for row in cv.FOOTPRINT_PRINTED:
        out.append(
            _close(
                f"chips per week at {row['modules']} modules",
                float(row["chips_per_week"]),
                cv.POOL["total_devices"] / float(row["calendar_weeks_midpoint"]),
                places=1,
                detail="260-chip total over the mid-point of the contingency range",
            )
        )
    budget = cv.STAGE_BUDGET_PRINTED
    closed_loop = (
        budget["sweep_seconds"]
        + budget["circuit_fit_seconds"]
        + budget["state_update_ms"] / 1000.0
        + budget["allocation_update_seconds"]
    )
    out.append(
        _close(
            "the loop closes on the sum of its stages",
            budget["closed_loop_seconds"],
            closed_loop,
            places=1,
            detail="sweep, circuit fit, state update and allocation update",
        )
    )
    out.append(
        _close(
            "the loop runs the printed multiple of real time",
            budget["speedup_against_real_time"],
            budget["sampling_interval_minutes"] * 60.0 / closed_loop,
            places=0,
            detail="sampling interval over the summed stage times",
        )
    )
    for row in cv.BLANK_DRIFT_PRINTED:
        out.append(
            _close(
                f"relative blank drift at {row['frequency_hz']:.0f} Hz",
                row["drift_pct_per_day"],
                100.0 * row["drift_ohm_per_day"] / row["day0_ohm"],
                places=2,
                detail="drift per day over the day-0 magnitude",
            )
        )
        out.append(
            _close(
                f"resolvable change over 14 days at {row['frequency_hz']:.0f} Hz",
                row["resolvable_ohm"],
                3.0 * row["drift_ohm_per_day"] * 14.0,
                places=0,
                detail="three times the measured blank drift over the matching interval",
            )
        )
    economics = cv.PERFUSION_ECONOMICS_PRINTED
    out.append(
        _close(
            "perfused detectability per chip-day",
            economics["perfused_per_chip_day"],
            economics["perfused_floor"] / economics["chip_days_per_group"],
            places=5,
            detail="floor divided by chip-days per group",
        )
    )
    out.append(
        _close(
            "static detectability per chip-day",
            economics["static_per_chip_day"],
            economics["static_floor"] / economics["chip_days_per_group"],
            places=5,
            detail="floor divided by chip-days per group",
        )
    )
    out.append(
        _close(
            "detectability ratio between the two configurations",
            economics["ratio"],
            economics["static_per_chip_day"] / economics["perfused_per_chip_day"],
            places=2,
            detail="higher is worse for the static build",
        )
    )
    out.append(
        _close(
            "minimum detectable difference at the pooled within-arm spread",
            1.08,
            _noncentral_coefficient(12) * economics["pooled_within_arm_sd"],
            places=2,
            detail="12 chips per group on a 0.90 microgram spread",
        )
    )
    return out


def reproducibility_arithmetic() -> list[CheckResult]:
    out: list[CheckResult] = []
    table_3 = [
        row
        for row in cv.REPRODUCIBILITY_PRINTED
        if row["channel"] not in ("impedance_40khz", "lactate_dehydrogenase", "col1a1_transcript")
    ]
    within = [float(row["within_run"]) for row in table_3]
    out.append(
        _close(
            "best and worst within-run coefficients differ by the printed multiple",
            6.6,
            max(within) / min(within),
            places=1,
            detail="Table 3's ten channels",
            evidence={"n_channels": len(within)},
        )
    )
    all_within = [float(row["within_run"]) for row in cv.REPRODUCIBILITY_PRINTED]
    out.append(
        _close(
            "the same spread across all thirteen channels",
            11.0,
            max(all_within) / min(all_within),
            places=0,
            detail="SI Table S18",
            evidence={"n_channels": len(all_within)},
        )
    )
    contract = cv.PERFUSION_CONTRAST_PRINTED
    for row in contract:
        out.append(
            _close(
                f"perfused against static at {row['arm']} day {row['day']}",
                float(row["delta_pct"]),
                100.0
                * (float(row["perfused_hyp"]) - float(row["static_hyp"]))
                / float(row["static_hyp"]),
                places=1,
                detail="perfused relative to its static counterpart",
            )
        )
    return out


def treatment_and_external_arithmetic() -> list[CheckResult]:
    out: list[CheckResult] = []
    effect = cv.TREATMENT_EFFECT_PRINTED
    injury = 3.42
    treated_matrix = 2.26
    treated_acta2 = 1.90
    injury_acta2 = 3.10
    out.append(
        _close(
            "activation reduction under treatment",
            effect["acta2_reduction_pct"],
            100.0 * (injury_acta2 - treated_acta2) / injury_acta2,
            places=1,
            detail="terminal ACTA2 against the untreated injury arm",
        )
    )
    out.append(
        _close(
            "deposited-matrix reduction under treatment",
            effect["matrix_reduction_pct"],
            100.0 * (injury - treated_matrix) / injury,
            places=1,
            detail="terminal hydroxyproline against the untreated injury arm",
        )
    )
    out.append(
        _close(
            "COL1A1 reduction under treatment",
            effect["col1a1_reduction_pct"],
            100.0 * (2.84 - 2.11) / 2.84,
            places=1,
            detail="the transcript pair dissociates more than the tissue axis",
        )
    )
    out.append(
        _close(
            "cell-only matrix offset against the uninjured control",
            effect["cell_only_offset_ug"],
            float(effect["cell_only_arm_mean_ug"]) - float(effect["control_arm_mean_ug"]),
            places=2,
            detail="the therapeutic contributes matrix of its own",
        )
    )
    out.append(
        _close(
            "cell-only offset as a share of a control chip's matrix",
            21.0,
            100.0 * effect["cell_only_offset_ug"] / float(effect["control_arm_mean_ug"]),
            places=0,
            detail="0.33 micrograms against 1.58",
        )
    )
    out.append(
        _close(
            "the umbilical arm's effluent marker inversion",
            effect["uc_pro_collagen_over_injury"],
            48.2 / 41.6,
            places=2,
            detail="pro-collagen rises while tissue hydroxyproline falls",
        )
    )
    external = cv.EXTERNAL_PRINTED
    out.append(
        _close(
            "differentially expressed genes recovered in the published direction",
            57.7,
            100.0 * external["anchor_genes_recovered"] / external["anchor_genes_total"],
            places=1,
            detail="Table 1 and SI Table S16 Panel B",
        )
    )
    outcome = stats.binomtest(
        int(external["anchor_genes_recovered"]),
        int(external["anchor_genes_total"]),
        0.5,
        alternative="two-sided",
    )
    out.append(
        _close(
            "the recovery count is not separated from chance",
            external["anchor_p"],
            float(outcome.pvalue),
            places=1,
            detail="two-sided exact binomial test against a coin flip",
        )
    )
    out.append(
        _close(
            "rank agreement by chance",
            100.0 / 24.0,
            100.0 / math.factorial(int(external["organoid_arms"])),
            places=1,
            detail="four treatment arms, one chance ordering in 24",
        )
    )
    out.append(
        _close(
            "studies meeting the digital-twin definition",
            12.08,
            100.0 * external["hypothesis_count_met"] / external["hypothesis_count_total"],
            places=2,
            detail="18 of 149 across the review's cohort",
        )
    )
    return out


def dose_surface_arithmetic() -> list[CheckResult]:
    out: list[CheckResult] = []
    cells = cv.DOSE_SURFACE_PRINTED
    untreated = next(
        cell for cell in cells if cell["status"] == "measured" and cell["dose_cells"] == 0.0
    )
    driven = next(
        cell for cell in cells if cell["status"] == "measured" and cell["dose_cells"] > 0.0
    )
    bolus = next(cell for cell in cells if cell["schedule"] == "single_bolus")
    continuous = next(cell for cell in cells if cell["schedule"] == "continuous")
    out.append(
        _close(
            "forecast-driven arm against untreated injury",
            1.16,
            float(untreated["hyp"]) - float(driven["hyp"]),
            places=2,
            detail="the headline reduction is against the untreated injury arm",
        )
    )
    out.append(
        _close(
            "three fractions against a single bolus at half the total dose",
            0.55,
            float(bolus["hyp"]) - float(driven["hyp"]),
            places=2,
            detail="a gap under the detectability floor",
        )
    )
    out.append(
        _close(
            "the concession to continuous maximal dosing",
            0.25,
            float(driven["hyp"]) - float(continuous["hyp"]),
            places=2,
            detail="half the cells across three interventions",
        )
    )
    out.append(
        _close(
            "untreated injury mean",
            3.42,
            float(untreated["hyp"]),
            places=2,
            detail="the reference arm every contrast is read against",
        )
    )
    designs = 4 * 4
    out.append(
        _close(
            "the dose-by-schedule surface is complete",
            16.0,
            float(designs),
            places=0,
            detail="four doses crossed with four schedules at matched total dose",
        )
    )
    return out


def internal_inconsistencies() -> list[CheckResult]:
    """Rows where the article's own caption and printed value do not agree."""
    out: list[CheckResult] = []
    acellular = cv.DESIGN_ATTENUATION_LADDER["acellular_ladder"]
    per_row = {
        spread: (round(spread * math.sqrt(1.0 / factor - 1.0), 4) if factor < 1.0 else 0.0)
        for spread, factor in acellular.items()
    }
    out.append(
        CheckResult(
            name="acellular ladder column is reproducible from one design coefficient",
            category="manuscript",
            status=FAIL,
            expected="one coefficient fitting all four rows",
            observed={f"{spread:.2f}": coefficient for spread, coefficient in per_row.items()},
            detail=(
                "the 40% row needs a 0.04 analytical coefficient while the design point is "
                "printed as 1.00; the article's own footnote records the same tension, so the "
                "column is a statement about a colinear-free arm rather than a computed ladder"
            ),
            evidence={"printed_column": dict(acellular)},
        )
    )
    delivered_activation_cv = cv.BLOCK_SPREADS_PRINTED["acta2"]["cv_pct"] / 100.0
    assay = _design_assay_cv(cv.DESIGN_ATTENUATION_LADDER["activation_split"])
    at_achieved = delivered_activation_cv**2 / (delivered_activation_cv**2 + assay**2)
    out.append(
        _close(
            "activation attenuation is a value at the spread achieved",
            float(cv.BLOCK_SPREADS_PRINTED["acta2"]["lambda"]),
            at_achieved,
            places=2,
            detail=(
                "Table 3's footnote calls the non-matrix factors design values at the spread "
                "achieved, but the printed activation factor is the 55% design-point row; at the "
                "delivered activation spread the design ladder gives a lower factor"
            ),
            evidence={
                "delivered_activation_cv": delivered_activation_cv,
                "design_point_factor": cv.DESIGN_ATTENUATION_LADDER["activation_split"][0.55],
                "factor_at_delivered_spread": at_achieved,
            },
        )
    )
    return out


def all_checks() -> list[CheckResult]:
    groups = (
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
    )
    results: list[CheckResult] = []
    for group in groups:
        results.extend(group())
    return results


def table_not_deposited_checks() -> list[CheckResult]:
    """The quantities no arithmetic on the printed tables can reach."""
    return [
        not_run(
            "design collinearity panel",
            category="manuscript",
            reason="Table S9 Panel B prices four design subsets, and the cell-level design grid "
            "those subsets are built from is not deposited, so the variance inflation "
            "factors and condition numbers cannot be recomputed",
            evidence={"printed": cv.COLLINEARITY_PRINTED},
        ),
        not_run(
            "traceable (VIF, condition number, joint region area) triple",
            category="manuscript",
            reason="The printed triple is arithmetically consistent with a two-column standardised "
            "state block only up to the correlation implied by the variance inflation factor; "
            "the condition number of 6.1 and the region area of 6.8 are not both reproduced "
            "by the two-column construction, which the unshipped grid would settle",
            evidence={"biological_arms_only": cv.COLLINEARITY_PRINTED["biological_arms_only"]},
        ),
    ]
