"""The paper-claim to code mapping, and the register of intentional departures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from obscal.verification.checks import CheckResult


@dataclass(frozen=True)
class Claim:
    """One manuscript claim, the code that carries it and the checks that read it."""

    identifier: str
    statement: str
    paper_location: str
    code_anchors: tuple[str, ...]
    checks: tuple[str, ...]

    def resolve(self, status_by_check: dict[str, str]) -> dict[str, Any]:
        observed = [status_by_check.get(name, "NOT_RUN") for name in self.checks]
        if "FAIL" in observed:
            status = "FAIL"
        elif "BLOCKED" in observed:
            status = "BLOCKED"
        elif not observed or "NOT_RUN" in observed:
            status = "NOT_RUN"
        else:
            status = "PASS"
        return {
            "id": self.identifier,
            "statement": self.statement,
            "paper_location": self.paper_location,
            "code_anchors": list(self.code_anchors),
            "checks": list(self.checks),
            "check_statuses": dict(zip(self.checks, observed, strict=True)),
            "verification": status,
        }


@dataclass(frozen=True)
class Deviation:
    """One intentional departure, or one engineering default where the manuscript is silent."""

    identifier: str
    paper_location: str
    departs: str
    reason: str

    def as_record(self) -> dict[str, str]:
        return {
            "id": self.identifier,
            "paper_location": self.paper_location,
            "departs": self.departs,
            "reason": self.reason,
        }


CLAIMS: tuple[Claim, ...] = (
    Claim(
        identifier="C1",
        statement=(
            "The two state axes, stellate activation and deposited matrix, are separately "
            "calibrated by breaking their collinearity with acellular and seeding-density "
            "ladders and by entering electrode-area coverage as a measured covariate."
        ),
        paper_location="Contribution 1; Sec. Calibration design and identifiability; Eq. (3)",
        code_anchors=(
            "src/obscal/calibration/regression.py::assemble_design",
            "src/obscal/calibration/regression.py::fit_pooled",
            "src/obscal/design/ladders.py::acellular_ladder",
            "src/obscal/design/ladders.py::seeding_density_ladder",
        ),
        checks=(
            "coverage is carried beside the features rather than inside them",
            "the pooled design matrix has full column rank",
            "the identifiability audit prices both state axes",
            "the pooled calibration recovers the pool's own matrix sensitivities",
            "the pooled calibration recovers the pool's own coverage sensitivities",
        ),
    ),
    Claim(
        identifier="C2",
        statement=(
            "Pooling 116 biological devices over 5 sensor features sets the plug-in "
            "covariance inflation at 1.064."
        ),
        paper_location="Contribution 1; Sec. Calibration design and identifiability",
        code_anchors=(
            "src/obscal/calibration/regression.py::plugin_inflation",
            "src/obscal/calibration/regression.py::residual_covariance",
        ),
        checks=(
            "plug-in covariance inflation",
            "the plug-in covariance inflation is the printed formula",
            "the residual covariance is positive definite",
        ),
    ),
    Claim(
        identifier="C3",
        statement=(
            "Each calibration-block device is divided at harvest and both halves are read on "
            "the same hydroxyproline assay, bounding that standard's own error variance from "
            "above and making the matrix attenuation factor a measured number."
        ),
        paper_location="Contribution 2; Sec. Split-chip replication and terminal-assay error",
        code_anchors=(
            "src/obscal/terminal/splitchip.py::split_chip_error",
            "src/obscal/terminal/splitchip.py::within_chip_fraction_from_halves",
            "src/obscal/terminal/splitchip.py::error_covariance",
        ),
        checks=(
            "the half-to-half difference standard deviation matches its brute-force value",
            "the single half-chip read error is the difference over root two",
            "the two routes to the within-chip share agree",
            "collapsing to the half-mean basis reproduces the terminal values",
        ),
    ),
    Claim(
        identifier="C4",
        statement=(
            "The errors-in-variables moment correction is applied to every calibration slope, "
            "and the correction moves each matrix slope away from zero."
        ),
        paper_location="Sec. Attenuation correction and the detectability floor; Eq. (4)",
        code_anchors=(
            "src/obscal/calibration/attenuation.py::moment_correction",
            "src/obscal/calibration/attenuation.py::per_axis_lambda",
        ),
        checks=(
            "the moment correction matches a brute-force recomputation",
            "the correction enlarges the delivered matrix sensitivity",
            "delivered attenuation on the matrix axis",
            "corrected slope equals naive over lambda (log_rb, matrix)",
        ),
    ),
    Claim(
        identifier="C5",
        statement=(
            "A damped oscillatory state-space twin consumes the calibrated front end, treats "
            "treatment as a state input rather than an observation channel, and forecasts both "
            "axes under mesenchymal stromal cell treatment."
        ),
        paper_location="Sec. Forecasting and terminal-assay allocation; Eq. (7)",
        code_anchors=(
            "src/obscal/twin/oscillatory.py::DampedOscillatoryBackbone",
            "src/obscal/twin/pipeline.py::fit_backbone",
            "src/obscal/twin/pipeline.py::predict_series",
        ),
        checks=(
            "the backbone consumes both state axes",
            "one forward pass returns the observed window and the forecast",
            "a backward pass carries a gradient into every parameter",
            "a saved checkpoint round-trips into identical forecasts",
            "the forecast is issued per axis for the requested horizon",
        ),
    ),
    Claim(
        identifier="C6",
        statement=(
            "The next destructive assay is placed by D-optimal design under a device budget, "
            "and the resulting curve is read against a fixed clock, a random subsample at "
            "matched rate, a re-implemented prior-art allocator and a clairvoyant oracle."
        ),
        paper_location="Contribution 3; Sec. Forecasting and terminal-assay allocation; Eq. (6)",
        code_anchors=(
            "src/obscal/allocation/doptimal.py::d_optimal_order",
            "src/obscal/allocation/doptimal.py::marginal_gain",
            "src/obscal/allocation/policies.py::prior_art_allocator_order",
            "src/obscal/allocation/curve.py::build_curve",
        ),
        checks=(
            "the allocator returns exactly the budgeted placements",
            "every accepted placement increases the design information",
            "the marginal gain matches a brute-force determinant difference",
            "every policy meets at the full budget",
            "the budgeted curve stays above the clairvoyant oracle below the full budget",
        ),
    ),
    Claim(
        identifier="C7",
        statement=(
            "The smallest resolvable change along a named axis follows from the covariance of "
            "the delivered estimator and the exact noncentral-t coefficient."
        ),
        paper_location="Sec. Attenuation correction and the detectability floor; Eq. (5)",
        code_anchors=(
            "src/obscal/power/noncentral.py::two_group_coefficient",
            "src/obscal/power/noncentral.py::directional_floor",
            "src/obscal/power/noncentral.py::estimator_covariance",
        ),
        checks=(
            "the noncentral-t coefficient at n = 4",
            "the noncentral-t coefficient at n = 12",
            "the noncentral-t coefficient at n = 24",
            "the coefficient is monotonically decreasing in the replication",
            "the estimator covariance is the inverse of the information matrix",
            "detectability floor (matrix_5_feature, sigma 0.74, n = 12)",
        ),
    ),
    Claim(
        identifier="C8",
        statement=(
            "The primary endpoint's reproducibility is reported on the run and batch axis "
            "rather than the operator or assay axis, and the primary endpoint grades poor on "
            "that programme's acceptance scale."
        ),
        paper_location="Sec. Terminal-assay error, attenuation and the variance decomposition; Table 3",
        code_anchors=(
            "src/obscal/terminal/reproducibility.py::channel_reproducibility",
            "src/obscal/terminal/reproducibility.py::grade_cv",
            "src/obscal/terminal/reproducibility.py::cv_spread_ratio",
        ),
        checks=(
            "best and worst within-run coefficients differ by the printed multiple",
            "the same spread across all thirteen channels",
            "the block spread is the quadrature sum of its two terms",
        ),
    ),
    Claim(
        identifier="C9",
        statement=(
            "Agreement against the destructive standard is reported as an intraclass "
            "correlation with Bland-Altman limits, and an estimate is never more reproducible "
            "than the assay it estimates."
        ),
        paper_location="Sec. Agreement against the gold-standard assay; Table 4",
        code_anchors=(
            "src/obscal/terminal/agreement.py::agreement",
            "src/obscal/terminal/agreement.py::icc_absolute_agreement",
            "src/obscal/terminal/agreement.py::rmse_from_loa",
        ),
        checks=(
            "the two ICC routes agree on the same table",
            "the Bland-Altman bias matches its brute-force mean",
            "the root-mean-square error rebuilt from the limits matches the direct value",
            "the estimate is held to an assay rather than a reader",
            "agreement row rebuilt from its own printed inputs (estimated_matrix_vs_hydroxyproline)",
        ),
    ),
    Claim(
        identifier="C10",
        statement=(
            "The platform's central asymmetry is a shape: matrix sensitivity peaks in the "
            "500 Hz to 1 kHz resistive band while coverage sensitivity peaks at 40 kHz, so an "
            "electrical fibrosis readout blind to coverage reports coverage more than matrix."
        ),
        paper_location="Discussion; Fig. 1b; SI Table S4",
        code_anchors=(
            "src/obscal/frontend/bands.py::matrix_information_shares",
            "src/obscal/frontend/bands.py::coverage_matrix_gap",
            "assets/sweep_grid.txt",
        ),
        checks=(
            "band information share at 100 Hz",
            "band information share at 500 Hz",
            "resistive band carries the printed share of the matrix information",
            "fused front-end coverage-to-matrix ratio",
            "the excitation grid is the printed seven-point set",
        ),
    ),
    Claim(
        identifier="C11",
        statement=(
            "The cell-free artefact channel is measured rather than assumed inert, and the "
            "resolvable-change threshold is three times the measured blank drift over the "
            "matching interval."
        ),
        paper_location="Sec. External validation and the artefact-confound control; SI Table S15",
        code_anchors=(
            "src/obscal/chip/sweep.py::resolvable_change",
            "src/obscal/frontend/blanks.py::schedule_prediction",
            "src/obscal/chip/sweep.py::resolvable_change_table",
        ),
        checks=(
            "relative blank drift at 100 Hz",
            "resolvable change over 14 days at 100 Hz",
            "the lack-of-fit statistic stays inside the unit interval",
        ),
    ),
    Claim(
        identifier="C12",
        statement=(
            "Treatment is a dose-by-schedule surface at matched cumulative dose, and both "
            "tissue axes fall monotonically with dose so no dose optimum is claimed."
        ),
        paper_location="Sec. Device, cultures and injury model; SI Table S10",
        code_anchors=(
            "src/obscal/design/surface.py::TreatmentSurface",
            "src/obscal/design/surface.py::monotonicity_verdict",
            "src/obscal/design/surface.py::constraint_axis_report",
        ),
        checks=(
            "forecast-driven arm against untreated injury",
            "three fractions against a single bolus at half the total dose",
            "the concession to continuous maximal dosing",
            "the dose-by-schedule surface is complete",
        ),
    ),
    Claim(
        identifier="C13",
        statement=(
            "Forty-five of 260 devices were withdrawn under pre-declared inclusion rules in six "
            "mutually exclusive classes, leaving 215 analysed."
        ),
        paper_location="Sec. Platform characterisation and culture longevity; SI Table S14",
        code_anchors=(
            "src/obscal/design/withdrawal.py::classify_device",
            "src/obscal/design/withdrawal.py::ledger_closure",
        ),
        checks=(
            "withdrawal classes sum to the withdrawn total",
            "withdrawn share of the pool",
            "analysed devices",
            "the withdrawal ledger closes against the pool",
        ),
    ),
    Claim(
        identifier="C14",
        statement=(
            "Contrasts use batch as a crossed factor, P is unadjusted in the tables and any "
            "count across a test family carries its Holm-corrected count."
        ),
        paper_location="Sec. Evaluation metrics and statistical analysis; SI Table S3 footnote",
        code_anchors=(
            "src/obscal/evaluation/contrasts.py::batch_adjusted_contrast",
            "src/obscal/evaluation/multiplicity.py::holm_step_down",
        ),
        checks=(
            "the Welch statistic matches a hand-computed value",
            "the batch-adjusted contrast recovers an injected arm effect",
            "the Holm step-down matches a brute-force running maximum",
            "Holm step-down retains the printed number of slopes",
            "Holm drops exactly the printed activation slopes among those clearing zero",
        ),
    ),
    Claim(
        identifier="C15",
        statement=(
            "Two public resources supply the external tests: an alcohol Liver-Chip series as a "
            "phenotype anchor and an independently published organoid line as a state-model "
            "test, with no public impedance set available for the calibration itself."
        ),
        paper_location="Sec. External validation and the artefact-confound control; Table 1",
        code_anchors=(
            "src/obscal/evaluation/discrimination.py::direction_recovery",
            "src/obscal/evaluation/discrimination.py::rank_agreement",
            "dataset_urls.txt",
        ),
        checks=(
            "differentially expressed genes recovered in the published direction",
            "the recovery count is not separated from chance",
            "rank agreement by chance",
            "the dataset register lists the verified resource addresses",
            "the register records the addresses that did not answer",
        ),
    ),
)


DEVIATIONS: tuple[Deviation, ...] = (
    Deviation(
        identifier="D1",
        paper_location="Data Availability, 'Derived data and code'",
        departs=(
            "The article deposits no per-device rows and states that no executable "
            "implementation accompanies it, so the release executes on a harness device pool "
            "rather than on the study's own records."
        ),
        reason=(
            "Without the study's rows no manuscript table value can be recomputed from this "
            "release. Every cohort-level quantity is therefore reported as the release's own on "
            "the harness pool, and the manuscript's arithmetic is checked separately from its "
            "printed tables."
        ),
    ),
    Deviation(
        identifier="D2",
        paper_location="Sec. Continuous sensing front end, geometric correction [29]",
        departs=(
            "The functional form of the geometric correction factor is not printed. The "
            "release computes it as the tissue-channel floor area over the floor area plus the "
            "perimeter term the current path adds through the channel walls."
        ),
        reason=(
            "The manuscript asserts only that the factor is computed from the printed geometry "
            "and never refitted per device, timepoint or arm. The release keeps that property "
            "and exposes the expression in one function so the choice is inspectable."
        ),
    ),
    Deviation(
        identifier="D3",
        paper_location="Table S17 footnote; SI per-stage budget",
        departs=(
            "The backbone's layer structure, optimiser, epoch budget, batch size, warm-up, "
            "weight decay, precision and gradient clipping are not printed. The release fixes "
            "the latent dimension, learning rate and seed count the manuscript does report and "
            "treats the rest as engineering defaults exposed in config."
        ),
        reason=(
            "Only the deployed learning rate, the five seeds and the latent dimension are "
            "recoverable from the record, so everything else is a labelled default rather than "
            "an invented manuscript value."
        ),
    ),
    Deviation(
        identifier="D4",
        paper_location="Table S2, mechanistic rung",
        departs=(
            "The mechanistic comparator's equations are not printed. The release implements a "
            "first-order threshold recurrence, fitted on held-in series and frozen."
        ),
        reason=(
            "The manuscript reports the rung's performance without its form, so a labelled "
            "recurrence is the only honest way to give the ladder a fourth rung."
        ),
    ),
    Deviation(
        identifier="D5",
        paper_location="Sec. Evaluation metrics and statistical analysis",
        departs=(
            "The manuscript names mixed models with batch as a crossed factor. The release "
            "fits a batch-adjusted fixed-effects linear contrast."
        ),
        reason=(
            "The run log that would carry the random-effect structure is not deposited. Because "
            "the pool is balanced across batches by construction, an additive batch adjustment "
            "recovers an injected arm effect with the same estimand."
        ),
    ),
    Deviation(
        identifier="D6",
        paper_location="Algorithm 2, step 27",
        departs=(
            "Predictive intervals are calibrated on held-out series in the release, because the "
            "manuscript states that the backbone supplies no native uncertainty quantification "
            "and that its intervals are conformalised after the fact."
        ),
        reason=(
            "The manuscript states the provenance rather than the estimator, so the release "
            "declares the finite-sample upper quantile it uses."
        ),
    ),
    Deviation(
        identifier="D7",
        paper_location="Sec. Forecasting and terminal-assay allocation, Eq. (6)",
        departs=(
            "The allocation curve's entry is the corrected slope's relative standard error, "
            "computed as the ordinary-least-squares standard error over the slope's absolute "
            "value divided by the attenuation factor."
        ),
        reason=(
            "The manuscript's own ablation states that reporting naive slopes gives 14.3% "
            "against 18.3% corrected, and 18.3/14.3 equals the reciprocal of the 0.78 factor it "
            "corrects at, which fixes the definition rather than leaving it open."
        ),
    ),
    Deviation(
        identifier="D8",
        paper_location="Table S11 footnote",
        departs=(
            "Allocation candidates are the devices the pool retains, so the release's full "
            "budget is its own retained count rather than the printed 148."
        ),
        reason=(
            "The harness pool's withdrawals fall under the pre-declared rules, so its retained "
            "count differs from the design's; using the printed count would be a manuscript "
            "value inserted into an executing design."
        ),
    ),
    Deviation(
        identifier="D9",
        paper_location="Table 3 footnote, split-chip block",
        departs=(
            "The activation entry of the terminal error covariance is a design-time analytical "
            "coefficient applied at the spread the pool delivers, taken from the manuscript's "
            "own Panel A construction."
        ),
        reason=(
            "Paired half reads exist for hydroxyproline alone, so only the matrix entry is "
            "identified; the manuscript says the other entries are design values at the spread "
            "achieved, which is what the release computes rather than assumes."
        ),
    ),
    Deviation(
        identifier="D10",
        paper_location="Table S1 Panel A",
        departs=(
            "Capacity, floating-point counts and fit times are not tuned to the printed "
            "parameter counts: the release reports the counts its own backbone carries at each "
            "latent dimension."
        ),
        reason=(
            "Reaching the printed parameter counts would require inventing the layer structure "
            "the manuscript does not print; the scaling shape, not the absolute count, is the "
            "claim under test."
        ),
    ),
    Deviation(
        identifier="D11",
        paper_location="Sec. Device, cultures and injury model",
        departs=(
            "The time basis of the pooled regression is a centred power basis with three "
            "columns, exposed as a configuration value."
        ),
        reason=(
            "The manuscript names a smooth basis and states that time enters as a covariate "
            "and an interaction without printing its form."
        ),
    ),
    Deviation(
        identifier="D12",
        paper_location="SI Table S9 Panel B",
        departs=(
            "The four design subsets the collinearity panel prices are not shipped as a "
            "cell-level grid, so the release audits the delivered design's own variance "
            "inflation factor and condition number instead of reproducing the printed panel."
        ),
        reason=(
            "Recomputing the printed panel would need the design grid the manuscript does not "
            "deposit; the audit quantity itself is implemented and exercised."
        ),
    ),
)


def status_by_check(results: list[CheckResult]) -> dict[str, str]:
    return {result.name: result.status for result in results}


def build_claim_map(results: list[CheckResult]) -> dict[str, Any]:
    statuses = status_by_check(results)
    claims = [claim.resolve(statuses) for claim in CLAIMS]
    unresolved = [
        {"claim": claim["id"], "check": name}
        for claim in claims
        for name in claim["checks"]
        if name not in statuses
    ]
    return {
        "claims": claims,
        "deviations": [deviation.as_record() for deviation in DEVIATIONS],
        "unmatched_check_names": unresolved,
        "n_claims": len(claims),
        "n_deviations": len(DEVIATIONS),
    }
