# ObsCal — calibration and terminal-assay allocation for a perfused liver-fibrosis chip

This repository carries the instrument group's implementation of the fitting, allocation
and forecasting procedures set out in *Bio-Robotic Digital Twin Modeling for Forecasting
Liver Fibrosis Improvement during Stem Cell-Based Treatment of Alcoholic Liver Disease*
(Cyborg and Bionic Systems). The article describes a perfused hepatocyte–stellate chip whose
two-axis fibrosis state — stellate activation and deposited matrix — is calibrated against
the destructive assay that consumes the device, estimated from a five-feature impedance and
optical front end, and carried forward under mesenchymal stromal cell treatment by a damped
oscillatory state–space twin. The code here implements that loop: the equivalent-circuit
fit of a multi-frequency sweep, the pooled calibration regression with its
errors-in-variables moment correction, the split-chip decomposition of the terminal assay's
own error, the detectability floor, the D-optimal placement of the next destructive assay
against four comparator brackets, the forecasting ladder, and the two verification families
described at the end of this file.

## What is and is not in this repository

Two facts about the article shape everything below.

The article deposits **no per-device rows**: the quantities behind its tables are carried as
group-level tables, and the Data Availability statement says so. It also ships **no
executable implementation**: "the estimation procedure ships as pseudocode in Algorithms 1
and 2 ... no executable implementation accompanies this article."

The release therefore runs its pipeline on a harness-supplied, schema-compatible device
pool built from `configs/cohort/fixture.yaml`, and reports its own numbers on that pool.
**No manuscript table value is claimed from this release.** The manuscript's arithmetic is
checked separately, from its own printed tables, by the `manuscript` check family described
below. The pool is a fixture in the software sense: deterministic, reproducible from its own
specification and seed, and never presented as material collected from the platform.

## Layout

```
configs/
  chip/         device geometry, culture and injury windows, batching
  frontend/     excitation grid, equivalent circuit, decoupling ladders
  calibration/  pooled fit, attenuation, identifiability audit, bootstrap
  allocation/   D-optimal criterion, budget grid, comparator policies
  twin/         backbone, training, forecast horizons, forecasting ladder
  cohort/       the harness device pool's own generating specification
  run/          one file per reported experiment: primary, ablation_*, supplementary_*, _harness
src/obscal/
  chip/         channel geometry, the study-constant geometric correction, equivalent-circuit fit
  frontend/     Eq. (2) feature assembly, frequency-resolved sensitivity, the blank channel
  design/       arm roster, chip budget, decoupling ladders, dose x schedule surface, exclusions
  calibration/  Eq. (3) and Eq. (4), identifiability diagnostics, device-level bootstrap
  terminal/     split-chip error, agreement against the standard, channel reproducibility
  power/        the exact noncentral-t coefficient, Eq. (5) detectability floors
  allocation/   Eq. (6) search, comparator brackets, the budget-against-precision curve
  twin/         Eq. (7) recurrence, the five forecasting rungs, calibrated intervals
  cohort/       device-record schema, harness pool builder, on-disk form
  evaluation/   Welch and batch-adjusted contrasts, Holm step-down, AUC, plain-text tables
  runtime/      seeding, atomic writes, typed configuration binding
  verification/ the two check families, the claim map, the integrity manifest, the driver
  study.py      joins the device pool to the calibration, allocation and twin layers
scripts/        launch scripts for the calibration, the forecast and the external fetch
tests/          unit tests per module, a minimal training loop and a single-batch overfit
assets/         the excitation grid and the per-band role it was chosen for
```

## Installation

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .
PYTHONPATH=src pytest -q
```

Conda:

```
conda env create -f environment.yml && conda activate obscal
```

Docker:

```
docker build -t obscal .
docker run --rm obscal -m obscal.cli.verify
```

The packages under `src/` are not installed by default in the container-free path, so the
launch scripts set `PYTHONPATH` themselves; `pyproject.toml` also declares
`pythonpath = ["src"]` for pytest, and honours a `PYTHON_BIN` environment variable
(`python3` by default) rather than assuming a `python` executable exists.

## Data

Only two external data resources are used, and both are public expression deposits read as
external anchors. They are listed with their access conditions and the addresses that
answered in `dataset_urls.txt`; the addresses that did not answer from the verification
host are recorded there too, with the reason. Neither deposit is redistributed here, and
neither is needed to run any command below.

- GEO **GSE175396** — human Liver-Chip model of alcohol-associated liver disease, 7 samples
  over 2 control and 5 exposed groups. Role: external injury-phenotype anchor. It carries no
  fibrogenic compartment and a single timepoint, so it anchors injury and never fibrosis.
- GEO **GSE63859** — independently published human liver organoid series, 19 samples.
  Role: independent external test of the state model. Transcriptomic only; it carries no
  electrical channel.

```
scripts/prepare_external.sh runs/external
```

fetches both records in SOFT format and prints the accession and title it received, so the
content can be matched against the identifiers the article prints. The two series matrices
are platform-specific and are not harmonised here.

The on-chip device records behind the article's own tables are not deposited and are not
obtainable; that absence is a property of the study, not of this link list. The harness
device pool is built locally instead:

```
python -m obscal.cli.build_cohort --out runs/cohort
```

It writes `devices.npz` plus `index.json` — 260 devices partitioned into the article's own
blocks (148 biological, 24 acellular ladder, 20 seeding-density ladder, 8 cell-free, 12
held-out geometry, 12 cross-line, 36 static counterpart). The pool's own generating
parameters live in `configs/cohort/fixture.yaml`; every one of them is a harness value, and
the execution checks verify that the pooled calibration recovers them.

## Running the pipeline

```
scripts/launch_calibration.sh configs/run/primary.yaml runs/primary
scripts/launch_forecast.sh   configs/run/primary.yaml runs/primary
```

or directly:

```
python -m obscal.cli.calibrate --run configs/run/primary.yaml --cohort runs/cohort --out runs/primary
python -m obscal.cli.allocate  --run configs/run/primary.yaml --cohort runs/cohort --out runs/primary
python -m obscal.cli.forecast  --run configs/run/primary.yaml --cohort runs/cohort --out runs/primary
```

Each writes a JSON record and a plain-text table under the output directory. Every
`configs/run/*.yaml` carries one reported experiment: `primary.yaml` is the delivered
system, `ablation_*.yaml` are the rows of the component ablation (single-frequency front
end, chopstick electrodes, no coverage covariate, whole-chip lysis, biological arms only,
no cell-free channel, pooled strata, naive slopes, fixed schedule, single axis, undamped
backbone, sham interrogation), and `supplementary_*.yaml` carry the SI-group analyses
(artefact-confound ablation, seed sensitivity, perfused-against-static, source
generalization, transfer and external resources, capacity scaling). `_harness.yaml` is for
the tests only and must not be used for reporting.

The commands are cheap on the harness pool. The article's own compute target — one 8-thread
workstation carrying a single graphics processor, a pooled calibration fit of 41 s at 116
devices over 5 features with 2000 bootstrap replicates, and a 14-minute backbone fit — is
reported in its Supplementary Materials and is *not* reproduced here: the release runs on
the harness pool with the deployed learning rate, five seeds, latent dimension 32, a
2000-replicate bootstrap and the printed 7-point excitation grid taken from the
configuration.

## Verifying the release

```
python -m obscal.cli.verify
```

The driver runs two families and writes four artefacts.

**The `manuscript` family** recomputes 270 quantities the article derives from its own
printed numbers: the arm totals and the chip budget, the plug-in covariance inflation, the
naive-against-corrected slope identity, the Holm step-down over the 15-cell calibration
family, the frequency-band information shares, the agreement rows rebuilt from their own
biases and limits, the split-chip error chain, the exact noncentral-t coefficients against
every detectability floor, the replication arithmetic, the allocation curve's structural
checks, the design attenuation ladder, the capacity power laws, the cost arithmetic, the
dose-by-schedule contrasts and the external direction test. 266 of them close. Two do not,
and both are recorded as they stand rather than reconciled:

- the activation attenuation factor printed in Table 3 is called a design value "at the
  spread achieved", but at the delivered activation spread the manuscript's own Panel A
  ladder gives 0.82 rather than the printed 0.89;
- the acellular-ladder column of SI Table S9 Panel A cannot be reproduced from a single
  design-time coefficient, which the article's own footnote also records.

A failure in this family is a discrepancy inside the article's tables, not a defect in this
release; `verification_report.json` reports a `code_status` beside the `overall_status` so
the two are not confused.

**The `execution` family** runs every mechanism the release implements, each against a
reference built from a definition, an enumeration or a brute-force recomputation rather
than from the code under test: a noiseless sweep against its generating circuit, the
recovered sensitivities against the pool's own generating sensitivities as a three-standard-
error test, the moment correction against an explicit matrix product, the split-chip error
against the paired halves, the two ICC routes against each other, the noncentral-t
coefficient against a draw-by-draw replication of the two-group test, the D-optimal criterion
against a determinant difference, the clairvoyant ceiling against the allocator's own curve,
the calibration design's rank repair, and a forward/loss/backward/update/save/load pass on
the twin including a single-batch overfit. It also covers the hygiene sweep: no host
absolute path, no email, no credential string, no compiled cache, no markdown beyond this
file, no library module writing to standard output, no configuration pinning a value the
manuscript reports as a result, and the stand-in vocabulary guard.

The artefacts are:

- `claim_to_code.json` — fifteen manuscript claims mapped to the modules that carry them and
  the checks that read them, plus a `deviations` array with every intentional departure and
  every engineering default the manuscript is silent about;
- `verification_report.json` — every check with its expected value, observed value and
  evidence, the family tallies and the manuscript discrepancies;
- `verification_summary.txt` — the same summary as plain text;
- `integrity_manifest.json` — a SHA-256 digest per shipped file plus a digest of the
  inventory, written last and excluding itself.

Every check carries exactly one of `PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`, decided from what
actually ran. `NOT_RUN` marks the two quantities the unshipped design grid would be needed
to reach, the live dataset-link probe in the default offline pass, and the bootstrap
interval when the bootstrap is disabled. `BLOCKED` marks a tool or container runtime that is
not available. Because both families carry outstanding entries, the release is
`PARTIALLY_VERIFIED`; the `code_status` line names the verdict over the release's own code
with the manuscript family excluded.

## Reproducibility

Every entry point seeds numpy, random and torch from the run's seed, and the pool builder is
a pure function of its specification and seed; building it twice yields identical devices,
which is checked. The atomic writers replace rather than append, produce mode 0644 files and
leave no temporary file behind. Two consecutive runs of the verification driver produce
byte-identical artefacts; the manifest test in `tests/` can only pass once the previous
run's artefacts are on disk, so run the driver twice with no edits in between before
treating a report as final.

## Licence and provenance

The release is the instrument group's own work, under the MIT licence in `LICENSE`.
`NOTICE` names every third-party method description implemented here, every runtime package
with its real licence and source address, and every public resource or catalogue identifier
the article points at. No third-party code, data, weights or material is bundled.
