# Standalone Publication Benchmark With Additional Controls

Entry point: `notebooks/registration_publication_controls_reproducible-allen_barseq.ipynb`.
Run from a fresh kernel, top to bottom. No other notebook, saved result table, or
fitted model is required. The previous publication notebook is unchanged.

## Plotting Without Rerunning the Experiment

Run the new **Export plotting data** cell once after scoring. If `cases`, `METHODS`,
and `OUTPUT` are still in the kernel, run only that export cell. There is no need
to repeat setup, fitting, registration, or evaluation. The exporter reads the
existing report's saved metrics and maps and resolves CCF division labels once.
If the kernel was reset, the setup/data-loading cells can reconstruct `cases`;
set `PLOT_REPORT` explicitly to the completed run's existing report directory.
Do not run the fitting or scoring cells just to export a completed report.

Then open `notebooks/registration_publication_plots-allen_barseq.ipynb`.
Choose the exported bundle (or use the displayed latest complete bundle), load
its dataframes, and rerun only the figure cells while editing. Plotting does not
load datasets, trained models, another notebook, or the atlas hierarchy, and does
not require PyTorch, CAST, PASTE, or PASTE2. It uses NumPy, pandas, SciPy, and Matplotlib.

Each `reports/plot_data/<timestamp>/` bundle contains:

- `scores.csv.gz`: full-precision metrics by case and method.
- `availability.csv.gz`: method completeness, including missing methods.
- `cases.csv.gz`: dataset, task, role, section, coverage, and direction.
- `measurements.csv.gz`: original coordinates and resolved division labels.
- `registered.csv.gz`: registered query coordinates for each saved method.
- `timings.csv.gz`: recorded fresh runtimes with case metadata.
- `manifest.json`: method roster, configuration, hardware, and source provenance.

CSV files load directly into pandas; no pickle or model checkpoint is involved.
Coordinates and scores retain full numerical precision. Four-significant-digit
display formatting does not change saved numbers. The manifest is written last,
so incomplete exports are not offered by the plotting notebook. Every export has
a new directory; earlier snapshots remain intact. The original report path is
provenance only, so the entire bundle is portable.

Figure settings include example method, maximum links, size, font, title
overrides, and optional cosine panels. The plotting cells reload the helpers so
layout-code edits take effect immediately. Figures go under root `figures/`.
The separate plotting-only commit manifest is
`docs/registration_plotting_files.txt`; the experiment remains independently
runnable without that optional notebook.

## Methods and Tasks

Both Allen MERFISH and BARseq use ten methods: Unregistered, Geometry rigid ICP,
PASTE rigid, CAST affine/non-affine, Splat rigid/affine/non-affine, PASTE2 rigid,
and MLP-MAE rigid. Settings match the controls experiment; the added methods
also require new Allen fits rather than silently omitting them from Allen plots.

### PASTE2 Inner-Solver Repair

PASTE2 1.0.1 reports a generic dummy-point error when its inner EMD solver warns.
The natural Allen cases hit that solver's one-million-iteration limit. The local
adapter now retries only that warning at five million, then twenty million
iterations. Other failures remain errors. Successful transport must have finite,
nonnegative weights and valid marginals; existing outer objective/convergence
audits remain enabled. No evaluation targets select the solver budget.

For this repair, restart the kernel and run top to bottom with
`IMPORT_COMPLETED_FITS=True`. Exact compatible fits can be copied from the prior
private publication workspace. The two failed cases can also reuse their saved
GLM-PCA costs and completed overlap candidates; their status remains incomplete
until fitting actually succeeds. Existing fits and reports are not deleted.
Resumed elapsed time is not reported as a fresh end-to-end fit time.

Allen development is pair 24/25; evaluation is pairs 28/29 and 32/33. Both natural
alignment and the combined rotation/smooth-deformation recovery task are included.
BARseq development is section 19; evaluation is sections 15, 23, and 27. Shared
parent tissue fractions are 100%, 60%, and 20%, with three crop directions only
for the partial-coverage cases. The existing training/test split, transformations,
sample caps, and scoring rules are unchanged. There are 34 cases and 340 scores.
Allen is transductive; BARseq uses held-out shared-region measurements. Do not
pool these evaluation designs or treat crop directions as independent specimens.

## Setup

1. Install the library and the dependencies in `requirements-registration.txt`
   and `requirements-registration-controls.txt` in the same environment.
2. Copy `.env.example` to `.env` and configure an absolute `SPATSPLAT_DATA_DIR`.
3. Run `scripts/download_registration_allen.py` and `scripts/download_barseq_pilot.py`
   according to their documented options. The notebook validates local paths.
4. Run this notebook. All fitting switches default to enabled.

The exact files to commit are listed in
`docs/registration_publication_controls_files.txt`. Checkpoints, figures, raw
data, `.env`, and other notebooks are deliberately not prerequisites.

## Cache Isolation

All fit outputs live under
`notebooks/checkpoints/registration_publication_controls/<ID>/private_fits/`.
The existing runners keep their internal directory structure beneath this private
root; source fingerprints still refer to the actual repository implementation.
No shared-cache model is written, deleted, moved, or linked by this notebook.
Reports are in the sibling `reports/` directory. Publication figures are under
`figures/registration_publication_controls/<ID>/`. Native diagnostic figures are
routed there too through a figure-directory link, never a model-cache link.

`IMPORT_COMPLETED_FITS=True` optionally copies exact-key completed fits from the
current benchmark/controls caches. The sources remain unchanged. Incomplete
fits are skipped. A clean checkout with no source caches trains normally.
Imports record their source path and completion-file checksum. All scores are
recomputed locally; another notebook's scoring output is not used.

To benefit fully from an evaluation currently running, wait until it completes
before running the import cell. Repeating the import cell can pick up later
completed fits, but never overwrites an existing private fit directory.
Set `IMPORT_COMPLETED_FITS=False` to disable copying; this does not discard
already-existing private fits. Existing private copies continue to work if the
original experiment's caches are deleted.

## Runtime

To fill missing evaluation runtimes, run
`notebooks/registration_timing_completion-allen_barseq.ipynb` top to bottom in the
original environment. It selects the latest complete exported plotting bundle
(or an explicit `BUNDLE_PATH`), checks hardware and implementation compatibility,
and reruns only stages without a completed fresh time. It retains existing
PASTE2/MLP measurements. A missing PASTE2 time requires a full fresh fit, not a
resumed overlap search. No scored publication fit is overwritten.

Fresh attempts live in `notebooks/checkpoints/registration_timing_completion/`.
Completed timing stages are skipped on restart; unfinished attempts are restarted
in empty directories. Failures remain unknown and are listed in `failures.csv`.
The export cell writes a new portable plotting bundle with byte-identical scores
and spatial-map files, plus updated timing data and provenance. Reload that bundle
in the plotting-only notebook to update the runtime figure. Development timing
gaps are intentionally not rerun because the runtime figure uses evaluation cases.

The main registration figure retains the compact measurement-correspondence
layout. Runtime is a separate figure, not another crowded main-figure panel.
An optional expression-score supplement contains coverage-weighted cosine.

Timings distinguish fresh completed work, resumed work, failed attempts, and
cache loading. Historical fit times are unknown unless actually recorded. A
cached fit is not assigned a zero-second training time. Valid source timing logs
are copied only when configuration, package versions, source fingerprints, case,
and the corresponding imported fit-cache path match. Event imports are deduplicated.

PASTE2 includes preprocessing, numerical retries, overlap selection, and final
alignment. CAST/Splat family totals include shared preparation and both affine
variants. The Splat rigid pose stage excludes field fitting. MLP fields and rigid
pose are separate stages. Evaluation, reconstruction diagnostics, and plotting
are not included. These are workload-stage timings, not identical standalone
method budgets. The family total cannot be charged to each splat variant as if
the work were repeated independently.

Runtime plots use evaluation cases only and keep Allen and BARseq separate.
The tables retain development cases separately. Dots are cases, diamonds are
medians, and lines span observed minima/maxima; they are not confidence intervals.
The log retains hardware metadata. This protocol runs PASTE/PASTE2 and CAST on
CPU, and MLP/splats on the configured device. Do not claim hardware-matched
algorithmic speedups or pool measurements from different hardware silently.

## Interpretation

The same ten methods appear in all quantitative panels. Missing fits block the
publication figure rather than dropping a method or substituting an unregistered
map. Development examples do not enter the evaluation aggregates. PASTE2 overlap
is estimated from training data, not supplied from known crop fractions. Its
numerical stabilization is documented in `docs/barseq_additional_controls.md`.
MLP uses the shared rigid pose objective, but its field training objective differs
from splats, so this is not a pure representation-only ablation.
