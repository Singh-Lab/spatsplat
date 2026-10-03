# SpatSplat

SpatSplat is a refactored research codebase for spatial transcriptomics reconstruction and transfer experiments. The reusable Python library lives in `spatsplat/`; paper-specific dataset loaders and notebook support code live in `paper/` and are intentionally not packaged as the public library.

## Repository Layout

```text
spatsplat/          Reusable library code
spatsplat/baselines MLP, Fourier MLP, KNN, and mean baselines
paper/data_loaders  Dataset-specific loaders used by the notebooks
notebooks/          Final analysis notebooks
scripts/            Data acquisition and setup utilities
figures/            Notebook figure exports
```

## Setup

Create or update the environment from the project root:

```bash
conda env create -f environment.yml
conda activate spatsplat
pip install -e .
```

Copy the environment template and set `SPATSPLAT_DATA_DIR` before running notebooks or download scripts:

```bash
cp .env.example .env
```

Edit `.env` for your machine:

```bash
SPATSPLAT_DATA_DIR="/path/to/spatsplat/data"
```

`.env` is intentionally ignored by git because it contains machine-local paths. Dataset loaders expect files under `SPATSPLAT_DATA_DIR`, usually in either `raw/<dataset_name>` or `processed/<dataset_name>`.

## Notebooks

The current notebooks are in `notebooks/`:

- `reconstruction-xenium-mouse_brain.ipynb`
- `exploratory-xenium-mouse_kidney_iri.ipynb`
- `conditional_field_synthesis-xenium-mouse_kidney_iri.ipynb`
- `geometry-xenium-mouse_brain.ipynb`
- `spatial_holdout-xenium-mouse_brain.ipynb`
- `pseudobatch_alignment-xenium-mouse_brain.ipynb`
- `reconstruction-merfish-mouse_brain.ipynb`
- `multislice_registration-merfish-mouse_brain.ipynb`
- `volumetric_reconstruction-allen_merfish-mouse_brain.ipynb`
- `splat_arrangement-allen_merfish-mouse_brain.ipynb`
- `splat_arrangement_latent-allen_merfish-mouse_brain.ipynb`
- `splat_arrangement_deterministic-allen_merfish-mouse_brain.ipynb`
- `conditional_field_synthesis-allen_merfish-mouse_brain.ipynb`
- `conditional_field_loso-allen_merfish-mouse_brain.ipynb`
- `splat_section_matching-embryonic_mouse_brain.ipynb`
- `cohort-zhuang_merfish-mouse_brain.ipynb`
- `conditional_field_synthesis-zhuang_merfish-mouse_brain.ipynb`
- `reconstruction-zhuang_merfish-mouse_brain.ipynb`
- `diagnostic-program_preservation-zhuang_merfish-mouse_brain.ipynb`
- `reconstruction-merfish-uterine_cancer.ipynb`
- `maturity_field-visium-embryonic_mouse_brain.ipynb`
- `reconstruction-nsclc_cosmx-lung_cancer.ipynb`
- `family_transfer-pdac_mets_visium-pancreatic_cancer.ipynb`

Notebook outputs are written to repo-local folders:

- Figures: `figures/`
- Checkpoints and arrangement experiment artifacts: `notebooks/checkpoints/`

The deterministic arrangement notebook reuses the registered arrangement artifacts via `SOURCE_RUN_ID`, without another notebook session. It trains a latent-plus-amplitude geometry regressor across training sections, selects checkpoints on a fixed validation subset, and reports full-section controls. Test evaluation is disabled by default (`RUN_TEST = False`). Historical flow results are optional comparisons, not training inputs. Its outputs use separate `splat_arrangement_deterministic` checkpoint and figure directories.

`conditional_field_synthesis-allen_merfish-mouse_brain.ipynb` tests depth-conditioned field generation using a frozen training autoencoder. It predicts complete splat parameters and trains against rendered expression, without per-section molecular inventories or reference geometry targets. It reuses only the source cohort configuration/coordinate metadata and autoencoder, with separate outputs under `conditional_field_synthesis`. Baselines include training-mean expression, spatial KNN, coordinate MLPs, and the existing volumetric splat field. Test evaluation is disabled by default.

`conditional_field_loso-allen_merfish-mouse_brain.ipynb` evaluates conditional generation across all seven retained pilot sections using fixed five-training/one-validation/one-test folds. Each fold refits coordinate normalization, the autoencoder, splat initialization, and diagnostic programs using its training sections only. It compares conditional and depth-independent splats, coordinate MLPs, KNN, and constant expression; reports interpolation and endpoint extrapolation separately; and averages seeds before sections. Completed fits are cached under `notebooks/checkpoints/conditional_field_loso/`, with figures under root `figures/conditional_field_loso-allen_merfish-mouse_brain/`. The full default run fits seven autoencoders and 56 fields. It needs source cohort metadata, not existing autoencoder or reference-field checkpoints.

`splat_section_matching-embryonic_mouse_brain.ipynb` explores matching the four local embryonic brain samples using independently fitted splat fields and bounded affine registration. It uses a shared decoder/gene panel, checks observed-expression agreement and support overlap, compares direct observed-map registration, and reports fitting-seed/direction variability and distortion flags. It is exploratory cohort matching, not held-out prediction; independent specimen identities remain unverified. Outputs use separate `splat_section_matching` checkpoint and figure directories.

## Data Acquisition

Some datasets have scripted download paths. Others require manual access because the public landing page does not expose a stable direct object URL suitable for a reproducible shell script.

### Scripted Datasets

Run scripts from the project root after configuring `.env`.

#### Xenium Mouse Brain Tiny Dataset

Used by the Xenium reconstruction, spatial holdout, and pseudobatch notebooks.

```bash
bash scripts/download_tiny_dataset.sh
```

Expected loader path:

```text
$SPATSPLAT_DATA_DIR/raw/tiny_dataset/
```

#### CosMx NSCLC

Used by `reconstruction-nsclc_cosmx-lung_cancer.ipynb`.

```bash
bash scripts/download_cosmx_nsclc_dataset.sh
```

Expected loader path:

```text
$SPATSPLAT_DATA_DIR/raw/cosmx_nsclc/nanostring_nsclc/
```

#### Embryonic Mouse Brain Atlas

Used by `maturity_field-visium-embryonic_mouse_brain.ipynb`.

```bash
bash scripts/download_embryonic_mouse_brain_atlas.sh
```

Expected loader path:

```text
$SPATSPLAT_DATA_DIR/raw/embryonic_mouse_brain_atlas/
```

#### PDAC Metastatic Visium

Used by `family_transfer-pdac_mets_visium-pancreatic_cancer.ipynb`.

Download the Zenodo RDS object:

```bash
bash scripts/download_pdac_mets_visium_zenodo_rds.sh
```

Then export the processed files used by the Python loader:

```bash
Rscript scripts/extract_pdac_mets_visium_from_rds.R
```

The extractor requires R with `Seurat` and `Matrix`; these are included in `environment.yml` for conda-based setup.

Expected paths:

```text
$SPATSPLAT_DATA_DIR/raw/pdac_mets_visium_gse272362/PDAC_Updated.rds
$SPATSPLAT_DATA_DIR/processed/pdac_mets_visium_gse272362/
```

### Mouse Kidney Injury and Repair (Xenium)

`exploratory-xenium-mouse_kidney_iri.ipynb` is the first stage of a conditional-synthesis investigation: inspect the cohort, verify a shared regional autoencoder, and fit independent splat fields before building an emitter or blending conditions. It does not depend on another notebook or its checkpoints.

The [published study](https://www.nature.com/articles/s41467-025-62599-9) provides a [processed AnnData archive](https://doi.org/10.6084/m9.figshare.28761695.v1), pinned here to Figshare file 53543495 with byte-count and MD5 checks. It contains 1,374,915 measurements, 299 retained measured genes, and 12 left/right kidney samples across sham, 4 hours, 12 hours, 2 days, 14 days and 6 weeks. The methods identify six animals: kidneys within a time point are paired samples, not independent biological replicates. Animal and state are confounded. The raw Xenium archive is also available as [GSE269719](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE269719), but the approximately 1.89 GiB processed file is sufficient for this notebook.

```bash
# Inspect the destination; no download by default.
python scripts/download_xenium_kidney_iri.py
python scripts/download_xenium_kidney_iri.py --download
```

The downloader writes to `$SPATSPLAT_DATA_DIR/raw/xenium_kidney_iri/Xenium.h5ad`. The loader reads measured integer counts from `raw/X`, never normalized `X`, Harmony coordinates or imputed expression. Targets are per-measurement `log1p(counts)` averaged over eligible 50/100/200 um regions; genes absent in one sample remain in the shared measured panel. All QC-passing observations are retained. The default reconstruction pilot is `ShamL`, `Day2L`, `Week6L`; set `FIT_SAMPLES=list(SAMPLES)` for all twelve. The notebook inventories all samples regardless of the fit subset.

First inspect the data, then enable `RUN_AE=True`, inspect in-sample regional AE fidelity, and enable `RUN_FIELDS=True`. Both stages load matching completed checkpoints by default; missing checkpoints never start training. Force flags explicitly restart training. Each independent field uses the common `AmplitudeSplatField`/`SplatConfig`, with configurable splat count and sharpness and a frozen shared decoder. These are full-fit regional reconstruction diagnostics, not held-out prediction or evidence of generalizable disease progression. Emitter, occupancy and blending are deferred until the reconstruction gate is satisfactory.

Outputs use `$SPATSPLAT_DATA_DIR/processed/xenium_kidney_iri/`, root `figures/xenium_kidney_iri/`, and `notebooks/checkpoints/xenium_kidney_iri/`.

The next-stage `conditional_field_synthesis-xenium-mouse_kidney_iri.ipynb` loads a completed independent run (default `e29bc647aaa3`, sharpness 1.0). It trains one categorical specimen-conditioned emitter with the same shared frozen decoder, 1,024 splats per field, regional objective and 5,000 updates per specimen. Independent fields are comparison references, not targets for matching splat positions. Set `RUN_EMITTER=True` for the first fit; a completed matching checkpoint is reused unless `FORCE_EMITTER_RETRAIN=True` is also set. The notebook reads source artifacts directly without executing the source notebook. For a new reconstruction run, set `SOURCE_RUN` to its run ID.

This is an in-sample endpoint-capacity experiment, not progression modeling or held-out validation. It compares exact regional averages, marker maps and fixed shared-state fractions. Physical coordinate frames are supplied. Emitter artifacts use `$SPATSPLAT_DATA_DIR/processed/kidney_iri_synthesis/`, root `figures/kidney_iri_synthesis/`, and `notebooks/checkpoints/kidney_iri_synthesis/` with provenance-keyed run directories.

The joint occupancy section warm-starts the completed emitter without changing its checkpoint. Inspect the coordinate-derived tissue/exterior/unknown support proxies, then set `RUN_OCCUPANCY=True`; leave both force-retrain flags false for normal cache reuse. Expression and occupancy share condition embeddings and train together, with the decoder frozen. Expression metrics remain unmasked. The section exports occupancy probabilities, binary masks, raw and masked canvas expression, and separate center/perimeter exterior diagnostics. Enclosed uncertain gaps are not treated as confirmed background; validation canvas blocks measure proxy agreement, not generalization to new specimens. Joint artifacts use the analogous `processed/kidney_iri_occupancy/`, root `figures/kidney_iri_occupancy/`, and `notebooks/checkpoints/kidney_iri_occupancy/` directories.

Center placement uses a fixed positive-support distance map instead of the earlier 150 um allowance. The mean squared physical distance is normalized by `center_distance_scale_um=50`, with `center_support_weight=0.1` ramped over 200 rounds. This is a soft training constraint, not post-hoc projection or an inference-time reference mask. Unknown gaps remain unlabeled for occupancy but are not preferred center locations. The full reference support anchors geometry, including monitoring blocks, so occupancy block metrics are monitoring diagnostics rather than independent held-out validation. Training also reports centers outside predicted occupancy; expression metrics remain ungated.

Kidney synthesis regional maps compare observations, independent splats and the completed joint expression-and-occupancy model. Primitive plots compare independent and joint splats on shared axes, using the joint model's predicted occupancy as a common background (independent fields do not emit occupancy). The size table counts the complete joint model. These results are written under the joint `kidney_iri_occupancy` run directories. The expression-only emitter is a warm-start dependency, not a fallback plotting model. Plotting changes do not invalidate trained checkpoints.

The final exploratory blending section interpolates learned specimen embeddings, defaulting to ShamL to Day2L at fractions 0, 0.25, 0.5, 0.75 and 1. Expression and occupancy read the same interpolated embedding; physical scale/origin metadata are interpolated explicitly. No training, emitted-splat averaging or endpoint-image averaging occurs. Endpoint parameters and rendered probes are checked for exact reproduction. Figures share marker color limits and report tissue area, connectivity, outside-center fractions and generated-canvas regional-state proportions. Blends are deterministic prototype transitions, not inferred disease progression or samples of biological variability. Arrays and manifests are saved under the joint run's `blends/<blend_id>/`; figures remain in the root joint figures directory. Existing training checkpoints are unchanged.

The following fixed-scaffold section keeps the chosen endpoint's occupancy, centers, radii, rotations and physical frame unchanged while blending only emitted latent profiles and amplitudes. `SCAFFOLD_SAMPLE` defaults to the left endpoint and may be switched to the right endpoint. The matching anchor fraction reproduces exactly; the opposite fraction is a molecular transplant onto the fixed scaffold, not a reconstruction of the other specimen. Shared slot indices need not have matching anatomical roles. Geometry and occupancy invariance are checked across fractions, with separate plots and exports under `scaffold_blends/<id>/`. No fitting or slot matching occurs, and checkpoint keys are unchanged.

### Allen Brain Cell Atlas MERFISH Cache

Used by:

- `multislice_registration-merfish-mouse_brain.ipynb`
- `volumetric_reconstruction-allen_merfish-mouse_brain.ipynb`

The independent `splat_arrangement-allen_merfish-mouse_brain.ipynb` experiment also uses this cache. It fits multiple 2D reference splat sets, clusters training-derived molecular programs, and learns a depth-conditioned flow for arranging a supplied program inventory. It does not load the volumetric notebook or its checkpoints.

`splat_arrangement_latent-allen_merfish-mouse_brain.ipynb` is a separate follow-up using compact frozen-encoder features and identity-preserving molecular/geometry pairs. It requires a completed registered arrangement artifact directory (set `SOURCE_RUN_ID`), not a running notebook session. It reuses the saved autoencoder and reference fields, verifies the original observation sample, and compares program-only, program-plus-amplitude, and profile-plus-amplitude conditioning. Existing decoded profiles are re-encoded as features, not treated as recovered fitted latents. Start with the single-section memorization diagnostic before the held-out comparison. Its checkpoints are stored separately under `notebooks/checkpoints/splat_arrangement_latent/`; figures go under root `figures/`.

These notebooks use `abc_atlas_access` and expect the Allen Brain Cell Atlas cache under:

```text
$SPATSPLAT_DATA_DIR/raw/abc_atlas/
```

You can override this location in `.env` if the cache already exists elsewhere:

```bash
SPATSPLAT_ABC_ATLAS_DIR="/path/to/abc_atlas"
```

The notebook calls `AbcProjectCache.from_cache_dir(...)` on this path. Install dependencies with `environment.yml`, then populate the cache using the standard `abc_atlas_access` workflow before running the notebook.

### Zhuang Multi-Donor MERFISH Preparation

`cohort-zhuang_merfish-mouse_brain.ipynb` audits the two coronal donors, Zhuang-ABCA-1 and Zhuang-ABCA-2, before selecting corresponding anatomical levels. The loader is `paper/data_loaders/zhuang_mouse_brain.py`. It pins `releases/20260415/manifest.json`, checks shared gene identifiers, joins CCF coordinates by observation ID, reports registration coverage, and proposes nearest-depth candidates for visual review. It does not train a generator or automatically accept anatomical matches.

Raw assets use the same `$SPATSPLAT_DATA_DIR/raw/abc_atlas/` cache and optional `SPATSPLAT_ABC_ATLAS_DIR` override as the other Allen notebooks. Derived inventories and sampled expression exports go to `$SPATSPLAT_DATA_DIR/processed/zhuang_brain_cohort/`. Figures go to the repository root `figures/zhuang_brain_cohort/` directory. The notebook enables metadata downloads but leaves expression downloads disabled until reviewed pairs are explicitly selected.

Optional command-line download stages (run in the spatsplat environment):

```bash
python scripts/download_zhuang_mouse_brain.py --stage genes
python scripts/download_zhuang_mouse_brain.py --stage plan
python scripts/download_zhuang_mouse_brain.py --stage metadata
python scripts/download_zhuang_mouse_brain.py --stage expression
```

The genes stage also acquires the pinned manifest when starting a fresh cache. Existing assets are reused. Metadata plus CCF coordinates require about 1.2 GB; the default log2 expression matrices add about 3 GB. Expression files are donor-wide, even when extracting a small section subset. Use `--expression-kind raw` only when raw counts are intended; the loader never silently converts between count and log2 targets. Two donors remain two biological replicates, not one replicate per section.

The independent `conditional_field_synthesis-zhuang_merfish-mouse_brain.ipynb` reconstructs three selected anatomical levels across the two Zhuang donors, then interpolates learned donor embeddings. It loads expression directly through the same loader/cache (downloads enabled by default), performs restricted geometry-only rigid alignment, trains a shared autoencoder, and supervises a conditional generator through rendered expression rather than reference splat positions. All six sections are training examples; intermediate fields are exploratory interpolation, not validated new-animal samples. Published log2 expression is used unchanged. Figures include shared-scale gene maps, consistently colored expression programs, and unmasked splat geometry; interpolation display support is explicitly derived from the endpoint observations, not generated anatomy. Outputs use `processed/zhuang_conditional_synthesis/` under the data directory, root `figures/zhuang_conditional_synthesis/`, and `notebooks/checkpoints/zhuang_conditional_synthesis/`. Training cells always retrain and never silently load checkpoints.

The original expression experiments described above are now opt-in: set `RUN_EXPRESSION_EXPERIMENT=True` to execute them. Their training cells retrain when enabled. They retain every eligible measurement and compare identically initialized frozen/trainable shared decoders over 100 section-balanced epochs. Composition and total-expression training precede MAE refinement with a frozen continuous geometry mapping. Each variant has separate checkpoints, endpoint metrics, and interpolation figures under the configuration-specific run directory. Legacy independent-field and loss-weight diagnostics are disabled by default; old exported runs remain available.

The primary synthesis workflow now follows the regional-fidelity analysis: leave `RUN_EXPRESSION_EXPERIMENT=False` and run the regional section directly after imports. It warm-starts from the archived frozen-decoder model selected by `REG_SOURCE_RUN`, retaining its frozen geometry while fitting molecular emission to means over 50/100/200 um squares. Predictions are averaged over all measurement coordinates within each selected region, not evaluated only at its center. Sections, scales and eligible regions receive equal sampling weight; a point-level auxiliary loss is optional and defaults to zero. The reference uses the same observed 100 um regions, PCA settings, five regional states and colors as the regional diagnostic. Evaluations compare the unchanged point-objective baseline with the regional fit using matched regional expression, marker edges, program distributions, spatial-shuffle controls and Frechet distances. These are training-endpoint comparisons, not held-out generalization. Set `RUN_REGIONAL_TRAINING=True` for the first fit; otherwise only matching completed checkpoints load. Regional outputs are separate under `processed/zhuang_regional_synthesis/` in the data directory, root `figures/zhuang_regional_synthesis/`, and `notebooks/checkpoints/zhuang_regional_synthesis/`. Interpolation maps show unmasked 100 um area-quadrature averages and do not claim generated anatomical support.

The appended joint-occupancy experiment is disabled by default (`ENABLE_OCCUPANCY=False`). It now requires a completed regional fit, uses that fit as its source, and retains the same multi-scale regional expression objective. To run it, enable occupancy in regional setup and set `RUN_JOINT_OCCUPANCY=True` for the first joint fit. By default it loads a matching completed checkpoint; a missing checkpoint never triggers training automatically. `FORCE_JOINT_RETRAIN=True` together with the training flag explicitly restarts the fit. Shared condition embeddings jointly emit expression residuals and a separate occupancy Gaussian field; unlike the initial regional fit, the new expression geometry residuals can move. Expression supervision and regional evaluation are not gated by occupancy. Before/after regional metrics and maps accompany the secondary measurement-level metrics. Exports retain soft occupancy probabilities and binary masks at the fixed 0.5 threshold, alongside expression splats and fixed-canvas gene maps. Occupancy supervision is a measurement-coordinate proxy, not a histological tissue mask: uncertain bands and enclosed gaps are excluded. Outputs are separate under `processed/zhuang_joint_occupancy/` in the data directory, root `figures/zhuang_joint_occupancy/`, and `notebooks/checkpoints/zhuang_joint_occupancy/`.

`reconstruction-zhuang_merfish-mouse_brain.ipynb` provides a separate Zhuang reconstruction benchmark using all eligible measurements (no subsampling cap) and the common gene panel. Each of six sections has its own AE and random 70/15/15 measurement split. Frozen/trainable AE-splats, MAE and biological-objective MLPs, Fourier MLPs, KNN, budget-matched KNN, and mean predictions use identical splits. AE training and splat initialization exclude held-out profiles. A separate full-section fitting diagnostic defaults to the first section and makes no held-out claim. The notebook uses published log2 targets, the established composition-first/MAE-refinement curriculum, root `figures/zhuang_reconstruction/`, data-directory `processed/zhuang_reconstruction/`, and `notebooks/checkpoints/zhuang_reconstruction/`. It does not load the synthesis notebook or its trained models.

### MERFISH Dataset Notes

The Zhuang reconstruction notebook also has an optional full-section capacity sweep. Its current settings hold the count at 1024 splats and test sharpness 0.001/0.003/0.01/0.03/0.1/0.25/0.5, reusing the archived frozen AE and fixed program prototypes while holding the objective and epoch budget fixed. The 0.5 setting retains the previous comparison point; metric and recall plots use logarithmic sharpness axes. Run its import cell and appended capacity cells with `RUN_CAPACITY_SWEEP=True`; there is no need to repeat the main benchmark. Metrics, program recall, marker recovery, and matched-location maps are saved separately under `zhuang_capacity` in the established processed-data, root-figure, and notebook-checkpoint directories. Configuration-specific directories preserve results from earlier count/sharpness sweeps.

To replot completed capacity sweeps without retraining, run the import cell, capacity settings cell, and final plotting cell, skipping the fitting cell. Plots combine saved runs whose manifests match except for the swept counts and sharpness lists. Duplicate settings prefer the current run, otherwise the newest compatible run. Combined CSVs retain source run IDs and do not overwrite the original per-run tables.

The reconstruction notebook includes saved-model program-preservation diagnostics. Run its import cell and the two code cells in that diagnostic section without rerunning training. They compare KNN, budget-matched KNN, and frozen/trainable splats on identical benchmark test rows across six sections, then separately inspect initial/learned decoded profiles and amplitude-weighted mixing in the 1024-splat, sharpness-0.25 full-fit case. The existing full-cohort dictionary is descriptive, not an independently held-out set of biological labels. Per-section recall, marker recovery, assignment-change margins, profile inventory, and contribution tables are written under `processed/zhuang_program_mechanisms/`; figures use root `figures/zhuang_program_mechanisms/`.

The controlled-initialization section compares two fresh full-section fits at 1024 splats and sharpness 0.25. Only the initial latent profiles change: the existing 16-neighbor profile average versus the individual measurement selected at each center. Geometry, 16-neighbor amplitudes, frozen AE, losses, seed, and training budget are identical. Run the import cell plus that section's setup, fitting, and plotting cells; skip fitting to replot saved results. Outputs use separate `zhuang_initialization_control` directories under processed data, root figures, and notebook checkpoints. Both initial and learned states are saved, and all results are explicitly in-sample.

The training-profile balancing section compares uniform training against capped inverse-square-root frequency weights, starting from identical center-profile fields. Eight expression groups are fitted using training measurements only, independently of the descriptive program dictionary. Weights average one with a maximum 3:1 ratio; only measurement reconstruction losses change, not geometry penalties or minibatch sampling. All evaluation metrics stay unweighted. Run the import cell and this section's setup, fitting, and plotting cells for two fresh fits, or skip fitting to replot. Group/weight audits, program proportions, marker recovery, and initial/final model states use separate `zhuang_profile_balance` directories under processed data, root figures, and notebook checkpoints. The synthesizer is unchanged; this is an in-sample objective test before considering transfer to synthesis.

The stronger-balancing section adds one fresh fit with inverse-frequency weights (exponent 1, maximum ratio 5). It reuses the exact saved groups and initial field from mild run `20d5cc7ab4c8`, verifies data/checkpoint provenance and unchanged training sources, and combines the archived uniform/mild results with the strong result. Run the import cell and the three stronger-balancing cells; skip fitting to replot. Separate configuration-specific directories preserve the earlier runs. All comparisons remain unweighted and in-sample.

The local-distribution section requires no training: run the import cell and its two code cells. It compares KNN and benchmark splats on test measurements separately from the full-fit Uniform/Mild/Strong experiment. Nonoverlapping windows span 5%, 10%, and 20% of the longest section dimension, with at least 20 measurements per window. Means, program proportions, profile diversity, projected Wasserstein distances, and marker distributions are compared with explicitly target-informed local-shuffle and local-mean controls. Scales with fewer than five eligible windows or below 50% measurement coverage remain exported but are omitted from comparison curves. Tables use `processed/zhuang_local_distributions/`, and plots use root `figures/zhuang_local_distributions/`.

The targeted P4 patch-rescue section starts from the learned uniform field in `zhuang_profile_balance/20d5cc7ab4c8`. Run the import cell, inspect the configurable core/margin preview, then run the refit and plotting cells. Only latent profiles are optimized, with all geometry, amplitudes and decoder parameters frozen and checked for exact equality afterward. The original 100-epoch bio-to-MAE curriculum is restarted on all measurements inside the core plus margin, without profile balancing. Marker recovery and spillover are reported separately for observed P4/non-P4 measurements in the core, margin and outside. This is a target-informed in-sample diagnostic, not cross-validation. Figures, processed tables and notebook checkpoints use separate `zhuang_patch_refit` directories; fitting always restarts from the archived field rather than silently loading a refit.

`diagnostic-program_preservation-zhuang_merfish-mouse_brain.ipynb` diagnoses program reassignment from saved Zhuang AEs and predictions without training. Its primary full-section comparison uses identical measurement IDs and a fixed observed-profile dictionary for observed expression, both pretrained AEs, independent frozen/trainable splats, and conditional frozen/trainable fields. It exports assignment proportions, program recall/precision, confusion matrices, matched-coordinate maps, and marker recovery under data-directory `processed/zhuang_program_diagnostics/` and root `figures/zhuang_program_diagnostics/`. Run IDs are explicit; the required archives/checkpoints must already exist. Optional benchmark-test context is separated from the full-fit comparison because conditional models saw those targets.

`diagnostic-p4_capacity-zhuang_merfish-mouse_brain.ipynb` runs two independent checks from saved archives without neural retraining. A fixed-geometry, nonnegative least-squares oracle fits Cldn11/Mog on the selected patch plus margin, bypassing the decoder and relaxing amplitude constraints; coefficient magnitudes and remote predictions expose pathological solutions. Separately, archived splat, KNN, MLP, Fourier MLP and mean predictions are compared on identical test measurements for P4 marker recovery, spillover, contrast and program recall. Whole-section test results are primary because the selected patch has very few test measurements. Run All loads the existing archives and executes both checks; missing archives never trigger model training. Tables and numerical coefficients use data-directory `processed/zhuang_marker_capacity/`, and figures use root `figures/zhuang_marker_capacity/`.

`diagnostic-regional_fidelity-zhuang_merfish-mouse_brain.ipynb` evaluates saved conditional endpoints at 50, 100 and 200 micrometer regional scales without neural retraining. It averages published expression over identical observation locations for observed, synthesized and available independent full-fit fields. A fixed real-training PCA supports regional-expression Frechet distance, alongside matched-region errors, marker correlations and adjacent-region contrast. Real-real donor pairs, a training-mean baseline and a spatial-shuffle oracle provide context. Five broad regional states share one dictionary and legend across every panel; they are not cell-type fractions. All six synthesis endpoints are in-sample, and only the first section currently has independent full-fit archives. Coverage and minimum-region checks prevent unsupported Frechet scores. Run All writes tables and evaluation references under data-directory `processed/zhuang_regional_fidelity/` and plots under root `figures/zhuang_regional_fidelity/`; no neural checkpoint is changed.

The MERFISH loaders currently expect manually acquired Vizgen-style flat files in `SPATSPLAT_DATA_DIR`. Public landing pages exist for the datasets below, but direct scripted download URLs are not yet part of this repository. Once a stable direct URL and exact object layout are confirmed, add a dedicated script in `scripts/` using the same `.env` and idempotent-download pattern as the existing dataset downloaders.

#### Mouse Brain Receptor Map

Used by:

- `reconstruction-merfish-mouse_brain.ipynb`
- `multislice_registration-merfish-mouse_brain.ipynb`

Landing pages:

- https://info.vizgen.com/mouse-brain-data
- https://info.vizgen.com/mouse-brain-map

Expected loader directory:

```text
$SPATSPLAT_DATA_DIR/raw/merfish-mousebraindata/
```

Expected files for a sample such as `S1R1`:

```text
cell_by_gene_S1R1.csv
detected_transcripts_S1R1.csv
cell_metadata_S1R1.csv
```

Recommended workflow:

1. Open the Vizgen landing page in a browser.
2. Complete any required registration or access flow.
3. Download the relevant sample files.
4. Place the files in `$SPATSPLAT_DATA_DIR/raw/merfish-mousebraindata/` using the names expected by the loader.

#### Human Uterine Cancer Protein Co-detection Showcase

Used by `reconstruction-merfish-uterine_cancer.ipynb`.

Landing page:

- https://info.vizgen.com/protein-co-detection-showcase

The showcase lists access for the CD45RA and CD45RO uterine cancer datasets. The current notebook uses the CD45RA co-stain dataset; Vizgen refers to this access target as `HumanUterineCancerPatient2-RACostain`, while the local loader expects the shorter folder name below.

Expected loader directory:

```text
$SPATSPLAT_DATA_DIR/raw/merfish-uterinecancer-p2-RACstain/
```

Expected files:

```text
cell_by_gene.csv
detected_transcripts.csv
cell_metadata.csv
```

Recommended workflow:

1. Open the Vizgen protein co-detection showcase page in a browser.
2. Access the `HumanUterineCancerPatient2-RACostain` / CD45RA dataset through the provided dataset access link.
3. Download the cell-by-gene, detected transcript, and cell metadata files.
4. Place them in `$SPATSPLAT_DATA_DIR/raw/merfish-uterinecancer-p2-RACstain/` with the filenames expected above.
