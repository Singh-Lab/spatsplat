# SpatSplat

SpatSplat is a refactored research codebase for spatial transcriptomics reconstruction and transfer experiments. The reusable Python library lives in `spatsplat/`; paper-specific dataset loaders and notebook support code live in `paper/` and are intentionally not packaged as the public library.

## Publication Registration Benchmark

Use [the self-contained Allen/BARseq notebook](notebooks/registration_publication_reproducible-allen_barseq.ipynb)
to prepare cases, fit all eight methods, evaluate them, and produce the compact
publication figure. No other notebook or existing checkpoint is required.
Follow the [registration-specific setup and data instructions](docs/registration_reproduction.md)
first; the general environment below does not include all CAST/PASTE dependencies.
The [commit file list](docs/registration_publication_files.txt) identifies its
required source, download helpers, configuration, and documentation.

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
- `spatial_holdout-xenium-mouse_brain.ipynb`
- `pseudobatch_alignment-xenium-mouse_brain.ipynb`
- `reconstruction-merfish-mouse_brain.ipynb`
- `multislice_registration-merfish-mouse_brain.ipynb`
- `volumetric_reconstruction-allen_merfish-mouse_brain.ipynb`
- `reconstruction-merfish-uterine_cancer.ipynb`
- `maturity_field-visium-embryonic_mouse_brain.ipynb`
- `reconstruction-nsclc_cosmx-lung_cancer.ipynb`
- `family_transfer-pdac_mets_visium-pancreatic_cancer.ipynb`

Notebook outputs are written to repo-local folders:

- Figures: `figures/`
- Long-running uterine cancer checkpoints: `notebooks/checkpoints/`

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

### Allen Brain Cell Atlas MERFISH Cache

Used by:

- `multislice_registration-merfish-mouse_brain.ipynb`
- `volumetric_reconstruction-allen_merfish-mouse_brain.ipynb`

This notebook uses `abc_atlas_access` and expects the Allen Brain Cell Atlas cache under:

```text
$SPATSPLAT_DATA_DIR/raw/abc_atlas/
```

You can override this location in `.env` if the cache already exists elsewhere:

```bash
SPATSPLAT_ABC_ATLAS_DIR="/path/to/abc_atlas"
```

The notebook calls `AbcProjectCache.from_cache_dir(...)` on this path. Install dependencies with `environment.yml`, then populate the cache using the standard `abc_atlas_access` workflow before running the notebook.

### MERFISH Dataset Notes

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

## Natural Volumetric Registration

`notebooks/registration_natural-volumetric_allen_merfish.ipynb` compares natural
slice alignment against the existing Allen pairwise methods, without artificial
distortions. It uses sections 24/25/26 for development and 28/29/30 plus 32/33/35
for evaluation. Two fixed flanks train a frozen volumetric field; the middle
section supplies expression to fit a rigid in-plane pose. Zero and 25-degree
splat-tilt limits are compared. Section 34 is absent in this donor; 32/33/35 keeps
the actual unequal z gaps (0.2 and 0.4) and is reported separately. The extra flank is explicitly additional input,
not a budget- or information-matched advantage attributed solely to 3D geometry.

The notebook reads sampling settings and baseline configurations from the original
Allen benchmark's `frozen_protocol.json`. Compatible pairwise fits are reused;
enable the corresponding `RUN_PAIRWISE` flag if an exact cache is missing.
`RUN_VOLUMETRIC=True` trains missing volumetric artifacts. Start with
`PHASE='development'`, inspect results, set `FREEZE_DEVELOPMENT=True`, and execute
the final freeze cell. Then use `PHASE='evaluation'` without tuning on those pairs.
Primary CCF agreement, observed-expression cosine, and coverage use the original
pairwise scoring routines and measurement samples. Coordinate displacement is
only a drift diagnostic, not recovery accuracy.

Data use the existing ABC Atlas directory. New checkpoints are under
`notebooks/checkpoints/allen_volumetric_natural/`, and PNG/PDF figures are under
`figures/allen_volumetric_natural/`. Plot-only notebook edits do not invalidate
the fitting caches. No other notebook needs to run in the same kernel, but the
original benchmark's frozen manifest is required.

The natural-registration notebook now compares `sharing_weights=[0., 0.05]`
at each tilt limit, with the splat budget fixed at 1,024. The soft sharing term
penalizes a splat whose minority flank supplies less than 20% of its combined
mean responsibility, weighted by detached usage. It does not force 50/50 support.
Only flanking training measurements enter this term. Dense responsibilities
provide gradients beyond top-k support; reconstruction still uses the original
top-k readout. Train/validation diagnostics report actual top-k shared usage,
dense/sparse sharing penalties, boundary accumulation, and z widths.
Field checkpoints still use validation reconstruction alone. Natural CCF and
expression scores, not shared support alone, determine whether the prior helps.

This is a new development protocol and cache identity: missing volumetric variants
train with `RUN_VOLUMETRIC=True` without forcing retraining of cached variants.
Compatible pairwise caches are searched across previous natural-registration runs.
Existing frozen protocols are not overwritten; archive an old natural-registration
freeze explicitly before freezing the new protocol after development review.
