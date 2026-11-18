# Reproducing the publication registration figure

## Entry point

Run `notebooks/registration_publication_reproducible-allen_barseq.ipynb` from a
fresh kernel, top to bottom. It prepares the data, fits all eight methods,
evaluates the 34 cases, and produces the compact A-G publication figure. It does
not execute or read any other notebook and does not require pre-existing model
checkpoints. It uses version-controlled Python helpers in `paper/` and `spatsplat/`.

The original development notebook is preserved. The new notebook defaults to
the accepted compact figure; alternative summaries and the expanded cosine
supplement are optional. Raw observations, annotations, gene panels, and training
settings are unchanged. Shared fit caches are reused if compatible; with an empty
cache, the notebook trains everything itself.

## Environment

The repository-wide `environment.yml` is not sufficient by itself: it does not
install the registration-specific CAST, DGL, and PASTE stack. Use the focused
requirements file, or install it into an appropriate existing Python 3.11
environment. These direct versions were recorded from the working environment;
installation into a completely fresh environment has not yet been tested.

Example setup from the repository root:

```bash
conda create -n spatsplat-registration python=3.11 pip
conda activate spatsplat-registration
# Install a PyTorch 2.10.0 build compatible with your GPU/driver first.
# The development environment uses torch 2.10.0+cu128; CPU is also supported.
python -m pip install -r requirements-registration.txt
python -m pip install -e . --no-deps
python -m ipykernel install --user --name spatsplat-registration --display-name 'spatsplat registration'
python -m pip check
```

Select that kernel in Jupyter/VS Code. CAST and PASTE run on CPU in this protocol,
including when CUDA is available; splat fitting uses CUDA when available. CPU
splat fitting is possible but can be much slower. CUDA/PyTorch compatibility is
machine-specific. Do not assume the old repository-wide CUDA 12.1 specification
reproduces the tested GPU environment.

Arial is requested for figures but is not distributed with the repository. If
not installed, Matplotlib falls back to Helvetica or DejaVu Sans. Install an
appropriately licensed font separately if exact typography is required.

## Data setup

Copy `.env.example` to `.env`, then set an absolute writable data directory:

```dotenv
SPATSPLAT_DATA_DIR="/absolute/path/to/registration-data"
```

Do not commit your `.env`. An optional absolute `SPATSPLAT_ABC_ATLAS_DIR` overrides
the default `$SPATSPLAT_DATA_DIR/raw/abc_atlas` directory. The notebook prints and
records the effective paths; other users do not need your directory layout.

Then run:

```bash
python scripts/download_registration_allen.py
python scripts/download_barseq_pilot.py
```

The Allen helper obtains the 638850 section metadata, reconstructed CCF labels,
released log2 expression, and CCF hierarchy through `abc_atlas_access`. These are
the only Allen resources needed here; DLPFC observations are not required.
It saves `registration_download.json` with the selected atlas manifest. To use a
specific release, supply `--manifest <exact-manifest-name>` from that record.
The loaders use the cache's last selected manifest. Do not change that manifest
between data preparation and evaluation. A new empty cache otherwise chooses the
release selected by the installed atlas client, which may differ from an old run.

The BARseq helper downloads its fixed version-1 MAT file and variable dictionary,
checking the recorded file sizes and SHA-256 hashes. Network access and sufficient
storage for the full released expression file are required. The notebook samples
observations after download; it does not download only the sampled observations.

## Running and outputs

Keep every `RUN` entry and `RUN_SPLAT_RIGID` enabled for a clean run. These flags
permit missing fits; they do not force retraining compatible completed caches.
The notebook imports dependencies and checks data configuration before fitting.
Missing/failed methods remain explicit and block the final publication plot.

Expected output: 34 cases x 8 methods = 272 score rows. The real Allen pairs,
controlled recovery cases, and unequal-coverage BARseq cases retain their
development/evaluation roles. Figure selection is fixed, not based on test scores.

- Fit caches: `notebooks/checkpoints/registration_standardized/`.
- New report: `notebooks/checkpoints/registration_publication_reproducible/reports/<ID>/`.
- Publication outputs: `figures/registration_publication_reproducible/<ID>/`.
- Run record: protocol, case inventory, scores, availability, fit status, and
  data/environment provenance in the report directory.

The model cache signature includes source and settings. Changing implementation
files or package versions can invalidate caches. Numerical equality across
hardware, package builds, and atlas releases is not guaranteed by a shared seed.
Archive report CSV/JSON files with a paper release if exact results matter.

## What to commit

`docs/registration_publication_files.txt` lists the specific repository files
needed for this deliverable, including the cloned notebook, helper dependency
closure, data downloaders, focused requirements, documentation, and tests. It is
deliberately explicit so unrelated experimental notebooks do not need to be added.
From the repository root, inspect and stage this list:

```bash
git status --short
while IFS= read -r path; do
    case "$path" in ''|'#'*) continue ;; esac
    git add -- "$path"
done < docs/registration_publication_files.txt
git diff --cached --stat
git diff --cached --check
```

Review already staged changes too; the command does not unstage unrelated work.
No git commit is created automatically.

Important transitive dependencies include `dlpfc_mlp_diagnostics.py` and
`data_loaders/dlpfc_sources.json`, even though this notebook fits neither an MLP
baseline nor DLPFC data. Shared imported modules currently reference them. The
whole `spatsplat` Python source tree is required both for imports and cache-source
fingerprinting, including its baseline subpackage and volumetric module.

Do not commit `.env`, data downloads, model checkpoints, generated figures,
`__pycache__`, or editor settings. No earlier notebook or earlier notebook output
is required. The methods Markdown is included for readers, not executed as code.

## Verification scope

The clone's code is syntax-checked, its local imports/dependency files are audited,
and it is checked in a temporary source-only copy without other notebooks or
checkpoints. Unit tests cover the common scorer, map replay, rigid extension, and
the standalone notebook contract. This is not a complete clean-data rerun:
downloads, model fits, and figure rendering were deliberately not executed while
preparing this deliverable.
