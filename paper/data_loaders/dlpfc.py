"""Pinned PASTE DLPFC inputs; annotations are retained only for audit/evaluation."""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv


SOURCES = json.loads(Path(__file__).with_name('dlpfc_sources.json').read_text())
GROUPS = {
    'I': ('151507', '151508', '151509', '151510'),
    'II': ('151669', '151670', '151671', '151672'),
    'III': ('151673', '151674', '151675', '151676'),
}
DEVELOPMENT_GROUP = 'I'
LAYERS = ('Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'WM')


def dataset_path():
    load_dotenv(Path(__file__).resolve().parents[2]/'.env')
    root = os.getenv('SPATSPLAT_DATA_DIR')
    if not root:
        raise RuntimeError('Set SPATSPLAT_DATA_DIR in the repository .env')
    return Path(root).expanduser()/'raw/dlpfc_paste'


def checksum(path):
    digest = hashlib.md5()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_dataset(*, allow_download=False, root=None):
    """Verify released files; download missing files atomically only on request."""
    root = Path(root) if root is not None else dataset_path()
    for entry in SOURCES['files']:
        path = root/entry['name']
        if path.exists():
            if path.stat().st_size != entry['size'] or checksum(path) != entry['md5']:
                raise ValueError(f'Checksum mismatch: {path}; inspect/remove the damaged file explicitly')
            print('Verified:', path.name, flush=True)
            continue
        if not allow_download:
            raise FileNotFoundError(f'{path} missing. Run scripts/download_dlpfc_dataset.py --download')
        root.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.part')
        print('Downloading:', entry['name'], flush=True)
        try:
            with urlopen(entry['url'], timeout=120) as response, temporary.open('wb') as target:
                for chunk in iter(lambda: response.read(2**20), b''):
                    target.write(chunk)
            if temporary.stat().st_size != entry['size'] or checksum(temporary) != entry['md5']:
                raise ValueError(f'Download checksum mismatch: {temporary}')
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return root


def pair_inventory():
    import pandas as pd
    rows = []
    for group, samples in GROUPS.items():
        for a, b, pair, distance in [(0,1,'AB',10), (2,3,'CD',10), (1,2,'BC',300)]:
            rows.append(dict(donor_group=group, role='development' if group==DEVELOPMENT_GROUP else 'evaluation',
                             pair=pair, reference=samples[a], query=samples[b],
                             separation_um=distance, primary=pair!='BC'))
    return pd.DataFrame(rows)


def load_section(sample, *, root=None, include_annotations=False):
    """Read the author's unchanged count matrix and spatial coordinates.

    Annotation-bearing obs/uns are removed by default. No new gene filtering,
    spot selection, coordinate rescaling, or count transformation is performed.
    """
    import anndata
    import numpy as np
    import pandas as pd
    from scipy import sparse

    sample = str(sample)
    if sample not in {s for group in GROUPS.values() for s in group}:
        raise ValueError(f'Unknown DLPFC section: {sample}')
    root = Path(root) if root is not None else dataset_path()
    path = root/f'{sample}_preprocessed.h5'
    if not path.exists():
        raise FileNotFoundError(f'{path} missing; run scripts/download_dlpfc_dataset.py --download')
    data = anndata.read_h5ad(path)
    if data.obs_names.has_duplicates or data.var_names.has_duplicates:
        raise ValueError('Duplicate spot or gene identifiers')
    if 'spatial' not in data.obsm or 'layer_guess_reordered' not in data.obs:
        raise ValueError('Missing published spatial coordinates or annotations')
    xy = np.asarray(data.obsm['spatial'])
    if xy.shape != (data.n_obs, 2) or not np.isfinite(xy).all():
        raise ValueError('Invalid spatial coordinates')
    counts = data.X.data if sparse.issparse(data.X) else np.asarray(data.X).ravel()
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.rint(counts)).all():
        raise ValueError('Expected unchanged finite nonnegative integer counts in X')
    if (np.asarray(data.X.sum(axis=1)).ravel() <= 0).any():
        raise ValueError('Zero-count spot in published input; audit before filtering')
    labels = data.obs['layer_guess_reordered'].dropna().astype(str)
    if set(labels)-set(LAYERS):
        raise ValueError('Unknown published layer annotation')
    data.uns['dlpfc_source'] = dict(sample=sample, doi=SOURCES['doi'])
    if not include_annotations:
        data.obs = pd.DataFrame(index=data.obs_names.copy())
        data.uns = {'dlpfc_source': data.uns['dlpfc_source']}
    return data


def audit_cohort(root=None):
    """Inventory only: no model fitting, matching, or benchmark scoring."""
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    inventory, layer_rows, gene_sets = [], [], {}
    for group, samples in GROUPS.items():
        subjects = set()
        for position, sample in zip('ABCD', samples):
            data = load_section(sample, root=root, include_annotations=True)
            subject = data.obs.subject.astype(str).unique()
            if len(subject) != 1:
                raise ValueError(f'Multiple subjects in section {sample}')
            subjects.add(subject[0])
            xy = np.asarray(data.obsm['spatial'])
            spacing = cKDTree(xy).query(xy, k=2)[0][:,1]
            labels = data.obs.layer_guess_reordered
            inventory.append(dict(sample=sample, donor_group=group, subject=subject[0], section=position,
                role='development' if group == DEVELOPMENT_GROUP else 'evaluation',
                spots=data.n_obs, genes=data.n_vars, annotated=int(labels.notna().sum()),
                unannotated=int(labels.isna().sum()), median_counts=float(np.median(np.asarray(data.X.sum(1)))),
                x_span=float(np.ptp(xy[:,0])), y_span=float(np.ptp(xy[:,1])),
                median_neighbor_spacing=float(np.median(spacing)),
                coordinate_matches_image_columns=bool(np.allclose(xy, data.obs[['imagecol','imagerow']].to_numpy()))))
            for label, count in labels.value_counts(dropna=False).items():
                layer_rows.append(dict(sample=sample, layer=str(label), spots=int(count)))
            gene_sets[sample] = set(data.var_names)
        if len(subjects) != 1:
            raise ValueError(f'Published donor grouping inconsistent: {group}')
    pairs = pair_inventory()
    pairs['shared_genes'] = [len(gene_sets[r.reference] & gene_sets[r.query]) for r in pairs.itertuples()]
    return pd.DataFrame(inventory), pd.DataFrame(layer_rows), pairs
