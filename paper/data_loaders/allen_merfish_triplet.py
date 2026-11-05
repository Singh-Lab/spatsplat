"""Small serial-section cohort from the existing ABC Atlas cache."""
from pathlib import Path
import os

import numpy as np


def load_allen_merfish_triplet(sections=(24, 25, 26), sample_cap=12000, seed=3107):
    import anndata
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

    data_dir = os.getenv('SPATSPLAT_DATA_DIR')
    if not data_dir:
        raise RuntimeError('Set SPATSPLAT_DATA_DIR in the repository .env')
    root = Path(os.getenv('SPATSPLAT_ABC_ATLAS_DIR', str(Path(data_dir)/'raw/abc_atlas')))
    cache = AbcProjectCache.from_cache_dir(root)
    directory = 'MERFISH-C57BL6J-638850'
    metadata = cache.get_metadata_dataframe(directory=directory, file_name='cell_metadata',
                                          dtype={'cell_label': str}).set_index('cell_label')
    metadata.index = metadata.index.astype(str)
    path = cache.get_file_path(directory=directory, file_name='C57BL6J-638850/log2')
    expression = anndata.read_h5ad(path, backed='r')
    rng = np.random.default_rng(seed)
    labels = [f'C57BL6J-638850.{i}' for i in sections]
    blocks, stats = [], []
    try:
        for label in labels:
            available = metadata.index[(metadata.brain_section_label.astype(str) == label)
                                       & metadata.index.isin(expression.obs_names)]
            if len(available) < 100:
                raise ValueError(f'Insufficient coordinate/expression overlap for {label}')
            selected = rng.choice(available.to_numpy(), min(sample_cap, len(available)), replace=False)
            positions = np.sort(expression.obs_names.get_indexer(selected))
            block = expression[positions].to_memory()
            values = block.X.toarray() if hasattr(block.X, 'toarray') else np.asarray(block.X)
            values = values.astype('float32')
            xy = metadata.loc[block.obs_names, ['x', 'y', 'z']].to_numpy('float32')
            valid = np.isfinite(xy).all(1) & np.isfinite(values).all(1) & (values >= 0).all(1) & (values.sum(1) > 0)
            blocks.append(dict(label=label, xyz=xy[valid], values=values[valid], ids=block.obs_names.to_numpy()[valid]))
            stats.append(dict(section=label, available=len(available), sampled=int(valid.sum()),
                              removed=int((~valid).sum()), z=float(np.median(xy[valid, 2]))))
        genes = expression.var_names.astype(str).tolist()
    finally:
        expression.file.close()
    z = np.array([row['z'] for row in stats])
    if not np.all(np.diff(z) > 0) or not np.allclose(np.diff(z), np.diff(z)[0], rtol=.01):
        raise ValueError('Choose three ordered, equally spaced sections')
    raw = np.concatenate([b['xyz'] for b in blocks])
    lo, hi = raw.min(0), raw.max(0)
    unit = float((hi-lo).max()) * 1.2
    origin = (lo+hi)/2 - unit/2
    for b in blocks:
        b['xyz'] = (b['xyz']-origin)/unit
    return dict(blocks=blocks, statistics=stats, genes=genes, origin=origin, unit=unit,
                spacing=float(np.diff(z)[0]/unit), expression_path=str(path))
