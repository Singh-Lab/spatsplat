"""Annotation-separated Allen MERFISH sections for registration development."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def load_registration_sections(sections, maximum=3000, seed=1023):
    import anndata
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

    if maximum < 100:
        raise ValueError('At least 100 measurements per section required')
    base = os.getenv('SPATSPLAT_DATA_DIR')
    if not base:
        raise RuntimeError('Configure SPATSPLAT_DATA_DIR in .env')
    root = Path(os.getenv('SPATSPLAT_ABC_ATLAS_DIR', str(Path(base)/'raw/abc_atlas')))
    cache = AbcProjectCache.from_cache_dir(root)
    directory = 'MERFISH-C57BL6J-638850'
    metadata = cache.get_metadata_dataframe(directory=directory, file_name='cell_metadata',
        dtype={'cell_label': str}).set_index('cell_label')
    ccf = cache.get_metadata_dataframe(directory=directory+'-CCF', file_name='reconstructed_coordinates',
        dtype={'cell_label': str}).set_index('cell_label')
    if not metadata.index.is_unique or not ccf.index.is_unique:
        raise ValueError('Ambiguous measurement identifiers')
    path = cache.get_file_path(directory=directory, file_name='C57BL6J-638850/log2')
    expression = anndata.read_h5ad(path, backed='r')
    blocks, rows = {}, []
    try:
        genes = expression.var['gene_symbol'].to_numpy(dtype=str)
        if len(np.unique(genes)) != len(genes):
            raise ValueError('Duplicate gene symbols; resolve explicitly before marker evaluation')
        for section in sections:
            label = f'C57BL6J-638850.{int(section):02d}'
            pool = metadata.loc[metadata.brain_section_label.eq(label)]
            pool = pool.loc[pool.index.isin(expression.obs_names)]
            pool = pool.loc[np.isfinite(pool[['x', 'y', 'z']]).all(axis=1)]
            if len(pool) < maximum:
                raise ValueError(f'{label}: insufficient coordinate/expression overlap for fixed sample cap')
            rng = np.random.default_rng(seed + int(section))
            chosen = rng.choice(np.sort(pool.index.to_numpy()), maximum, replace=False)
            indices = np.sort(expression.obs_names.get_indexer(chosen))
            block = expression[indices].to_memory()
            values = sparse.csr_matrix(block.X, dtype=np.float32)
            if not np.isfinite(values.data).all() or (values.data < 0).any():
                raise ValueError('Expected finite nonnegative released log2 expression')
            valid = np.asarray(values.sum(1)).ravel() > 0
            ids = block.obs_names.to_numpy(dtype=str)[valid]
            coordinates = pool.loc[ids, ['x', 'y']].to_numpy(dtype=float)
            labels = pd.to_numeric(ccf.reindex(ids).parcellation_index, errors='coerce')
            labels = labels.where(labels > 0).to_numpy(dtype=float)
            z = float(pool.z.median())
            blocks[int(section)] = dict(xy=coordinates, counts=values[valid], ids=ids,
                                       labels=labels, z=z, section=label)
            rows.append(dict(section=label, available=len(pool), sampled=len(ids), z=z,
                annotation_fraction=float(np.isfinite(labels).mean()),
                x_extent=float(np.ptp(coordinates[:, 0])), y_extent=float(np.ptp(coordinates[:, 1]))))
    finally:
        expression.file.close()
    return dict(blocks=blocks, genes=genes, inventory=pd.DataFrame(rows),
                expression_path=str(path), atlas_root=str(root), expression_scale='released log2')
