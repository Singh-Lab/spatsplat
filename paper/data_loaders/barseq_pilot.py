"""BARseq pilot inspection using section coordinates, not corrected or CCF positions."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat
from dotenv import load_dotenv

FILENAME = 'wholecortexBARseq_filt_neurons_fixedbent_CCF.mat'

def data_directory():
    load_dotenv(Path(__file__).resolve().parents[2]/'.env')
    base = os.getenv('SPATSPLAT_DATA_DIR')
    if not base: raise RuntimeError('Configure SPATSPLAT_DATA_DIR in .env')
    return Path(base).expanduser()/'raw/barseq_pilot'

def load_barseq_pilot(path=None, *, exclude_qc_filtered=True):
    path = Path(path) if path is not None else data_directory()/FILENAME
    if not path.exists():
        raise FileNotFoundError(f'{path}: run scripts/download_barseq_pilot.py first')
    r = loadmat(path, simplify_cells=True)['filt_neurons']
    required = {'expmat','id','pos','pos40x','fixedpos','slice','genes','clustid','CCF'}
    if not required.issubset(r): raise ValueError(f'Missing fields: {required-set(r)}')
    counts = sparse.csr_matrix(r['expmat'], dtype=np.float32)
    genes = np.asarray(r['genes'])[:,0].astype(str)
    ids = np.asarray(r['id']).reshape(-1)
    sections = np.asarray(r['slice']).reshape(-1).astype(int)
    labels = np.asarray(r['clustid']).reshape(-1).astype(str)
    xy = np.asarray(r['pos'], dtype=float)
    if counts.shape != (len(ids),len(genes)) or xy.shape != (len(ids),2):
        raise ValueError('Inconsistent expression/coordinate dimensions')
    if len(np.unique(ids)) != len(ids) or len(np.unique(genes)) != len(genes):
        raise ValueError('Duplicate measurement or gene identifiers')
    if not np.isfinite(counts.data).all() or np.any(counts.data < 0):
        raise ValueError('Invalid expression values')
    gene_keep = ~np.char.startswith(genes,'unused-')
    excluded_genes = genes[~gene_keep].tolist()
    counts = counts[:,gene_keep]
    totals = np.asarray(counts.sum(axis=1)).ravel()
    finite = np.isfinite(xy).all(axis=1)
    valid = finite & (totals > 0)
    if exclude_qc_filtered: valid &= labels != 'qc-filtered'
    frame = pd.DataFrame(dict(section=sections, raw=1, retained=valid.astype(int),
        author_qc_filtered=(labels=='qc-filtered').astype(int),
        zero_biological_counts=(totals==0).astype(int), invalid_xy=(~finite).astype(int)))
    inventory = frame.groupby('section',sort=True).sum().reset_index()
    coords = {name: np.asarray(r[key],dtype=float)[valid] for name,key in
              [('section','pos'),('manually_corrected','fixedpos'),('image_local','pos40x'),('ccf','CCF')]}
    obs = pd.DataFrame(dict(measurement_id=ids[valid].astype(str), section=sections[valid],
        author_cluster=labels[valid], total_counts=totals[valid],
        detected_genes=counts[valid].getnnz(axis=1)))
    return dict(counts=counts[valid].tocsr(), genes=genes[gene_keep], obs=obs,
        xy=coords['section'], coordinates=coords, inventory=inventory,
        source=str(path), excluded_genes=excluded_genes,
        provenance=dict(coordinate_field='pos', coordinate_units='unverified native section units',
            section_order='1 posterior to 40 anterior per author dictionary',
            physical_z_spacing=None, exclude_qc_filtered=exclude_qc_filtered,
            note='pos40x is not assumed to be globally stitched. fixedpos is manually corrected; CCF is registered.'))
