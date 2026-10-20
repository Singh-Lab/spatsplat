"""Annotation-blind spatial matching and explicit anatomical scoring contracts."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance

from .data_loaders.dlpfc import LAYERS


def spatial_coupling(reference, mapped_query, *, historical_normalization=False):
    """Uniform-marginal EMD with Euclidean coordinate cost, no expression/labels.

    Historical mode reproduces independent centering/minimum-spacing scaling
    from the PASTE STUtility analysis. Default preserves the registered frame.
    This transport is a readout, not a new coordinate transformation.
    """
    import ot
    a, b = (np.array(x, dtype=float, copy=True) for x in (reference, mapped_query))
    for x in (a,b):
        if x.ndim != 2 or x.shape[1] != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError('Expected finite Nx2 coordinates, N >= 2')
        if historical_normalization:
            minimum = cKDTree(x).query(x,k=2)[0][:,1].min()
            if minimum <= 0:
                raise ValueError('Historical normalization is undefined for duplicate coordinates')
            x -= x.mean(0)
            x /= minimum
    costs = distance.cdist(a,b,metric='euclidean')
    coupling, log = ot.emd(np.full(len(a),1/len(a)), np.full(len(b),1/len(b)), costs,
                           numItermax=1000000, log=True)
    if log.get('warning'):
        raise RuntimeError(f"Spatial transport did not converge: {log['warning']}")
    return coupling


def layer_agreement(reference_labels, query_labels, coupling):
    """Absolute matching mass plus conditional accuracy and its denominator.

    Missing labels are unscored, never treated as an eighth matching category.
    Native and coordinate-derived couplings use exactly the same scorer.
    """
    a,b = np.asarray(reference_labels,dtype=object),np.asarray(query_labels,dtype=object)
    pi = np.asarray(coupling,dtype=float)
    if pi.shape != (len(a),len(b)) or not np.isfinite(pi).all() or (pi < 0).any() or not np.isclose(pi.sum(),1,atol=1e-6):
        raise ValueError('Expected a nonnegative unit-mass coupling with matching label dimensions')
    valid_a,valid_b = ~pd.isna(a),~pd.isna(b)
    if set(a[valid_a])-set(LAYERS) or set(b[valid_b])-set(LAYERS):
        raise ValueError('Unknown anatomical label')
    a = np.array([str(x) if valid else '' for x,valid in zip(a,valid_a)])
    b = np.array([str(x) if valid else '' for x,valid in zip(b,valid_b)])
    scored_mass = float(pi[np.ix_(valid_a,valid_b)].sum())
    matches = sum(float(pi[np.ix_(a==label,b==label)].sum()) for label in LAYERS)
    # This bound uses the actual coupling marginals, not raw minimum spot counts.
    row_mass,col_mass = pi.sum(1),pi.sum(0)
    ceiling = sum(min(float(row_mass[a==label].sum()),float(col_mass[b==label].sum())) for label in LAYERS)
    return dict(matching_mass=matches, scored_mass=scored_mass,
                agreement_given_scored=matches/scored_mass if scored_mass else float('nan'),
                marginal_label_ceiling=ceiling)
