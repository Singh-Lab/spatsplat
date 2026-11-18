"""Training-only external baselines with explicit held-out coordinate readouts."""
import numpy as np
from scipy import sparse


def training_pair(blocks, reference, query, genes):
    """Never supply validation/test observations to CAST or PASTE fitting."""
    pair = dict(reference=reference, query=query, genes=np.asarray(genes, dtype=str),
                xy=[], counts=[], spots=[])
    for section in (reference, query):
        block = blocks[section]
        split = block['splits']
        train = np.asarray(split['train'])
        if len(train) < 3 or len(np.unique(train)) != len(train):
            raise ValueError('Need at least three distinct training measurements')
        for other in ('val', 'test'):
            if np.intersect1d(train, split[other]).size:
                raise ValueError('Training indices overlap validation/test')
        pair['xy'].append(np.asarray(block['xy'][train], dtype=float).copy())
        pair['counts'].append(sparse.csr_matrix(block['counts'][train], dtype=np.float32))
        pair['spots'].append(np.asarray(block['ids'][train], dtype=str))
    return pair


def extend_affine(training_xy, mapped_training_xy, query_xy, *, rigid=False):
    """Recover the actual affine map from its training-coordinate evaluations.

    This is not interpolation of expression or non-affine displacements. Reject
    outputs that cannot be explained by one orientation-preserving affine map.
    """
    x, y, query = [np.asarray(a, dtype=float) for a in
                   (training_xy, mapped_training_xy, query_xy)]
    if x.ndim != 2 or x.shape[1] != 2 or x.shape != y.shape or query.ndim != 2 or query.shape[1] != 2:
        raise ValueError('Expected matching training Nx2 arrays and query Mx2')
    if not all(np.isfinite(a).all() for a in (x, y, query)):
        raise ValueError('Nonfinite coordinates')
    design = np.column_stack((x, np.ones(len(x))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 3:
        raise ValueError('Degenerate coordinate support')
    residual = float(np.max(np.abs(design @ coefficients-y)))
    if residual > 1e-5 * max(1., float(np.ptp(y, axis=0).max())):
        raise ValueError('Coordinate output is not affine; cannot extend to held-out positions')
    linear = coefficients[:2]
    determinant = float(np.linalg.det(linear))
    if determinant <= 1e-10:
        raise ValueError('Reflected or collapsed map')
    if rigid and not np.allclose(linear.T @ linear, np.eye(2), atol=1e-4, rtol=1e-4):
        raise ValueError('PASTE rigid readout is not a proper rigid map')
    mapped = np.column_stack((query, np.ones(len(query)))) @ coefficients
    return mapped, coefficients, dict(affine_residual=residual, determinant=determinant,
                                     singular_values=np.linalg.svd(linear, compute_uv=False).tolist())
