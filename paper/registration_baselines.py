"""Optional PASTE/CAST adapters and field-independent registration diagnostics.

Coordinates returned by both adapters put B in A's input coordinate frame.
External dependencies are imported only inside their respective runners.
"""

from pathlib import Path
from functools import wraps
import hashlib
import importlib.metadata
import json
import random
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree, Delaunay


def select_coordinate_state(initial, states, objectives, valid=None):
    """Select by the native training objective, including the initial state."""
    objectives = np.asarray(objectives, dtype=float)
    if len(objectives) != len(states) + 1:
        raise ValueError('Expected one initial objective plus one per state')
    for index in np.argsort(objectives, kind='stable'):
        candidate = _numpy(initial if index == 0 else states[index - 1])
        if (np.isfinite(objectives[index]) and np.isfinite(candidate).all()
                and (valid is None or valid(candidate))):
            return candidate.copy(), int(index)
    raise ValueError('No finite registration checkpoint')


class OrientationGuard:
    """Reject global orientation reversal on a fixed triangulation; folds are scored separately."""
    def __init__(self, reference):
        reference = _numpy(reference)
        self.triangles = Delaunay(reference).simplices
        self.before = self.area(reference)

    def area(self, coordinates):
        t = _numpy(coordinates)[self.triangles]
        u, v = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    def __call__(self, coordinates):
        ratio = self.area(coordinates) / self.before
        return bool(np.isfinite(ratio).all() and np.median(ratio) > 0)


def proper_procrustes(a, b, coupling):
    """PASTE's coupling-weighted rigid fit, constrained to determinant +1."""
    coupling = np.asarray(coupling, dtype=float)
    coupling = coupling / coupling.sum()
    center_a = coupling.sum(1) @ a
    center_b = coupling.sum(0) @ b
    u, _, vt = np.linalg.svd((b - center_b).T @ coupling.T @ (a - center_a))
    correction = np.eye(2)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    return (b - center_b) @ rotation + center_a


class ObjectiveCheckpoint:
    """Keep matching model parameters and objective values before an update."""
    def __init__(self):
        self.objective = float('inf')
        self.step = None
        self.state = None

    def consider(self, model, objective, step):
        value = float(objective)
        if np.isfinite(value) and value < self.objective:
            self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.objective, self.step = value, int(step)

    def restore(self, model):
        if self.state is None:
            raise RuntimeError('No finite registration checkpoint')
        model.load_state_dict(self.state)


def deformation_diagnostics(original, mapped, *, relative_area_tolerance=1e-8):
    """Triangle-orientation diagnostic, not a proof of a diffeomorphic warp.

    Exclude numerically degenerate *source* triangles using twice-area / longest
    squared edge, a dimensionless criterion fixed before examining the mapping.
    Mapped collapse is never excluded. Counts make the scoring denominator explicit.
    """
    original, mapped = np.asarray(original, dtype=np.float64), np.asarray(mapped, dtype=np.float64)
    if (original.shape != mapped.shape or original.ndim != 2 or original.shape[1] != 2
            or not np.isfinite(original).all() or not np.isfinite(mapped).all()):
        raise ValueError('Invalid mapped coordinates')
    if not 0 < relative_area_tolerance < 1:
        raise ValueError('relative_area_tolerance must be between zero and one')
    triangles = Delaunay(original).simplices

    def area(x):
        t = x[triangles]
        u, v = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]
        return u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]

    before, after = area(original), area(mapped)
    t = original[triangles]
    edge_squared = np.maximum.reduce([
        np.sum((t[:, i] - t[:, j])**2, axis=1) for i,j in ((0,1),(1,2),(2,0))])
    valid = np.abs(before) > relative_area_tolerance * edge_squared
    if not valid.any():
        raise ValueError('No numerically nondegenerate source triangles to score')
    ratio = after[valid] / before[valid]
    # A global reflection is permitted by PASTE/CAST and is not a local fold.
    orientation = -1 if np.median(ratio) < 0 else 1
    return {'local_fold_fraction': float(np.mean(ratio * orientation < 0)),
            'near_collapsed_triangle_fraction': float(np.mean(np.abs(ratio) < 0.01)),
            'median_absolute_area_ratio': float(np.median(np.abs(ratio))),
            'global_reflection': bool(orientation == -1),
            'triangles_total': int(len(triangles)),
            'triangles_scored': int(valid.sum()),
            'degenerate_source_triangles_excluded': int((~valid).sum()),
            'relative_area_tolerance': float(relative_area_tolerance)}


def validate_pair(xy_a, xy_b, expression_a, expression_b):
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (xy_a, xy_b, expression_a, expression_b)]
    a, b, ea, eb = arrays
    if any(x.ndim != 2 or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Inputs must be finite two-dimensional arrays')
    if a.shape[1] != 2 or b.shape[1] != 2 or min(len(a), len(b)) < 3:
        raise ValueError('Each section needs at least three 2D measurements')
    if len(a) != len(ea) or len(b) != len(eb) or ea.shape[1] != eb.shape[1]:
        raise ValueError('Expression rows and shared gene columns must match')
    if ea.shape[1] == 0 or min(ea.min(), eb.min()) < 0:
        raise ValueError('Expression must be nonnegative with a shared gene panel')
    return arrays


def subset_indices(n, maximum=3000, seed=1023):
    if maximum < 3:
        raise ValueError('maximum must be at least three')
    return np.sort(np.random.default_rng(seed).choice(n, min(n, maximum), replace=False))


def _numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def _seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_paste(xy_a, xy_b, expression_a, expression_b, *, alpha=0.1,
              iterations=200, gpu=False, seed=1023, allow_reflection=True):
    """PASTE full-overlap OT followed by its rigid Procrustes stacking."""
    import anndata
    import paste
    import ot

    if getattr(ot, '__version__', '0.9.4') != '0.9.4':
        raise ImportError(
            'This PASTE adapter requires POT==0.9.4. POT 0.9.5 and 0.9.7 '
            'have incompatible line-search APIs for paste-bio 1.4.0. '
            'Install paste-bio==1.4.0 and POT==0.9.4 together.')
    a, b, ea, eb = validate_pair(xy_a, xy_b, expression_a, expression_b)
    _seed(seed)
    slices = [anndata.AnnData(ea), anndata.AnnData(eb)]
    for adata, xy in zip(slices, (a, b)):
        adata.obsm['spatial'] = xy.copy()
    backend = ot.backend.TorchBackend() if gpu else ot.backend.NumpyBackend()
    coupling = paste.pairwise_align(
        *slices, alpha=alpha, dissimilarity='kl', norm=True,
        numItermax=iterations, use_gpu=gpu, backend=backend)
    if not allow_reflection:
        return proper_procrustes(a, b, _numpy(coupling))
    stacked = paste.stack_slices_pairwise(slices, [_numpy(coupling)])
    # PASTE centers A as well as B; restore the original A frame.
    offset = (a - stacked[0].obsm['spatial']).mean(axis=0)
    if not np.allclose(stacked[0].obsm['spatial'] + offset, a, atol=1e-6):
        raise RuntimeError('Unexpected PASTE reference-frame transformation')
    return np.asarray(stacked[1].obsm['spatial']) + offset


def _preserve_plot_settings(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        import matplotlib.pyplot as plt
        existing = set(plt.get_fignums())
        with plt.rc_context():
            try:
                return function(*args, **kwargs)
            finally:
                for number in set(plt.get_fignums()) - existing:
                    plt.close(number)
    return wrapped


@_preserve_plot_settings
def run_cast(xy_a, xy_b, expression_a, expression_b, *, output_dir,
             mark_epochs=400, affine_iterations=500, ffd_iterations=400,
             gpu=False, seed=1023, coordinate_scale=1000.0, figure_dir=None,
             affine_step_multiplier=0.1, ffd_step_size=5.0, return_details=False,
             allow_reflection=True, affine_diff_step=50., ffd_diff_step=5.,
             embedding_cache_dir=None):
    """CAST Mark + Stack with explicit defaults from the official implementation.

Input expression is already log-transformed in the Allen notebook. Do not
log-transform it again. The scalar coordinate conversion preserves aspect ratio
and puts normalized coordinates on the scale expected by Stack's step sizes.
"""
    from CAST import CAST_MARK
    from CAST.CAST_Stack import (prelocate, affine_trans_t, corr_dist,
                                 max_minus_value_t, BSpline_GD, Affine_GD)
    from CAST.models.model_GCNII import Args
    import dgl
    import torch

    a, b, ea, eb = validate_pair(xy_a, xy_b, expression_a, expression_b)
    if coordinate_scale <= 0:
        raise ValueError('coordinate_scale must be positive')
    if affine_iterations < 1 or ffd_iterations < 0 or mark_epochs < 1:
        raise ValueError('Invalid CAST iteration counts')
    if min(affine_step_multiplier, ffd_step_size, affine_diff_step, ffd_diff_step) <= 0:
        raise ValueError('CAST step sizes must be positive')
    _seed(seed)
    dgl.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coords = {'A': a * coordinate_scale, 'B': b * coordinate_scale}
    args = Args(dataname='registration', gpu=0 if gpu else -1,
                epochs=mark_epochs, lr1=1e-3, wd1=0, lambd=1e-3,
                n_layers=9, der=0.5, dfr=0.3, use_encoder=True, encoder_dim=512)
    cast_device = args.device
    mark_path = None
    if embedding_cache_dir is not None:
        mark_config = dict(version=1, epochs=mark_epochs, seed=seed, gpu=gpu,
                           torch=torch.__version__, dgl=dgl.__version__,
                           cast=importlib.metadata.version('CAST'))
        digest = hashlib.sha256(json.dumps(mark_config, sort_keys=True).encode())
        for value in (a, b, ea, eb):
            digest.update(str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        mark_path = Path(embedding_cache_dir) / (digest.hexdigest()[:20] + '.pt')
    embedding_cache_hit = mark_path is not None and mark_path.exists()
    if embedding_cache_hit:
        embeddings = torch.load(mark_path, map_location=cast_device, weights_only=True)
    else:
        # Mark's graph uses a fixed scale, independently of Stack's numerical scale sweep.
        embeddings = CAST_MARK(
            {'A': a * 1000., 'B': b * 1000.}, {'A': ea, 'B': eb}, str(output_dir), args=args,
            gpu_t=None if gpu else -1, if_plot=False)
        if mark_path is not None:
            mark_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = mark_path.with_suffix('.tmp.pt')
            torch.save({k: v.detach().cpu() for k, v in embeddings.items()}, temporary)
            temporary.replace(mark_path)
    alpha = (torch.tensor([1/3000, 1/3000, 1/100, 5, 5], device=cast_device)
             .reshape(5, 1) * affine_step_multiplier)
    mean_q, mean_r = coords['B'].mean(0), coords['A'].mean(0)
    query = torch.as_tensor(coords['B'] - mean_q, dtype=torch.float32, device=cast_device)
    ref = torch.as_tensor(coords['A'] - mean_r, dtype=torch.float32, device=cast_device)
    cost = max_minus_value_t(torch.as_tensor(
        corr_dist(_numpy(embeddings['B']), _numpy(embeddings['A'])),
        device=cast_device, dtype=torch.float32))
    theta = prelocate(query, ref, cost, 500, str(output_dir),
                     d_list=[3, 2, 1, .5, 1/3], ifplot=False,
                     mirror_t=None if allow_reflection else [1])
    initial = affine_trans_t(theta, query)
    # CAST_STACK does not forward diff_step to Affine_GD. Call the official routine
    # directly so the affine finite-difference scale is explicit and tunable.
    affine = Affine_GD(initial, ref, cost, str(output_dir), bleeding=500,
                       dist_penalty=0, diff_step=affine_diff_step, alpha_basis=alpha,
                       iterations=affine_iterations, coords_log=True, mid_visual=False,
                       early_stop_thres=None)
    logs = {'affine_J': affine[0], 'affine_it_theta': affine[2],
            'affine_coords_log': affine[3], 'coords_ref': ref}
    torch.save(logs, output_dir / 'B_align_to_A_result_log.data')
    guard = None if allow_reflection else OrientationGuard(query)
    affine_centered, affine_step = select_coordinate_state(
        initial, logs['affine_coords_log'], logs['affine_J'], valid=guard)
    affine_mapped = (affine_centered + mean_r) / coordinate_scale
    selected = affine_centered
    ffd_step = 0
    ffd_objectives = [float(logs['affine_J'][affine_step])]
    if ffd_iterations:
        # Use the official FFD optimizer, starting from the selected affine state.
        q = torch.as_tensor(affine_centered, device=cast_device, dtype=torch.float32)
        offset = q.min(0)[0]
        ffd = BSpline_GD(q-offset, ref-offset, cost, ffd_iterations, str(output_dir),
                        bleeding=500, dist_penalty=0, alpha_basis=ffd_step_size,
                        diff_step=ffd_diff_step, mesh_size=8, coords_log=True, mid_visual=False,
                        max_xy=(q-offset).max(0)[0].cpu(), renew_mesh_trans=True)
        shifted, ffd_step = select_coordinate_state(q-offset, ffd[4], ffd[3], valid=guard)
        selected = shifted + _numpy(offset)
        ffd_objectives = list(map(float, ffd[3]))
        torch.save({'objectives': ffd_objectives, 'coordinates': ffd[4],
                    'selected_step': ffd_step, 'offset': offset.cpu()},
                   output_dir / 'selected_affine_ffd_log.pt')
    mapped = (selected + mean_r) / coordinate_scale
    diagnostics = {
        'selection_rule': 'minimum finite native training objective, including initial state',
        'allow_reflection': allow_reflection, 'embedding_cache_hit': embedding_cache_hit,
        'affine_diff_step': affine_diff_step, 'ffd_diff_step': ffd_diff_step,
        'affine_selected_at_budget': affine_step == affine_iterations,
        'ffd_selected_at_budget': bool(ffd_iterations and ffd_step == ffd_iterations),
        'affine_tail_relative_improvement': float((logs['affine_J'][max(0, len(logs['affine_J'])-101)]
            - logs['affine_J'][-1]) / max(abs(logs['affine_J'][max(0, len(logs['affine_J'])-101)]), 1e-12)),
        'affine_selected_step': affine_step, 'ffd_selected_step': ffd_step,
        'affine_initial_objective': float(logs['affine_J'][0]),
        'affine_selected_objective': float(logs['affine_J'][affine_step]),
        'affine_final_objective': float(logs['affine_J'][-1]),
        'ffd_initial_objective': ffd_objectives[0],
        'ffd_selected_objective': ffd_objectives[ffd_step],
        'ffd_final_objective': ffd_objectives[-1],
        'deformation': deformation_diagnostics(b, mapped),
    }
    np.savez_compressed(output_dir / 'selected_coordinates.npz',
                        affine_mapped_b=affine_mapped, mapped_b=mapped,
                        diagnostics=json.dumps(diagnostics))
    if figure_dir is not None:
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        for artifact in output_dir.iterdir():
            if artifact.suffix.lower() in {'.pdf', '.png', '.svg', '.jpg'}:
                shutil.move(str(artifact), str(figure_dir / artifact.name))
    if return_details:
        return {'mapped_b': mapped, 'affine_mapped_b': affine_mapped, 'diagnostics': diagnostics}
    return mapped


def cached_registration(method, xy_a, xy_b, expression_a, expression_b, *,
                        cache_dir, run=False, force=False, options=None,
                        provenance=None, artifact_dir=None):
    """Reuse an exact input/config cache; never implicitly launch a new run."""
    if method not in {'PASTE', 'CAST'}:
        raise ValueError('method must be PASTE or CAST')
    arrays = validate_pair(xy_a, xy_b, expression_a, expression_b)
    config = {'method': method, 'options': dict(options or {}),
              'provenance': provenance or {}, 'adapter_version': 3 if method == 'CAST' else 2}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for x in arrays:
        digest.update(str(x.shape).encode())
        digest.update(np.ascontiguousarray(x).tobytes())
    key = digest.hexdigest()[:20]
    path = Path(cache_dir) / f'{method.lower()}_{key}.npz'
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as data:
            mapped = data['mapped_b']
            metadata = json.loads(str(data['metadata'].item()))
            extra = {'affine_mapped_b': data['affine_mapped_b']} if 'affine_mapped_b' in data else {}
        if mapped.shape != arrays[1].shape or not np.isfinite(mapped).all():
            raise ValueError(f'Invalid registration cache: {path}')
        print(f'Loaded {method} alignment: {path.name}', flush=True)
        return {'mapped_b': mapped, 'metadata': metadata, 'cache_hit': True, **extra}
    if not run:
        print(f'Skipped {method} (training disabled; no usable cache): {path.name}', flush=True)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(options or {})
    if method == 'CAST':
        kwargs['output_dir'] = Path(cache_dir) / f'cast_{key}'
        kwargs['return_details'] = True
        if artifact_dir is not None:
            kwargs['figure_dir'] = Path(artifact_dir) / f'cast_{key}'
    print(f'Fitting {method} alignment: {path.name}', flush=True)
    start = time.perf_counter()
    result = (run_paste if method == 'PASTE' else run_cast)(*arrays, **kwargs)
    extra = {}
    diagnostics = {}
    if method == 'CAST':
        mapped = result['mapped_b']
        extra['affine_mapped_b'] = result['affine_mapped_b']
        diagnostics = result['diagnostics']
    else:
        mapped = result
    if mapped.shape != arrays[1].shape or not np.isfinite(mapped).all():
        raise RuntimeError(f'{method} returned invalid coordinates')
    versions = {}
    for package in ['paste-bio', 'POT', 'CAST', 'dgl', 'torch', 'numpy']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {**config, 'seconds': time.perf_counter() - start,
                'versions': versions, 'cache_key': key, 'diagnostics': diagnostics}
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, mapped_b=mapped, metadata=json.dumps(metadata), **extra)
    temporary.replace(path)
    print(f'Completed {method} alignment in {metadata["seconds"]:.4g} seconds: {path.name}', flush=True)
    return {'mapped_b': mapped, 'metadata': metadata, 'cache_hit': False, **extra}


def evaluate_alignment(xy_a, mapped_b, expression_a, expression_b, *,
                       radius, k=5, labels_a=None, labels_b=None):
    """Symmetric measurement transfer, not a learned-field reconstruction score.

Scores are descriptive/in-sample if alignment saw these profiles. Coverage and
coverage-weighted cosine prevent disjoint alignments benefiting from exclusion.
The radius is fixed once from reference A, never refit to each method's output.
"""
    a, b, ea, eb = validate_pair(xy_a, mapped_b, expression_a, expression_b)
    if radius <= 0 or k < 1:
        raise ValueError('radius and k must be positive')
    if (labels_a is None) != (labels_b is None):
        raise ValueError('Supply both label arrays or neither')
    if labels_a is not None and (len(labels_a) != len(a) or len(labels_b) != len(b)):
        raise ValueError('Label lengths must match coordinates')
    profiles = [x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12) for x in (ea, eb)]
    rows = []
    for source, target, ps, pt, ls, lt in [
        (a, b, profiles[0], profiles[1], labels_a, labels_b),
        (b, a, profiles[1], profiles[0], labels_b, labels_a),
    ]:
        distances, neighbors = cKDTree(source).query(target, k=list(range(1, min(k, len(source))+1)))
        inside = distances <= radius
        weights = inside / np.maximum(distances, 1e-8)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        pred = np.einsum('nk,nkg->ng', weights, ps[neighbors])
        valid = inside.any(axis=1) & (pt.sum(axis=1) > 0) & (pred.sum(axis=1) > 0)
        cosine = (pred * pt).sum(axis=1) / np.maximum(
            np.linalg.norm(pred, axis=1) * np.linalg.norm(pt, axis=1), 1e-12)
        row = {'coverage': float(inside.any(axis=1).mean()),
               'expression_valid_fraction': float(valid.mean()),
               'profile_cosine': float(cosine[valid].mean()) if valid.any() else np.nan,
               'coverage_weighted_cosine': float(np.where(valid, cosine, 0).mean()),
               'profile_l1': float(np.abs(pred[valid]-pt[valid]).sum(axis=1).mean()) if valid.any() else np.nan,
               'median_neighbor_distance': float(np.median(distances[:, 0]))}
        if ls is not None:
            ls, lt = np.asarray(ls, dtype=str), np.asarray(lt, dtype=str)
            labeled = (ls[neighbors[:, 0]] != '') & (lt != '') & inside[:, 0]
            row['annotation_coverage'] = float(labeled.mean())
            row['annotation_agreement'] = float((ls[neighbors[labeled, 0]] == lt[labeled]).mean()) if labeled.any() else np.nan
        rows.append(row)
    result = {name: float(np.mean([r[name] for r in rows])) for name in rows[0]}
    result.update({'coverage_b_to_a': rows[0]['coverage'], 'coverage_a_to_b': rows[1]['coverage']})
    return result
