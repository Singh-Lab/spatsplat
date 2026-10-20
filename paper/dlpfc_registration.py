"""Development-only DLPFC comparison orchestration; never consumes layer labels when fitting."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree, distance

from .data_loaders.dlpfc import load_section, pair_inventory, LAYERS
from .dlpfc_registration_metrics import spatial_coupling, layer_agreement
from .registration_baselines import (run_cast, proper_procrustes,
                                     deformation_diagnostics, _seed)


METHODS = ('Unregistered', 'Geometry rigid ICP', 'PASTE', 'CAST affine',
           'CAST non-affine', 'Splat affine', 'Splat non-affine')


def versions():
    result = {}
    for name in ('numpy', 'scipy', 'torch', 'anndata', 'paste-bio', 'POT', 'CAST', 'dgl'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not installed'
    return result


def prepare_pair(reference, query, root):
    """All released spots; ordered shared genes. Only development pairs are enabled."""
    pairs = pair_inventory()
    selected = pairs[(pairs['reference'] == reference) & (pairs['query'] == query)]
    if len(selected) != 1 or selected.iloc[0].role != 'development':
        raise ValueError('Only prespecified development pairs are enabled before protocol freeze')
    a, b = [load_section(s, root=root) for s in (reference, query)]
    genes = a.var_names[a.var_names.isin(b.var_names)]
    a, b = a[:, genes].copy(), b[:, genes].copy()
    counts = [sparse.csr_matrix(x.X, dtype=np.float32) for x in (a, b)]
    if any(np.any(np.asarray(x.sum(1)).ravel() <= 0) for x in counts):
        raise ValueError('Shared panel creates empty spots; audit before changing eligibility')
    raw = [np.asarray(x.obsm['spatial'], dtype=np.float64) for x in (a, b)]
    together = np.vstack(raw)
    scale = float(np.ptp(together, axis=0).max()) * 1.2
    origin = (together.min(0) + together.max(0)) / 2 - scale / 2
    # One shared translation/scalar: no per-section recentering of the unregistered control.
    xy = [(x - origin) / scale for x in raw]
    return dict(reference=reference, query=query, counts=counts, xy=xy,
                genes=genes.to_numpy(dtype=str), spots=[x.obs_names.to_numpy(dtype=str) for x in (a,b)],
                origin=origin, scale=scale)


def pair_key(pair, config):
    digest = hashlib.sha256(json.dumps(dict(config=config, versions=versions()), sort_keys=True).encode())
    for name in ('dlpfc_registration.py', 'dlpfc_splat_engine.py', 'registration_baselines.py',
                 'registration_perturbations.py', 'dlpfc_paste_solver.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    for path in sorted((Path(__file__).resolve().parents[1] / 'spatsplat').rglob('*.py')):
        digest.update(path.read_bytes())
    for x in [*pair['xy'], pair['genes'], *pair['spots']]:
        digest.update(str(x.shape).encode())
        digest.update(np.ascontiguousarray(x).tobytes())
    for x in pair['counts']:
        for values in (x.data, x.indices, x.indptr):
            digest.update(values.tobytes())
    return digest.hexdigest()[:20]


def rigid_icp(a, b, *, iterations=100):
    """Four rotation starts, proper rigid least squares; no expression or annotations."""
    tree = cKDTree(a)
    best, score = None, float('inf')
    for angle in np.arange(4) * np.pi / 2:
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        mapped = (b - b.mean(0)) @ rotation.T + a.mean(0)
        previous = float('inf')
        for _ in range(iterations):
            _, nearest = tree.query(mapped)
            target = a[nearest]
            u, _, vt = np.linalg.svd((mapped-mapped.mean(0)).T @ (target-target.mean(0)))
            correction = np.diag([1., np.linalg.det(u @ vt)])
            mapped = (mapped-mapped.mean(0)) @ (u @ correction @ vt) + target.mean(0)
            value = float(np.mean(tree.query(mapped)[0]**2))
            if abs(previous-value) < 1e-12:
                break
            previous = value
        if value < score:
            best, score = mapped.copy(), value
    return {'mapped': best, 'objective': np.array(score)}


def fit_paste(pair, *, alpha=.1, iterations=400, gpu=False, seed=1023,
              inner_iterations=1000000):
    """Published spatial OT initialization and KL/count input; retain native coupling.

    Coordinate readout uses proper rigid stacking (no reflections), a declared
    difference from unconstrained published stacking. Native coupling is unchanged.
    """
    import anndata
    import ot
    from .dlpfc_paste_solver import checked_pairwise_align
    if ot.__version__ != '0.9.4':
        raise ImportError('The installed paste-bio adapter requires POT==0.9.4')
    _seed(seed)
    slices = [anndata.AnnData(x.copy()) for x in pair['counts']]
    for adata, xy in zip(slices, pair['xy']):
        adata.obsm['spatial'] = xy.copy()
    # Same centering, minimum-spacing normalization and Euclidean EMD as PASTE's
    # initialization, with an explicit checked budget rather than POT's default.
    initial = spatial_coupling(*pair['xy'], historical_normalization=True)
    coupling, diagnostics = checked_pairwise_align(*slices, alpha=alpha, G_init=initial,
        inner_iterations=inner_iterations,
        dissimilarity='kl', norm=True, numItermax=iterations, use_gpu=gpu,
        backend=ot.backend.TorchBackend() if gpu else ot.backend.NumpyBackend())
    coupling = coupling.detach().cpu().numpy() if hasattr(coupling, 'detach') else np.asarray(coupling)
    if (not np.isfinite(coupling).all() or (coupling < -1e-12).any()
            or not np.allclose(coupling.sum(1), 1/len(pair['xy'][0]), atol=1e-7)
            or not np.allclose(coupling.sum(0), 1/len(pair['xy'][1]), atol=1e-7)):
        raise RuntimeError('PASTE returned invalid transport marginals')
    return dict(mapped=proper_procrustes(*pair['xy'], coupling), native=coupling,
                diagnostics=np.array(json.dumps(diagnostics)))


def fit_cast(pair, output, figures, options):
    # Explicit DLPFC preprocessing, unlike the already-log2 Allen inputs.
    features = []
    for counts in pair['counts']:
        normalized = sparse.diags(1e4 / np.asarray(counts.sum(1)).ravel()) @ counts
        normalized = normalized.tocsr()
        normalized.data = np.log1p(normalized.data)
        features.append(normalized.toarray())
    result = run_cast(*pair['xy'], *features, output_dir=output,
        figure_dir=figures, return_details=True, allow_reflection=False, **options)
    return dict(mapped=result['mapped_b'], affine=result['affine_mapped_b'],
                diagnostics=np.array(json.dumps(result['diagnostics'])))


def fit_splats(pair, output, *, engine_options, ae_epochs=150, min_steps=1500,
               seed=1023, device='cuda'):
    import torch
    from spatsplat import GeneProfileAutoencoder, train_profile_autoencoder
    from .dlpfc_splat_engine import build_engine
    from .registration_perturbations import splat_anchors
    _seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    cfg = engine_options['splat_config']
    counts = [torch.from_numpy(x.toarray()) for x in pair['counts']]
    profiles = [x / x.sum(1, keepdim=True) for x in counts]
    xy = [torch.tensor(x, dtype=torch.float32, device=device) for x in pair['xy']]
    ae = GeneProfileAutoencoder(len(pair['genes']), cfg['hidden_dim'], cfg['latent_dim']).to(device)
    engine = build_engine(engine_options, ae, device)
    ae_path = output / 'autoencoder.pt'
    if ae_path.exists():
        ae.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
        print('Loaded shared AE', flush=True)
    else:
        combined = torch.cat(profiles)
        permutation = torch.randperm(len(combined))
        n_val = max(1, int(.1 * len(combined)))
        def progress(epoch, loss, metrics):
            if epoch == 0 or (epoch + 1) % 25 == 0:
                print(f'AE epoch {epoch+1}: train={loss:.4g}, validation cosine={metrics["cosine"]:.4g}', flush=True)
        history = train_profile_autoencoder(ae, combined, permutation[n_val:], permutation[:n_val],
            epochs=ae_epochs, batch_size=512, on_epoch_end=progress)
        torch.save(ae.state_dict(), ae_path)
        (output / 'ae_history.json').write_text(json.dumps(history))
    ae.eval()
    models = []
    for i, (positions, values, profile) in enumerate(zip(xy, counts, profiles)):
        _seed(seed+i)
        # This is unsupervised registration: all spots may be used for reconstruction.
        idx = torch.arange(len(values))
        model = engine['init_section_splat_model'](positions, profile, values.sum(1,keepdim=True), idx)
        path = output / f'field_{i}.pt'
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            print(f'Loaded field {i}', flush=True)
        else:
            mask, off = engine['build_section_tissue_support'](positions.cpu().numpy())
            fit = engine['train_section_splat_model'](positions, values,
                dict(train=idx, val=idx, test=idx), model, label=f'Section {i}', seed=seed+i,
                off_tissue_points=off.to(device), tissue_mask_tensor=mask.to(device), min_steps=min_steps)
            model = fit['model']
            torch.save(model.state_dict(), path)
            (output / f'field_{i}_history.json').write_text(json.dumps(
                dict(history=fit['history'], training_budget=fit['training_budget'],
                     reconstruction=fit['summary']['train'])))
        model.eval().requires_grad_(False)
        models.append(model)
    _seed(seed + 100)
    queries = engine['sample_registration_queries'](xy[0])
    affine = engine['run_affine_registration'](*models, queries, allow_reflection=False)
    anchors_b, anchors_a, weights = splat_anchors(*models, *xy, affine['transform'])
    nonaffine = engine['run_nonaffine_registration'](*models, queries, affine_result=affine,
        anchor_means_b=anchors_b, anchor_means_a=anchors_a, anchor_weights=weights,
        orientation_xy_b=xy[1], support_xy_a=xy[0], allow_reflection=False)
    with torch.no_grad():
        result = dict(affine=affine['transform'].inverse(xy[1]).cpu().numpy(),
                      mapped=nonaffine['warp'].inverse_approx(xy[1]).cpu().numpy())
    torch.save(dict(affine=affine['transform'].state_dict(), warp=nonaffine['warp'].state_dict()),
               output / 'transforms.pt')
    diagnostics = dict(affine_selected_step=affine['selected_step'],
        nonaffine_selected_step=nonaffine['selected_step'], **nonaffine['inverse_diagnostics'])
    (output / 'registration_history.json').write_text(json.dumps(
        dict(affine=affine['history'], nonaffine=nonaffine['history'], diagnostics=diagnostics)))
    result['diagnostics'] = np.array(json.dumps(diagnostics))
    return result


def cached_fit(family, pair, config, cache_dir, figure_dir, *, run=False, force=False):
    """Exact input/config/code/version cache; failures are logged, never substituted."""
    if force and not run:
        raise ValueError('force=True requires run=True')
    key = pair_key(pair, dict(family=family, options=config))
    folder = Path(cache_dir) / family / key
    path = folder / 'alignment.npz'
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as saved:
            result = {k: saved[k] for k in saved.files}
        print(f'Loaded {family}: {key}', flush=True)
        return result, 'cached'
    if not run:
        print(f'Skipped {family}: enable its RUN flag to fit', flush=True)
        return None, 'not run'
    folder.mkdir(parents=True, exist_ok=True)
    # Explicit force uses a fresh child directory, preserving previous fitted fields.
    working = folder / ('work_' + str(time.time_ns()) if force else 'work')
    working.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f'Fitting {family}: {key}', flush=True)
    try:
        if family == 'PASTE':
            result = fit_paste(pair, **config)
        elif family == 'CAST':
            result = fit_cast(pair, working, Path(figure_dir)/family/key, config)
        elif family == 'Splat':
            result = fit_splats(pair, working, **config)
        elif family == 'Geometry':
            result = rigid_icp(*pair['xy'], **config)
        else:
            raise ValueError(f'Unknown method family: {family}')
        for name in ('mapped', 'affine'):
            if name in result and (result[name].shape != pair['xy'][1].shape or not np.isfinite(result[name]).all()):
                raise ValueError(f'Invalid {name} output')
        result['metadata'] = np.array(json.dumps(dict(config=config, versions=versions(),
            seconds=time.perf_counter()-started, key=key, work=str(working))))
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **result)
        temporary.replace(path)
        return result, 'completed'
    except Exception as exc:
        (folder / 'failure.json').write_text(json.dumps(dict(error=repr(exc), config=config,
            seconds=time.perf_counter()-started), indent=2))
        raise


def score_maps(pair, mappings, labels, native=None):
    """Common spatial EMD, historical readout, physical support, per-layer scores."""
    a, b = pair['xy']
    spacing = np.median(cKDTree(a).query(a,k=2)[0][:,1])
    rows, layers = [], []
    for method, mapped in mappings.items():
        print('Scoring:', method, flush=True)
        common = spatial_coupling(a, mapped)
        historical = native if method == 'PASTE' and native is not None else spatial_coupling(
            a, mapped, historical_normalization=True)
        row = dict(method=method, **layer_agreement(*labels, common),
            published_style_agreement=layer_agreement(*labels, historical)['matching_mass'],
            correspondence_type='native PASTE' if method == 'PASTE' else 'historically normalized spatial OT')
        d = cKDTree(a).query(mapped)[0] / spacing
        reverse = cKDTree(mapped).query(a)[0] / spacing
        row.update(query_coverage_2_spacings=float(np.mean(d <= 2)),
                   reference_coverage_2_spacings=float(np.mean(reverse <= 2)),
                   median_nearest_distance_spacings=float(np.median(d)),
                   transport_distance_spacings=float(np.sum(common*distance.cdist(a,mapped))/spacing),
                   **deformation_diagnostics(b, mapped))
        rows.append(row)
        for label in LAYERS:
            ia, ib = np.asarray(labels[0]) == label, np.asarray(labels[1]) == label
            mass = common[np.ix_(ia,ib)].sum()
            denom = common[ia].sum()
            layers.append(dict(method=method, layer=label, matching_mass=float(mass),
                reference_mass=float(denom), recall_given_reference=float(mass/denom) if denom else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(layers)


def plot_overlays(pair, mappings, labels):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = dict(zip(LAYERS, ['#4c78a8', '#f58518', '#54a24b', '#e45756',
                              '#b279a2', '#72b7b2', '#79706e']))
    fig, axes = plt.subplots(2, 4, figsize=(18,9), layout='constrained')
    all_xy = np.vstack([pair['xy'][0], *mappings.values()])
    lo, hi = all_xy.min(0)-.03, all_xy.max(0)+.03
    for ax, method in zip(axes.flat, METHODS):
        ax.set_title(method)
        if method not in mappings:
            ax.text(.5,.5,'Not run / failed',ha='center',transform=ax.transAxes)
        else:
            for xy, lab, marker, alpha in [(pair['xy'][0],labels[0],'o',.35),
                                           (mappings[method],labels[1],'+',.8)]:
                ax.scatter(xy[:,0],xy[:,1],c=[colors.get(x,'#bbbbbb') for x in lab],
                           s=6,marker=marker,alpha=alpha,linewidths=.45,rasterized=True)
        ax.set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]),aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
    axes.flat[-1].axis('off')
    handles = [Line2D([],[],marker='o',linestyle='',color=c,label=k) for k,c in colors.items()]
    axes.flat[-1].legend(handles=handles,loc='center',frameon=False)
    fig.suptitle(f'{pair["query"]} to {pair["reference"]} | circles: reference; crosses: query')
    return fig
