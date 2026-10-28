"""Controlled BARseq coverage experiment; true overlap is evaluation-only."""
from pathlib import Path
import hashlib
import json
import numpy as np


def make_cases(xy, counts, ids, *, seed=1023):
    """Disjoint acquisitions and globally fixed splits before nested cropping.

    Each view keeps a prescribed quantile of the parent coordinate projection.
    Reference keeps the low end, query the high end. This gives nominal shared
    parent fractions 1, .6, .2, without looking at expression or method results.
    """
    xy = np.asarray(xy, dtype=float)
    ids = np.asarray(ids, dtype=str)
    if len(xy) < 100 or len(np.unique(ids)) != len(ids):
        raise ValueError('Need at least 100 uniquely identified measurements')
    unit = 4 * float(np.ptp(xy, axis=0).max())
    if not np.isfinite(xy).all() or unit <= 0:
        raise ValueError('Invalid spatial support')
    origin = (xy.min(0) + xy.max(0))/2
    base = (xy-origin)/unit + .5
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xy))
    pools = np.array_split(order, 2)
    roles = np.empty(len(xy), dtype='<U5')
    for pool in pools:
        a, b = int(.6*len(pool)), int(.8*len(pool))
        roles[pool[:a]] = 'train'; roles[pool[a:b]] = 'val'; roles[pool[b:]] = 'test'
    angle = np.deg2rad(35.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    translation = np.array([.06, -.04])
    cases = []
    for keep in (1., .8, .6):
        for direction in ((0,) if keep == 1 else (0, 60, 120)):
            axis = np.array([np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))])
            projection = base @ axis
            low, high = np.quantile(projection, [1-keep, keep])
            shared = (projection >= low) & (projection <= high)
            blocks = {}
            for name, pool, eligible in [('reference', pools[0], projection <= high),
                                         ('query', pools[1], projection >= low)]:
                idx = np.sort(pool[eligible[pool]])
                splits = {key: np.flatnonzero(roles[idx] == key) for key in ('train', 'val', 'test')}
                if min(map(len, splits.values())) < 10:
                    raise ValueError('Too few measurements after cropping; increase MAXIMUM')
                points = base[idx].copy()
                if name == 'query':
                    points = (points-.5) @ rotation.T + .5 + translation
                blocks[name] = dict(xy=points.astype('float32'), counts=np.asarray(counts[idx], dtype='float32'),
                    ids=ids[idx], splits=splits, truth_xy=base[idx], shared=shared[idx])
            if min(np.count_nonzero(b['shared'][b['splits']['test']]) for b in blocks.values()) < 5:
                raise ValueError('Too few shared-region held-out measurements')
            cases.append(dict(name=f'keep_{keep:g}_direction_{direction}', keep=keep, direction=direction,
                nominal_overlap=2*keep-1, empirical_parent_overlap=float(shared.mean()),
                unit=unit, blocks=blocks))
    return cases


def run_case(case, genes, root, settings, *, device, run):
    """Fit from observed views only; truth positions are touched only by scoring."""
    import torch
    import pandas as pd
    from scipy.spatial import cKDTree
    from spatsplat import GeneProfileAutoencoder, train_profile_autoencoder
    from .dlpfc_splat_engine import build_engine
    from .barseq_registration_pilot import fit_rigid
    from .barseq_external_baselines import training_pair, extend_affine
    from .dlpfc_registration import cached_fit, versions
    from .allen_registration_cosine import expression_scores

    root = Path(root)
    blocks = case['blocks']
    spec = dict(case=case['name'], settings=settings, device=device, versions=versions(),
                torch=str(torch.__version__), numpy=np.__version__)
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for p in [Path(__file__), *sorted((root/'spatsplat').rglob('*.py')),
              *sorted((root/'paper').glob('*.py'))]:
        h.update(p.read_bytes())
    for b in blocks.values():
        for a in [b['xy'], b['counts'], b['ids'], *b['splits'].values()]:
            a = np.ascontiguousarray(a)
            h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
    h.update(np.asarray(genes, dtype=str).tobytes())
    output = root/'notebooks/checkpoints/barseq_unequal_sampling'/h.hexdigest()[:20]
    output.mkdir(parents=True, exist_ok=True)
    (output/'specification.json').write_text(json.dumps(spec, indent=2))
    print(case['name'], '| cache:', output.name, flush=True)
    seed = settings['seed']
    torch.manual_seed(seed)
    def profiles(y):
        return y/y.sum(1, keepdim=True).clamp_min(1e-8)
    ae = GeneProfileAutoencoder(len(genes), 128, 32).to(device)
    train = torch.cat([profiles(torch.tensor(b['counts'][b['splits']['train']])) for b in blocks.values()])
    val = torch.cat([profiles(torch.tensor(b['counts'][b['splits']['val']])) for b in blocks.values()])
    path = output/'autoencoder.pt'
    if path.exists():
        ae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        print('Loaded autoencoder', flush=True)
    else:
        if not run['fields']: raise FileNotFoundError('Enable RUN[fields] for missing autoencoder')
        train_profile_autoencoder(ae, torch.cat((train, val)), torch.arange(len(train)),
            torch.arange(len(train), len(train)+len(val)), epochs=settings['ae_epochs'], batch_size=512,
            optimizer=torch.optim.Adam(ae.parameters(), lr=.001))
        torch.save(ae.state_dict(), path)
    ae.eval().requires_grad_(False)
    engine = build_engine(settings['field'], ae, device)
    models = {}; tensors = {}; reconstruction = []
    for i, (name, b) in enumerate(blocks.items()):
        torch.manual_seed(seed+i)
        x = torch.tensor(b['xy'], device=device)
        y = torch.tensor(b['counts'], device=device)
        split = {k: torch.tensor(v, device=device) for k, v in b['splits'].items()}
        tr = split['train']
        model = engine['init_section_splat_model'](x[tr], profiles(y[tr]), y[tr].sum(1, keepdim=True),
            torch.arange(len(tr), device=device))
        path = output/f'field_{name}.pt'
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            print('Loaded field:', name, flush=True)
        else:
            if not run['fields']: raise FileNotFoundError(f'Enable RUN[fields] for {name}')
            mask, off = engine['build_section_tissue_support'](b['xy'][b['splits']['train']])
            fit = engine['train_section_splat_model'](x, y, split, model, label=name, seed=seed+i,
                off_tissue_points=off.to(device), tissue_mask_tensor=mask.to(device), min_steps=settings['min_field_steps'])
            model = fit['model']; torch.save(model.state_dict(), path)
        models[name] = model.eval().requires_grad_(False)
        tensors[name] = (x, split)
        te = b['splits']['test']; observed = b['counts'][te]
        with torch.no_grad():
            predicted = torch.cat([model(z).exp().cpu() for z in x[split['test']].split(512)]).numpy()
        tr = b['splits']['train']
        neighbors = cKDTree(b['xy'][tr]).query(b['xy'][te], k=min(16, len(tr)))[1]
        for label, pred in [('Splat', predicted), ('KNN', b['counts'][tr][neighbors].mean(1)),
                            ('Constant', np.broadcast_to(b['counts'][tr].mean(0), observed.shape))]:
            cosine = (pred*observed).sum(1)/np.maximum(np.linalg.norm(pred, axis=1)*np.linalg.norm(observed, axis=1), 1e-12)
            reconstruction.append(dict(view=name, method=label, cosine=float(cosine.mean()), mae=float(np.abs(pred-observed).mean())))

    rb, qb = blocks['reference'], blocks['query']
    mappings = {'Unregistered': qb['xy'].copy()}
    for label, expression in [('Geometry-only rigid', False), ('Splat expression-only rigid', True)]:
        path = output/('pose_expression.pt' if expression else 'pose_geometry.pt')
        if path.exists():
            result = torch.load(path, map_location='cpu', weights_only=True)
            print('Loaded pose:', label, flush=True)
        else:
            if not run['registration']: raise FileNotFoundError('Enable RUN[registration]')
            result = fit_rigid(tensors['reference'][0], tensors['query'][0], models['reference'], models['query'],
                tensors['reference'][1], tensors['query'][1], settings['registration'], use_expression=expression)
            torch.save(result, path)
        mappings[label] = result['mapped'].numpy()
    pair = training_pair(blocks, 'reference', 'query', genes)
    status = []
    for family in ('PASTE', 'CAST'):
        try:
            result, state = cached_fit(family, pair, settings['external'][family],
                root/'notebooks/checkpoints/barseq_unequal_sampling_external',
                root/'figures/barseq_unequal_sampling/native', run=run[family])
            if result is not None:
                mapped_train = result['affine'] if family == 'CAST' else result['mapped']
                mapped, _, _ = extend_affine(pair['xy'][1], mapped_train, qb['xy'], rigid=family == 'PASTE')
                mappings['CAST affine' if family == 'CAST' else 'PASTE rigid'] = mapped
            status.append(dict(method=family, status=state))
        except Exception as exc:
            status.append(dict(method=family, status='failed', error=repr(exc)))
            print(f'{family} failed: {exc}', flush=True)
    # Oracle correspondence is available only here, after all maps are fitted.
    mappings['Known transform (oracle)'] = qb['truth_xy'].copy()
    rt = rb['splits']['test']; rt = rt[rb['shared'][rt]]
    qt = qb['splits']['test']; qt = qt[qb['shared'][qt]]
    radius = 2*float(np.median(cKDTree(rb['xy'][rt]).query(rb['xy'][rt], k=2)[0][:, 1]))
    rows = []
    for label, mapped in mappings.items():
        error = np.linalg.norm(mapped[qt]-qb['truth_xy'][qt], axis=1)
        metrics = expression_scores(rb['xy'][rt], mapped[qt], rb['counts'][rt], qb['counts'][qt],
            radius=radius, k=5, permutations=99, seed=seed)
        rows.append(dict(method=label, median_error_native=float(np.median(error)*case['unit']),
            q90_error_native=float(np.quantile(error, .9)*case['unit']),
            median_error_neighbor_spacings=float(np.median(error)/(radius/2)),
            reference_test_shared=len(rt), query_test_shared=len(qt), **metrics))
    scores = pd.DataFrame(rows)
    scores.to_csv(output/'scores.csv', index=False)
    pd.DataFrame(status).to_csv(output/'external_status.csv', index=False)
    pd.DataFrame(reconstruction).to_csv(output/'reconstruction.csv', index=False)
    np.savez_compressed(output/'maps.npz', **{str(i): a for i, a in enumerate(mappings.values())})
    (output/'map_labels.json').write_text(json.dumps(list(mappings)))
    return dict(scores=scores, mappings=mappings, reconstruction=pd.DataFrame(reconstruction),
                status=pd.DataFrame(status), output=output)
