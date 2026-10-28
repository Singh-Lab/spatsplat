"""Rigid expression-only extension reusing the completed standardized fields.

The seven-method runner remains unchanged so its expensive fit caches stay valid.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch

from . import standardized_registration as base
from .standardized_maps import check_replay
from paper.barseq_registration_pilot import Rigid2D, fit_rigid

METHODS = base.METHODS[:-2] + ('Splat rigid',) + base.METHODS[-2:]
DEFAULTS = dict(angle_starts=12, refine_starts=3, steps=300, lr=.002,
                batch_size=512, geometry_weight=0., seed=1023)


def signature(root):
    digest = hashlib.sha256()
    for name in ('paper/workflows/standardized_rigid.py', 'paper/barseq_registration_pilot.py'):
        digest.update((Path(root)/name).read_bytes())
    return digest.hexdigest()


def load_fields(pair, work, config):
    """Restore frozen fields; never train or use held-out expression here."""
    from spatsplat import GeneProfileAutoencoder
    from paper.dlpfc_splat_engine import build_engine
    device = config['device']
    cfg = config['engine_options']['splat_config']
    ae = GeneProfileAutoencoder(len(pair['genes']), cfg['hidden_dim'], cfg['latent_dim']).to(device)
    ae.load_state_dict(torch.load(work/'autoencoder.pt', map_location=device, weights_only=True))
    ae.eval().requires_grad_(False)
    engine = build_engine(config['engine_options'], ae, device)
    models, xy = [], []
    for i in range(2):
        torch.manual_seed(config['seed']+i)
        positions = torch.as_tensor(pair['xy'][i], dtype=torch.float32, device=device)
        counts = torch.from_numpy(pair['counts'][i].toarray())
        total = counts.sum(1, keepdim=True)
        model = engine['init_section_splat_model'](positions, counts/total, total, torch.arange(len(counts)))
        model.load_state_dict(torch.load(work/f'field_{i}.pt', map_location=device, weights_only=True))
        models.append(model.eval().requires_grad_(False))
        xy.append(positions)
    return models, xy


def map_rigid(xy, state):
    transform = Rigid2D('cpu')
    transform.load_state_dict(state)
    with torch.no_grad():
        return transform(torch.as_tensor(xy, dtype=torch.float32)).numpy()


def add_rigid(cases, fitted, status, config, root, output, settings=None, *, run=True):
    """Reuse field caches, fitting a proper-rigid map on supplied training data only."""
    settings = dict(DEFAULTS if settings is None else settings)
    if settings['geometry_weight'] != 0:
        raise ValueError('Splat rigid is the expression-only control')
    root, output = Path(root), Path(output)
    rows = []
    for case in cases:
        name = case['name']
        try:
            parent = status[(status.case == name) & (status.family == 'Splat') & (status.status == 'available')]
            if len(parent) != 1:
                raise RuntimeError('Complete the standardized Splat fit first')
            work = Path(parent.iloc[0]['cache'])/'work'
            digest = hashlib.sha256(json.dumps(dict(settings=settings, implementation=signature(root)), sort_keys=True).encode())
            for file in ('autoencoder.pt', 'field_0.pt', 'field_1.pt'):
                digest.update((work/file).read_bytes())
            for x in case['pair']['xy']:
                digest.update(np.ascontiguousarray(x).tobytes())
            folder = root/'notebooks/checkpoints/registration_standardized/rigid'/digest.hexdigest()[:20]
            path = folder/'rigid.pt'
            if path.exists():
                result = torch.load(path, map_location='cpu', weights_only=True)
                print('Loaded Splat rigid:', name, flush=True)
            else:
                if not run:
                    raise FileNotFoundError('Enable RUN_SPLAT_RIGID to fit the missing rigid map')
                print('Fitting Splat rigid on frozen fields:', name, flush=True)
                models, xy = load_fields(case['pair'], work, config)
                # These are all supplied fitting observations, not the BARseq evaluation split.
                # Pose selection uses decoded fields at fitting coordinates on both datasets.
                splits = [dict(train=torch.arange(len(x)), val=torch.arange(len(x))) for x in xy]
                result = fit_rigid(*xy, *models, *splits, settings, use_expression=True)
                folder.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp.pt')
                torch.save(result, temporary)
                temporary.replace(path)
                del models, xy
            replay = map_rigid(case['pair']['xy'][1], result['state'])
            audit = check_replay(dict(mapped=replay), dict(mapped=result['mapped']))
            mapped = map_rigid(case['query_xy'], result['state'])
            if not np.isfinite(mapped).all():
                raise ValueError('Nonfinite rigid coordinates')
            fitted[name]['Splat rigid'] = mapped
            rows.append(dict(case=name, dataset=case['dataset'], task=case['task'], family='Splat rigid',
                             status='available', cache=str(folder)))
            target = output/'cases'/name
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target/'rigid_map.npz', mapped=mapped)
            (target/'rigid_audit.json').write_text(json.dumps(audit, indent=2))
        except Exception as exc:
            rows.append(dict(case=name, dataset=case['dataset'], task=case['task'], family='Splat rigid',
                             status='failed or missing', error=repr(exc)))
            print(name, 'Splat rigid:', repr(exc), flush=True)
        pd.DataFrame(rows).to_csv(output/'rigid_fit_status.csv', index=False)
    return fitted, pd.concat([status, pd.DataFrame(rows)], ignore_index=True)


def evaluate_cases(cases, fitted, output, **kwargs):
    """Apply exactly the existing coordinate-based scorer to the additional map.

    The legacy scorer has a fixed roster. Its affine slot is only a scoring alias:
    no fitting is done and the resulting rows are relabeled before publication.
    Intermediate legacy reports stay separate from the combined eight-method table.
    """
    output = Path(output)
    for folder in ('base_metrics', 'rigid_metrics'):
        (output/folder).mkdir(parents=True, exist_ok=True)
    scores, availability = base.evaluate_cases(cases, fitted, output/'base_metrics', **kwargs)
    rigid_maps = {c['name']: ({'Splat affine': fitted[c['name']]['Splat rigid']}
                  if 'Splat rigid' in fitted[c['name']] else {}) for c in cases}
    extra, extra_available = base.evaluate_cases(cases, rigid_maps, output/'rigid_metrics', **kwargs)
    if not extra.empty:
        extra['method'] = 'Splat rigid'
        scores = pd.concat([scores, extra], ignore_index=True)
    extra_available = extra_available[extra_available.method == 'Splat affine'].copy()
    extra_available['method'] = 'Splat rigid'
    availability = pd.concat([availability, extra_available], ignore_index=True)
    scores.to_csv(output/'scores.csv', index=False)
    availability.to_csv(output/'method_availability.csv', index=False)
    return scores, availability
