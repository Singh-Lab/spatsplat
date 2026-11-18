"""Allen benchmark orchestration; frozen DLPFC implementations remain untouched."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance

from .dlpfc_registration import pair_key, fit_splats, rigid_icp, fit_paste
from .dlpfc_registration_metrics import spatial_coupling
from .dlpfc_mlp_diagnostics import load_or_fit_mlps, affine_control
from .registration_baselines import run_cast, deformation_diagnostics

PAIRS = [('development', 24, 25), ('evaluation', 28, 29), ('evaluation', 32, 33)]
CONDITIONS = ('natural', 'same_section_affine', 'same_section_smooth')


def make_case(cohort, reference, query, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown condition')
    a = cohort['blocks'][reference]
    b = cohort['blocks'][query] if condition == 'natural' else a
    # The fixed padding accommodates every predefined perturbation; no per-method normalization.
    raw = np.vstack([a['xy'], cohort['blocks'][query]['xy']])
    unit = float(np.ptp(raw, axis=0).max()) * 2
    origin = (raw.min(0) + raw.max(0))/2 - unit/2
    x, truth = (a['xy']-origin)/unit, (b['xy']-origin)/unit
    y = truth.copy()
    if condition != 'natural':
        angle = np.deg2rad(20)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        y = (y-.5) @ rotation.T + .5 + np.array([.035, -.025])
        if condition == 'same_section_smooth':
            y[:, 0] += .025*np.sin(2*np.pi*y[:, 1])
    if np.any(np.vstack([x, y]) <= 0) or np.any(np.vstack([x, y]) >= 1):
        raise ValueError('Predeclared frame does not contain the perturbation')
    pair = dict(reference=a['section'], query=b['section'], xy=[x, y],
        counts=[a['counts'], b['counts']], genes=cohort['genes'], spots=[a['ids'], b['ids']])
    # Evaluation targets and anatomical labels are never supplied to method fitting.
    evaluation = dict(labels=[a['labels'], b['labels']], truth=truth if condition != 'natural' else None,
                      coordinate_unit=unit, condition=condition)
    return pair, evaluation


def protocol_key(config):
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    root = Path(__file__).resolve().parents[1]
    for name in ['paper/allen_registration_benchmark.py', 'paper/data_loaders/allen_registration.py',
                 'paper/dlpfc_mlp_diagnostics.py', 'paper/dlpfc_registration.py',
                 'paper/dlpfc_splat_engine.py', 'paper/registration_baselines.py',
                 'paper/registration_perturbations.py', 'paper/dlpfc_paste_solver.py',
                 'paper/dlpfc_registration_metrics.py']:
        h.update((root/name).read_bytes())
    for path in sorted((root/'spatsplat').rglob('*.py')):
        h.update(path.read_bytes())
    return h.hexdigest()[:20]


def cached_method(family, pair, config, root, figures, *, run=False):
    key = pair_key(pair, dict(family=family, settings=config, implementation=protocol_key(config)))
    folder = Path(root)/family/key
    path = folder/'alignment.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}, folder
    if not run:
        raise FileNotFoundError(f'Missing {family} cache; enable RUN[{family!r}]')
    folder.mkdir(parents=True, exist_ok=True)
    if family == 'Geometry':
        result = rigid_icp(*pair['xy'], **config)
    elif family == 'PASTE':
        # Same declared KL-on-nonnegative-log2 convention as the earlier Allen experiment.
        result = fit_paste(pair, **config)
    elif family == 'CAST':
        values = [x.toarray() for x in pair['counts']]
        fit = run_cast(*pair['xy'], *values, output_dir=folder/'work', figure_dir=Path(figures)/key,
                       return_details=True, allow_reflection=False, **config)
        result = dict(mapped=fit['mapped_b'], affine=fit['affine_mapped_b'])
    elif family == 'Splat':
        result = fit_splats(pair, folder/'work', **config)
    elif family == 'MLP':
        models, model_folder = load_or_fit_mlps(pair, config['field'], folder/'fields', run=True)
        mapped = affine_control(pair, models, config['registration'], folder/'registration', run=True)
        result = dict(mapped=mapped, model_folder=np.array(str(model_folder)))
    else:
        raise ValueError('Unknown method family')
    for key in ('mapped', 'affine'):
        if key in result and (result[key].shape != pair['xy'][1].shape or not np.isfinite(result[key]).all()):
            raise ValueError(f'Invalid {family} mapping')
    result['metadata'] = np.array(json.dumps(dict(config=config, work=str(folder/'work'))))
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **result)
    temporary.replace(path)
    return result, folder


def annotation_agreement(labels, coupling):
    a, b = [np.asarray(x, dtype=float) for x in labels]
    valid_a, valid_b = np.isfinite(a), np.isfinite(b)
    if coupling.shape != (len(a), len(b)) or not np.isfinite(coupling).all() or (coupling < 0).any() or not np.isclose(coupling.sum(), 1):
        raise ValueError('Invalid coupling')
    scored = float(coupling[np.ix_(valid_a, valid_b)].sum())
    rows = []
    for label in np.union1d(a[valid_a], b[valid_b]):
        mass = float(coupling[np.ix_(a == label, b == label)].sum())
        ref = float(coupling[a == label].sum())
        rows.append(dict(parcellation=int(label), matching_mass=mass, reference_mass=ref,
                         recall_given_reference=mass/ref if ref else np.nan))
    matched = sum(row['matching_mass'] for row in rows)
    return dict(matching_mass=matched, scored_mass=scored,
                agreement_given_scored=matched/scored if scored else np.nan), rows


def score_case(pair, evaluation, mappings, native=None):
    a, original = pair['xy']
    spacing = float(np.median(cKDTree(a).query(a, k=2)[0][:, 1]))
    if spacing <= 0:
        raise ValueError('Duplicate coordinates invalidate spacing metrics')
    metrics, layers = [], []
    for method, mapped in mappings.items():
        pi = spatial_coupling(a, mapped)
        row, per_region = annotation_agreement(evaluation['labels'], pi)
        historical = native if method == 'PASTE' and native is not None else spatial_coupling(a, mapped, historical_normalization=True)
        secondary, _ = annotation_agreement(evaluation['labels'], historical)
        row.update(method=method, historical_agreement=secondary['matching_mass'],
            historical_readout='native PASTE' if method == 'PASTE' and native is not None else 'normalized spatial OT',
            query_coverage_2_spacings=float(np.mean(cKDTree(a).query(mapped)[0] <= 2*spacing)),
            reference_coverage_2_spacings=float(np.mean(cKDTree(mapped).query(a)[0] <= 2*spacing)),
            transport_distance_spacings=float(np.sum(pi*distance.cdist(a, mapped))/spacing),
            **deformation_diagnostics(original, mapped))
        truth = evaluation['truth']
        if truth is not None:
            error = np.linalg.norm(mapped-truth, axis=1)
            row.update(median_recovery_error_spacings=float(np.median(error)/spacing),
                       p90_recovery_error_spacings=float(np.quantile(error, .9)/spacing))
        metrics.append(row)
        layers.extend(dict(method=method, **r) for r in per_region)
    return pd.DataFrame(metrics), pd.DataFrame(layers)
