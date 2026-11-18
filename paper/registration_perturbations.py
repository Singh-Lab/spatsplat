"""Controlled registration stress tests; coordinates remain in one fixed frame."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from spatsplat import compute_splat_weights


def perturbation_cases(seeds=(1103, 1104, 1105), levels=((15., .08), (35., .15))):
    """Translations are fractions of the original shared section coordinate scale."""
    cases = [dict(name='identity', level=0, seed=0, angle_degrees=0., translation=[0., 0.])]
    for level, (angle, distance) in enumerate(levels, 1):
        if angle < 0 or distance < 0 or not np.isfinite([angle, distance]).all():
            raise ValueError('Perturbation magnitudes must be finite and nonnegative')
        for seed in seeds:
            rng = np.random.default_rng(seed)
            direction = rng.uniform(-np.pi, np.pi)
            cases.append(dict(name=f'level{level}_seed{seed}', level=level, seed=int(seed),
                              angle_degrees=float(angle * rng.choice([-1, 1])),
                              translation=(distance * np.array([np.cos(direction), np.sin(direction)])).tolist()))
    if len({c['name'] for c in cases}) != len(cases):
        raise ValueError('Seeds must be distinct')
    return cases


def transform_coordinates(xy, case, *, inverse=False):
    theta = np.deg2rad(case['angle_degrees'])
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    translation = np.asarray(case['translation'])
    xy = np.asarray(xy, dtype=np.float64)
    if inverse:
        return (xy - .5 - translation) @ rotation + .5
    return (xy - .5) @ rotation.T + .5 + translation


def common_padded_frame(xy_a, xy_b, cases, margin=.05):
    """Use a rotation-invariant bound, never re-center or clip a perturbed section."""
    if not 0 < margin < .5:
        raise ValueError('margin must lie between 0 and 0.5')
    for xy in (xy_a, xy_b):
        if np.asarray(xy).ndim != 2 or np.asarray(xy).shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError('Expected finite N x 2 coordinates')
    radius = max(np.linalg.norm(np.asarray(xy) - .5, axis=1).max() for xy in (xy_a, xy_b))
    translation = max(np.linalg.norm(c['translation']) for c in cases)
    divisor = max(1., (radius + translation) / (.5 - margin))
    a = (np.asarray(xy_a) - .5) / divisor + .5
    targets = {c['name']: (transform_coordinates(xy_b, c) - .5) / divisor + .5 for c in cases}
    return a, targets, float(divisor)


def experiment_fingerprint(config, arrays):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for value in arrays:
        value = np.ascontiguousarray(value)
        if value.dtype.hasobject:
            raise ValueError('Object arrays cannot be fingerprinted reproducibly')
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()[:20]


def cached_field(path, build, fit, *, run=False, force=False):
    """Only weights and scalar diagnostics are serialized, never Python models."""
    path = Path(path)
    if not (path.exists() and not force) and not run:
        return None
    model = build()
    if path.exists() and not force:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['state_dict'])
        print(f'Loaded field: {path.name}', flush=True)
    else:
        result = fit(model)
        model = result['model']
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.pt')
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'training_budget': result['training_budget']}, temporary)
        temporary.replace(path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def cached_mappings(path, fit, *, run=False, force=False):
    path = Path(path)
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as saved:
            result = {k: saved[k] for k in saved.files}
        if any(v.ndim != 2 or v.shape[1] != 2 or not np.isfinite(v).all() for v in result.values()):
            raise ValueError(f'Invalid coordinate cache: {path}')
        print(f'Loaded alignment: {path.name}', flush=True)
        return result
    if not run:
        print(f'Skipped alignment (training disabled; no usable cache): {path.name}', flush=True)
        return {}
    print(f'Fitting alignment: {path.name}', flush=True)
    start = time.perf_counter()
    result = fit()
    if not result or any(v.ndim != 2 or v.shape[1] != 2 or not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Alignment returned invalid coordinates')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **result)
    temporary.replace(path)
    print(f'Completed {path.name} in {time.perf_counter() - start:.4g} seconds', flush=True)
    return result


@torch.no_grad()
def splat_anchors(model_a, model_b, xy_a, xy_b, affine, cosine_threshold=.8):
    """The notebook's active, nearest-center, profile-filtered anchor rule."""
    active = []
    profiles = []
    for model, xy in [(model_a, xy_a), (model_b, xy_b)]:
        active.append(compute_splat_weights(model, xy).mean(0) > 1e-4)
        logits, _ = model.decoder(model.latents)
        profile = torch.softmax(logits, dim=1)
        profiles.append(torch.nn.functional.normalize(profile, dim=1))
    mapped_b = affine.inverse(model_b.means)
    active[1] &= ((mapped_b >= 0) & (mapped_b <= 1)).all(1)
    a, b = model_a.means[active[0]], model_b.means[active[1]]
    if not len(a) or not len(b):
        return b[:0], a[:0], b.new_empty(0)
    nearest = torch.cdist(mapped_b[active[1]], a).argmin(1)
    cosine = (profiles[1][active[1]] * profiles[0][active[0]][nearest]).sum(1)
    keep = cosine >= cosine_threshold
    return b[keep], a[nearest][keep], cosine[keep]


def consistency_metrics(mapped, unperturbed, coordinate_divisor):
    """Equivariance to input perturbation, NOT anatomical registration accuracy."""
    mapped, unperturbed = np.asarray(mapped), np.asarray(unperturbed)
    if mapped.shape != unperturbed.shape or mapped.ndim != 2 or mapped.shape[1] != 2:
        raise ValueError('Corresponding B measurement coordinates are required')
    distance = np.linalg.norm(mapped - unperturbed, axis=1) * coordinate_divisor
    return {'consistency_rms': float(np.sqrt(np.mean(distance**2))),
            'consistency_median': float(np.median(distance)),
            'consistency_q90': float(np.quantile(distance, .9))}
