"""Development-only MLP field control using the frozen affine registration engine."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from spatsplat.baselines.mlp import CountMLP
from spatsplat.training import mae_count_loss
from .dlpfc_registration import pair_key
from .dlpfc_splat_engine import build_engine
from .registration_baselines import _seed


def train_mlp(xy, counts, *, hidden_dim=128, epochs=300, min_steps=2000,
              batch_size=512, lr=1e-3, seed=1023, device='cuda'):
    """Full-section reconstruction, not held-out prediction or label supervision."""
    if min(len(xy), hidden_dim, epochs, batch_size, lr) <= 0 or min_steps < 0:
        raise ValueError('Invalid MLP training settings')
    _seed(seed)
    model = CountMLP(2, hidden_dim, counts.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    positions = torch.as_tensor(xy, dtype=torch.float32, device=device)
    values = torch.as_tensor(counts, dtype=torch.float32)  # Transfer only each target batch.
    updates = math.ceil(len(xy) / batch_size)
    epochs = max(epochs, math.ceil(min_steps / updates))
    history = []
    for epoch in range(epochs):
        permutation = torch.randperm(len(xy), device=device)
        total = 0.
        for indices in permutation.split(batch_size):
            loss = mae_count_loss(model(positions[indices]), values[indices.cpu()].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite MLP training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        history.append(dict(epoch=epoch + 1, train_mae=total / len(xy)))
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch + 1 == epochs:
            print(f'MLP epoch {epoch+1}: count MAE {history[-1]["train_mae"]:.4g}', flush=True)
    return model.eval().requires_grad_(False), history, epochs * updates


def load_or_fit_mlps(pair, config, root, *, run=False, force=False):
    if force and not run:
        raise ValueError('force requires run=True')
    key = hashlib.sha256((pair_key(pair, config) + Path(__file__).read_text()).encode()).hexdigest()[:20]
    folder = Path(root) / key
    models = []
    for i, sample in enumerate((pair['reference'], pair['query'])):
        path = folder / f'{sample}.pt'
        if path.exists() and not force:
            saved = torch.load(path, map_location='cpu', weights_only=True)
            model = CountMLP(2, config['hidden_dim'], len(pair['genes'])).to(config['device'])
            model.load_state_dict(saved['state'])
            model.eval().requires_grad_(False)
            print('Loaded MLP', sample, flush=True)
        elif run:
            options = {**config, 'seed': config['seed'] + i}
            model, history, steps = train_mlp(pair['xy'][i], pair['counts'][i].toarray(), **options)
            folder.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp.pt')
            torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            history=history, optimizer_steps=steps, config=options), temporary)
            temporary.replace(path)
        else:
            raise FileNotFoundError(f'Missing MLP field {path}; enable RUN_MLP')
        models.append(model)
    return models, folder


def affine_control(pair, models, config, root, *, run=False):
    """Identical registration options and sampling seed for both representations."""
    h = hashlib.sha256((pair_key(pair, config) + Path(__file__).read_text()).encode())
    for model in models:
        for name, value in sorted(model.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            h.update(str((name, array.shape, array.dtype.str)).encode())
            h.update(array.tobytes())
    path = Path(root) / h.hexdigest()[:20] / 'affine.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            mapped = data['mapped']
    elif run:
        engine = build_engine(config['engine_options'], None, config['device'])
        xy = [torch.tensor(x, dtype=torch.float32, device=config['device']) for x in pair['xy']]
        _seed(config['seed'] + 100)
        queries = engine['sample_registration_queries'](xy[0])
        fit = engine['run_affine_registration'](*models, queries, allow_reflection=False)
        with torch.no_grad():
            mapped = fit['transform'].inverse(xy[1]).cpu().numpy()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fit['transform'].state_dict(), path.parent / 'transform.pt')
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, mapped=mapped, metadata=np.array(json.dumps(dict(
            config=config, selected_step=fit['selected_step'], history=fit['history']))))
        temporary.replace(path)
    else:
        raise FileNotFoundError(f'Missing registration {path}; enable RUN_REGISTRATION')
    if mapped.shape != pair['xy'][1].shape or not np.isfinite(mapped).all():
        raise ValueError('Invalid affine mapping')
    return mapped
