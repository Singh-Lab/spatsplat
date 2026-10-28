"""Recorded-transform version of the existing CAST adapter.

The fitting body and native-objective checkpoint selection are preserved from
paper.registration_baselines.run_cast. Added state capture enables exact native
FFD replay at held-out coordinates; no interpolation of mapped spots is used.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import numpy as np
from paper.registration_baselines import (_preserve_plot_settings,validate_pair,_seed,
    _numpy,select_coordinate_state,OrientationGuard,deformation_diagnostics)

@_preserve_plot_settings
def run_cast_recorded(xy_a, xy_b, expression_a, expression_b, *, output_dir,
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
    # Record the selected native transforms, not an interpolated displacement map.
    map_state=dict(prelocate_theta=theta.detach().cpu().clone(),
        affine_theta=None if affine_step==0 else affine[2][affine_step-1].detach().cpu().clone(),
        mean_query=torch.as_tensor(mean_q).cpu(),mean_reference=torch.as_tensor(mean_r).cpu(),
        coordinate_scale=float(coordinate_scale),meshes=[],offset=None,max_xy=None)
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
        map_state.update(meshes=[m.detach().cpu().clone() for m in ffd[1][:ffd_step]],
            offset=offset.detach().cpu(),max_xy=(q-offset).max(0)[0].detach().cpu())
        torch.save({'objectives': ffd_objectives, 'coordinates': ffd[4],
                    'selected_step': ffd_step, 'offset': offset.cpu()},
                   output_dir / 'selected_affine_ffd_log.pt')
    mapped = (selected + mean_r) / coordinate_scale
    torch.save(map_state,output_dir/'native_map.pt')
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

