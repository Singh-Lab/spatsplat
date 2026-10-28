"""Out-of-sample coordinate readouts of the fitted native registration maps."""
import numpy as np
import torch
from paper.barseq_external_baselines import extend_affine


def splat_maps(xy, state, options):
    """Replay the engine's affine inverse and contractive bilinear-grid inverse.

    The optimizer is unchanged. Inputs are coordinates only; no target expression,
    overlap mask, or true registration is accepted by this readout.
    """
    x=torch.as_tensor(xy,dtype=torch.float32)
    def inverse(points,affine):
        return (points-affine['translation'])@torch.linalg.inv(affine['linear']).T
    with torch.no_grad():
        affine=inverse(x,state['affine'])
        warp=state['warp']
        base={'linear':warp['base_affine.linear'],'translation':warp['base_affine.translation']}
        estimate=inverse(x,base)
        def forward(points):
            grid=points.view(1,-1,1,2)*2-1
            displacement=torch.nn.functional.grid_sample(warp['displacement'],grid,
                mode='bilinear',padding_mode='border',align_corners=True).view(2,-1).T
            return displacement
        for _ in range(options['nonaffine_inverse_steps']):
            updated=inverse(x-forward(estimate),base)
            delta=torch.linalg.vector_norm(updated-estimate,dim=1).max()
            estimate=updated
            if float(delta)<=options['nonaffine_inverse_tolerance']: break
        roundtrip=estimate@base['linear'].T+base['translation']+forward(estimate)
        error=torch.linalg.vector_norm(roundtrip-x,dim=1)
    return {'affine':affine.numpy(),'mapped':estimate.numpy()},dict(
        roundtrip_max=float(error.max()),
        unconverged_fraction=float((error>10*options['nonaffine_inverse_tolerance']).float().mean()))


def cast_maps(xy,state):
    """Apply the selected native affine state and each selected native FFD step."""
    from CAST.CAST_Stack import affine_trans_t, BSpline_GD_preparation, BSpline_GD_uv_ij_calculate, BSpline_renew_coords
    scale=float(state['coordinate_scale'])
    x=torch.tensor(np.asarray(xy)*scale-state['mean_query'].numpy(),dtype=torch.float32)
    with torch.no_grad():
        x=affine_trans_t(state['prelocate_theta'],x)
        if state['affine_theta'] is not None: x=affine_trans_t(state['affine_theta'],x)
        affine=(x+state['mean_reference'])/scale
        if state['meshes']:
            x=x-state['offset']
            _,_,kls,_,delta=BSpline_GD_preparation(state['max_xy'],8,'cpu',None)
            for mesh in state['meshes']:
                uv,ij=BSpline_GD_uv_ij_calculate(x,delta,'cpu')
                x=BSpline_renew_coords(uv,kls,ij,mesh)
            x=x+state['offset']
        mapped=(x+state['mean_reference'])/scale
    return dict(affine=affine.numpy(),mapped=mapped.numpy())


def check_replay(replayed,native, *, tolerance=2e-5):
    """Never substitute interpolation of training displacements for a native map."""
    errors={}
    for key in replayed:
        a,b=np.asarray(replayed[key]),np.asarray(native[key])
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'Invalid native/replayed map: {key}')
        error=float(np.max(np.abs(a-b)))
        if error>tolerance: raise ValueError(f'{key} replay mismatch: {error:.4g} > {tolerance:.4g}')
        errors[key+'_replay_max_error']=error
    return errors


def readout(family,pair,result,work,xy,options):
    if family in ('Geometry','PASTE'):
        mapped,_,audit=extend_affine(pair['xy'][1],result['mapped'],xy,rigid=True)
        return dict(mapped=mapped),audit
    if family=='Splat':
        state=torch.load(work/'transforms.pt',map_location='cpu',weights_only=True)
        native,audit=splat_maps(pair['xy'][1],state,options['engine_options'])
        audit.update(check_replay(native,result))
        mapped,heldout=splat_maps(xy,state,options['engine_options'])
        audit.update({'all_coordinates_'+k:v for k,v in heldout.items()})
        return mapped,audit
    if family=='CAST':
        state=torch.load(work/'native_map.pt',map_location='cpu',weights_only=True)
        native=cast_maps(pair['xy'][1],state)
        audit=check_replay(native,result)
        return cast_maps(xy,state),audit
    raise ValueError(f'Unknown method family: {family}')
