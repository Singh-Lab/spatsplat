"""Expose PASTE 1.4's inner EMD budget without modifying installed/global code."""
from types import FunctionType
import warnings

import numpy as np


def checked_pairwise_align(*slices, inner_iterations=1000000, **options):
    import paste
    if inner_iterations < 1:
        raise ValueError('inner_iterations must be positive')
    original = paste.pairwise_align
    scope = dict(original.__globals__)
    fgw = scope['my_fused_gromov_wasserstein']
    diagnostics = {}

    def checked_fgw(*args, **kwargs):
        # pairwise_align accepts **kwargs but does not forward them to FGW.
        # Its FGW helper does forward numItermaxEmd to POT's conditional gradient.
        kwargs['numItermaxEmd'] = inner_iterations
        coupling, log = fgw(*args, **kwargs)
        losses = np.asarray([float(x) for x in log['loss']])
        if not np.isfinite(losses).all() or len(losses) < 2:
            raise RuntimeError('PASTE returned an invalid objective history')
        delta = abs(losses[-1] - losses[-2])
        converged = delta < 1e-9 or delta / max(abs(losses[-1]), 1e-300) < 1e-9
        diagnostics.update(inner_emd_iterations=inner_iterations,
            outer_iterations_completed=len(losses)-1, outer_converged=bool(converged),
            objective_history=losses.tolist(), final_objective=float(losses[-1]))
        if not converged:
            raise RuntimeError('PASTE outer objective has not converged; increase iterations')
        return coupling, log

    scope['my_fused_gromov_wasserstein'] = checked_fgw
    local_align = FunctionType(original.__code__, scope, original.__name__, original.__defaults__, original.__closure__)
    local_align.__kwdefaults__ = original.__kwdefaults__
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='numItermax reached before optimality.*')
        result = local_align(*slices, **options)
    return result, diagnostics
