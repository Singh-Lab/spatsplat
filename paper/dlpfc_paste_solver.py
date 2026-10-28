"""Correct and audit PASTE 1.4 FGW without modifying installed/global code."""
from inspect import signature
from types import FunctionType
import warnings

import numpy as np


SOLVER_REVISION = 'weighted-linesearch-v2'


def _clone(function, scope):
    local = FunctionType(function.__code__, scope, function.__name__,
                         function.__defaults__, function.__closure__)
    local.__kwdefaults__ = function.__kwdefaults__
    return local


def _checked_fgw(fgw, *args, inner_iterations=1000000, **kwargs):
    import ot
    kwargs['numItermaxEmd'] = inner_iterations
    kwargs['log'] = True
    bound = signature(fgw).bind(*args, **kwargs)
    bound.apply_defaults()
    parameters = bound.arguments
    M, C1, C2 = (parameters[k] for k in ('M', 'C1', 'C2'))
    p, q, alpha = (parameters[k] for k in ('p', 'q', 'alpha'))
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('PASTE alpha must be in [0, 1]')
    scope = dict(fgw.__globals__)
    original_line_search = scope['solve_gromov_linesearch']

    def weighted_line_search(*line_args, **line_kwargs):
        # PASTE 1.4 passes M=0 and reg=1 here, inconsistent with its CG cost.
        line_kwargs.update(M=(1-alpha)*M, reg=alpha)
        return original_line_search(*line_args, **line_kwargs)

    scope['solve_gromov_linesearch'] = weighted_line_search
    coupling, log = _clone(fgw, scope)(*args, **kwargs)
    losses = np.asarray([float(x) for x in log['loss']], dtype=float)
    if len(losses) < 2 or not np.isfinite(losses).all():
        raise RuntimeError('PASTE returned an invalid objective history')

    nx = ot.backend.get_backend(M, C1, C2, coupling)
    const, h1, h2 = ot.gromov.init_matrix(C1, C2, p, q, parameters['loss_fun'])
    expression = float(nx.sum(M*coupling))
    geometry = float(ot.gromov.gwloss(const, h1, h2, coupling))
    recomputed = (1-alpha)*expression + alpha*geometry
    # Float32 distance contractions can lose precision, especially near zero.
    dtype = np.asarray(nx.to_numpy(coupling)).dtype
    eps = np.finfo(dtype).eps
    tolerance = max(1e-7, 64*eps*max(1., abs(losses[0]), abs(recomputed)))
    if not np.isfinite(recomputed) or np.any(losses < -tolerance) or recomputed < -tolerance:
        raise RuntimeError('PASTE returned a nonfinite or negative FGW objective')
    error = abs(float(losses[-1])-recomputed)
    if error > tolerance:
        raise RuntimeError(f'PASTE logged/recomputed objective mismatch: {error:.4g} > {tolerance:.4g}')
    delta = abs(losses[-1]-losses[-2])
    converged = (delta < parameters['tol_abs'] or
                 delta/max(abs(losses[-1]), 1e-300) < parameters['tol_rel'])
    diagnostics = dict(solver_revision=SOLVER_REVISION, inner_emd_iterations=inner_iterations,
        outer_iterations_completed=len(losses)-1, outer_converged=bool(converged),
        objective_history=losses.tolist(), final_objective=float(losses[-1]),
        recomputed_objective=recomputed, objective_consistency_error=error,
        objective_consistency_tolerance=tolerance, expression_cost=expression,
        geometry_cost=geometry, weighted_expression_cost=(1-alpha)*expression,
        weighted_geometry_cost=alpha*geometry)
    if not converged:
        raise RuntimeError('PASTE outer objective has not converged; increase iterations')
    return coupling, log, diagnostics


def checked_pairwise_align(*slices, inner_iterations=1000000, **options):
    import paste
    if inner_iterations < 1:
        raise ValueError('inner_iterations must be positive')
    original = paste.pairwise_align
    scope = dict(original.__globals__)
    fgw = scope['my_fused_gromov_wasserstein']
    diagnostics = {}

    def checked_fgw(*args, **kwargs):
        coupling, log, audit = _checked_fgw(fgw, *args, inner_iterations=inner_iterations, **kwargs)
        diagnostics.update(audit)
        return coupling, log

    scope['my_fused_gromov_wasserstein'] = checked_fgw
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='numItermax reached before optimality.*')
        result = _clone(original, scope)(*slices, **options)
    return result, diagnostics
