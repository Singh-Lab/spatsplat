"""Training-only PASTE2 and MLP-MAE rigid additions; existing benchmark unchanged."""
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
from types import FunctionType, SimpleNamespace
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from paper.dlpfc_registration import pair_key, versions
from paper.registration_baselines import proper_procrustes, _seed
from paper.barseq_external_baselines import extend_affine
from paper.barseq_registration_pilot import fit_rigid
from paper.dlpfc_mlp_diagnostics import load_or_fit_mlps
from .standardized_rigid import map_rigid, load_fields
from .standardized_maps import check_replay
from . import standardized_registration as base

EXTRA_METHODS = ('PASTE2 rigid', 'MLP-MAE rigid')
PRE_RETRY_SOURCE = '6f61afc631ab59970134b91dd56b7b0b2ef74c817f90740d432b02f857d0a22a'
PRE_EMD_RETRY_SOURCE = 'b933532ef5d0b67e0d75c667c66786d2d4ea141fab558d1f3a42e164787db354'


def clone(function, **updates):
    scope = dict(function.__globals__); scope.update(updates)
    result = FunctionType(function.__code__, scope, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def validate_partial(coupling, p, q, mass, tolerance=1e-6):
    coupling = np.asarray(coupling, dtype=float)
    if coupling.shape != (len(p), len(q)) or not np.isfinite(coupling).all():
        raise ValueError('Invalid partial coupling shape/values')
    if (coupling.min() < -tolerance or abs(coupling.sum()-mass) > tolerance or
            np.any(coupling.sum(1)-p > tolerance) or np.any(coupling.sum(0)-q > tolerance)):
        raise ValueError('PASTE2 violates partial-transport mass/marginal constraints')


def edge_inconsistency(graph, labels):
    edges = list(graph.edges())
    if not edges:
        raise ValueError('PASTE2 overlap selection produced an empty graph')
    return float(np.mean([labels[a] != labels[b] for a, b in edges]))


def emd_with_retry(solver, a, b, cost, *, attempts, **kwargs):
    """Retry only a reported iteration limit; never accept an unconverged LP."""
    for limit in (1000000, 5000000, 20000000):
        coupling, log = solver(a, b, cost, **dict(kwargs, numItermax=limit, log=True))
        warning = log.get('warning')
        attempts.append(dict(iteration_limit=limit, warning=warning))
        if warning is None:
            if (not np.isfinite(coupling).all() or coupling.min() < -1e-7 or
                    not np.allclose(coupling.sum(1), a, atol=1e-7, rtol=1e-6) or
                    not np.allclose(coupling.sum(0), b, atol=1e-7, rtol=1e-6)):
                raise RuntimeError('PASTE2 inner transport returned invalid balanced marginals')
            return coupling, log
        if 'numItermax' not in str(warning):
            raise RuntimeError(f'PASTE2 inner transport failed: {warning}')
        print(f'PASTE2 inner transport reached {limit} iterations; retrying if budget remains.', flush=True)
    raise RuntimeError('PASTE2 inner transport did not converge within 20000000 iterations')


def paste2_adapter(iterations=1000):
    """Local compatibility/audit wrappers; never mutate installed PASTE2 globals."""
    if importlib.metadata.version('paste2') != '1.0.1':
        raise ImportError('This adapter is audited for paste2==1.0.1')
    from paste2 import PASTE2 as native
    from paste2 import model_selection as selection
    audit = []

    def checked_solver(M, C1, C2, p, q, **kwargs):
        emd_attempts = []
        def emd(*args, **options):
            return emd_with_retry(native.ot.lp.emd, *args, attempts=emd_attempts, **options)
        local_ot = SimpleNamespace(**vars(native.ot))
        local_ot.lp = SimpleNamespace(**vars(native.ot.lp))
        local_ot.lp.emd = emd
        solver = clone(native.partial_fused_gromov_wasserstein, ot=local_ot)
        mass = min(float(kwargs['m']), float(np.sum(p)), float(np.sum(q)))
        if not 0 < mass <= 1:
            raise ValueError('Invalid PASTE2 transported mass')
        kwargs.update(m=mass, numItermax=iterations, log=True)
        # Native default starts with unit mass even for a partial problem.
        if kwargs.get('G0') is None:
            kwargs['G0'] = mass*np.outer(p/np.sum(p), q/np.sum(q))
        validate_partial(kwargs['G0'], p, q, mass)
        initial = native.fgwloss_partial(kwargs['alpha'], M, C1, C2, kwargs['G0'])
        coupling, log = solver(M, C1, C2, p, q, **kwargs)
        validate_partial(coupling, p, q, mass)
        losses = np.asarray([initial, *log['loss']], dtype=float)
        rows, cols = coupling.sum(1), coupling.sum(0)
        geometry = rows@(C1*C1)@rows + cols@(C2*C2)@cols - 2*np.sum((C1@coupling@C2.T)*coupling)
        recomputed = (1-kwargs['alpha'])*np.sum(M*coupling)+kwargs['alpha']*geometry
        tol = 1e-6*max(1., abs(float(initial)), abs(float(recomputed)))
        if (len(losses) < 2 or not np.isfinite(losses).all() or losses.min() < -tol
                or not np.isfinite(recomputed) or abs(log['partial_fgw_cost']-recomputed) > tol
                or np.any(np.diff(losses) > tol)):
            raise RuntimeError('PASTE2 objective failed finite/consistency/descent audit')
        converged = bool(abs(losses[-1]-losses[-2]) <= max(1e-9, 1e-9*abs(losses[-1])))
        record = dict(mass=mass, final_objective=float(recomputed), steps=len(losses)-1,
                      converged=converged, objective_history=losses.tolist(), emd_attempts=emd_attempts)
        audit.append(record)
        if not converged:
            raise RuntimeError('PASTE2 did not converge; increase its iteration budget')
        return coupling, log

    dense = lambda x: x.toarray() if sparse.issparse(x) else np.asarray(x)
    given = clone(native.partial_pairwise_align_given_cost_matrix, partial_fused_gromov_wasserstein=checked_solver)
    pairwise = clone(native.partial_pairwise_align, to_dense_array=dense,
                     partial_fused_gromov_wasserstein=checked_solver)
    # Same edge-disagreement fraction, without deprecated pandas chained writes.
    edge_score = clone(selection.calculate_convex_hull_edge_inconsistency,
                       edge_inconsistency_score=edge_inconsistency)
    selector = clone(selection.select_overlap_fraction, to_dense_array=dense,
                     calculate_convex_hull_edge_inconsistency=edge_score,
                     partial_pairwise_align_given_cost_matrix=given)
    return pairwise, selector, audit


def stable_glmpca_cost(glm, args, kwargs, seed, audit_path):
    """Retry numerical divergence only, with a fixed training-only penalty ladder."""
    from paste2.glmpca import GlmpcaError
    native = glm.__globals__['glmpca']
    attempts = []
    for penalty in (1, 10, 100, 1000):
        def regularized(*a, **kw):
            kw['penalty'] = penalty
            return native(*a, **kw)
        _seed(seed)
        try:
            cost = clone(glm, glmpca=regularized)(*args, **kwargs)
            if not np.isfinite(cost).all() or cost.max() <= 0:
                raise GlmpcaError('Numerical divergence: invalid expression costs')
        except GlmpcaError as exc:
            if 'Numerical divergence' not in str(exc):
                raise
            attempts.append(dict(penalty=penalty, status='diverged', error=str(exc)))
            audit_path.write_text(json.dumps(attempts, indent=2))
            print(f'PASTE2 GLM-PCA diverged at penalty={penalty}; retrying if budget remains.', flush=True)
            continue
        attempts.append(dict(penalty=penalty, status='finite', seed=seed))
        audit_path.write_text(json.dumps(attempts, indent=2))
        return cost
    raise RuntimeError('PASTE2 GLM-PCA diverged at penalties 1, 10, 100, 1000; see glmpca_audit.json')


def fit_paste2(pair, folder, config):
    """Only the training pair enters overlap selection and partial alignment."""
    import anndata
    _seed(config['seed'])
    pairwise, selector, audit = paste2_adapter(config['iterations'])
    slices = [anndata.AnnData(x.toarray().astype(float)) for x in pair['counts']]
    for adata, xy in zip(slices, pair['xy']):
        adata.var_names = pair['genes']
        adata.obsm['spatial'] = xy.copy()
    # Reuse the same training-derived GLM-PCA cost for overlap selection and final fit.
    glm = selector.__globals__['glmpca_distance']
    cost_path = folder/'glmpca_cost.npy'
    def cached_cost(*args, **kwargs):
        if cost_path.exists():
            cost = np.load(cost_path, allow_pickle=False)
        else:
            cost = stable_glmpca_cost(glm, args, kwargs, config['seed'], folder/'glmpca_audit.json')
            if not np.isfinite(cost).all() or cost.max() <= 0:
                raise ValueError('Nonfinite or uninformative PASTE2 expression costs')
            temporary = cost_path.with_suffix('.tmp.npy')
            np.save(temporary, cost); temporary.replace(cost_path)
        if cost.shape != (len(slices[0]), len(slices[1])) or not np.isfinite(cost).all():
            raise ValueError('Invalid cached PASTE2 expression costs')
        return cost
    given = selector.__globals__['partial_pairwise_align_given_cost_matrix']
    def cached_candidate(a, b, *, s, **kwargs):
        path = folder/f'overlap_{s:.2f}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                pi, objective = saved['coupling'], float(saved['objective'])
            validate_partial(pi, np.full(len(a),1/len(a)), np.full(len(b),1/len(b)), s)
            print(f'Loaded PASTE2 overlap candidate {s:.2f}', flush=True)
            return pi, objective
        pi, objective = given(a, b, s=s, **kwargs)
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, coupling=pi, objective=objective); temporary.replace(path)
        (folder/f'overlap_{s:.2f}_audit.json').write_text(json.dumps(audit[-1], indent=2))
        return pi, objective
    selector = clone(selector, glmpca_distance=cached_cost,
                     partial_pairwise_align_given_cost_matrix=cached_candidate)
    selection_path = folder/'overlap_selection.json'
    if selection_path.exists():
        overlap = float(json.loads(selection_path.read_text())['selected_overlap'])
    else:
        overlap = float(selector(*slices, alpha=config['alpha']))
        selection_path.write_text(json.dumps(dict(selected_overlap=overlap,
            rule='official training-only edge-inconsistency selector', truth_used=False), indent=2))
    if not 0 < overlap <= 1:
        raise ValueError('Invalid selected overlap')
    pairwise = clone(pairwise, glmpca_distance=cached_cost)
    coupling, objective = pairwise(*slices, s=overlap, alpha=config['alpha'],
        dissimilarity='glmpca', norm=True, return_obj=True, verbose=False)
    mapped = proper_procrustes(*pair['xy'], coupling)
    return dict(mapped=mapped, coupling=coupling, selected_overlap=np.array(overlap),
                diagnostics=np.array(json.dumps(dict(final=audit[-1], objective=objective))))


def extension_key(pair, options, root, source_signature, *, legacy=False, source_override=None):
    versions_extra = versions()
    for name in ('paste2','pandas','scanpy','statsmodels'):
        try: versions_extra[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions_extra[name] = 'missing'
    h = hashlib.sha256()
    for path in ('paper/workflows/barseq_additional_controls.py','paper/dlpfc_mlp_diagnostics.py',
                 'paper/barseq_registration_pilot.py'):
        h.update((Path(root)/path).read_bytes())
    return pair_key(pair, dict(options=options, versions=versions_extra,
                              base_source=source_signature,
                              extension_source=source_override or (PRE_RETRY_SOURCE if legacy else h.hexdigest())))


def field_diagnostics(case, models, method):
    rows = []
    for side, model in enumerate(models):
        device = next(model.parameters()).device
        idx = case['reference_eval'] if side == 0 else case['query_eval']
        samples = [('train', case['pair']['xy'][side], case['pair']['counts'][side].toarray()),
                   ('test_shared', case['reference_xy'][idx] if side == 0 else case['query_xy'][idx],
                    case['reference_values'][idx] if side == 0 else case['query_values'][idx])]
        for split, xy, counts in samples:
            with torch.no_grad():
                pred = torch.cat([model(torch.as_tensor(x, dtype=torch.float32, device=device)).exp().cpu()
                                  for x in np.array_split(xy, max(1, int(np.ceil(len(xy)/512))))]).numpy()
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite reconstruction')
            cosine = np.sum(pred*counts, axis=1)/np.maximum(np.linalg.norm(pred,axis=1)*np.linalg.norm(counts,axis=1),1e-12)
            rows.append(dict(case=case['name'], role=case['role'], method=method,
                view=('reference','query')[side], split=split, measurements=len(xy),
                mae=float(np.abs(pred-counts).mean()), profile_cosine=float(cosine.mean()),
                parameters=sum(p.numel() for p in model.parameters())))
    return rows


def run_additions(cases, fitted, status, base_options, options, root, output, signature, run):
    """A missing new method never substitutes another method's map."""
    root, output = Path(root), Path(output)
    statuses, diagnostics = [], []
    for case in cases:
        pair = case['pair']; key = extension_key(pair, options, root, signature)
        folder = root/'notebooks/checkpoints/barseq_additional_controls/fits'/key
        # The retry leaves all previously successful default-penalty fits unchanged.
        # Reuse only the exact preceding implementation/config/data/version key.
        previous = folder.parent/extension_key(pair, options, root, signature, legacy=True)
        if not folder.exists() and previous.is_dir():
            shutil.copytree(previous, folder)
            (folder/'pre_retry_cache.json').write_text(json.dumps(dict(source=str(previous)), indent=2))
            print('Reused pre-retry caches for', case['name'], flush=True)
        for method in EXTRA_METHODS:
            try:
                work = folder/('paste2' if method == 'PASTE2 rigid' else 'mlp')
                path = work/'alignment.npz'
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        result = {k:saved[k] for k in saved.files}
                    print('Loaded', case['name'], method, flush=True)
                else:
                    if not run[method]: raise FileNotFoundError(f'Enable RUN_EXTRA[{method!r}]')
                    work.mkdir(parents=True, exist_ok=True)
                    if method == 'PASTE2 rigid':
                        result = fit_paste2(pair, work, options['paste2'])
                    else:
                        models, _ = load_or_fit_mlps(pair, options['mlp'], work/'fields', run=True)
                        xy = [torch.as_tensor(x, dtype=torch.float32, device=options['mlp']['device']) for x in pair['xy']]
                        splits = [dict(train=torch.arange(len(x)), val=torch.arange(len(x))) for x in xy]
                        result_fit = fit_rigid(*xy, *models, *splits, options['rigid'], use_expression=True)
                        torch.save(result_fit, work/'rigid.pt')
                        native = result_fit['mapped'].numpy()
                        check_replay(dict(mapped=map_rigid(pair['xy'][1],result_fit['state'])),dict(mapped=native))
                        result = dict(mapped=native)
                    _, _, audit = extend_affine(pair['xy'][1], result['mapped'], pair['xy'][1], rigid=True)
                    result['replay_audit'] = np.array(json.dumps(audit))
                    temporary = path.with_suffix('.tmp.npz')
                    np.savez_compressed(temporary, **result); temporary.replace(path)
                mapped, _, audit = extend_affine(pair['xy'][1], result['mapped'], case['query_xy'], rigid=True)
                fitted[case['name']][method] = mapped
                target = output/'cases'/case['name']; target.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target/('paste2_map.npz' if method=='PASTE2 rigid' else 'mlp_map.npz'),mapped=mapped)
                row = dict(case=case['name'],method=method,status='available',cache=str(work),**audit)
                if method=='PASTE2 rigid': row['selected_overlap']=float(result['selected_overlap'])
            except Exception as exc:
                row = dict(case=case['name'],method=method,status='failed or missing',error=repr(exc))
                print(case['name'],method,repr(exc),flush=True)
            statuses.append(row)
            pd.DataFrame(statuses).to_csv(output/'additional_fit_status.csv',index=False)
        # Post-fit diagnostics cannot influence either registration objective/checkpoint.
        for method in ('MLP-MAE rigid','Splat rigid'):
            if method not in fitted[case['name']]: continue
            try:
                if method=='MLP-MAE rigid':
                    models, _ = load_or_fit_mlps(pair, options['mlp'], folder/'mlp/fields',run=False)
                else:
                    parent=status[(status.case==case['name'])&(status.family=='Splat')&(status.status=='available')]
                    models, _ = load_fields(pair,Path(parent.iloc[0]['cache'])/'work',base_options['Splat'])
                diagnostics.extend(field_diagnostics(case,models,method))
                del models
            except Exception as exc:
                diagnostics.append(dict(case=case['name'],method=method,error=repr(exc)))
            pd.DataFrame(diagnostics).to_csv(output/'field_diagnostics.csv',index=False)
    return fitted,pd.DataFrame(statuses),pd.DataFrame(diagnostics)


def score_additions(cases, fitted, output, **kwargs):
    """Use the identical common scorer via a renamed slot; no refitting occurs."""
    tables, availability = [], []
    for method in EXTRA_METHODS:
        maps = {c['name']:({'Splat affine':fitted[c['name']][method]}
                if method in fitted[c['name']] else {}) for c in cases}
        folder = Path(output)/method.lower().replace(' ','_'); folder.mkdir(parents=True,exist_ok=True)
        scores, present = base.evaluate_cases(cases,maps,folder,**kwargs)
        if not scores.empty:
            scores['method']=method; tables.append(scores)
        present=present[present.method=='Splat affine'].copy(); present['method']=method
        availability.append(present)
    return (pd.concat(tables,ignore_index=True) if tables else pd.DataFrame(),
            pd.concat(availability,ignore_index=True))
