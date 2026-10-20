"""In-sample field fidelity diagnostics against a spatially constant profile."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
import torch


def cosine_rows(a, b):
    numerator = np.einsum('ij,ij->i', a, np.broadcast_to(b, a.shape), dtype=np.float64)
    denominator = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=-1)
    return np.divide(numerator, denominator, out=np.full(len(a), np.nan), where=denominator > 1e-15)


def profile_metrics(observed, predicted, mean_profile, genes):
    """R2 relative to the declared constant profile, with undefined correlations NaN."""
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if observed.shape != predicted.shape or observed.ndim != 2 or observed.shape[1] != len(genes):
        raise ValueError('Profile shapes and gene panel must match')
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError('Profiles must be finite')
    rows, error_total, constant_total = [], 0., 0.
    for start in range(0, len(genes), 256):
        stop = start+256
        y, p = observed[:, start:stop].astype(float), predicted[:, start:stop].astype(float)
        baseline = np.asarray(mean_profile[start:stop], dtype=float)
        yc, pc = y-y.mean(0), p-p.mean(0)
        vy, vp = np.sum(yc*yc, axis=0), np.sum(pc*pc, axis=0)
        error = np.sum((p-y)**2, axis=0)
        constant_error = np.sum((y-baseline)**2, axis=0)
        variable_y = np.ptp(y, axis=0) > 1e-12
        variable_p = np.ptp(p, axis=0) > 1e-12
        corr = np.divide(np.sum(yc*pc, axis=0), np.sqrt(vy*vp),
                         out=np.full(len(vy), np.nan), where=variable_y & variable_p)
        ratio = np.divide(vp, vy, out=np.full(len(vy), np.nan), where=variable_y)
        r2 = 1-np.divide(error, constant_error, out=np.full(len(vy), np.nan), where=constant_error > 1e-20)
        rows.append(pd.DataFrame(dict(gene=genes[start:stop], correlation=corr,
            variance_ratio=ratio, r2_over_constant=r2, observed_variance=vy/len(y),
            predicted_variance=vp/len(y), mean_observed=y.mean(0), mean_predicted=p.mean(0))))
        error_total += float(error.sum()); constant_total += float(constant_error.sum())
    residual_observed = observed-mean_profile
    residual_predicted = predicted-mean_profile
    centered = cosine_rows(residual_observed, residual_predicted)
    spot = pd.DataFrame(dict(profile_cosine=cosine_rows(observed,predicted),
        constant_profile_cosine=cosine_rows(observed,mean_profile), centered_cosine=centered,
        profile_mae=np.mean(np.abs(observed-predicted),axis=1),
        constant_profile_mae=np.mean(np.abs(observed-mean_profile),axis=1)))
    summary = dict(profile_cosine=float(spot.profile_cosine.mean()),
        constant_profile_cosine=float(spot.constant_profile_cosine.mean()),
        centered_cosine=float(spot.centered_cosine.mean()),
        centered_cosine_defined_fraction=float(np.isfinite(centered).mean()),
        r2_over_constant=1-error_total/constant_total if constant_total > 1e-20 else np.nan,
        profile_mae=float(spot.profile_mae.mean()), constant_profile_mae=float(spot.constant_profile_mae.mean()))
    return summary, pd.concat(rows, ignore_index=True), spot


def neighborhood_operator(xy, k=16):
    if k < 1 or len(xy) < 1:
        raise ValueError('Positive neighborhood size and nonempty coordinates required')
    k = min(k, len(xy))
    distances, indices = cKDTree(xy).query(xy, k=k)
    indices = np.asarray(indices).reshape(len(xy), k)
    distances = np.asarray(distances).reshape(len(xy), k)
    operator = sparse.csr_matrix((np.full(len(xy)*k,1/k),
        (np.repeat(np.arange(len(xy)),k),indices.ravel())),shape=(len(xy),len(xy)),dtype=np.float32)
    return operator, distances[:,-1]


@torch.no_grad()
def evaluate_field(model, xy, counts, genes, markers, *, device='cpu', batch_size=256, neighbors=16):
    """Frozen model inference only. Do not retain full gene matrices across sections."""
    if batch_size < 1 or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Use a positive batch size and a frozen model')
    totals = np.asarray(counts.sum(1)).ravel()
    if np.any(totals <= 0):
        raise ValueError('Profiles require positive observed totals')
    observed = counts.toarray().astype(np.float32)
    observed /= totals[:,None]
    predicted = np.empty_like(observed)
    predicted_totals = np.empty(len(xy), dtype=np.float32)
    count_mae = np.empty(len(xy), dtype=np.float32)
    for start in range(0,len(xy),batch_size):
        stop = start+batch_size
        rates = model(torch.tensor(xy[start:stop],dtype=torch.float32,device=device)).exp().cpu().numpy()
        predicted_totals[start:stop] = rates.sum(1)
        predicted[start:stop] = rates / np.maximum(rates.sum(1,keepdims=True),1e-12)
        count_mae[start:stop] = np.abs(rates-counts[start:stop].toarray()).mean(1)
    mean_profile = observed.mean(0, dtype=np.float64).astype(np.float32)
    marker_indices = [i for i,g in enumerate(genes) if g in markers]
    maps = dict(genes=np.asarray(genes)[marker_indices], observed=observed[:,marker_indices].copy(),
        predicted=predicted[:,marker_indices].copy(), constant=mean_profile[marker_indices].copy())
    summaries, tables = [], []
    summary, table, spot = profile_metrics(observed,predicted,mean_profile,genes)
    summaries.append(dict(view='Spot',**summary)); tables.append(table.assign(view='Spot'))
    spot['count_mae'] = count_mae
    spot['observed_total'] = totals; spot['predicted_total'] = predicted_totals
    spot['signed_log_total_error'] = np.log1p(predicted_totals)-np.log1p(totals)
    operator, radius = neighborhood_operator(xy, neighbors)
    smooth_observed, smooth_predicted = operator@observed, operator@predicted
    summary, table, _ = profile_metrics(smooth_observed,smooth_predicted,mean_profile,genes)
    summaries.append(dict(view=f'{neighbors}-neighbor',**summary)); tables.append(table.assign(view=f'{neighbors}-neighbor'))
    maps['neighborhood_observed'] = smooth_observed[:,marker_indices]
    maps['neighborhood_predicted'] = smooth_predicted[:,marker_indices]
    return dict(summary=pd.DataFrame(summaries),gene_metrics=pd.concat(tables,ignore_index=True),
                spot_metrics=spot,marker_maps=maps,neighborhood_radius=radius)
