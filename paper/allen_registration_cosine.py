"""Observed-expression alignment scores, with geometry-matched shuffle controls."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def expression_scores(reference, mapped, values_a, values_b, *, radius, k=5,
                      permutations=99, seed=1023):
    if radius <= 0 or k < 1 or permutations < 1:
        raise ValueError('Positive radius, k and permutation count required')
    profiles = []
    for xy, values in [(reference, values_a), (mapped, values_b)]:
        values = np.asarray(values, dtype=float)
        if (len(xy) != len(values) or len(xy) < 2 or not np.isfinite(xy).all()
                or not np.isfinite(values).all() or (values < 0).any()
                or (values.sum(1) <= 0).any()):
            raise ValueError('Invalid coordinates or expression profiles')
        profiles.append(values / values.sum(1, keepdims=True))
    observed, null, coverage = [], [], []
    rng = np.random.default_rng(seed)
    for source, target, ps, pt in [(reference, mapped, *profiles),
                                  (mapped, reference, profiles[1], profiles[0])]:
        distances, indices = cKDTree(source).query(target, k=list(range(1, min(k, len(source))+1)))
        weights = (distances <= radius) / np.maximum(distances, 1e-8)
        weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
        valid = weights.sum(1) > 0

        def score(source_profiles):
            pred = np.einsum('nk,nkg->ng', weights, source_profiles[indices])
            cosine = (pred * pt).sum(1) / np.maximum(
                np.linalg.norm(pred, axis=1)*np.linalg.norm(pt, axis=1), 1e-12)
            return np.array([cosine[valid].mean() if valid.any() else np.nan,
                             np.where(valid, cosine, 0).mean()])

        observed.append(score(ps))
        coverage.append(valid.mean())
        # Permute expression only. Geometry, neighbors, weights and support stay fixed.
        null.append(np.array([score(ps[rng.permutation(len(ps))]) for _ in range(permutations)]))
    observed = np.mean(observed, axis=0)
    null = np.mean(null, axis=0)
    result = dict(coverage=float(np.mean(coverage)), query_coverage=coverage[0],
                  reference_coverage=coverage[1], radius=radius, neighbors=k, permutations=permutations)
    for i, name in enumerate(['profile_cosine', 'coverage_weighted_cosine']):
        result[name] = observed[i]
        result[name+'_shuffle_mean'] = float(null[:, i].mean())
        result[name+'_shuffle_q05'] = float(np.quantile(null[:, i], .05))
        result[name+'_shuffle_q95'] = float(np.quantile(null[:, i], .95))
        result[name+'_gain_over_shuffle'] = observed[i] - result[name+'_shuffle_mean']
    return result


def cached_scores(cases, root, *, k=5, permutations=99, seed=1023):
    rows = []
    for case in cases:
        pair = case['pair']
        values = [x.toarray() for x in pair['counts']]
        reference = pair['xy'][0]
        radius = 2*float(np.median(cKDTree(reference).query(reference, k=2)[0][:, 1]))
        for method, mapped in case['mappings'].items():
            h = hashlib.sha256(Path(__file__).read_bytes())
            h.update(str((radius, k, permutations, seed)).encode())
            for array in [reference, mapped, *values]:
                array = np.ascontiguousarray(array)
                h.update(str((array.shape, array.dtype.str)).encode()); h.update(array.tobytes())
            path = Path(root)/h.hexdigest()[:20]/'scores.csv'
            if path.exists():
                result = pd.read_csv(path).iloc[0].to_dict()
            else:
                print('Scoring expression cosine:', case['name'], method, flush=True)
                result = expression_scores(reference, mapped, *values, radius=radius,
                                           k=k, permutations=permutations, seed=seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([result]).to_csv(path, index=False)
            rows.append(dict(case=case['name'], condition=case['evaluation']['condition'],
                             method=method, **result))
    return pd.DataFrame(rows)


def cosine_figures(scores, methods=None):
    import matplotlib.pyplot as plt
    from .allen_registration_plots import METHODS, COLORS, case_label, case_title, method_label

    methods = list(METHODS if methods is None else methods)
    scores = scores[scores.method.isin(methods)]
    figures = {}
    for case, group in scores.groupby('case', sort=False):
        if group.method.duplicated().any() or set(group.method) != set(methods):
            raise ValueError('Expected one cosine result for every method')
        table = group.set_index('method').reindex(methods)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), layout='constrained')
        y = np.arange(len(table))
        for ax, metric, title in zip(axes, ['profile_cosine', 'coverage_weighted_cosine'],
            ['Cosine among supported measurements', 'Coverage-weighted cosine (unsupported = 0)']):
            for i, (method, row) in enumerate(table.iterrows()):
                low, high = row[metric+'_shuffle_q05'], row[metric+'_shuffle_q95']
                ax.plot([low, high], [i, i], color='#777777', linewidth=3,
                        label='Shuffle 5th-95th percentiles' if i==0 else None)
                ax.scatter(row[metric+'_shuffle_mean'], i, marker='x', color='#555555', s=40,
                           label='Shuffle mean' if i==0 else None)
                ax.scatter(row[metric], i, color=COLORS[method], s=45,
                           label='Observed expression' if i==0 else None, zorder=3)
                if np.isfinite(row[metric]):
                    ax.annotate(f'{row[metric]:.4g}', (row[metric], i), xytext=(5, -9),
                                textcoords='offset points', fontsize=9)
                else:
                    ax.text(.02, i, 'N/A: no support in one or both directions', va='center', fontsize=8)
            baseline = table.loc['Unregistered', metric]
            if np.isfinite(baseline):
                ax.axvline(baseline, color='#aaaaaa', ls='--', lw=1,
                           label='No additional registration')
            ax.set_yticks(y, [f'{method_label(m)} ({100*table.loc[m,"coverage"]:.4g}% coverage)'
                              for m in table.index])
            ax.invert_yaxis(); ax.set(xlim=(0, 1.08), title=title, xlabel='Cosine similarity')
            ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
        axes[1].legend(loc='upper right', fontsize=8)
        neighbors = int(group.neighbors.iloc[0])
        fig.suptitle(case_title(case)+f'\nObserved log2-expression profiles; symmetric {neighbors}-neighbor transfer; higher is better')
        figures[case_label(case)+'_expression_cosine'] = fig
    return figures
