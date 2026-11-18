"""Post-hoc CCF hierarchy diagnostics over fixed registration mappings."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .dlpfc_registration_metrics import spatial_coupling

UNANNOTATED = 'Unannotated'
LEVELS = ('parcellation', 'structure', 'division')


def load_hierarchy():
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    data = os.getenv('SPATSPLAT_DATA_DIR')
    if not data:
        raise RuntimeError('Configure SPATSPLAT_DATA_DIR')
    root = Path(os.getenv('SPATSPLAT_ABC_ATLAS_DIR', str(Path(data)/'raw/abc_atlas')))
    cache = AbcProjectCache.from_cache_dir(root)
    table = cache.get_metadata_dataframe(directory='Allen-CCF-2020',
        file_name='parcellation_to_parcellation_term_membership_acronym')
    table = table.set_index('parcellation_index')
    if not table.index.is_unique or not {'structure', 'division'}.issubset(table.columns):
        raise ValueError('Unexpected Allen hierarchy schema')
    return table


def region_labels(labels, hierarchy, level):
    if level not in LEVELS:
        raise ValueError('Unknown hierarchy level')
    result = []
    for value in labels:
        if not np.isfinite(value) or value <= 0:
            result.append(UNANNOTATED)
        elif level == 'parcellation':
            result.append(str(int(value)))
        else:
            if int(value) not in hierarchy.index:
                raise ValueError(f'Missing atlas mapping for parcellation {value}')
            term = hierarchy.loc[int(value), level]
            if pd.isna(term) or str(term).lower() == 'unassigned':
                raise ValueError(f'Known parcellation {value} lacks a {level} annotation')
            result.append(str(term))
    return np.array(result)


def confusion_metrics(labels_a, labels_b, coupling):
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    pi = sparse.csr_matrix(coupling)
    if pi.shape != (len(a), len(b)) or not np.isfinite(pi.data).all() or (pi.data < 0).any() or not np.isclose(pi.sum(), 1):
        raise ValueError('Expected finite nonnegative unit-mass transport')
    categories = sorted((set(a) | set(b)) - {UNANNOTATED})
    categories += [UNANNOTATED]
    lookup = {label: i for i, label in enumerate(categories)}
    ai, bi = [[lookup[label] for label in labels] for labels in (a, b)]
    left = sparse.csr_matrix((np.ones(len(a)), (ai, np.arange(len(a)))), shape=(len(categories), len(a)))
    right = sparse.csr_matrix((np.ones(len(b)), (np.arange(len(b)), bi)), shape=(len(b), len(categories)))
    mass = (left @ pi @ right).toarray()
    matrix = pd.DataFrame(mass, index=categories, columns=categories)
    reference, query = mass.sum(1), mass.sum(0)
    matched = np.diag(mass).copy(); matched[-1] = 0
    scored = float(mass[:-1, :-1].sum())
    ceiling = float(np.minimum(reference[:-1], query[:-1]).sum())
    rows = pd.DataFrame(dict(region=categories, reference_mass=reference, query_mass=query,
        matching_mass=matched, recall=np.divide(matched, reference,
            out=np.full(len(reference), np.nan), where=reference > 0)))
    rows.loc[rows.region.eq(UNANNOTATED), 'recall'] = np.nan
    summary = dict(matching_mass=float(matched.sum()), scored_mass=scored,
        agreement_given_scored=float(matched.sum()/scored) if scored else np.nan,
        marginal_label_ceiling=ceiling,
        reference_unannotated_mass=float(reference[-1]), query_unannotated_mass=float(query[-1]))
    return matrix, rows, summary


def analyze_cases(cases, hierarchy, output, methods):
    """Reuse primary spatial couplings for all three hierarchy levels."""
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    hierarchy.to_csv(root/'atlas_hierarchy.csv')
    summaries, regions, confusion = [], [], {}
    for case in cases:
        reference = case['pair']['xy'][0]
        for method in methods:
            mapped = case['mappings'][method]
            h = hashlib.sha256()
            h.update(Path(__file__).with_name('dlpfc_registration_metrics.py').read_bytes())
            for array in (reference, mapped):
                array = np.ascontiguousarray(array)
                h.update(str((array.shape, array.dtype.str)).encode()); h.update(array.tobytes())
            path = root/'couplings'/f'{h.hexdigest()[:20]}.npz'
            if path.exists():
                pi = sparse.load_npz(path)
            else:
                print('CCF diagnostic transport:', case['name'], method, flush=True)
                pi = sparse.csr_matrix(spatial_coupling(reference, mapped))
                path.parent.mkdir(parents=True, exist_ok=True); sparse.save_npz(path, pi)
            for level in LEVELS:
                labels = [region_labels(x, hierarchy, level) for x in case['evaluation']['labels']]
                matrix, table, summary = confusion_metrics(*labels, pi)
                key = (case['name'], method, level)
                confusion[key] = matrix
                folder = root/case['name']/method.replace(' ', '_')
                folder.mkdir(parents=True, exist_ok=True)
                matrix.to_csv(folder/f'{level}_transport_mass.csv', index_label='reference_region')
                denominator = matrix.sum(axis=1).replace(0, np.nan)
                matrix.div(denominator, axis=0).to_csv(folder/f'{level}_row_normalized.csv', index_label='reference_region')
                summaries.append(dict(case=case['name'], method=method, level=level, **summary))
                regions.append(table.assign(case=case['name'], method=method, level=level))
    summary = pd.DataFrame(summaries)
    region = pd.concat(regions, ignore_index=True)
    summary.to_csv(root/'hierarchy_agreement.csv', index=False)
    region.to_csv(root/'region_agreement.csv', index=False)
    return summary, region, confusion


def diagnostic_figures(summary, region, confusion, methods, maximum_regions=20,
                       confusion_methods=None, hierarchy=None):
    import matplotlib.pyplot as plt
    from .allen_registration_plots import COLORS, case_label, case_title, method_label
    figures = {}
    confusion_methods = methods if confusion_methods is None else confusion_methods
    def label(region, level):
        if level == 'parcellation' and hierarchy is not None and str(region).isdigit():
            index = int(region)
            if index in hierarchy.index and 'substructure' in hierarchy:
                return f'{region}: {hierarchy.loc[index, "substructure"]}'
        return region
    for case, group in summary.groupby('case', sort=False):
        if 'Splat affine' in methods:
            table = group.pivot(index='method', columns='level', values='matching_mass').reindex(
                index=methods, columns=LEVELS)
            differences = 100 * table.subtract(table.loc['Splat affine'], axis='columns')
            compared = [method for method in methods if method != 'Splat affine']
            fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True, layout='constrained')
            for ax, level, title in zip(axes, LEVELS, ['Exact parcellation', 'Atlas structure', 'Atlas division']):
                values = differences.loc[compared, level].to_numpy(dtype=float)
                positions = np.arange(len(compared))
                ax.axvline(0, color='#444444', lw=1, ls='--')
                ax.scatter(values, positions, c=[COLORS[m] for m in compared], s=45, zorder=3)
                for y, value in zip(positions, values):
                    if np.isfinite(value):
                        ax.annotate(f'{value:+.4g}', (value, y), xytext=(6 if value >= 0 else -6, 0),
                                    textcoords='offset points', va='center',
                                    ha='left' if value >= 0 else 'right', fontsize=9)
                    else:
                        ax.text(.5, y, 'N/A', transform=ax.get_yaxis_transform(), va='center')
                finite = np.abs(values[np.isfinite(values)])
                limit = max(float(finite.max()) if finite.size else 0, .1) * 1.5
                ax.set_xlim(-limit, limit)
                ax.set_title(title)
                ax.set_xlabel('Agreement difference (percentage points)')
                ax.set_yticks(positions, [method_label(m) for m in compared])
                ax.grid(axis='x', alpha=.2)
            axes[0].invert_yaxis()
            fig.suptitle(case_title(case)+'\nDifference from Splat affine: positive = higher agreement; negative = lower\n'
                         'Each panel has its own symmetric scale; dashed line = Splat affine', fontsize=11)
            figures[case_label(case)+'_ccf_hierarchy_difference'] = fig
        fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
        for method in methods:
            table = group[group.method.eq(method)].set_index('level').reindex(LEVELS)
            ax.plot(range(3), 100*table.matching_mass, marker='o', color=COLORS[method], label=method_label(method))
        ax.set(xticks=range(3), xticklabels=['Exact parcellation', 'Atlas structure', 'Atlas division'],
               ylabel='Matching CCF transport mass (%)', ylim=(0, 100),
               title=case_title(case)+'\nSame correspondence weights; coarser labels necessarily increase or preserve agreement')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        figures[case_label(case)+'_ccf_hierarchy'] = fig
        for level in LEVELS:
            part = region[region.case.eq(case) & region.level.eq(level)]
            # Reference marginals are fixed and independent of method performance.
            support = part.groupby('region').reference_mass.first().drop(UNANNOTATED, errors='ignore')
            selected = support[support > 0].sort_values(ascending=False).head(maximum_regions).index.tolist()
            if not selected:
                continue
            recall = part.pivot(index='region', columns='method', values='recall').reindex(index=selected, columns=methods)
            fig, ax = plt.subplots(figsize=(13, max(5, .32*len(selected)+2)), layout='constrained')
            im = ax.imshow(np.ma.masked_invalid(recall.to_numpy()), vmin=0, vmax=1, aspect='auto', cmap='viridis')
            ax.set_xticks(range(len(methods)), [method_label(m) for m in methods], rotation=35, ha='right')
            ax.set_yticks(range(len(selected)), [f'{label(r, level)} ({100*support[r]:.3g}% mass)' for r in selected])
            ax.set_title(case_title(case)+f'\n{level}: reference-region recall; up to {maximum_regions} largest reference regions')
            fig.colorbar(im, ax=ax, label='Matching mass / reference-region mass')
            figures[case_label(case)+'_'+level+'_region_recall'] = fig
            for method in confusion_methods:
                matrix = confusion[(case, method, level)]
                shown = matrix.reindex(index=selected, columns=selected).copy()
                # Hide omitted/unknown regions without renormalizing their mass away.
                shown = shown.div(matrix.loc[selected].sum(axis=1), axis=0)
                fig, ax = plt.subplots(figsize=(12, max(6, .34*len(selected)+2)), layout='constrained')
                im = ax.imshow(shown, vmin=0, vmax=1, aspect='auto', cmap='viridis')
                ax.set_xticks(range(len(shown.columns)), [label(r, level) for r in shown.columns], rotation=90)
                ax.set_yticks(range(len(selected)), [label(r, level) for r in selected])
                ax.set(xlabel='Query region', ylabel='Reference region',
                       title=case_title(case)+f'\n{method_label(method)} | {level}: correspondence among displayed regions\n'
                       'Full-row normalization retained; omitted/unannotated mass is not shown (rows may sum below 1)')
                fig.colorbar(im, ax=ax, label='Fraction of reference-region transport mass')
                figures[case_label(case)+'_'+method.replace(' ', '_')+'_'+level+'_confusion'] = fig
    return figures
