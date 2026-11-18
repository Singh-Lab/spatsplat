"""One results-only registration figure, with fixed example and both evaluation pairs."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .allen_ccf_diagnostics import region_labels, UNANNOTATED
from .allen_registration_plots import case_label, method_label, COLORS

CONDITIONS = ('natural', 'same_section_affine', 'same_section_smooth')


def central_figure(cases, registration, cosine, hierarchy, methods,
                   example=(28, 29), pairs=((28, 29), (32, 33))):
    """Example chosen by pair order, not scores; quantitative panels retain every pair."""
    if any(case['role'] != 'evaluation' for case in cases):
        raise ValueError('Publication figure requires evaluation cases only')
    if 'Splat affine' not in methods:
        raise ValueError('Splat affine is the predeclared visual method')
    registration = registration.copy(); cosine = cosine.copy()
    for table in (registration, cosine):
        table['case'] = table['case'].map(case_label)
        if table.duplicated(['case', 'method']).any():
            raise ValueError('Duplicate case/method results')
    selected = []
    for condition in CONDITIONS:
        name = case_label(f'{example[0]}_to_{example[1]}_{condition}')
        matches = [c for c in cases if case_label(c['name']) == name]
        if len(matches) != 1:
            raise ValueError(f'Missing or duplicate example: {name}')
        selected.append(matches[0])
    divisions = [[region_labels(labels, hierarchy, 'division') for labels in c['evaluation']['labels']]
                 for c in selected]
    regions = sorted(set(np.concatenate([x for labels in divisions for x in labels])) - {UNANNOTATED})
    palette = {r: plt.get_cmap('tab20')(i % 20) for i, r in enumerate(regions)}
    if len(regions) > 20:
        raise ValueError('More than 20 divisions; choose an explicit distinguishable palette')
    palette[UNANNOTATED] = '#bbbbbb'
    fig = plt.figure(figsize=(20, 16), layout='constrained')
    grid = fig.add_gridspec(5, 3, height_ratios=[1, 1, .22, 1.4, .12])
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(3)]
                     for row in (0, 1, 3)])
    legend_ax = fig.add_subplot(grid[2, :])
    legend_ax.set_axis_off()
    comparison_legend_ax = fig.add_subplot(grid[4, :])
    comparison_legend_ax.set_axis_off()
    titles = [f'Cross-section: {example[1]} to {example[0]}',
              f'Rigid recovery: section {example[0]}', f'Smooth recovery: section {example[0]}']
    limits = np.vstack([xy for c in selected for xy in
                       [c['pair']['xy'][0], c['pair']['xy'][1], c['mappings']['Splat affine']]])
    lo, hi = limits.min(0)-.015, limits.max(0)+.015
    for col, (case, labels) in enumerate(zip(selected, divisions)):
        for row, method in enumerate(['Unregistered', 'Splat affine']):
            ax = axes[row, col]
            mapped = case['pair']['xy'][1] if row == 0 else case['mappings'][method]
            ax.scatter(*case['pair']['xy'][0].T, c=[palette[r] for r in labels[0]],
                       s=6, marker='o', alpha=.65, edgecolors='none',
                       rasterized=True, zorder=1)
            ax.scatter(*mapped.T, facecolors='none', edgecolors=[palette[r] for r in labels[1]],
                       s=14, marker='o', alpha=.9, linewidths=.55,
                       rasterized=True, zorder=2)
            letter = 'ABCDEF'[row*3+col]
            stage = 'Before additional registration' if row == 0 else 'After splat affine'
            ax.set(title=f'{letter}  {titles[col]}\n{stage}', xlim=(lo[0], hi[0]),
                   ylim=(hi[1], lo[1]), aspect='equal', xticks=[], yticks=[])

    symbols = ['o', 'D']
    if len(pairs) != len(symbols):
        raise ValueError('Figure is designed for the two predefined evaluation pairs')
    recovery_max = 1.
    for col, condition in enumerate(CONDITIONS):
        ax = axes[2, col]
        table = cosine if condition == 'natural' else registration
        metric = 'coverage_weighted_cosine' if condition == 'natural' else 'median_recovery_error_spacings'
        for p, pair in enumerate(pairs):
            key = case_label(f'{pair[0]}_to_{pair[1]}_{condition}')
            subset = table[table.case.eq(key) & table.method.isin(methods)].set_index('method')
            if set(subset.index) != set(methods):
                raise ValueError(f'Incomplete quantitative results for {key}')
            subset = subset.reindex(methods)
            values = subset[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError('Non-finite central figure metric')
            y = np.arange(len(methods)) + (-.12 if p == 0 else .12)
            ax.scatter(values, y, marker=symbols[p], s=50,
                       c=[COLORS[m] for m in methods], zorder=3)
            if condition == 'natural':
                null = subset['coverage_weighted_cosine_shuffle_mean'].to_numpy(dtype=float)
                ax.scatter(null, y, marker='|', s=90, color='#777777', zorder=2)
                for yi, start, end in zip(y, null, values):
                    ax.plot([start, end], [yi, yi], color='#aaaaaa', lw=.8, zorder=1)
            else:
                recovery_max = max(recovery_max, float(values.max()))
        ax.set_yticks(range(len(methods)), [method_label(m) for m in methods])
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
        if col == 0:
            ax.set(title='G  Cross-section expression agreement\nHigher is better',
                   xlabel='Coverage-weighted cosine (observed expression)')
            ax.margins(x=.12)
        else:
            ax.set_xscale('symlog', linthresh=.1)
            ax.set(title=('H  Rigid recovery' if col == 1 else 'I  Smooth recovery')+'\nLower is better',
                   xlabel='Median error / reference spacing\nLinear below 0.1; logarithmic above')
    for ax in axes[2, 1:]:
        ax.set_xlim(0, recovery_max*1.4)
    pair_handles = [Line2D([], [], marker=symbols[i], linestyle='', color='black',
                          label=f'Pair {a}/{b}; recovery section {a}') for i, (a, b) in enumerate(pairs)]
    pair_handles.append(Line2D([], [], marker='|', linestyle='', color='#777777',
                               label='G: geometry-matched shuffle mean'))
    comparison_legend_ax.legend(handles=pair_handles, loc='center',
                                ncol=len(pair_handles), columnspacing=3,
                                fontsize=9, frameon=False)
    region_handles = [Line2D([], [], marker='o', linestyle='', color=palette[r], label=r)
                      for r in regions + ([UNANNOTATED] if any(UNANNOTATED in x for labs in divisions for x in labs) else [])]
    slide_handles = [
        Line2D([], [], marker='o', linestyle='', markerfacecolor='#555555',
               markeredgecolor='none', markersize=5, label='Reference (filled)'),
        Line2D([], [], marker='o', linestyle='', markerfacecolor='none',
               markeredgecolor='#555555', markersize=7, label='Query (hollow)')]
    legend_ax.legend(handles=slide_handles+region_handles, loc='center',
                     ncol=min(8, len(region_handles)+2),
                     title='A-F: slide identity and CCF divisions', fontsize=9,
                     frameon=False)
    fig.suptitle('Splat fields as a representation for spatial transcriptomic registration\n'
                 'One fixed visual example; quantitative results from both evaluation pairs', fontsize=16)
    return fig


def compact_figure(cases, registration, cosine, hierarchy, methods,
                   pairs=((28, 29), (32, 33))):
    """Both natural pairs, with all-method agreement and recovery summaries."""
    if len(pairs) != 2 or any(c['role'] != 'evaluation' for c in cases):
        raise ValueError('Expected two evaluation pairs')
    registration = registration.copy()
    cosine = cosine.copy()
    for table in (registration, cosine):
        table['case'] = table['case'].map(case_label)
        if table.duplicated(['case', 'method']).any():
            raise ValueError('Duplicate case/method results')

    def values(table, pair, condition, metric):
        key = case_label(f'{pair[0]}_to_{pair[1]}_{condition}')
        subset = table[table.case.eq(key) & table.method.isin(methods)].set_index('method')
        if set(subset.index) != set(methods):
            raise ValueError(f'Incomplete results for {key}')
        result = subset.reindex(methods)[metric].to_numpy(dtype=float)
        if not np.isfinite(result).all():
            raise ValueError(f'Non-finite {metric} for {key}')
        return result

    selected = []
    for a, b in pairs:
        key = case_label(f'{a}_to_{b}_natural')
        matches = [c for c in cases if case_label(c['name']) == key]
        if len(matches) != 1:
            raise ValueError(f'Missing or duplicate example: {key}')
        selected.append(matches[0])
    labels = [[region_labels(x, hierarchy, 'division') for x in c['evaluation']['labels']]
              for c in selected]
    regions = sorted(set(np.concatenate([x for pair in labels for x in pair])))
    annotated = [r for r in regions if r != UNANNOTATED]
    if len(annotated) > 20:
        raise ValueError('More than 20 divisions require an explicit palette')
    palette = {r: plt.get_cmap('tab20')(i) for i, r in enumerate(annotated)}
    palette[UNANNOTATED] = '#bbbbbb'
    fig = plt.figure(figsize=(12, 8), layout='constrained')
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, .28], width_ratios=[1, 1, 1.35])
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(2)])
    for row, (case, labs, (a, b)) in enumerate(zip(selected, labels, pairs)):
        reference = case['pair']['xy'][0]
        queries = [case['pair']['xy'][1], case['mappings']['Splat affine']]
        bounds = np.vstack([reference] + queries)
        lo, hi = bounds.min(0)-.015, bounds.max(0)+.015
        for col, query in enumerate(queries):
            ax = axes[row, col]
            ax.scatter(*reference.T, c=[palette[r] for r in labs[0]], s=4,
                       alpha=.65, edgecolors='none', rasterized=True)
            ax.scatter(*query.T, facecolors='none', edgecolors=[palette[r] for r in labs[1]],
                       s=10, alpha=.9, linewidths=.4, rasterized=True)
            stage = 'Before registration' if col == 0 else 'Splat affine'
            ax.set(title=f'{"ABDE"[row*2+col]}  {b} to {a}: {stage}',
                   xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), aspect='equal', xticks=[], yticks=[])
    symbols = ['o', 'D']
    colors = [COLORS[m] for m in methods]
    for i, pair in enumerate(pairs):
        y = np.arange(len(methods)) + (i-.5)*.22
        axes[0, 2].scatter(values(cosine, pair, 'natural', 'coverage_weighted_cosine'),
                           y, c=colors, marker=symbols[i], s=30)
        for j, condition in enumerate(CONDITIONS[1:]):
            error = values(registration, pair, condition, 'median_recovery_error_spacings')
            axes[1, 2].scatter(error, y+(j-.5)*.1, marker=symbols[i], s=30,
                               facecolors=colors if j == 0 else 'none', edgecolors=colors)
    for ax in axes[:, 2]:
        ax.set_yticks(range(len(methods)), [method_label(m) for m in methods], fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    axes[0, 2].set(title='C  Cross-section agreement', xlabel='Coverage-weighted cosine (higher is better)')
    axes[1, 2].set_xscale('symlog', linthresh=.1)
    axes[1, 2].set_xlim(left=0)
    axes[1, 2].set(title='F  Controlled recovery', xlabel='Median error / spacing (lower is better)')
    region_ax = fig.add_subplot(grid[2, :2]); region_ax.set_axis_off()
    handles = [Line2D([], [], marker='o', linestyle='', color=palette[r], label=r) for r in regions]
    handles = [Line2D([], [], marker='o', linestyle='', color='#555555', label='Reference (filled)'),
               Line2D([], [], marker='o', linestyle='', color='#555555', markerfacecolor='none',
                      label='Query (hollow)')] + handles
    region_ax.legend(handles=handles, loc='center', ncol=5, fontsize=7, frameon=False,
                     title='Slide identity and CCF divisions')
    metric_ax = fig.add_subplot(grid[2, 2]); metric_ax.set_axis_off()
    handles = [Line2D([], [], marker=symbols[i], linestyle='', color='#555555',
                      label=f'Pair {a}/{b}') for i, (a, b) in enumerate(pairs)]
    handles += [Line2D([], [], marker='o', linestyle='', color='#555555',
                       markerfacecolor='#555555' if fill else 'none', label=label)
                for fill, label in [(True, 'F: rigid'), (False, 'F: smooth')]]
    metric_ax.legend(handles=handles, loc='center', ncol=2, fontsize=8, frameon=False)
    return fig


HELVETICA_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic',
    'mathtext.bf': 'sans:bold',
    'pdf.fonttype': 42,
}


@plt.rc_context(HELVETICA_STYLE)
def all_overlay_figure(cases, hierarchy, registration, cosine, methods,
                       pairs=((28, 29), (32, 33))):
    """Twelve overlays with two all-method comparison panels on the right."""
    if len(pairs) != 2 or any(c['role'] != 'evaluation' for c in cases):
        raise ValueError('Expected two evaluation pairs')
    selected = []
    for a, b in pairs:
        for condition in CONDITIONS:
            key = case_label(f'{a}_to_{b}_{condition}')
            matches = [c for c in cases if case_label(c['name']) == key]
            if len(matches) != 1:
                raise ValueError(f'Missing or duplicate case: {key}')
            selected.append(matches[0])
    labels = [[region_labels(x, hierarchy, 'division') for x in c['evaluation']['labels']]
              for c in selected]
    regions = sorted(set(np.concatenate([x for pair in labels for x in pair])))
    annotated = [r for r in regions if r != UNANNOTATED]
    if len(annotated) > 20:
        raise ValueError('More than 20 divisions require an explicit palette')
    palette = {r: plt.get_cmap('tab20')(i) for i, r in enumerate(annotated)}
    palette[UNANNOTATED] = '#bbbbbb'
    fig = plt.figure(figsize=(15, 11), layout='constrained')
    grid = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, .3],
                           width_ratios=[1, 1, 1, 1.5])
    for p, (a, b) in enumerate(pairs):
        # One frame per pair makes all three perturbations directly comparable.
        bounds = np.vstack([xy for c in selected[p*3:p*3+3] for xy in
                            [*c['pair']['xy'], c['mappings']['Splat affine']]])
        lo, hi = bounds.min(0)-.015, bounds.max(0)+.015
        for col in range(3):
            case, labs = selected[p*3+col], labels[p*3+col]
            title = [f'{b} to {a}: natural', f'Section {a}: rigid', f'Section {a}: smooth'][col]
            for stage in range(2):
                row = p*2+stage
                ax = fig.add_subplot(grid[row, col])
                query = case['pair']['xy'][1] if stage == 0 else case['mappings']['Splat affine']
                ax.scatter(*case['pair']['xy'][0].T, c=[palette[r] for r in labs[0]],
                           s=3, alpha=.65, edgecolors='none', rasterized=True)
                ax.scatter(*query.T, facecolors='none', edgecolors=[palette[r] for r in labs[1]],
                           s=8, alpha=.9, linewidths=.35, rasterized=True)
                ax.set(title=f'{"ABCDEFGHIJKL"[row*3+col]}  {title}',
                       xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), aspect='equal', xticks=[], yticks=[])
                ax.title.set_fontsize(9)
                if col == 0:
                    ax.set_ylabel('Before registration' if stage == 0 else 'After splat affine', fontsize=9)
    legend_ax = fig.add_subplot(grid[4, :3]); legend_ax.set_axis_off()
    handles = [Line2D([], [], marker='o', linestyle='', color='#555555', label='Reference (filled)'),
               Line2D([], [], marker='o', linestyle='', color='#555555', markerfacecolor='none',
                      label='Query (hollow)')]
    handles += [Line2D([], [], marker='o', linestyle='', color=palette[r], label=r) for r in regions]
    legend_ax.legend(handles=handles, loc='center', ncol=6, fontsize=7,
                     title='Slide identity and CCF divisions', frameon=False)
    registration = registration.copy()
    cosine = cosine.copy()
    for table in (registration, cosine):
        table['case'] = table['case'].map(case_label)
        if table.duplicated(['case', 'method']).any():
            raise ValueError('Duplicate case/method results')

    def values(table, pair, condition, metric):
        key = case_label(f'{pair[0]}_to_{pair[1]}_{condition}')
        subset = table[table.case.eq(key) & table.method.isin(methods)].set_index('method')
        if set(subset.index) != set(methods):
            raise ValueError(f'Incomplete results for {key}')
        result = subset.reindex(methods)[metric].to_numpy(dtype=float)
        if not np.isfinite(result).all():
            raise ValueError(f'Non-finite {metric} for {key}')
        return result

    agreement_ax = fig.add_subplot(grid[:2, 3])
    recovery_ax = fig.add_subplot(grid[2:4, 3])
    symbols = ['o', 'D']
    colors = [COLORS[m] for m in methods]
    for i, pair in enumerate(pairs):
        y = np.arange(len(methods)) + (i-.5)*.24
        agreement_ax.scatter(values(cosine, pair, 'natural', 'coverage_weighted_cosine'),
                             y, c=colors, marker=symbols[i], s=35)
        for j, condition in enumerate(CONDITIONS[1:]):
            recovery_ax.scatter(values(registration, pair, condition, 'median_recovery_error_spacings'),
                                y+(j-.5)*.11, marker=symbols[i], s=35,
                                facecolors=colors if j == 0 else 'none', edgecolors=colors)
    for ax in (agreement_ax, recovery_ax):
        ax.set_yticks(range(len(methods)), [method_label(m) for m in methods], fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    agreement_ax.set(title='M  Cross-section agreement',
                     xlabel='Coverage-weighted cosine (higher is better)')
    recovery_ax.set_xscale('symlog', linthresh=.1)
    recovery_ax.set_xlim(left=0)
    recovery_ax.set(title='N  Controlled recovery',
                    xlabel='Median error / reference spacing (lower is better)\nLinear below 0.1; logarithmic above')
    metric_ax = fig.add_subplot(grid[4, 3]); metric_ax.set_axis_off()
    handles = [Line2D([], [], marker=symbols[i], linestyle='', color='#555555',
                      label=f'M: {b} to {a}') for i, (a, b) in enumerate(pairs)]
    handles.append(Line2D([], [], marker='o', linestyle='', color='#555555',
                          label='N: rigid (filled)'))
    handles += [Line2D([], [], marker=symbols[i], linestyle='', color='#555555',
                       label=f'N: section {a}') for i, (a, b) in enumerate(pairs)]
    handles.append(Line2D([], [], marker='o', linestyle='', color='#555555',
                          markerfacecolor='none', label='N: smooth (hollow)'))
    metric_ax.legend(handles=handles, loc='center', ncol=2, fontsize=8, frameon=False)
    return fig
