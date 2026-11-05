"""Data-only registration galleries corresponding to main panels B, D, and G."""
import numpy as np


GROUPS = {
    'B': ('Allen', 'natural', 'Allen natural registration'),
    'D': ('Allen', 'combined_distortion', 'Allen controlled recovery'),
    'G': ('BARseq', 'unequal_coverage', 'BARseq unequal coverage'),
}


def gallery_cases(cases, panel):
    dataset, task, _ = GROUPS[panel]
    selected = [c for c in cases if c['role']=='evaluation'
                and c['dataset']==dataset and c['task']==task]
    return sorted(selected, key=lambda c: (c['section'], -c['shared_fraction'],
                                           c['direction'], c['name']))


def page_data(case, fitted, scores, methods):
    """Validate every map/score and use common limits without clipping failures."""
    ref = np.asarray(case['reference_xy'])
    query = np.asarray(case['query_xy'])
    if ref.ndim!=2 or ref.shape[1]!=2 or not len(ref):
        raise ValueError('Expected nonempty reference coordinates')
    metric = 'ccf_division_agreement' if case['task']=='natural' else 'positional_accuracy'
    rows = scores[scores.case==case['name']]
    if rows.method.duplicated().any():
        raise ValueError(f'Duplicate scores: {case["name"]}')
    values = rows.set_index('method').reindex(methods)[metric].to_numpy(dtype=float)
    maps = [np.asarray(fitted[case['name']][m]) for m in methods]
    if not np.isfinite(values).all() or any(x.shape!=query.shape for x in maps):
        raise ValueError(f'Incomplete scores or invalid maps: {case["name"]}')
    points = np.vstack([ref, query, *maps])
    if not np.isfinite(points).all():
        raise ValueError(f'Nonfinite coordinates: {case["name"]}')
    low, high = points.min(0), points.max(0)
    padding = max(float(np.ptp(points, axis=0).max())*.04, 1e-6)
    return maps, values, metric, low-padding, high+padding


def gallery_page(case, fitted, scores, methods, panel, *, font='Arial',
                 figsize=(12, 6.5), columns=5, point_size=2.5):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    methods = tuple(methods)
    if not methods or columns<1:
        raise ValueError('Provide methods and a positive column count')
    maps, values, metric, low, high = page_data(case, fitted, scores, methods)
    label = 'CCF division agreement' if metric=='ccf_division_agreement' else 'Positional accuracy'
    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': [font, 'DejaVu Sans'],
                         'font.size': 8, 'axes.titlesize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(int(np.ceil(len(methods)/columns)), columns,
                                 figsize=figsize, squeeze=False)
        ref = case['reference_xy']
        for ax, method, mapped, value in zip(axes.flat, methods, maps, values):
            ax.scatter(*ref.T, s=point_size, c='#666666', alpha=.55,
                       edgecolors='none', rasterized=True)
            ax.scatter(*mapped.T, s=point_size*1.8, facecolors='none',
                       edgecolors='#168b8b', linewidths=.25, alpha=.65, rasterized=True)
            ax.set(title=f'{method}\n{label}: {value:.4g}',
                   xlim=(low[0], high[0]), ylim=(high[1], low[1]),
                   aspect='equal', xticks=[], yticks=[])
        for ax in list(axes.flat)[len(methods):]:
            ax.set_visible(False)
        if panel=='B':
            parts = case['name'].split('_')
            detail = f'Sections {parts[2]} to {parts[1]}'
        elif panel=='D':
            detail = f'Section {case["section"]:g}: combined rotation and smooth deformation'
        else:
            detail = (f'Section {case["section"]:g} | Shared parent tissue: '
                      f'{100*case["shared_fraction"]:.0f}% | Crop direction: {case["direction"]:g} degrees')
        fig.suptitle(f'Main panel {panel}: {GROUPS[panel][2]}\n{detail}', fontsize=11, y=.98)
        handles = [Line2D([], [], marker='o', linestyle='none', color='#666666',
                          markersize=5, label='Reference'),
                   Line2D([], [], marker='o', linestyle='none', markerfacecolor='none',
                          color='#168b8b', markersize=5, label='Query (method coordinates)')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.035),
                   ncol=2, frameon=False)
        fig.text(.5, .018, 'All saved measurements shown; identical limits across methods; no error-link lines.',
                 ha='center', fontsize=7)
        fig.subplots_adjust(left=.025, right=.985, bottom=.13, top=.82, wspace=.15, hspace=.35)
    return fig
