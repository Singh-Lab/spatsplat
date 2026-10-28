"""Results-only publication figure for the common Allen/BARseq roster."""
import numpy as np
from .standardized_rigid import METHODS

EXAMPLES = ('allen_28_29_natural', 'allen_28_29_combined_distortion',
            'barseq_23_keep_0.8_direction_0')

BARSEQ_SECTION_STYLES = ((15, 's', '///'), (23, '^', '\\\\\\'), (27, 'D', 'xxx'))


def validate_inputs(cases, fitted, scores, example_method, include_cosine=True):
    lookup = {c['name']: c for c in cases}
    for name in EXAMPLES:
        if name not in lookup or example_method not in fitted.get(name, {}):
            raise ValueError(f'Missing example map: {name}, {example_method}')
    evaluation = [c for c in cases if c['role'] == 'evaluation']
    for c in evaluation:
        rows = scores[scores.case == c['name']]
        if rows.method.duplicated().any() or set(rows.method) != set(METHODS):
            raise ValueError(f'Incomplete method scores: {c["name"]}')
        metrics = ['ccf_division_agreement' if c['task'] == 'natural' else 'positional_accuracy']
        if include_cosine:
            metrics.append('coverage_weighted_cosine')
        if not np.isfinite(rows[metrics].to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite figure metrics: {c["name"]}')
    return lookup


def measurement_correspondences(reference, mapped_query, *, maximum=60, seed=1023):
    """Mutual nearest neighbors within two reference spacings; no label filtering."""
    from scipy.spatial import cKDTree
    reference, mapped_query = np.asarray(reference), np.asarray(mapped_query)
    if maximum < 1 or len(reference) < 2 or len(mapped_query) < 1:
        raise ValueError('Need positive maximum and nonempty measurement arrays')
    if not np.isfinite(reference).all() or not np.isfinite(mapped_query).all():
        raise ValueError('Nonfinite measurement coordinates')
    tree = cKDTree(reference)
    spacing = float(np.median(tree.query(reference, k=2)[0][:, 1]))
    if spacing <= 0:
        raise ValueError('Nonpositive reference spacing')
    distance, ref_index = tree.query(mapped_query)
    reverse = cKDTree(mapped_query).query(reference)[1]
    query_index = np.flatnonzero((reverse[ref_index] == np.arange(len(mapped_query))) & (distance <= 2*spacing))
    eligible = len(query_index)
    if eligible > maximum:
        query_index = np.sort(np.random.default_rng(seed).choice(query_index, maximum, replace=False))
    return dict(reference_index=ref_index[query_index], query_index=query_index,
                eligible=eligible, shown=len(query_index), radius=2*spacing)


def draw_measurement_links(ax, case, mapped, reference_colors, query_colors, *, title, maximum=60):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba_array
    ref, query = case['reference_xy'], case['query_xy']
    reference_colors = to_rgba_array(reference_colors)
    query_colors = to_rgba_array(query_colors)
    if len(reference_colors) == 1:
        reference_colors = np.repeat(reference_colors, len(ref), axis=0)
    if len(query_colors) == 1:
        query_colors = np.repeat(query_colors, len(query), axis=0)
    matches = measurement_correspondences(ref, mapped, maximum=maximum)
    # Display original section geometries; the registered map determines pairs only.
    width = max(np.ptp(ref[:, 0]), np.ptp(query[:, 0]), 1e-6)
    shift = np.array([ref[:, 0].max()-query[:, 0].min()+.18*width, 0.])
    right = query+shift
    ax.scatter(*ref.T, s=3, c=reference_colors, alpha=.5, edgecolors='none', rasterized=True)
    ax.scatter(*right.T, s=3, c=query_colors, alpha=.5, edgecolors='none', rasterized=True)
    ri, qi = matches['reference_index'], matches['query_index']
    if len(qi):
        segments = np.stack([ref[ri], right[qi]], axis=1)
        ax.add_collection(LineCollection(segments, colors='#666666', linewidths=.45, alpha=.3, zorder=1))
        ax.scatter(*ref[ri].T, s=9, c=reference_colors[ri], edgecolors='black', linewidths=.2, zorder=3)
        ax.scatter(*right[qi].T, s=9, c=query_colors[qi], edgecolors='black', linewidths=.2, zorder=3)
    points = np.vstack([ref, right]); low, high = points.min(0), points.max(0)
    padding = .04*max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    ax.set(title=title, aspect='equal', xticks=[], yticks=[],
           xlim=(low[0]-padding, high[0]+padding), ylim=(high[1]+padding, low[1]-padding))
    # Keep the allocated panel rectangle; pad data limits rather than shrinking axes.
    ax.set_aspect('equal', adjustable='datalim')
    ax.text(.02, .02, f'{matches["shown"]} / {matches["eligible"]} eligible links shown',
            transform=ax.transAxes, fontsize=7, va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
    ax.text(.02, 1.01, 'Reference', transform=ax.transAxes, fontsize=7)
    ax.text(.98, 1.01, 'Query (original coordinates)', transform=ax.transAxes, ha='right', fontsize=7)


def publication_figure(cases, fitted, scores, *, example_method='Splat rigid', include_cosine=False,
                       measurement_links=False, maximum_links=60):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    from paper.allen_ccf_diagnostics import load_hierarchy, region_labels, UNANNOTATED
    lookup = validate_inputs(cases, fitted, scores, example_method, include_cosine)
    chosen = [lookup[name] for name in EXAMPLES]
    hierarchy = load_hierarchy()
    labels = [[region_labels(x, hierarchy, 'division') for x in c['labels']] for c in chosen[:2]]
    regions = sorted(set(np.concatenate([x for pair in labels for x in pair])) - {UNANNOTATED})
    if len(regions) > 20:
        raise ValueError('More than 20 divisions require an explicit palette')
    palette = {r: plt.get_cmap('tab20')(i) for i, r in enumerate(regions)}
    palette[UNANNOTATED] = '#bbbbbb'
    colors = dict(zip(METHODS, ['#888888', '#3c78a8', '#c17a26', '#925f9e',
                              '#bd83bd', '#d1495b', '#5c9a72', '#135f43']))
    with plt.rc_context({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        columns = 4 if include_cosine else 3
        def panel_label(row, col):
            index = row*columns+col
            if measurement_links:
                index -= min(row, 2)
                if row < 2 and col >= 2:
                    index -= 1
            return chr(65+index)
        fig = plt.figure(figsize=(17 if include_cosine else (11.5 if measurement_links else 13),
                                  9 if measurement_links else 10), layout='constrained')
        grid = fig.add_gridspec(5, columns, height_ratios=[1, 1, 1, .16, .16],
                               width_ratios=[1, 1, 1.55, 1.55][:columns])
        axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(columns)] for r in range(3)])
        if measurement_links:
            # Let E/F share the full spatial block with a narrow internal gutter.
            axes[2, 0].remove(); axes[2, 1].remove()
            spatial_pair = grid[2, :2].subgridspec(1, 2, wspace=.02)
            axes[2, 0] = fig.add_subplot(spatial_pair[0, 0])
            axes[2, 1] = fig.add_subplot(spatial_pair[0, 1],
                                       sharex=axes[2, 0], sharey=axes[2, 0])
        titles = ['Allen: sections 29 to 28', 'Allen: combined distortion', 'BARseq: unequal coverage']
        for row, case in enumerate(chosen):
            ref, query = case['reference_xy'], case['query_xy']
            mapped = fitted[case['name']][example_method]
            points = np.vstack([ref, query, mapped])
            low, high = points.min(0)-.012, points.max(0)+.012
            if measurement_links and row == 2:
                padding = max(.012, .05*float(np.ptp(points, axis=0).max()))
                low, high = points.min(0)-padding, points.max(0)+padding
            rc, qc = ([palette[x] for x in labels[row][0]], [palette[x] for x in labels[row][1]]) if row < 2 else ('#666666', '#168b8b')
            if measurement_links and row < 2:
                axes[row, 0].remove(); axes[row, 1].remove()
                ax = fig.add_subplot(grid[row, :2])
                panel = panel_label(row, 0)
                draw_measurement_links(ax, case, mapped, rc, qc, maximum=maximum_links,
                    title=f'{panel}  {titles[row]}\n{example_method}: inferred measurement correspondences')
                continue
            for col, xy in enumerate((query, mapped)):
                ax = axes[row, col]
                ax.scatter(*ref.T, s=3, c=rc, alpha=.65, edgecolors='none', rasterized=True)
                ax.scatter(*xy.T, s=8, facecolors='none', edgecolors=qc, linewidths=.35, alpha=.8, rasterized=True)
                stage = 'Before registration' if col == 0 else example_method
                ax.set(title=f'{panel_label(row,col)}  {titles[row]}\n{stage}', aspect='equal',
                       xlim=(low[0], high[0]), ylim=(high[1], low[1]), xticks=[], yticks=[])
                if measurement_links:
                    # Equal physical x/y scaling follows from box ratio = data ratio.
                    # Disable data-aspect adjustment entirely: it must not recenter
                    # or tighten the common before/after view during drawing/export.
                    ax.set_aspect('auto')
                    ax.set_box_aspect(float((high[1]-low[1])/(high[0]-low[0])))
                    ax.set_anchor('N')
                    ax.set_xlim(low[0], high[0])
                    ax.set_ylim(high[1], low[1])
                    ax.set_autoscale_on(False)
        for row, task in enumerate(('natural', 'combined_distortion')):
            metrics = ['ccf_division_agreement' if row == 0 else 'positional_accuracy']
            if include_cosine:
                metrics.append('coverage_weighted_cosine')
            for col, metric in zip((2, 3), metrics):
                ax = axes[row, col]
                values = []
                for j, (section, marker) in enumerate(((28, 'o'), (32, 'D'))):
                    g = scores[(scores.dataset == 'Allen') & (scores.task == task) & (scores.section == section)]
                    v = g.set_index('method').loc[list(METHODS), metric].to_numpy()
                    values.extend(v)
                    ax.scatter(v, np.arange(len(METHODS))+(j-.5)*.22, c=[colors[m] for m in METHODS], marker=marker, s=22)
                ax.set_yticks(range(len(METHODS)), METHODS if col == 2 else ['']*len(METHODS))
                ax.invert_yaxis()
                limits = (-.02, 1.02)
                if metric == 'ccf_division_agreement':
                    limits = (max(0., min(.6, min(values)-.01)), min(1., max(.8, max(values)+.01)))
                    ax.xaxis.set_major_locator(MultipleLocator(.05))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                label = {'ccf_division_agreement': 'CCF division agreement', 'positional_accuracy': 'Positional accuracy',
                         'coverage_weighted_cosine': 'Coverage-weighted cosine'}[metric]
                ax.set(xlim=limits, title=f'{panel_label(row,col)}  {label}', xlabel=label)
                ax.grid(axis='x', alpha=.2)
                ax.set_axisbelow(True)
        follow = scores[(scores.dataset == 'BARseq') & (scores.role == 'evaluation')]
        coverage_panels = [(2, 'positional_accuracy', 'Positional accuracy')]
        if include_cosine:
            coverage_panels.append((3, 'coverage_weighted_cosine', 'Coverage-weighted cosine'))
        for col, metric, label in coverage_panels:
            ax = axes[2, col]
            for k, level in enumerate((1., .6, .2)):
                for j, method in enumerate(METHODS):
                    g = follow[(follow.method == method) & np.isclose(follow.shared_fraction, level)].sort_values(['section', 'direction'])
                    if g.empty:
                        raise ValueError(f'Missing BARseq group: {method}, {level}')
                    position = k+(j-(len(METHODS)-1)/2)*.105
                    ax.boxplot([g[metric].to_numpy()], positions=[position], widths=.085,
                               manage_ticks=False, showfliers=False, patch_artist=True,
                               boxprops=dict(facecolor=colors[method], alpha=.3),
                               medianprops=dict(color=colors[method]))
                    offsets = np.linspace(-.025, .025, len(g))
                    for section, marker, hatch in BARSEQ_SECTION_STYLES:
                        mask = g.section.to_numpy() == section
                        ax.scatter(position+offsets[mask], g.loc[mask, metric], s=26,
                                   marker=marker, c=colors[method], edgecolors='#333333',
                                   linewidths=.35, hatch=hatch, zorder=3)
            # Separate coverage groups without changing their shared metric scale.
            for boundary in (.5, 1.5):
                ax.axvline(boundary, color='#888888', linestyle=(0, (3, 3)),
                           linewidth=.8, zorder=1)
            ax.set_xticks(range(3), ['100%', '60%', '20%'])
            ax.set(xlim=(-.5, 2.5), ylim=(-.02, 1.02), xlabel='Shared fraction of parent tissue',
                   ylabel=label, title=f'{panel_label(2,col)}  {label}')
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        symbols = fig.add_subplot(grid[3, :]); symbols.axis('off')
        section_handles = [symbols.scatter([], [], marker=marker, s=48,
                           facecolors='white', edgecolors='#333333', linewidths=.6,
                           hatch=hatch, label=f'BARseq: section {section}')
                           for section, marker, hatch in BARSEQ_SECTION_STYLES]
        symbols.legend(handles=[
            Line2D([], [], marker='o', linestyle='', color='#555555',
                   label='Allen: both slices filled; BARseq: reference filled' if measurement_links else 'Spatial: reference filled'),
            Line2D([], [], marker='o', linestyle='', color='#555555', markerfacecolor='none',
                   label='BARseq overlays: query hollow' if measurement_links else 'Spatial: query hollow'),
            Line2D([], [], marker='o', linestyle='', color='black', label='Allen: pair 28/29 or section 28'),
            Line2D([], [], marker='D', linestyle='', color='black', label='Allen: pair 32/33 or section 32'),
            *section_handles], loc='center', ncol=4, frameon=False)
        legend = fig.add_subplot(grid[4, :]); legend.axis('off')
        legend.legend(handles=[Patch(facecolor=colors[m], label=m) for m in METHODS],
                      loc='center', ncol=4, frameon=False)
    return fig
