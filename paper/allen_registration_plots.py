"""Results-only plots; deliberately excluded from fitting/protocol cache keys."""
import numpy as np
import matplotlib.pyplot as plt
import re


def case_label(name):
    """Correct the control's public name without invalidating fitted caches."""
    match = re.fullmatch(r'(\d+)_to_(\d+)_(natural|same_section_affine|same_section_smooth)', name)
    if match is None:
        return name
    reference, query, condition = match.groups()
    if condition == 'natural':
        return f'{query}_to_{reference}_cross_section_alignment'
    kind = 'rigid' if condition == 'same_section_affine' else 'smooth'
    return f'{reference}_same_section_{kind}_recovery'


def case_title(name):
    name = case_label(name)
    match = re.fullmatch(r'(\d+)_to_(\d+)_cross_section_alignment', name)
    if match:
        return f'Section {match[1]} to section {match[2]}: cross-section alignment'
    match = re.fullmatch(r'(\d+)_same_section_(rigid|smooth)_recovery', name)
    if match:
        return f'Section {match[1]}: same-section {match[2]} recovery'
    return name


def condition_label(name):
    return {'natural': 'Cross-section alignment (no imposed distortion)',
            'same_section_affine': 'Same-section rigid recovery (rotation + translation)',
            'same_section_smooth': 'Same-section smooth recovery (rigid + sinusoidal displacement)'}.get(name, name)


def method_label(name):
    return {'Unregistered': 'No additional registration',
            'Geometry rigid ICP': 'Geometry-only rigid ICP',
            'PASTE': 'PASTE (rigid readout)'}.get(name, name)


METRIC_LABELS = {
    'case': 'Experiment', 'method': 'Method',
    'matching_mass': 'Primary CCF agreement (fraction of total mass)',
    'scored_mass': 'Annotated-pair mass (fraction)',
    'agreement_given_scored': 'CCF agreement within annotated pairs',
    'historical_agreement': 'Alternative CCF agreement (method-specific readout)',
    'query_coverage_2_spacings': 'Query near reference (fraction)',
    'reference_coverage_2_spacings': 'Reference near query (fraction)',
    'local_fold_fraction': 'Reversed triangles (fraction)',
    'median_recovery_error_spacings': 'Median recovery error (reference spacings)',
    'p90_recovery_error_spacings': '90th-percentile recovery error (reference spacings)',
}


def display_metrics(table):
    table = table.copy()
    table['case'] = table['case'].map(case_title)
    table['method'] = table['method'].map(method_label)
    return table.rename(columns=METRIC_LABELS)

METHODS = ['Unregistered', 'Geometry rigid ICP', 'PASTE', 'CAST affine',
           'CAST non-affine', 'MLP-MAE affine', 'Splat affine', 'Splat non-affine']
COLORS = dict(zip(METHODS, ['#888888', '#aa8844', '#4477aa', '#cc6677',
                          '#882255', '#aa4499', '#228833', '#66aa77']))


def _bars(ax, table, column, title, *, percent=False, reference=None):
    values = table[column].to_numpy(dtype=float) * (100 if percent else 1)
    y = np.arange(len(table))
    ax.barh(y, values, color=[COLORS.get(m, '#4477aa') for m in table.index], height=.65)
    ax.set_yticks(y, ['PASTE (native coupling)' if m == 'PASTE' and column == 'historical_agreement'
                     else method_label(m) for m in table.index])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Percent' if percent else column)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    if percent:
        ax.set_xlim(0, 110)
        ax.set_xticks([0, 25, 50, 75, 100])
    for i, value in enumerate(values):
        if np.isfinite(value):
            ax.annotate(f'{value:.4g}', (value, i), xytext=(4, 0),
                        textcoords='offset points', va='center', fontsize=9)
        else:
            ax.text(.02, i, 'N/A', transform=ax.get_yaxis_transform(), va='center')
    if reference is not None and np.isfinite(reference):
        ax.axvline(reference, color='#333333', ls='--', lw=1, label='No additional registration')
        ax.legend(loc='lower right', fontsize=8)


def registration_figures(metrics, methods=None):
    """Never pool conditions or pairs; repeated measures are not error bars."""
    methods = list(METHODS if methods is None else methods)
    metrics = metrics[metrics.method.isin(methods)]
    if metrics.duplicated(['case', 'method']).any():
        raise ValueError('Expected one result per case and method')
    figures = {}
    for case, group in metrics.groupby('case', sort=False):
        case = case_label(case)
        if set(group.method) != set(methods):
            raise ValueError(f'Incomplete method comparison: {case}')
        table = group.set_index('method').reindex(methods)
        for column, title, suffix, note in [
            ('matching_mass', 'Primary CCF agreement: common spatial OT', 'primary_agreement',
             'Percent of total transport mass with matching CCF labels'),
            ('historical_agreement', 'Alternative CCF agreement: method-specific readout', 'alternative_agreement',
             'Native PASTE coupling; normalized spatial OT for other methods')]:
            fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
            _bars(ax, table, column, title, percent=True,
                  reference=100*table.loc['Unregistered', column])
            fig.suptitle(f'{case_title(case)} | higher is better\n{note}', fontsize=11)
            figures[case+'_'+suffix] = fig

        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), layout='constrained')
        for ax, column, title in zip(axes,
            ['query_coverage_2_spacings', 'reference_coverage_2_spacings'],
            ['Query near reference (higher is better)', 'Reference near query (higher is better)']):
            _bars(ax, table, column, title, percent=True)
        fig.suptitle(f'{case_title(case)}\nCoverage radius = 2 x median reference nearest-neighbor spacing', fontsize=11)
        figures[case+'_geometry'] = fig

        if group.condition.iloc[0] != 'natural':
            fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
            y = np.arange(len(table))
            for column, marker, label in [('median_recovery_error_spacings', 'o', 'Median'),
                                          ('p90_recovery_error_spacings', 'x', '90th percentile')]:
                ax.scatter(table[column], y, marker=marker, s=45, label=label,
                           c=[COLORS[m] for m in table.index])
            for i, row in enumerate(table.itertuples()):
                lo, hi = row.median_recovery_error_spacings, row.p90_recovery_error_spacings
                ax.plot([lo, hi], [i, i], color=COLORS[row.Index], alpha=.6)
                ax.annotate(f'{lo:.4g} / {hi:.4g}', (hi, i), xytext=(5, 0),
                            textcoords='offset points', va='center', fontsize=9)
            ax.set_xscale('symlog', linthresh=.1)
            maximum = float(table.p90_recovery_error_spacings.max())
            ax.set_xlim(0, max(1, maximum)*5)
            ax.set_yticks(y, [method_label(m) for m in table.index]); ax.invert_yaxis()
            ax.set_xlabel('Recovery error / median reference spacing (linear below 0.1; logarithmic above)')
            ax.set_title(f'{case_title(case)}\nPositional error (lower is better); labels: median / 90th percentile')
            ax.grid(axis='x', alpha=.2); ax.legend(loc='lower right')
            figures[case+'_recovery'] = fig
    return figures


def reconstruction_figure(reconstruction, methods=('MLP-MAE', 'Splat')):
    reconstruction = reconstruction[reconstruction.method.isin(methods)]
    if reconstruction.duplicated(['section', 'method']).any():
        raise ValueError('Expected one reconstruction per section and method')
    sections = list(reconstruction.section.unique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    for ax, column, title in zip(axes, ['profile_cosine', 'centered_cosine', 'r2_over_constant'],
        ['Profile cosine', 'Mean-centered profile cosine', 'Profile R2 relative to section mean']):
        for i, method in enumerate(methods):
            offset = (i-(len(methods)-1)/2)*.32
            color = {'MLP-MAE': '#aa4499', 'Splat': '#228833'}[method]
            table = reconstruction[reconstruction.method.eq(method)].set_index('section').reindex(sections)
            values = table[column].to_numpy()
            x = np.arange(len(sections)) + offset
            ax.bar(x, values, width=.3, color=color, label=method)
            for pos, value in zip(x, values):
                ax.annotate(f'{value:.4g}', (pos, value), xytext=(0, 4 if value>=0 else -4),
                            textcoords='offset points', ha='center', va='bottom' if value>=0 else 'top', fontsize=9)
        if column == 'profile_cosine':
            constant = reconstruction.groupby('section').constant_profile_cosine.first().reindex(sections)
            ax.scatter(np.arange(len(sections)), constant, marker='_', s=250, color='black',
                       label='Constant section-mean profile', zorder=3)
        ax.axhline(0, color='black', lw=.7)
        ax.set_xticks(np.arange(len(sections)), [s.rsplit('.', 1)[-1] for s in sections])
        ax.set(title=title, xlabel='Section'); ax.margins(y=.2)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(fontsize=9)
    fig.suptitle('Cross-section experiment: field reconstruction (in-sample; higher is better)')
    return fig
