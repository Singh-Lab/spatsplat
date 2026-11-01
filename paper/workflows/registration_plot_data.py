"""Portable plotting dataframes. Loading never imports models or accesses datasets."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False, na_values=[''])


def export_plot_bundle(report, cases, methods):
    """Export existing scored results/maps, resolving atlas labels once if necessary."""
    report = Path(report)
    scores = read_csv(report/'scores.csv')
    availability = read_csv(report/'method_availability.csv')
    hierarchy = None
    metadata, measurements, registered = [], [], []
    for case in cases:
        name = case['name']
        metadata.append(dict(case=name, **{k:case[k] for k in
            ('dataset','task','role','section','shared_fraction','retained_fraction','direction')}))
        if case.get('division_labels') is not None:
            divisions = case['division_labels']
        elif case['labels'] is not None:
            from paper.allen_ccf_diagnostics import load_hierarchy, region_labels
            if hierarchy is None:
                hierarchy = load_hierarchy()
            divisions = [region_labels(labels, hierarchy, 'division') for labels in case['labels']]
        else:
            divisions = [None, None]
        folder = report/'cases'/name
        # Use exactly the coordinates saved for this scored run, not regenerated data.
        with np.load(folder/'evaluation_coordinates.npz', allow_pickle=False) as data:
            for side, labels in zip(('reference','query'), divisions):
                xy = data[side]
                if not np.array_equal(xy, case[side+'_xy']):
                    raise ValueError(f'Case coordinates differ from saved report: {name}, {side}')
                measurements.append(pd.DataFrame(dict(case=name, side=side,
                    measurement_index=np.arange(len(xy)), x=xy[:,0], y=xy[:,1],
                    division=labels if labels is not None else np.repeat('',len(xy)))))
        names = json.loads((folder/'map_labels.json').read_text())
        with np.load(folder/'maps.npz', allow_pickle=False) as data:
            maps = {method:data[str(i)] for i,method in enumerate(names)}
        for method, filename in [('Splat rigid','rigid_map.npz'),
                                 ('PASTE2 rigid','paste2_map.npz'), ('MLP-MAE rigid','mlp_map.npz')]:
            if (folder/filename).is_file():
                with np.load(folder/filename, allow_pickle=False) as data:
                    maps[method] = data['mapped']
        for method, xy in maps.items():
            if method in methods:
                if len(xy)!=len(case['query_xy']) or not np.isfinite(xy).all():
                    raise ValueError(f'Invalid saved map: {name}, {method}')
                registered.append(pd.DataFrame(dict(case=name, method=method,
                    measurement_index=np.arange(len(xy)), x=xy[:,0], y=xy[:,1])))
    inventory = pd.DataFrame(metadata)
    events_path = report/'timing_events.jsonl'
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    if not events and (report/'timing_summary.csv').is_file():
        timings = read_csv(report/'timing_summary.csv')
    else:
        rows = []
        for case in cases:
            for stage in sorted({e['stage'] for e in events}):
                group = [e for e in events if e['case']==case['name'] and e['stage']==stage]
                fresh = [e for e in group if e['status']=='completed' and e['activity']=='fresh']
                rows.append(dict(case=case['name'],stage=stage,
                    measured_fresh_minutes=fresh[-1]['wall_seconds']/60 if fresh else np.nan))
        timings = pd.DataFrame(rows,columns=['case','stage','measured_fresh_minutes'])
    timings = timings.drop(columns=[k for k in inventory if k!='case'],errors='ignore').merge(
        inventory,on='case',how='left',validate='many_to_one')
    bundle = report/'plot_data'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    bundle.mkdir(parents=True)
    tables = dict(scores=scores,availability=availability,cases=inventory,timings=timings,
                  measurements=pd.concat(measurements,ignore_index=True),
                  registered=pd.concat(registered,ignore_index=True))
    for key, frame in tables.items():
        frame.to_csv(bundle/f'{key}.csv.gz',index=False,float_format='%.17g',compression='gzip')
    protocol = json.loads((report/'protocol.json').read_text())
    manifest = dict(schema_version=1,source_report=str(report),methods=list(methods),
        protocol=protocol,hardware=list({json.dumps(e['hardware'],sort_keys=True) for e in events}),
        tables={key:f'{key}.csv.gz' for key in tables},
        expected_results=len(cases)*len(methods),
        complete=bool(len(availability)==len(cases)*len(methods) and availability.available.all()))
    # Manifest is the completion marker; failed exports are never discoverable bundles.
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return bundle


def load_plot_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle/'manifest.json').read_text())
    if manifest['schema_version']!=1:
        raise ValueError('Unsupported plotting-data schema')
    tables = {}
    for name, filename in manifest['tables'].items():
        path = (bundle/filename).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError('Plot table must be inside its bundle')
        tables[name] = read_csv(path)
    return manifest, tables


def plotting_inputs(tables):
    """Translate editable dataframes into the established publication plot interface."""
    cases, fitted = [], {}
    observations, registered = tables['measurements'], tables['registered']
    if observations.duplicated(['case','side','measurement_index']).any():
        raise ValueError('Duplicate measurement rows')
    if registered.duplicated(['case','method','measurement_index']).any():
        raise ValueError('Duplicate registered-coordinate rows')
    for row in tables['cases'].to_dict('records'):
        name = row.pop('case'); case = dict(name=name,**row)
        divisions = []
        for side in ('reference','query'):
            group = observations[(observations.case==name)&(observations.side==side)].sort_values('measurement_index')
            if group.empty or not np.array_equal(group.measurement_index,np.arange(len(group))):
                raise ValueError(f'Missing measurement rows: {name}, {side}')
            case[side+'_xy'] = group[['x','y']].to_numpy(dtype=float)
            if not np.isfinite(case[side+'_xy']).all():
                raise ValueError('Nonfinite coordinates')
            divisions.append(group.division.fillna('Unannotated').astype(str).to_numpy())
        case['division_labels'] = divisions
        cases.append(case); fitted[name] = {}
        for method, group in registered[registered.case==name].groupby('method',sort=False):
            group = group.sort_values('measurement_index')
            if not np.array_equal(group.measurement_index,np.arange(len(case['query_xy']))):
                raise ValueError(f'Missing registered measurements: {name}, {method}')
            xy = group[['x','y']].to_numpy(dtype=float)
            if not np.isfinite(xy).all():
                raise ValueError('Nonfinite registered coordinates')
            fitted[name][method] = xy
    return cases, fitted


def runtime_figure(timings, *, role='evaluation'):
    """Workload-stage timings, not a hardware-matched per-method speed ranking."""
    import matplotlib.pyplot as plt
    data = timings[timings.role==role]
    stages = ['Geometry family', 'PASTE family', 'CAST family', 'Splat family',
              'Splat rigid pose', 'PASTE2 full pipeline', 'MLP fields', 'MLP rigid pose']
    with plt.rc_context({'font.family':'sans-serif', 'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                         'font.size':9, 'pdf.fonttype':42}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout='constrained')
        for ax, dataset in zip(axes, ('Allen', 'BARseq')):
            for index, stage in enumerate(stages):
                group = data[(data.dataset==dataset)&(data.stage==stage)]
                seconds = group.measured_fresh_minutes.dropna().to_numpy()*60
                if len(seconds):
                    ax.plot([seconds.min(), seconds.max()], [index,index], color='#555555', linewidth=1)
                    ax.scatter(seconds, index+np.linspace(-.12,.12,len(seconds)), s=15,
                               color='#267f86', alpha=.6)
                    ax.scatter([np.median(seconds)], [index], marker='D', s=25, color='#b34e32')
                ax.text(1.01, index, f'n={len(seconds)}', transform=ax.get_yaxis_transform(),
                        va='center', fontsize=8)
            ax.set(xscale='log', xlabel='Measured fresh stage time (seconds; log scale)',
                   title=f'{dataset}: {role} cases')
            ax.grid(axis='x', alpha=.2)
        axes[0].set_yticks(range(len(stages)), stages)
        axes[0].invert_yaxis()
        fig.suptitle('Registration workload timing: points = cases; diamonds = medians')
        return fig
