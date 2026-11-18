"""Fine anatomical agreement from saved maps; never fits registration models."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def regional_scores(reference_labels, query_labels, coupling):
    from scipy import sparse
    a, b = np.asarray(reference_labels), np.asarray(query_labels)
    pi = sparse.csr_matrix(coupling)
    if pi.shape != (len(a), len(b)) or not np.isfinite(pi.data).all() or (pi.data<0).any() or not np.isclose(pi.sum(), 1):
        raise ValueError('Invalid correspondence mass')
    rows = []
    for region in sorted((set(a)|set(b))-{'Unannotated'}):
        ia, ib = np.flatnonzero(a==region), np.flatnonzero(b==region)
        mass = float(pi[ia][:,ib].sum())
        ra, qb = float(pi[ia].sum()), float(pi[:,ib].sum())
        rows.append(dict(region=region, reference_count=len(ia), query_count=len(ib),
            reference_mass=ra, query_mass=qb, matching_mass=mass,
            reference_fraction=mass/ra if ra else np.nan,
            query_fraction=mass/qb if qb else np.nan))
    return pd.DataFrame(rows)


def prepare_region_table(root, bundle, manifest, tables, *, recompute=False):
    """Resolve labels once and cache portable scores alongside the source bundle.

    A fresh run needs the original Allen data and POT. Loading the resulting CSV
    or plotting it requires neither. Couplings use exactly the primary scorer.
    """
    from .registration_plot_data import read_csv
    root, bundle = Path(root), Path(bundle)
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update((root/'paper/dlpfc_registration_metrics.py').read_bytes())
    digest.update((root/'paper/allen_ccf_diagnostics.py').read_bytes())
    for key in ('measurements','registered','cases','scores'):
        digest.update((bundle/manifest['tables'][key]).read_bytes())
    folder = bundle/'region_agreement'/digest.hexdigest()[:20]
    path = folder/'region_scores.csv'
    if path.exists() and not recompute:
        return read_csv(path), folder
    from dotenv import load_dotenv
    load_dotenv(root/'.env')
    from paper.data_loaders.allen_registration import load_registration_sections
    from paper.allen_ccf_diagnostics import load_hierarchy, region_labels, confusion_metrics
    from paper.dlpfc_registration_metrics import spatial_coupling
    from .registration_plot_data import plotting_inputs
    cases, maps = plotting_inputs(tables)
    cases = [c for c in cases if c['dataset']=='Allen' and c['task']=='natural' and c['role']=='evaluation']
    pairs = {f'allen_{a}_{b}_natural': (a,b) for role,a,b in manifest['protocol']['allen_pairs'] if role=='evaluation'}
    sections = sorted({s for c in cases for s in pairs[c['name']]})
    cohort = load_registration_sections(sections, maximum=manifest['protocol']['allen_maximum'],
                                        seed=manifest['protocol']['seed'])
    hierarchy = load_hierarchy()
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for c in cases:
        a,b = [cohort['blocks'][s] for s in pairs[c['name']]]
        raw = np.vstack([a['xy'],b['xy']])
        unit = float(np.ptp(raw,axis=0).max())*2
        origin = (raw.min(0)+raw.max(0))/2-unit/2
        labels = [region_labels(block['labels'],hierarchy,'structure') for block in (a,b)]
        divisions = [region_labels(block['labels'],hierarchy,'division') for block in (a,b)]
        for side,block,division in zip(('reference','query'),(a,b),divisions):
            if not np.array_equal((block['xy']-origin)/unit,c[side+'_xy']):
                raise ValueError(f'Atlas measurements do not match saved coordinates: {c["name"]}')
            expected = tables['measurements']
            expected = expected[(expected.case==c['name'])&(expected.side==side)].sort_values('measurement_index')
            if not np.array_equal(division,expected.division.to_numpy()):
                raise ValueError('Atlas divisions changed relative to the scored bundle')
        for method in manifest['methods']:
            print('Regional correspondence:',c['name'],method,flush=True)
            key = hashlib.sha256()
            for xy in (c['reference_xy'],maps[c['name']][method]):
                key.update(np.ascontiguousarray(xy).tobytes())
            cache = folder/(key.hexdigest()[:20]+'.npz')
            from scipy import sparse
            if cache.exists() and not recompute:
                pi = sparse.load_npz(cache)
            else:
                pi = sparse.csr_matrix(spatial_coupling(c['reference_xy'],maps[c['name']][method]))
                sparse.save_npz(cache,pi)
            _,_,summary = confusion_metrics(*divisions,pi)
            score = tables['scores']
            original = score[(score.case==c['name'])&(score.method==method)].ccf_division_agreement
            if len(original)!=1 or not np.isclose(summary['agreement_given_scored'],original.iloc[0],atol=1e-7,rtol=1e-6):
                raise ValueError('Recomputed division agreement does not match the main figure')
            frames.append(regional_scores(*labels,pi).assign(case=c['name'],method=method))
    if not frames:
        raise ValueError('No natural Allen evaluation cases')
    table = pd.concat(frames,ignore_index=True)
    table.to_csv(path,index=False,float_format='%.17g')
    hierarchy.to_csv(folder/'atlas_hierarchy.csv')
    return table,folder


def region_figures(table, methods, *, min_count=10, rows_per_page=30, font='Arial'):
    """Yield pages without dropping small/absent regions or mixing section pairs."""
    import matplotlib.pyplot as plt
    if rows_per_page<1:
        raise ValueError('rows_per_page must be positive')
    for case,group in table.groupby('case',sort=True):
        if group.duplicated(['region','method']).any():
            raise ValueError('Duplicate regional method scores')
        for region, block in group.groupby('region'):
            if not set(methods).issubset(set(block.method)):
                raise ValueError(f'Missing method scores for {case}: {region}')
        counts = group[['region','reference_count','query_count']].drop_duplicates()
        if counts.region.duplicated().any():
            raise ValueError('Inconsistent region counts across methods')
        counts['size'] = counts[['reference_count','query_count']].min(axis=1)
        counts = counts.sort_values(['size','region']).set_index('region')
        for start in range(0,len(counts),rows_per_page):
            page = counts.iloc[start:start+rows_per_page]
            with plt.rc_context({'font.family':'sans-serif','font.sans-serif':[font,'DejaVu Sans'],
                                 'font.size':8,'pdf.fonttype':42}):
                fig,axes = plt.subplots(1,2,figsize=(13,max(4,2.8+.24*len(page))),sharey=True,
                                        layout='constrained')
                cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#dddddd')
                labels = [f'{r}{" *" if v["size"]<min_count else ""}   [{int(v.reference_count)} / {int(v.query_count)}]'
                          for r,v in page.iterrows()]
                for ax,key,title in zip(axes,('reference_fraction','query_fraction'),
                                        ('Reference-region matching fraction','Query-region matching fraction')):
                    data = group.pivot(index='region',columns='method',values=key).reindex(index=page.index,columns=methods)
                    im = ax.imshow(np.ma.masked_invalid(data.to_numpy()),vmin=0,vmax=1,cmap=cmap,aspect='auto',interpolation='nearest')
                    ax.set_xticks(range(len(methods)),methods,rotation=55,ha='right')
                    ax.set_title(title)
                    ax.set_yticks(range(len(page)),labels)
                axes[0].set_ylabel('CCF structure [reference / query measurements]; smaller first')
                fig.colorbar(im,ax=axes,shrink=.7,label='Same-region correspondence mass / region mass')
                parts = case.split('_')
                fig.suptitle(f'Allen {parts[2]} to {parts[1]}: finer anatomical agreement (main panel B)\n'
                             f'Page {start//rows_per_page+1}; * fewer than {min_count} measurements in either section; grey = absent denominator',fontsize=10)
                yield f'{case}_regions_{start//rows_per_page+1:02d}',fig
