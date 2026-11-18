"""Fixed seven-method registration protocol for Allen and BARseq.

No dataset-specific choice of splat objective or geometry optimizer. Dataset
scales and evaluation designs remain explicit instead of being pooled.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse

METHODS=('Unregistered','Geometry rigid ICP','PASTE rigid','CAST affine','CAST non-affine','Splat affine','Splat non-affine')
FAMILY_METHODS={'Geometry':('Geometry rigid ICP',),'PASTE':('PASTE rigid',),
                'CAST':('CAST affine','CAST non-affine'),'Splat':('Splat affine','Splat non-affine')}


def source_signature(root):
    root=Path(root)
    digest=hashlib.sha256()
    sources=[root/'paper/workflows'/n for n in ('standardized_registration.py','standardized_maps.py','standardized_cast.py')]
    sources += [root/'paper'/n for n in ('dlpfc_registration.py','dlpfc_splat_engine.py',
        'dlpfc_paste_solver.py','registration_baselines.py','registration_perturbations.py',
        'barseq_unequal_sampling.py','barseq_external_baselines.py','allen_registration_benchmark.py',
        'allen_registration_cosine.py','dlpfc_registration_metrics.py')]
    sources += [root/'paper/data_loaders'/n for n in ('allen_registration.py','barseq_pilot.py')]
    sources += sorted((root/'spatsplat').rglob('*.py'))
    for path in sources:
        digest.update(str(path.relative_to(root)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def cast_features(pair,dataset):
    """Keep established assay preprocessing; use identical CAST architecture/settings."""
    if dataset=='Allen': return [x.toarray() for x in pair['counts']]
    if dataset!='BARseq': raise ValueError('Unknown dataset')
    features=[]
    for counts in pair['counts']:
        total=np.asarray(counts.sum(1)).ravel()
        if np.any(total<=0): raise ValueError('Empty profile')
        normalized=(sparse.diags(1e4/total)@counts).tocsr()
        normalized.data=np.log1p(normalized.data)
        features.append(normalized.toarray())
    return features


def cache_fit(family,pair,dataset,config,root,signature, *, run):
    from paper.dlpfc_registration import pair_key,fit_splats,fit_paste,rigid_icp,versions
    from .standardized_cast import run_cast_recorded
    from .standardized_maps import readout
    root=Path(root)
    key=pair_key(pair,dict(family=family,dataset=dataset,options=config,implementation=signature))
    folder=root/'notebooks/checkpoints/registration_standardized/fits'/family/key
    path=folder/'alignment.npz'; work=folder/'work'
    if path.exists():
        with np.load(path,allow_pickle=False) as saved: result={k:saved[k] for k in saved.files}
        print(f'Loaded {family}: {key}',flush=True)
    else:
        if not run: raise FileNotFoundError(f'Missing {family} fit; enable RUN[{family!r}]')
        work.mkdir(parents=True,exist_ok=True)
        print(f'Fitting {dataset} {family}: {key}',flush=True)
        if family=='Geometry': result=rigid_icp(*pair['xy'],**config)
        elif family=='PASTE': result=fit_paste(pair,**config)
        elif family=='Splat': result=fit_splats(pair,work,**config)
        elif family=='CAST':
            native=run_cast_recorded(*pair['xy'],*cast_features(pair,dataset),output_dir=work,
                figure_dir=root/'figures/registration_standardized/native'/key,
                return_details=True,allow_reflection=False,**config)
            result=dict(affine=native['affine_mapped_b'],mapped=native['mapped_b'],
                        diagnostics=np.array(json.dumps(native['diagnostics'])))
        else: raise ValueError('Unknown family')
        # Audit exact map replay before accepting a completed fit cache.
        _,audit=readout(family,pair,result,work,pair['xy'][1],config)
        result['metadata']=np.array(json.dumps(dict(family=family,dataset=dataset,config=config,
            source_signature=signature,versions=versions(),replay_audit=audit)))
        temporary=path.with_suffix('.tmp.npz'); np.savez_compressed(temporary,**result); temporary.replace(path)
    return result,work


def allen_cases(cohort,pairs):
    from paper.allen_registration_benchmark import make_case
    cases=[]
    for role,a,b in pairs:
        for condition,task in [('natural','natural'),('same_section_smooth','combined_distortion')]:
            pair,evaluation=make_case(cohort,a,b,condition)
            cases.append(dict(name=f'allen_{a}_{b}_{task}',dataset='Allen',task=task,role=role,
                pair=pair,reference_xy=pair['xy'][0],query_xy=pair['xy'][1],
                reference_values=pair['counts'][0].toarray(),query_values=pair['counts'][1].toarray(),
                reference_eval=np.arange(len(pair['xy'][0])),query_eval=np.arange(len(pair['xy'][1])),
                truth=evaluation['truth'],labels=evaluation['labels'],section=a,
                retained_fraction=1.,shared_fraction=1.,direction=0,
                evaluation_design='transductive; annotations and true transform withheld'))
    return cases


def barseq_cases(data,sections, *, maximum=8000,seed=1023):
    from paper.barseq_unequal_sampling import make_cases
    from paper.barseq_external_baselines import training_pair
    cases=[]
    for section in sections:
        idx=np.flatnonzero(data['obs'].section.to_numpy()==section)
        idx=np.sort(np.random.default_rng(seed).choice(idx,min(maximum,len(idx)),replace=False))
        for c in make_cases(data['xy'][idx],data['counts'][idx].toarray(),
                            data['obs'].measurement_id.to_numpy()[idx],seed=seed):
            blocks=c['blocks']; a,b=blocks['reference'],blocks['query']
            pair=training_pair(blocks,'reference','query',data['genes'])
            rt=a['splits']['test']; qt=b['splits']['test']
            # Evaluation mask is never passed to a fitting function.
            rt=rt[a['shared'][rt]]; qt=qt[b['shared'][qt]]
            cases.append(dict(name=f"barseq_{section}_{c['name']}",dataset='BARseq',task='unequal_coverage',
                role='initial' if section==19 else 'evaluation',pair=pair,
                reference_xy=a['xy'],query_xy=b['xy'],reference_values=a['counts'],query_values=b['counts'],
                reference_eval=rt,query_eval=qt,truth=b['truth_xy'],labels=None,section=section,
                retained_fraction=c['keep'],shared_fraction=c['nominal_overlap'],direction=c['direction'],
                evaluation_design='held-out disjoint acquisitions; train only; reserved validation unused'))
    return cases


def run_cases(cases,options,root,output,run,signature):
    """Keep failed families explicit; never replace them with unregistered maps."""
    from .standardized_maps import readout
    root,output=Path(root),Path(output)
    fitted={}; status=[]
    for c in cases:
        mappings={'Unregistered':c['query_xy'].copy()}; audits={}
        for family in FAMILY_METHODS:
            try:
                result,work=cache_fit(family,c['pair'],c['dataset'],options[family],root,signature,run=run[family])
                mapped,audit=readout(family,c['pair'],result,work,c['query_xy'],options[family])
                if any(x.shape!=c['query_xy'].shape or not np.isfinite(x).all() for x in mapped.values()):
                    raise ValueError('Invalid out-of-sample mapping')
                if family in ('CAST','Splat'):
                    mappings[family+' affine']=mapped['affine']; mappings[family+' non-affine']=mapped['mapped']
                else: mappings[FAMILY_METHODS[family][0]]=mapped['mapped']
                audits[family]=audit
                row=dict(case=c['name'],dataset=c['dataset'],task=c['task'],family=family,status='available',cache=str(work.parent))
            except Exception as exc:
                row=dict(case=c['name'],dataset=c['dataset'],task=c['task'],family=family,status='failed or missing',error=repr(exc))
                print(c['name'],family,repr(exc),flush=True)
            status.append(row); pd.DataFrame(status).to_csv(output/'fit_status.csv',index=False)
        fitted[c['name']]=mappings
        folder=output/'cases'/c['name']; folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'maps.npz',**{str(i):x for i,x in enumerate(mappings.values())})
        (folder/'map_labels.json').write_text(json.dumps(list(mappings)))
        (folder/'readout_audit.json').write_text(json.dumps(audits,indent=2))
        np.savez_compressed(folder/'evaluation_coordinates.npz',reference=c['reference_xy'],query=c['query_xy'],
            reference_eval=c['reference_eval'],query_eval=c['query_eval'],
            truth=np.empty((0,2)) if c['truth'] is None else c['truth'])
    return fitted,pd.DataFrame(status)


def evaluate_cases(cases,fitted,output, *, permutations=99,seed=1023):
    from paper.allen_registration_cosine import expression_scores
    from paper.registration_baselines import deformation_diagnostics
    from paper.dlpfc_registration_metrics import spatial_coupling
    from paper.allen_registration_benchmark import annotation_agreement
    from paper.allen_ccf_diagnostics import load_hierarchy,region_labels,confusion_metrics
    output=Path(output); rows=[]; availability=[]
    hierarchy=load_hierarchy() if any(c['dataset']=='Allen' for c in cases) else None
    for c in cases:
        a,q=c['reference_eval'],c['query_eval']
        ref=c['reference_xy'][a]
        spacing=float(np.median(cKDTree(ref).query(ref,k=2)[0][:,1]))
        if spacing<=0: raise ValueError('Nonpositive reference spacing')
        for method in METHODS:
            available=method in fitted[c['name']]
            availability.append(dict(case=c['name'],method=method,available=available))
            if not available: continue
            mapped=fitted[c['name']][method]
            row=dict(case=c['name'],dataset=c['dataset'],task=c['task'],role=c['role'],section=c['section'],
                method=method,retained_fraction=c['retained_fraction'],shared_fraction=c['shared_fraction'],
                direction=c['direction'],evaluation_design=c['evaluation_design'],
                reference_evaluated=len(a),query_evaluated=len(q),reference_spacing=spacing)
            row.update(expression_scores(ref,mapped[q],c['reference_values'][a],c['query_values'][q],
                radius=2*spacing,k=5,permutations=permutations,seed=seed))
            row.update(deformation_diagnostics(c['query_xy'],mapped))
            if c['truth'] is not None:
                error=np.linalg.norm(mapped[q]-c['truth'][q],axis=1)/spacing
                row.update(median_error_neighbor_spacings=float(np.median(error)),
                    p90_error_neighbor_spacings=float(np.quantile(error,.9)),positional_accuracy=float(1/(1+np.median(error))))
            if c['dataset']=='Allen' and c['task']=='natural':
                coupling=spatial_coupling(ref,mapped[q])
                labs=[c['labels'][0][a],c['labels'][1][q]]
                exact,_=annotation_agreement(labs,coupling)
                _,_,division=confusion_metrics(*[region_labels(x,hierarchy,'division') for x in labs],coupling)
                row.update(ccf_exact_agreement=exact['agreement_given_scored'],ccf_scored_mass=exact['scored_mass'],
                    ccf_division_agreement=division['agreement_given_scored'])
            rows.append(row)
        pd.DataFrame(rows).to_csv(output/'scores.csv',index=False)
        pd.DataFrame(availability).to_csv(output/'method_availability.csv',index=False)
    return pd.DataFrame(rows),pd.DataFrame(availability)
