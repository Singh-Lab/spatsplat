"""Fill missing evaluation timings without changing scored publication results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import torch

from .registration_timing import TimingLog
from .registration_plot_data import load_plot_bundle
from . import standardized_registration as base
from . import standardized_rigid as rigid
from . import barseq_additional_controls as extra

STAGES = ('Geometry family', 'PASTE family', 'CAST family', 'Splat family',
          'Splat rigid pose', 'PASTE2 full pipeline', 'MLP fields', 'MLP rigid pose')


def prepare(root, bundle):
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    manifest, tables = load_plot_bundle(bundle)
    tables['_source_report'] = manifest['source_report']
    if not manifest['complete']:
        raise ValueError('Select a completed publication bundle')
    digest = hashlib.sha256((bundle/'manifest.json').read_bytes())
    digest.update((bundle/'timings.csv.gz').read_bytes())
    digest.update(Path(__file__).read_bytes())
    output = root/'notebooks/checkpoints/registration_timing_completion'/digest.hexdigest()[:20]
    output.mkdir(parents=True, exist_ok=True)
    log = TimingLog(output)
    recorded = {json.dumps(json.loads(h),sort_keys=True) for h in manifest['hardware']}
    if recorded and json.dumps(log.hardware,sort_keys=True) not in recorded:
        raise RuntimeError(f'Timing hardware differs from the existing measurements. Current: {log.hardware}; recorded: {recorded}')
    config = manifest['protocol']
    if base.source_signature(root)!=config['source_signature'] or rigid.signature(root)!=config['rigid_signature']:
        raise RuntimeError('Registration implementation differs from the saved protocol; do not mix benchmark timings silently')
    if config['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('The saved protocol requires CUDA')
    (output/'protocol.json').write_text(json.dumps(dict(source_bundle=str(bundle),
        config=config, hardware=log.hardware, stages=STAGES, role='evaluation',
        paste2_adapter_sha256=hashlib.sha256((root/'paper/workflows/barseq_additional_controls.py').read_bytes()).hexdigest()),indent=2))
    return manifest, tables, output, log


def completed_events(output):
    path = Path(output)/'timing_events.jsonl'
    if not path.exists():
        return {}
    completed = {}
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event['status']=='completed' and event['activity']=='fresh':
            completed[(event['case'],event['stage'])] = event
    return completed


def missing_plan(tables, output):
    known = tables['timings']
    if known.duplicated(['case','stage']).any():
        raise ValueError('Duplicate timing rows in source bundle')
    measured = {(r.case,r.stage) for r in known.itertuples()
                if pd.notna(r.measured_fresh_minutes)}
    measured.update(completed_events(output))
    rows = []
    for case in tables['cases'].to_dict('records'):
        if case['role']!='evaluation':
            continue
        for stage in STAGES:
            if (case['case'],stage) not in measured:
                rows.append(dict(case=case['case'],dataset=case['dataset'],stage=stage))
    return pd.DataFrame(rows,columns=['case','dataset','stage'])


def load_evaluation_cases(config, tables):
    from paper.data_loaders.allen_registration import load_registration_sections
    from paper.data_loaders.barseq_pilot import load_barseq_pilot
    pairs = [tuple(p) for p in config['allen_pairs'] if p[0]=='evaluation']
    allen = load_registration_sections(sorted({s for _,a,b in pairs for s in (a,b)}),
        maximum=config['allen_maximum'],seed=config['seed'])
    cases = base.allen_cases(allen,pairs)
    sections = tables['cases'].query("dataset=='BARseq' and role=='evaluation'").section.unique().tolist()
    cases += base.barseq_cases(load_barseq_pilot(),sections,
        maximum=config['barseq_maximum'],seed=config['seed'])
    expected = set(tables['cases'].query("role=='evaluation'")['case'])
    if {c['name'] for c in cases} != expected:
        raise ValueError('Regenerated cases differ from the selected bundle')
    for case in cases:
        for side in ('reference','query'):
            saved = tables['measurements']
            xy = saved[(saved.case==case['name'])&(saved.side==side)].sort_values('measurement_index')[['x','y']].to_numpy()
            if not np.array_equal(xy,case[side+'_xy']):
                raise ValueError(f'Coordinates do not match the saved experiment: {case["name"]}, {side}')
    return cases


def run_missing(cases, config, tables, root, output, log):
    """Fit only missing stages; never score or publish these newly fitted maps."""
    root, output = Path(root), Path(output)
    failures = []
    artifact_path = output/'field_artifacts.json'
    artifacts = json.loads(artifact_path.read_text()) if artifact_path.exists() else {}
    for case in cases:
        name = case['name']; pair = case['pair']
        plan = missing_plan(tables,output)
        needed = set(plan.loc[plan.case==name,'stage'])
        if not needed:
            continue
        print(name, 'missing:', ', '.join(sorted(needed)), flush=True)
        for family in ('Geometry','PASTE','CAST','Splat'):
            stage = family+' family'
            if stage not in needed:
                continue
            # A failed/interrupted attempt is never resumed and called a fresh fit.
            attempt = output/'attempts'/name/family/uuid.uuid4().hex
            attempt.mkdir(parents=True)
            native_figures = root/'figures/registration_timing_completion'/output.name/name/family/attempt.name
            native_figures.mkdir(parents=True,exist_ok=True)
            (attempt/'figures').symlink_to(native_figures,target_is_directory=True)
            try:
                with log.measure(name,stage,'fresh'):
                    _, work = base.cache_fit(family,pair,case['dataset'],config['options'][family],
                        attempt,config['source_signature'],run=True)
                    if family=='Splat':
                        artifacts[name] = str(work)
                        artifact_path.write_text(json.dumps(artifacts,indent=2))
            except Exception as exc:
                failures.append(dict(case=name,stage=stage,error=repr(exc)))
                print(stage,repr(exc),flush=True)
        if 'Splat rigid pose' in needed:
            try:
                if name not in artifacts:
                    previous = Path(tables.get('_source_report',''))/'base_fit_status.csv'
                    if previous.is_file():
                        saved = pd.read_csv(previous)
                        saved = saved[(saved.case==name)&(saved.family=='Splat')&(saved.status=='available')]
                        if len(saved)==1:
                            from paper.dlpfc_registration import pair_key
                            expected = pair_key(pair,dict(family='Splat',dataset=case['dataset'],
                                options=config['options']['Splat'],implementation=config['source_signature']))
                            cache = Path(saved.iloc[0]['cache'])
                            if cache.name==expected:
                                artifacts[name] = str(cache/'work')
                    if name not in artifacts:
                        raise RuntimeError('A completed Splat family fit is needed to time the rigid pose')
                models, xy = rigid.load_fields(pair,Path(artifacts[name]),config['options']['Splat'])
                splits = [dict(train=torch.arange(len(x)),val=torch.arange(len(x))) for x in xy]
                with log.measure(name,'Splat rigid pose','fresh'):
                    result = rigid.fit_rigid(*xy,*models,*splits,config['rigid_settings'],use_expression=True)
                del models,xy,result
            except Exception as exc:
                failures.append(dict(case=name,stage='Splat rigid pose',error=repr(exc)))
                print('Splat rigid pose',repr(exc),flush=True)
        if 'PASTE2 full pipeline' in needed:
            work = output/'attempts'/name/'PASTE2'/uuid.uuid4().hex
            work.mkdir(parents=True)
            try:
                with log.measure(name,'PASTE2 full pipeline','fresh'):
                    result = extra.fit_paste2(pair,work,config['extra_options']['paste2'])
                del result
            except Exception as exc:
                failures.append(dict(case=name,stage='PASTE2 full pipeline',error=repr(exc)))
                print('PASTE2',repr(exc),flush=True)
        if needed & {'MLP fields','MLP rigid pose'}:
            work = output/'attempts'/name/'MLP'/uuid.uuid4().hex
            try:
                if 'MLP fields' in needed:
                    with log.measure(name,'MLP fields','fresh'):
                        models,_ = extra.load_or_fit_mlps(pair,config['extra_options']['mlp'],work,run=True)
                else:
                    # Prerequisite only: retain the already measured field-training time.
                    models,_ = extra.load_or_fit_mlps(pair,config['extra_options']['mlp'],work,run=True)
                if 'MLP rigid pose' in needed:
                    xy = [torch.as_tensor(x,dtype=torch.float32,device=config['device']) for x in pair['xy']]
                    splits = [dict(train=torch.arange(len(x)),val=torch.arange(len(x))) for x in xy]
                    with log.measure(name,'MLP rigid pose','fresh'):
                        result = extra.fit_rigid(*xy,*models,*splits,config['rigid_settings'],use_expression=True)
                    del xy,result
                del models
            except Exception as exc:
                failures.append(dict(case=name,stage='MLP',error=repr(exc)))
                print('MLP',repr(exc),flush=True)
        pd.DataFrame(failures,columns=['case','stage','error']).to_csv(output/'failures.csv',index=False)
        missing_plan(tables,output).to_csv(output/'remaining.csv',index=False)
    return missing_plan(tables,output)


def publish_timings(bundle, tables, output):
    """Create a new portable snapshot; scores and spatial maps remain byte-identical."""
    bundle, output = Path(bundle), Path(output)
    manifest = json.loads((bundle/'manifest.json').read_text())
    timings = tables['timings'].copy()
    events = completed_events(output)
    for (case,stage),event in events.items():
        mask = (timings.case==case)&(timings.stage==stage)
        if mask.any():
            if timings.loc[mask,'measured_fresh_minutes'].notna().any():
                continue
            timings.loc[mask,'measured_fresh_minutes'] = event['wall_seconds']/60
        else:
            metadata = tables['cases'].set_index('case').loc[case].to_dict()
            timings = pd.concat([timings,pd.DataFrame([dict(case=case,stage=stage,
                measured_fresh_minutes=event['wall_seconds']/60,**metadata)])],ignore_index=True)
    destination = bundle.parent/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination.mkdir()
    for name,filename in manifest['tables'].items():
        if Path(filename).name!=filename:
            raise ValueError('Expected a flat bundle table filename')
        if name!='timings':
            shutil.copy2(bundle/filename,destination/filename)
    timings.to_csv(destination/manifest['tables']['timings'],index=False,float_format='%.17g',compression='gzip')
    manifest['hardware'] = sorted(set(manifest['hardware'])|
        {json.dumps(e['hardware'],sort_keys=True) for e in events.values()})
    manifest['timing_completion'] = dict(source_bundle=str(bundle),report=str(output),
        scope='evaluation',remaining=missing_plan(tables,output).to_dict('records'))
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    timings.to_csv(output/'combined_timings.csv',index=False,float_format='%.17g')
    return destination
