"""Isolated publication artifacts and optional, verified import of completed fits."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from paper.dlpfc_registration import pair_key, versions
from . import standardized_rigid as rigid
from . import barseq_additional_controls as extra


def prepare_workspace(root, config, cases):
    root = Path(root).resolve()
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for case in cases:
        digest.update(case['name'].encode())
        digest.update(pair_key(case['pair'], {}).encode())
        for key in ('reference_xy', 'query_xy', 'reference_values', 'query_values',
                    'reference_eval', 'query_eval', 'truth'):
            if case[key] is not None:
                x = np.ascontiguousarray(case[key])
                digest.update(str((x.shape, x.dtype)).encode()); digest.update(x.tobytes())
        if case['labels'] is not None:
            for x in case['labels']:
                digest.update(np.ascontiguousarray(x).tobytes())
    identifier = digest.hexdigest()[:20]
    home = root/'notebooks/checkpoints/registration_publication_controls'/identifier
    fit_root, output = home/'private_fits', home/'reports'
    figures = root/'figures/registration_publication_controls'/identifier
    for folder in (fit_root, output, figures):
        folder.mkdir(parents=True, exist_ok=True)
    # Existing runners write native CAST diagnostics relative to their artifact root.
    # Keep these under root figures too; no fit cache is linked to another experiment.
    native = figures/'native'; native.mkdir(exist_ok=True)
    link = fit_root/'figures'
    if not link.exists():
        link.symlink_to(native, target_is_directory=True)
    if link.resolve() != native.resolve():
        raise ValueError('Unexpected native-figure destination')
    (output/'protocol.json').write_text(json.dumps(config, indent=2))
    return identifier, fit_root, output, figures


def copy_completed(source, destination, marker):
    """Copy, never link/move, a finished cache; ignore incomplete running fits."""
    source, destination = Path(source), Path(destination)
    if destination.exists() or not (source/marker).is_file():
        return False
    if source.resolve() == destination.resolve():
        raise ValueError('Source and private destination must differ')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.importing')
    if temporary.exists():
        raise RuntimeError(f'Interrupted cache import; inspect {temporary}')
    checksum = hashlib.sha256((source/marker).read_bytes()).hexdigest()
    shutil.copytree(source, temporary)
    if (hashlib.sha256((source/marker).read_bytes()).hexdigest() != checksum or
            hashlib.sha256((temporary/marker).read_bytes()).hexdigest() != checksum):
        shutil.rmtree(temporary)
        raise RuntimeError('Source fit changed during import; retry once the run finishes')
    (temporary/'publication_import.json').write_text(json.dumps(dict(
        source=str(source), marker=marker, sha256=checksum), indent=2))
    temporary.rename(destination)
    return True


def rigid_key(case, work, settings, root):
    files = [work/f for f in ('autoencoder.pt', 'field_0.pt', 'field_1.pt')]
    if not all(f.is_file() for f in files):
        return None
    digest = hashlib.sha256(json.dumps(dict(settings=settings,
        implementation=rigid.signature(root)), sort_keys=True).encode())
    for file in files:
        digest.update(file.read_bytes())
    for xy in case['pair']['xy']:
        digest.update(np.ascontiguousarray(xy).tobytes())
    return digest.hexdigest()[:20]


def seed_completed_fits(root, fit_root, output, cases, options, extra_options, settings, signature):
    """Optional optimization. Absence of all source caches is a supported clean start."""
    root, fit_root, output = Path(root), Path(fit_root), Path(output)
    if not fit_root.resolve().is_relative_to(root/'notebooks/checkpoints/registration_publication_controls'):
        raise ValueError('Imports must target the private publication workspace')
    rows, eligible_timing = [], {}
    # Numerical repair: reuse exact-key fits from earlier private workspaces too.
    private_homes = root/'notebooks/checkpoints/registration_publication_controls'
    sources = [root, *sorted(private_homes.glob('*/private_fits'),
                             key=lambda p:p.stat().st_mtime, reverse=True)]
    sources = [p for p in sources if p.resolve()!=fit_root.resolve()]
    failed_private = {}
    for source in sources:
        status_path = source.parent/'reports/additional_fit_status.csv'
        if source.name!='private_fits' or not status_path.is_file():
            continue
        try:
            saved_status = pd.read_csv(status_path)
            failed_private[source] = set(saved_status.loc[
                (saved_status.method=='PASTE2 rigid') & (saved_status.status=='failed or missing'), 'case'])
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
    for case in cases:
        entries = []
        for family in ('Geometry', 'PASTE', 'CAST', 'Splat'):
            key = pair_key(case['pair'], dict(family=family, dataset=case['dataset'],
                options=options[family], implementation=signature))
            relative = Path('notebooks/checkpoints/registration_standardized/fits')/family/key
            entries.append((relative, 'alignment.npz', (family+' family',), (relative,)))
            if family == 'Splat':
                key = rigid_key(case, root/relative/'work', settings, root)
                if key:
                    relative_rigid = Path('notebooks/checkpoints/registration_standardized/rigid')/key
                    entries.append((relative_rigid, 'rigid.pt', ('Splat rigid pose',), (relative_rigid,)))
                else:
                    for source in sources:
                        previous_key = rigid_key(case, source/relative/'work', settings, root)
                        if previous_key:
                            relative_rigid = Path('notebooks/checkpoints/registration_standardized/rigid')/previous_key
                            entries.append((relative_rigid, 'rigid.pt', ('Splat rigid pose',), (relative_rigid,)))
                            break
        key = extra.extension_key(case['pair'], extra_options, root, signature)
        previous_key = extra.extension_key(case['pair'], extra_options, root, signature,
                                          source_override=extra.PRE_EMD_RETRY_SOURCE)
        for method, stages in [('paste2', ('PASTE2 full pipeline',)),
                               ('mlp', ('MLP fields', 'MLP rigid pose'))]:
            relative = Path('notebooks/checkpoints/barseq_additional_controls/fits')/key/method
            previous_relative = relative.parent.parent/previous_key/method
            entries.append((relative, 'alignment.npz', stages, (relative,previous_relative)))
        for relative, marker, stages, alternatives in entries:
            target = fit_root/relative
            candidates = [source/alternate for source in sources for alternate in alternatives]
            source = next((p for p in candidates if (p/marker).is_file()), candidates[0])
            copy_marker = marker
            if not (source/marker).is_file() and stages==('PASTE2 full pipeline',):
                partial = next((source/alternate for source in sources
                    if case['name'] in failed_private.get(source,set())
                    for alternate in alternatives if (source/alternate/'glmpca_cost.npy').is_file()), None)
                if partial is not None:
                    source, copy_marker = partial, 'glmpca_cost.npy'
            copied = copy_completed(source, target, copy_marker)
            if (target/'publication_import.json').is_file():
                provenance = json.loads((target/'publication_import.json').read_text())
                for stage in stages:
                    eligible_timing[(case['name'], stage)] = provenance['source']
            rows.append(dict(case=case['name'], stages='; '.join(stages), copied=copied,
                available=(target/marker).is_file(), source=str(source), destination=str(target)))
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output/'cache_import_inventory.csv', index=False)
    import_timings(root, output, eligible_timing, options, extra_options, settings, signature)
    return inventory


def import_timings(root, output, eligible, options, extra_options, settings, signature):
    """Only matching protocols and imported fit stages may inherit measured times."""
    canonical = lambda value: json.dumps(value, sort_keys=True)
    path = output/'timing_events.jsonl'
    def fingerprint(event):
        return canonical({k:v for k,v in event.items() if k != 'imported_from'})
    existing = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    seen = {fingerprint(event) for event in existing}
    reports = list((root/'notebooks/checkpoints/barseq_additional_controls/reports').glob('*'))
    reports += list((root/'notebooks/checkpoints/registration_publication_controls').glob('*/reports'))
    for report in sorted(reports):
        if report.resolve()==output.resolve():
            continue
        if not (report/'protocol.json').is_file() or not (report/'timing_events.jsonl').is_file():
            continue
        config = json.loads((report/'protocol.json').read_text())
        expected = dict(options=options, extra_options=extra_options, rigid_settings=settings,
                        source_signature=signature, rigid_signature=rigid.signature(root),
                        versions=versions(), paste2_version=importlib.metadata.version('paste2'),
                        extra_source=hashlib.sha256((root/'paper/workflows/barseq_additional_controls.py').read_bytes()).hexdigest())
        previous_helper = '7f770fd09b0108537e3bfed2c60acc3fe46423f1e35343d90e6f0fff3f09102e'
        if (config.get('extra_source') not in (expected['extra_source'], previous_helper) or
                any(canonical(config.get(k)) != canonical(v) for k,v in expected.items() if k!='extra_source')):
            continue
        source_stages = {}
        for filename in ('base_fit_status.csv', 'additional_fit_status.csv'):
            status_path = report/filename
            if not status_path.is_file():
                continue
            try:
                statuses = pd.read_csv(status_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            for record in statuses.to_dict('records'):
                if record.get('status') != 'available':
                    continue
                if filename == 'base_fit_status.csv':
                    family = record['family']
                    stages = ('Splat rigid pose',) if family=='Splat rigid' else (family+' family',)
                else:
                    stages = ('PASTE2 full pipeline',) if record['method']=='PASTE2 rigid' else ('MLP fields','MLP rigid pose')
                for stage in stages:
                    source_stages[(record['case'], stage)] = record['cache']
        # The current run may still be appending. An unfinished final line is ignored.
        for line in (report/'timing_events.jsonl').read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (event['case'], event['stage'])
            if (key not in eligible or source_stages.get(key) != eligible[key]
                    or fingerprint(event) in seen):
                continue
            seen.add(fingerprint(event)); event['imported_from'] = str(report)
            with path.open('a') as stream:
                stream.write(json.dumps(event)+'\n')


from .registration_plot_data import runtime_figure
