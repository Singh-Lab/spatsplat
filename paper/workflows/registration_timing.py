"""Opt-in wall-clock instrumentation without changing numerical cache keys."""
from contextlib import contextmanager
import json
import platform
from pathlib import Path
import time

import pandas as pd
import torch

from . import standardized_registration as base
from . import standardized_rigid as rigid
from . import barseq_additional_controls as extra


def synchronize():
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)


class TimingLog:
    def __init__(self, output):
        self.path = Path(output)/'timing_events.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hardware = dict(host=platform.node(), cpu=platform.processor(),
            torch_threads=torch.get_num_threads(),
            gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

    @contextmanager
    def measure(self, case, stage, activity):
        synchronize()
        start = time.perf_counter()
        status = 'completed'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            synchronize()
            record = dict(case=case, stage=stage, activity=activity, status=status,
                          wall_seconds=time.perf_counter()-start, timestamp=time.time(),
                          hardware=self.hardware)
            with self.path.open('a') as stream:
                stream.write(json.dumps(record)+'\n')

    def summary(self, cases):
        records = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        stages = ('Geometry family', 'PASTE family', 'CAST family', 'Splat family',
                  'Splat rigid pose', 'PASTE2 full pipeline', 'MLP fields', 'MLP rigid pose')
        rows = []
        for case in cases:
            for stage in stages:
                events = [r for r in records if r['case']==case['name'] and r['stage']==stage]
                fits = [r for r in events if r['activity']=='fresh' and r['status']=='completed']
                last = events[-1] if events else {}
                rows.append(dict(case=case['name'], section=case['section'],
                    shared_fraction=case['shared_fraction'], stage=stage,
                    measured_fresh_minutes=fits[-1]['wall_seconds']/60 if fits else float('nan'),
                    latest_activity=last.get('activity', 'cached or not run; no timing'),
                    latest_status=last.get('status', 'unknown'),
                    latest_call_seconds=last.get('wall_seconds', float('nan'))))
        table = pd.DataFrame(rows)
        table.to_csv(self.path.parent/'timing_summary.csv', index=False)
        return table


def instrument(output, *, source_root=None):
    """Return locally wrapped runners; installed/global functions stay unchanged."""
    log = TimingLog(output)
    current = {'case': None}

    def timed_cache(family, pair, dataset, config, root, signature, *, run):
        from paper.dlpfc_registration import pair_key
        key = pair_key(pair, dict(family=family, dataset=dataset, options=config, implementation=signature))
        folder = Path(root)/'notebooks/checkpoints/registration_standardized/fits'/family/key
        activity = ('cache_load' if (folder/'alignment.npz').exists() else
                    'resumed' if folder.exists() and any(folder.rglob('*')) else 'fresh')
        with log.measure(current['case'], family+' family', activity):
            return base.cache_fit(family, pair, dataset, config, root, signature, run=run)

    def timed_splat_pose(*args, **kwargs):
        with log.measure(current['case'], 'Splat rigid pose', 'fresh'):
            return rigid.fit_rigid(*args, **kwargs)

    def timed_paste2(pair, folder, config):
        activity = 'resumed' if any(folder.iterdir()) else 'fresh'
        with log.measure(current['case'], 'PASTE2 full pipeline', activity):
            return extra.fit_paste2(pair, folder, config)

    def timed_mlp_fields(pair, config, folder, **kwargs):
        # Diagnostics reload fields with run=False; do not count these as training.
        if not kwargs.get('run', False):
            return extra.load_or_fit_mlps(pair, config, folder, **kwargs)
        activity = 'cache_or_resumed' if folder.exists() and any(folder.rglob('*.pt')) else 'fresh'
        with log.measure(current['case'], 'MLP fields', activity):
            return extra.load_or_fit_mlps(pair, config, folder, **kwargs)

    def timed_mlp_pose(*args, **kwargs):
        with log.measure(current['case'], 'MLP rigid pose', 'fresh'):
            return extra.fit_rigid(*args, **kwargs)

    run_base = extra.clone(base.run_cases, cache_fit=timed_cache)
    source_signature = rigid.signature if source_root is None else lambda unused: rigid.signature(source_root)
    source_key = extra.extension_key if source_root is None else (
        lambda pair, options, unused, signature, **kwargs:
        extra.extension_key(pair, options, source_root, signature, **kwargs))
    run_rigid = extra.clone(rigid.add_rigid, fit_rigid=timed_splat_pose, signature=source_signature)
    run_extra = extra.clone(extra.run_additions, fit_paste2=timed_paste2,
        load_or_fit_mlps=timed_mlp_fields, fit_rigid=timed_mlp_pose, extension_key=source_key)

    def run_cases(cases, options, root, output, run, signature):
        fitted, tables = {}, []
        for case in cases:
            current['case'] = case['name']
            maps, status = run_base([case], options, root, output, run, signature)
            fitted.update(maps); tables.append(status)
            pd.concat(tables, ignore_index=True).to_csv(Path(output)/'fit_status.csv', index=False)
        return fitted, pd.concat(tables, ignore_index=True)

    def add_rigid(cases, fitted, status, config, root, output, settings, *, run):
        for case in cases:
            current['case'] = case['name']
            fitted, status = run_rigid([case], fitted, status, config, root, output, settings, run=run)
        status[status.family=='Splat rigid'].to_csv(Path(output)/'rigid_fit_status.csv', index=False)
        return fitted, status

    def run_additions(cases, fitted, status, base_options, options, root, output, signature, run):
        tables, diagnostics = [], []
        for case in cases:
            current['case'] = case['name']
            fitted, table, diagnostic = run_extra([case], fitted, status, base_options,
                options, root, output, signature, run)
            tables.append(table); diagnostics.append(diagnostic)
            pd.concat(tables, ignore_index=True).to_csv(Path(output)/'additional_fit_status.csv', index=False)
            pd.concat(diagnostics, ignore_index=True).to_csv(Path(output)/'field_diagnostics.csv', index=False)
        return fitted, pd.concat(tables, ignore_index=True), pd.concat(diagnostics, ignore_index=True)

    return run_cases, add_rigid, run_additions, log
