import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from paper.workflows import registration_timing_completion as completion
from paper.workflows.registration_timing import TimingLog


class TimingCompletionTests(unittest.TestCase):
    def tables(self):
        return dict(cases=pd.DataFrame([
            dict(case='eval', dataset='Allen', role='evaluation'),
            dict(case='dev', dataset='Allen', role='development')]),
            timings=pd.DataFrame([dict(case='eval', stage=s, measured_fresh_minutes=2.)
                                 for s in completion.STAGES]))

    def test_plan_preserves_known_and_ignores_failed_or_resumed(self):
        tables = self.tables()
        tables['timings'].loc[:2, 'measured_fresh_minutes'] = np.nan
        with tempfile.TemporaryDirectory() as folder:
            events = [dict(case='eval', stage=stage, activity=activity, status=status)
                      for stage, activity, status in [
                          ('Geometry family', 'fresh', 'completed'),
                          ('PASTE family', 'fresh', 'failed'),
                          ('CAST family', 'resumed', 'completed')]]
            (Path(folder)/'timing_events.jsonl').write_text(
                ''.join(json.dumps(e)+'\n' for e in events))
            plan = completion.missing_plan(tables, folder)
            self.assertEqual(set(plan.stage), {'PASTE family', 'CAST family'})
            self.assertEqual(set(plan.case), {'eval'})

    def test_run_isolated_and_resumes_completed_stages(self):
        tables = self.tables()
        tables['timings'].loc[0, 'measured_fresh_minutes'] = np.nan
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root/'timings'
            log = TimingLog(output)
            cases = [dict(name='eval', dataset='Allen', pair={})]
            config = dict(options={'Geometry': {}}, source_signature='test')
            def fit(family, pair, dataset, options, attempt, signature, run):
                self.assertTrue(attempt.is_relative_to(output/'attempts'))
                self.assertTrue((attempt/'figures').resolve().is_relative_to(root/'figures'))
                return {}, attempt/'work'
            with patch.object(completion.base, 'cache_fit', side_effect=fit) as mocked:
                self.assertTrue(completion.run_missing(cases, config, tables, root, output, log).empty)
                self.assertTrue(completion.run_missing(cases, config, tables, root, output, log).empty)
                self.assertEqual(mocked.call_count, 1)

    def test_export_changes_only_missing_timings(self):
        tables = self.tables()
        tables['timings'].loc[0, 'measured_fresh_minutes'] = np.nan
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder)/'original'
            output = Path(folder)/'timing'
            bundle.mkdir(); output.mkdir()
            tables['timings'].to_csv(bundle/'timings.csv.gz', index=False)
            (bundle/'scores.csv.gz').write_bytes(b'unchanged scores and maps')
            manifest = dict(tables={'timings': 'timings.csv.gz', 'scores': 'scores.csv.gz'}, hardware=[])
            (bundle/'manifest.json').write_text(json.dumps(manifest))
            original = (bundle/'timings.csv.gz').read_bytes()
            events = [dict(case='eval', stage=s, activity='fresh', status='completed',
                           wall_seconds=60., hardware={}) for s in completion.STAGES[:2]]
            (output/'timing_events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            destination = completion.publish_timings(bundle, tables, output)
            updated = pd.read_csv(destination/'timings.csv.gz')
            self.assertEqual(updated.iloc[0].measured_fresh_minutes, 1.)
            self.assertEqual(updated.iloc[1].measured_fresh_minutes, 2.)
            self.assertEqual((bundle/'timings.csv.gz').read_bytes(), original)
            self.assertEqual((destination/'scores.csv.gz').read_bytes(), (bundle/'scores.csv.gz').read_bytes())
            self.assertEqual(json.loads((destination/'manifest.json').read_text())['timing_completion']['remaining'], [])

    def test_notebook_clean_and_parseable(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root/'notebooks/registration_timing_completion-allen_barseq.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])


if __name__ == '__main__':
    unittest.main()
