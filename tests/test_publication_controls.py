import ast
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
import torch

from paper.workflows import publication_controls as publication
from paper.workflows import registration_timing as timing
from paper.workflows import barseq_additional_controls as extra
from paper.workflows import standardized_rigid as rigid
from paper.workflows import standardized_publication as figure


class PublicationControlsTests(unittest.TestCase):
    def test_inner_transport_retries_only_iteration_limit(self):
        from unittest.mock import Mock
        weights = np.array([.5,.5]); coupling = np.diag(weights)
        solver = Mock(side_effect=[(coupling, {'warning':'numItermax reached before optimality'}),
                                   (coupling, {'warning':None})])
        attempts = []
        result, _ = extra.emd_with_retry(solver, weights, weights, np.zeros((2,2)), attempts=attempts)
        np.testing.assert_equal(result,coupling)
        self.assertEqual([x.kwargs['numItermax'] for x in solver.call_args_list],[1000000,5000000])
        self.assertEqual(len(attempts),2)
        solver = Mock(return_value=(coupling, {'warning':'infeasible'}))
        with self.assertRaisesRegex(RuntimeError,'infeasible'):
            extra.emd_with_retry(solver, weights, weights, np.zeros((2,2)), attempts=[])
        self.assertEqual(solver.call_count,1)

    def test_inner_transport_rejects_invalid_marginals_and_exhaustion(self):
        from unittest.mock import Mock
        weights = np.array([.5,.5])
        solver = Mock(return_value=(np.zeros((2,2)),{'warning':None}))
        with self.assertRaisesRegex(RuntimeError,'marginals'):
            extra.emd_with_retry(solver, weights, weights, np.zeros((2,2)), attempts=[])
        solver = Mock(return_value=(np.diag(weights),{'warning':'numItermax reached'}))
        with self.assertRaisesRegex(RuntimeError,'20000000'):
            extra.emd_with_retry(solver, weights, weights, np.zeros((2,2)), attempts=[])
        self.assertEqual(solver.call_count,3)

    def test_repair_import_preserves_partial_paste2_from_private_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            home = root/'notebooks/checkpoints/registration_publication_controls'
            old = home/'old/private_fits'; new = home/'new/private_fits'
            output = home/'new/reports'; output.mkdir(parents=True)
            relative = Path('notebooks/checkpoints/barseq_additional_controls/fits/old-key/paste2')
            source = old/relative; source.mkdir(parents=True)
            old_report = old.parent/'reports'; old_report.mkdir()
            pd.DataFrame([dict(case='case',method='PASTE2 rigid',status='failed or missing')]).to_csv(
                old_report/'additional_fit_status.csv',index=False)
            np.save(source/'glmpca_cost.npy',np.eye(2))
            (source/'overlap_0.99.npz').write_bytes(b'completed candidate')
            def key(*args, **kwargs):
                return 'old-key' if kwargs.get('source_override') else 'new-key'
            with patch.object(publication,'pair_key',return_value='base-key'), \
                    patch.object(publication,'rigid_key',return_value=None), \
                    patch.object(extra,'extension_key',side_effect=key), \
                    patch.object(publication,'import_timings'):
                table = publication.seed_completed_fits(root,new,output,
                    [dict(name='case',dataset='Allen',pair={})],
                    {k:{} for k in ('Geometry','PASTE','CAST','Splat')},{},{},'signature')
            target = new/relative.parent.parent/'new-key/paste2'
            self.assertTrue((target/'glmpca_cost.npy').exists())
            self.assertTrue((target/'overlap_0.99.npz').exists())
            self.assertTrue((source/'glmpca_cost.npy').exists())
            row = table[table.stages=='PASTE2 full pipeline'].iloc[0]
            self.assertTrue(row.copied)
            self.assertFalse(row.available)

    def test_copy_requires_completion_and_is_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'source'; target = root/'private'
            source.mkdir(); (source/'partial.txt').write_text('partial')
            self.assertFalse(publication.copy_completed(source, target, 'alignment.npz'))
            (source/'alignment.npz').write_bytes(b'completed fixture')
            self.assertTrue(publication.copy_completed(source, target, 'alignment.npz'))
            (source/'alignment.npz').unlink()
            self.assertEqual((target/'alignment.npz').read_bytes(), b'completed fixture')
            self.assertFalse(target.is_symlink())
            self.assertFalse(publication.copy_completed(source, target, 'alignment.npz'))

    def test_extra_runner_hashes_source_but_writes_private_cache(self):
        xy = np.array([[0.,0.],[1.,0.],[0.,1.]])
        case = dict(name='fixture', pair={'xy':[xy,xy]}, query_xy=xy)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source'; private = Path(folder)/'private'; output = Path(folder)/'reports'
            source.mkdir(); output.mkdir()
            def key(pair, options, root, signature, **kwargs):
                self.assertEqual(root, source)
                return 'old-key' if kwargs.get('legacy') else 'new-key'
            def fit(pair, work, config):
                self.assertTrue(work.is_relative_to(private))
                return dict(mapped=xy.copy(), selected_overlap=np.array(.5))
            with patch.object(extra, 'EXTRA_METHODS', ('PASTE2 rigid',)), \
                    patch.object(extra, 'extension_key', side_effect=key), \
                    patch.object(extra, 'fit_paste2', side_effect=fit), \
                    patch.object(timing, 'synchronize'):
                _, _, run, _ = timing.instrument(output, source_root=source)
                fitted, status, _ = run([case], {'fixture':{}}, pd.DataFrame(), {},
                    {'paste2':{}}, private, output, 'signature', {'PASTE2 rigid':True})
            self.assertEqual(status.status.tolist(), ['available'])
            self.assertIn('PASTE2 rigid', fitted['fixture'])
            self.assertTrue((private/'notebooks/checkpoints/barseq_additional_controls/fits/new-key/paste2/alignment.npz').is_file())
            self.assertFalse((source/'notebooks').exists())

    def test_rigid_runner_uses_real_source_signature_with_private_artifacts(self):
        xy = np.array([[0.,0.],[1.,0.],[0.,1.]])
        case = dict(name='fixture', dataset='BARseq', task='unequal_coverage',
                    pair={'xy':[xy,xy]}, query_xy=xy)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source'; private = Path(folder)/'private'; output = Path(folder)/'reports'
            source.mkdir(); output.mkdir(); work = private/'fields/work'; work.mkdir(parents=True)
            for name in ('autoencoder.pt','field_0.pt','field_1.pt'):
                (work/name).write_bytes(b'fingerprint fixture')
            status = pd.DataFrame([dict(case='fixture', family='Splat',status='available',cache=str(work.parent))])
            def signature(root):
                self.assertEqual(root, source)
                return 'source-signature'
            with patch.object(rigid, 'signature', side_effect=signature), \
                    patch.object(rigid, 'load_fields', return_value=([None,None],[torch.tensor(xy),torch.tensor(xy)])), \
                    patch.object(rigid, 'fit_rigid', return_value=dict(state={},mapped=torch.tensor(xy))), \
                    patch.object(rigid, 'map_rigid', side_effect=lambda x,state: np.asarray(x)), \
                    patch.object(timing, 'synchronize'):
                _, run, _, _ = timing.instrument(output, source_root=source)
                _, result = run([case], {'fixture':{}}, status, {}, private, output,
                                dict(rigid.DEFAULTS), run=True)
            row = result[result.family=='Splat rigid'].iloc[0]
            self.assertEqual(row.status, 'available')
            self.assertTrue(Path(row.cache).is_relative_to(private))
            self.assertFalse((source/'notebooks').exists())

    def test_extended_figure_roster_is_opt_in(self):
        methods = (*rigid.METHODS, *extra.EXTRA_METHODS)
        cases = [dict(name=name,role='evaluation',task='combined_distortion') for name in figure.EXAMPLES]
        maps = {c['name']:{'Splat rigid':np.zeros((2,2))} for c in cases}
        scores = pd.DataFrame([dict(case=c['name'],method=method,positional_accuracy=.5)
                               for c in cases for method in methods])
        figure.validate_inputs(cases,maps,scores,'Splat rigid',False,methods=methods)
        with self.assertRaisesRegex(ValueError, 'Incomplete method scores'):
            figure.validate_inputs(cases,maps,scores,'Splat rigid',False)
        self.assertEqual(len(figure.METHODS),8)

    def test_notebook_and_manifest_are_standalone(self):
        root = Path(__file__).resolve().parents[1]
        name = 'notebooks/registration_publication_controls_reproducible-allen_barseq.ipynb'
        notebook = nbformat.read(root/name,as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                self.assertEqual(cell.outputs,[])
                self.assertIsNone(cell.execution_count)
                self.assertNotIn('.ipynb',cell.source)
        manifest = (root/'docs/registration_publication_controls_files.txt').read_text().splitlines()
        files = [f for f in manifest if f and not f.startswith('#')]
        self.assertEqual([f for f in files if f.endswith('.ipynb')],[name])
        self.assertTrue(all((root/f).is_file() for f in files))

    def test_source_only_checkout_imports_without_other_notebooks_or_caches(self):
        root = Path(__file__).resolve().parents[1]
        files = [f for f in (root/'docs/registration_publication_controls_files.txt').read_text().splitlines()
                 if f and not f.startswith('#')]
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            for file in files:
                (target/file).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root/file, target/file)
            code = '''
from pathlib import Path
import paper, spatsplat
from paper.workflows import publication_controls, registration_timing, standardized_publication
from paper.workflows.standardized_registration import source_signature
from paper.workflows.standardized_rigid import signature
root = Path.cwd()
assert Path(paper.__file__).resolve().is_relative_to(root)
assert Path(spatsplat.__file__).resolve().is_relative_to(root)
assert len(source_signature(root)) == 64
assert len(signature(root)) == 64
assert len(list((root/'notebooks').glob('*.ipynb'))) == 1
assert not (root/'notebooks/checkpoints').exists()
assert not (root/'figures').exists()
'''
            result = subprocess.run([sys.executable,'-c',code],cwd=target,text=True,
                capture_output=True,timeout=120,env=dict(os.environ,PYTHONPATH=str(target),
                PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(target/'mpl'),NUMBA_CACHE_DIR=str(target/'numba')))
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)

    def test_timing_import_checks_exact_cache_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); output = root/'private/reports'; output.mkdir(parents=True)
            report = root/'notebooks/checkpoints/barseq_additional_controls/reports/fixture'
            report.mkdir(parents=True)
            helper = root/'paper/workflows/barseq_additional_controls.py'
            helper.parent.mkdir(parents=True); helper.write_text('fixture')
            config = dict(options={},extra_options={},rigid_settings={},source_signature='base',
                rigid_signature='rigid',versions={},paste2_version='1.0.1',
                extra_source=hashlib.sha256(helper.read_bytes()).hexdigest())
            (report/'protocol.json').write_text(json.dumps(config))
            event = dict(case='case',stage='PASTE2 full pipeline',activity='fresh',status='completed',
                         wall_seconds=10,timestamp=1,hardware={})
            (report/'timing_events.jsonl').write_text(json.dumps(event)+'\n')
            pd.DataFrame([dict(case='case',method='PASTE2 rigid',status='available',cache='/exact/cache')]).to_csv(
                report/'additional_fit_status.csv',index=False)
            with patch.object(publication.rigid,'signature',return_value='rigid'), \
                    patch.object(publication,'versions',return_value={}), \
                    patch.object(publication.importlib.metadata,'version',return_value='1.0.1'):
                publication.import_timings(root,output,{('case','PASTE2 full pipeline'):'/wrong/cache'}, {},{},{},'base')
                self.assertFalse((output/'timing_events.jsonl').exists())
                for _ in range(2):
                    publication.import_timings(root,output,{('case','PASTE2 full pipeline'):'/exact/cache'}, {},{},{},'base')
            records = (output/'timing_events.jsonl').read_text().splitlines()
            self.assertEqual(len(records),1)
            self.assertEqual(json.loads(records[0])['wall_seconds'],10)


if __name__ == '__main__':
    unittest.main()
