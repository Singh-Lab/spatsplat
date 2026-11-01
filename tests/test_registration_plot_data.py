import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import nbformat
import numpy as np
import pandas as pd

from paper.workflows.registration_plot_data import export_plot_bundle, load_plot_bundle, plotting_inputs


class PlotDataTests(unittest.TestCase):
    def test_dataframe_roundtrip_is_portable_and_preserves_precision(self):
        xy = np.array([[1/3, .1], [.2345678901234567,.9], [.8,.7123456789012345]])
        methods = ['Unregistered','Splat rigid']
        case = dict(name='fixture',dataset='Allen',task='natural',role='evaluation',
            section=28,shared_fraction=1.,retained_fraction=1.,direction=0,
            reference_xy=xy,query_xy=xy.copy(),labels=None,
            division_labels=[np.array(['NA','CTX','Unannotated'])]*2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); report = root/'report'; maps = report/'cases/fixture'
            maps.mkdir(parents=True)
            np.savez(maps/'evaluation_coordinates.npz',reference=xy,query=xy)
            np.savez(maps/'maps.npz',**{'0':xy})
            (maps/'map_labels.json').write_text(json.dumps(['Unregistered']))
            np.savez(maps/'rigid_map.npz',mapped=xy+.1)
            pd.DataFrame([dict(case='fixture',method=m,ccf_division_agreement=.12345678901234567)
                          for m in methods]).to_csv(report/'scores.csv',index=False)
            pd.DataFrame([dict(case='fixture',method=m,available=True) for m in methods]).to_csv(
                report/'method_availability.csv',index=False)
            (report/'protocol.json').write_text('{}')
            event = dict(case='fixture',stage='Splat rigid pose',status='completed',activity='fresh',
                         wall_seconds=12,hardware={'cpu':'fixture'})
            (report/'timing_events.jsonl').write_text(json.dumps(event)+'\n')
            bundle = export_plot_bundle(report,[case],methods)
            portable = root/'portable'; shutil.copytree(bundle,portable)
            shutil.rmtree(report)
            manifest,tables = load_plot_bundle(portable)
            self.assertTrue(manifest['complete'])
            restored,fitted = plotting_inputs(tables)
            np.testing.assert_array_equal(restored[0]['reference_xy'],xy)
            np.testing.assert_array_equal(fitted['fixture']['Splat rigid'],xy+.1)
            self.assertEqual(restored[0]['division_labels'][0][0],'NA')
            self.assertEqual(tables['scores'].ccf_division_agreement.iloc[0],.12345678901234567)
            self.assertEqual(tables['timings'].measured_fresh_minutes.iloc[0],.2)
            bad = dict(tables,registered=pd.concat([tables['registered'],tables['registered'].iloc[:1]]))
            with self.assertRaisesRegex(ValueError,'Duplicate registered'):
                plotting_inputs(bad)

    def test_plotting_modules_import_without_training_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            for name in ('paper/__init__.py','paper/workflows/__init__.py',
                         'paper/workflows/registration_plot_data.py','paper/workflows/standardized_publication.py'):
                (target/name).parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(root/name,target/name)
            code = '''
import sys
from paper.workflows import registration_plot_data, standardized_publication
assert 'torch' not in sys.modules
assert 'spatsplat' not in sys.modules
assert 'abc_atlas_access' not in sys.modules
assert 'paper.workflows.standardized_rigid' not in sys.modules
assert 'paper.workflows.publication_controls' not in sys.modules
'''
            result = subprocess.run([sys.executable,'-c',code],cwd=target,text=True,capture_output=True,
                timeout=60,env=dict(os.environ,PYTHONPATH=str(target),PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)

    def test_plotting_notebook_has_no_training_or_scoring_calls(self):
        root = Path(__file__).resolve().parents[1]
        notebook = nbformat.read(root/'notebooks/registration_publication_plots-allen_barseq.ipynb',as_version=4)
        nbformat.validate(notebook)
        banned = {'run_cases','run_additions','add_rigid','evaluate_cases','score_additions',
                  'load_barseq_pilot','load_registration_sections','seed_completed_fits'}
        for cell in notebook.cells:
            if cell.cell_type=='code':
                tree = ast.parse(cell.source)
                self.assertEqual(cell.outputs,[])
                for node in ast.walk(tree):
                    if isinstance(node,ast.Call):
                        name = getattr(node.func,'id',getattr(node.func,'attr',''))
                        self.assertNotIn(name,banned)


if __name__=='__main__':
    unittest.main()
