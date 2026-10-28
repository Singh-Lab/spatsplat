"""Source-only checks: no datasets, fitting, or figure rendering."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = 'notebooks/registration_publication_reproducible-allen_barseq.ipynb'


class StandaloneRegistrationTests(unittest.TestCase):
    def test_notebook_is_clean_and_has_no_other_notebook_execution(self):
        notebook = nbformat.read(ROOT/NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            self.assertEqual(cell.outputs, [])
            self.assertIsNone(cell.execution_count)
            ast.parse(cell.source)
            self.assertNotIn('.ipynb', cell.source)
            self.assertNotIn('%run', cell.source)
        source = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
        for required in ('load_registration_sections(', 'load_barseq_pilot(', 'run_cases(',
                         'add_rigid(', 'evaluate_cases(', 'measurement_links=True'):
            self.assertIn(required, source)
        self.assertIn('RUN_SPLAT_RIGID=True', source)
        self.assertIn('RUN={family:True', source)

    def test_source_only_copy_imports_without_earlier_notebooks_or_caches(self):
        manifest = ROOT/'docs/registration_publication_files.txt'
        paths = [x.strip() for x in manifest.read_text().splitlines() if x.strip() and not x.startswith('#')]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual([x for x in paths if x.endswith('.ipynb')], [NOTEBOOK])
        for path in paths:
            self.assertTrue((ROOT/path).is_file(), path)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in paths:
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/name, target)
            code = '''
from pathlib import Path
import sys, nbformat
root=Path.cwd()
sys.path.insert(0,str(root))
import paper, spatsplat
assert Path(paper.__file__).resolve().is_relative_to(root)
assert Path(spatsplat.__file__).resolve().is_relative_to(root)
from paper.workflows import standardized_registration, standardized_rigid, standardized_publication
from paper.data_loaders import allen_registration, barseq_pilot
assert len(standardized_registration.source_signature(root))==64
assert len(standardized_rigid.signature(root))==64
n=nbformat.read(root/'notebooks/registration_publication_reproducible-allen_barseq.ipynb',as_version=4)
# Execute setup/import/configuration only, never data loading or fitting.
exec(n.cells[1].source, {})
assert not (root/'notebooks/checkpoints').exists()
assert not (root/'figures').exists()
print('Source-only import/configuration succeeded')
'''
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root),
                       MPLCONFIGDIR=str(root/'mpl-cache'), NUMBA_CACHE_DIR=str(root/'numba-cache'),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            result = subprocess.run([sys.executable, '-c', code], cwd=root, env=env,
                                    text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)


if __name__ == '__main__':
    unittest.main()
