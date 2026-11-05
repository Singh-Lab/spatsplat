import ast
import json
from pathlib import Path
import unittest
import numpy as np
from paper.workflows.registration_region_agreement import regional_scores


class RegionTests(unittest.TestCase):
    def test_direction_denominators_absent_and_unannotated(self):
        # A includes a link to unannotated tissue, which must remain in its denominator.
        a = ['A','A','B','Unannotated']
        b = ['A','Unannotated','C','Unannotated']
        pi = np.eye(4)/4
        t = regional_scores(a,b,pi).set_index('region')
        self.assertNotIn('Unannotated',t.index)
        self.assertEqual(t.loc['A','reference_fraction'],.5)
        self.assertEqual(t.loc['A','query_fraction'],1.)
        self.assertEqual(t.loc['B','reference_fraction'],0.)
        self.assertTrue(np.isnan(t.loc['B','query_fraction']))
        self.assertTrue(np.isnan(t.loc['C','reference_fraction']))
        self.assertEqual(t.loc['C','query_fraction'],0.)
        self.assertEqual(t.loc['A','reference_count'],2)

    def test_invalid_mass(self):
        with self.assertRaises(ValueError):
            regional_scores(['A'],['A'],np.array([[2.]]))

    def test_notebook_cells_parse_without_running(self):
        root = Path(__file__).resolve().parents[1]
        n = json.loads((root/'notebooks/registration_publication_plots-allen_barseq.ipynb').read_text())
        cells = [c for c in n['cells'] if c['cell_type']=='code' and 'registration_region_agreement' in ''.join(c['source'])]
        self.assertEqual(len(cells),2)
        for c in cells:
            ast.parse(''.join(c['source']))
            self.assertEqual(c['outputs'],[])


if __name__=='__main__':
    unittest.main()
