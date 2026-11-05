"""Validate gallery grouping and data without rendering or running notebooks."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from paper.workflows.registration_supplement import gallery_cases, page_data


class GalleryTests(unittest.TestCase):
    def test_group_order_and_exclusion(self):
        cases = [dict(name=str(i), dataset='BARseq', task='unequal_coverage',
                      role='evaluation', section=s, shared_fraction=f, direction=d)
                 for i, (s,f,d) in enumerate([(23,.2,120),(15,.6,60),(15,1.,0),(15,.6,0)])]
        cases.append(dict(cases[0], name='dev', role='initial'))
        self.assertEqual([c['name'] for c in gallery_cases(cases, 'G')], ['2','3','1','0'])
        self.assertEqual(gallery_cases(cases, 'B'), [])

    def test_common_bounds_include_bad_maps_and_require_scores(self):
        xy = np.array([[0.,0.], [1.,1.]])
        case = dict(name='test', task='natural', reference_xy=xy, query_xy=xy)
        maps = {'test': {'Unregistered': xy, 'Bad': xy+100}}
        scores = pd.DataFrame(dict(case=['test']*2, method=['Unregistered','Bad'],
                                   ccf_division_agreement=[.7,.1]))
        _, values, metric, low, high = page_data(case, maps, scores, ['Unregistered','Bad'])
        self.assertTrue((low<0).all())
        self.assertTrue((high>101).all())
        self.assertEqual(metric, 'ccf_division_agreement')
        np.testing.assert_array_equal(values, [.7,.1])
        with self.assertRaises(ValueError):
            page_data(case, maps, scores.iloc[:1], ['Unregistered','Bad'])

    def test_new_notebook_cell_is_unexecuted_and_parseable(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root/'notebooks/registration_publication_plots-allen_barseq.ipynb').read_text())
        cells = [c for c in nb['cells'] if c['cell_type']=='code'
                 and 'GALLERY_PANELS' in ''.join(c['source'])]
        self.assertEqual(len(cells), 1)
        ast.parse(''.join(cells[0]['source']))
        self.assertEqual(cells[0]['outputs'], [])
        self.assertIsNone(cells[0]['execution_count'])


if __name__=='__main__':
    unittest.main()
