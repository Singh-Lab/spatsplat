import unittest
import numpy as np
from scipy import sparse

from paper.allen_registration_benchmark import make_case, annotation_agreement


class AllenBenchmarkTests(unittest.TestCase):
    def setUp(self):
        xy = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        self.cohort = dict(genes=np.array(['A', 'B']), blocks={})
        for section in [24, 25]:
            self.cohort['blocks'][section] = dict(xy=xy+.01*(section-24),
                counts=sparse.csr_matrix(np.ones((4, 2))), ids=np.array(['a','b','c','d']),
                labels=np.array([1., 1., 2., np.nan]), section=str(section))

    def test_targets_excluded_from_fit(self):
        pair, evaluation = make_case(self.cohort, 24, 25, 'natural')
        self.assertNotIn('labels', pair)
        self.assertNotIn('truth', pair)
        self.assertIsNone(evaluation['truth'])

    def test_same_section_identity_and_bounded_perturbations(self):
        for name in ['same_section_affine', 'same_section_smooth']:
            pair, evaluation = make_case(self.cohort, 24, 25, name)
            np.testing.assert_array_equal(pair['xy'][0], evaluation['truth'])
            np.testing.assert_array_equal(pair['counts'][0].toarray(), pair['counts'][1].toarray())
            self.assertFalse(np.allclose(pair['xy'][1], evaluation['truth']))
            self.assertTrue(np.all((pair['xy'][1] > 0) & (pair['xy'][1] < 1)))

    def test_missing_labels_do_not_match(self):
        labels = [np.array([1., 2., np.nan]), np.array([1., 2., np.nan])]
        summary, rows = annotation_agreement(labels, np.eye(3)/3)
        self.assertAlmostEqual(summary['matching_mass'], 2/3)
        self.assertAlmostEqual(summary['scored_mass'], 2/3)
        self.assertEqual(summary['agreement_given_scored'], 1)
        self.assertEqual(len(rows), 2)

    def test_absent_reference_label_recall_is_undefined(self):
        _, rows = annotation_agreement([np.array([1., 1.]), np.array([1., 2.])], np.ones((2, 2))/4)
        self.assertTrue(np.isnan(next(r for r in rows if r['parcellation']==2)['recall_given_reference']))

    def test_invalid_coupling(self):
        with self.assertRaises(ValueError):
            annotation_agreement([np.ones(2), np.ones(2)], np.ones((2, 2)))
