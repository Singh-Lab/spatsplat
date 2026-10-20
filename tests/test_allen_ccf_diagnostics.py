import unittest
import numpy as np
import pandas as pd
from paper.allen_ccf_diagnostics import confusion_metrics, region_labels, UNANNOTATED


class CCFDiagnosticsTests(unittest.TestCase):
    def test_unannotated_mass_and_denominator(self):
        matrix, regions, score = confusion_metrics(['1', UNANNOTATED], ['1', UNANNOTATED], np.eye(2)/2)
        self.assertAlmostEqual(score['matching_mass'], .5)
        self.assertAlmostEqual(score['scored_mass'], .5)
        self.assertAlmostEqual(score['agreement_given_scored'], 1)
        self.assertAlmostEqual(matrix.to_numpy().sum(), 1)
        self.assertTrue(np.isnan(regions.set_index('region').loc[UNANNOTATED, 'recall']))

    def test_reference_recall_includes_unannotated_query(self):
        _, regions, _ = confusion_metrics(['1', '1'], ['1', UNANNOTATED], np.ones((2, 2))/4)
        self.assertAlmostEqual(regions.set_index('region').loc['1', 'recall'], .5)

    def test_coarse_gain_preserves_scored_mass(self):
        atlas = pd.DataFrame(dict(structure=['A','A'], division=['D','D']), index=[1,2])
        pi = np.array([[0, .5], [.5, 0]])
        fine = [region_labels([1,2], atlas, 'parcellation')]*2
        coarse = [region_labels([1,2], atlas, 'structure')]*2
        _, _, f = confusion_metrics(*fine, pi)
        _, _, c = confusion_metrics(*coarse, pi)
        self.assertEqual(f['matching_mass'], 0)
        self.assertEqual(c['matching_mass'], 1)
        self.assertEqual(f['scored_mass'], c['scored_mass'])

    def test_missing_mapping_fails(self):
        with self.assertRaises(ValueError):
            region_labels([9], pd.DataFrame({'structure':['A']}, index=[1]), 'structure')

    def test_proportions_limit_agreement(self):
        _, _, score = confusion_metrics(['A','A'], ['A','B'], np.ones((2,2))/4)
        self.assertEqual(score['marginal_label_ceiling'], .5)
