import unittest
import numpy as np
from paper.allen_registration_cosine import expression_scores


class CosineTests(unittest.TestCase):
    def test_identity_and_shuffle(self):
        xy=np.column_stack([np.arange(8),np.zeros(8)])
        values=np.eye(8)
        scores=expression_scores(xy,xy,values,values,radius=.2,permutations=19)
        self.assertAlmostEqual(scores['profile_cosine'],1)
        self.assertAlmostEqual(scores['coverage_weighted_cosine'],1)
        self.assertLess(scores['profile_cosine_shuffle_mean'],.4)

    def test_unsupported_not_rewarded(self):
        xy=np.column_stack([np.arange(4),np.zeros(4)])
        scores=expression_scores(xy,xy+100,np.ones((4,2)),np.ones((4,2)),radius=.2,permutations=3)
        self.assertEqual(scores['coverage'],0)
        self.assertEqual(scores['coverage_weighted_cosine'],0)
        self.assertTrue(np.isnan(scores['profile_cosine']))

    def test_matches_existing_metric(self):
        from paper.registration_baselines import evaluate_alignment
        rng=np.random.default_rng(9)
        a,b=rng.random((15,2)),rng.random((17,2))
        x,y=rng.random((15,4)),rng.random((17,4))
        old=evaluate_alignment(a,b,x,y,radius=.3,k=5)
        new=expression_scores(a,b,x,y,radius=.3,k=5,permutations=3)
        for key in ['profile_cosine','coverage_weighted_cosine','coverage']:
            self.assertAlmostEqual(old[key],new[key],places=10)
