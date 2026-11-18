import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from paper.workflows import standardized_rigid as rigid
from paper.barseq_registration_pilot import Rigid2D


class RigidExtensionTests(unittest.TestCase):
    def test_roster_and_rigid_readout(self):
        self.assertEqual(len(rigid.METHODS), 8)
        self.assertEqual(len(set(rigid.METHODS)), 8)
        self.assertEqual(rigid.DEFAULTS['geometry_weight'], 0.)
        t = Rigid2D('cpu')
        with torch.no_grad():
            t.angle.fill_(.5)
            t.translation.copy_(torch.tensor([.03, -.04]))
        points = np.random.default_rng(1).uniform(size=(10, 2)).astype('float32')
        mapped = rigid.map_rigid(points, t.state_dict())
        np.testing.assert_allclose(mapped, t(torch.from_numpy(points)).detach().numpy())
        np.testing.assert_allclose(np.linalg.norm(mapped[1:]-mapped[0], axis=1),
                                   np.linalg.norm(points[1:]-points[0], axis=1), atol=1e-6)

    def test_extra_scoring_uses_rigid_map_and_keeps_missing_explicit(self):
        points = np.ones((2, 2))
        cases = [dict(name='a'), dict(name='b')]
        fitted = {'a': {'Splat rigid': points}, 'b': {}}
        def score(cases, mappings, output, **kwargs):
            rows, availability = [], []
            for c in cases:
                for method in rigid.base.METHODS:
                    present = method in mappings[c['name']]
                    availability.append(dict(case=c['name'], method=method, available=present))
                    if present:
                        np.testing.assert_array_equal(mappings[c['name']][method], points)
                        rows.append(dict(case=c['name'], method=method, profile_cosine=.75))
            return pd.DataFrame(rows), pd.DataFrame(availability)
        with tempfile.TemporaryDirectory() as folder, patch.object(rigid.base, 'evaluate_cases', side_effect=score):
            scores, availability = rigid.evaluate_cases(cases, fitted, Path(folder))
        self.assertEqual(scores.method.tolist(), ['Splat rigid'])
        self.assertEqual(len(availability), 16)
        self.assertEqual(availability[availability.method == 'Splat rigid'].available.tolist(), [True, False])


if __name__ == '__main__':
    unittest.main()
