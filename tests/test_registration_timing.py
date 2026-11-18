import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from paper.workflows.registration_timing import TimingLog


class TimingTests(unittest.TestCase):
    def test_fit_time_survives_cache_reload_without_becoming_load_time(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('paper.workflows.registration_timing.synchronize'), \
                patch('paper.workflows.registration_timing.time.perf_counter', side_effect=[0,120,200,201]):
            log = TimingLog(folder)
            with log.measure('case', 'PASTE2 full pipeline', 'fresh'):
                pass
            with log.measure('case', 'PASTE2 full pipeline', 'cache_load'):
                pass
            table = log.summary([dict(name='case', section=19, shared_fraction=1)])
            row = table[table.stage=='PASTE2 full pipeline'].iloc[0]
            self.assertEqual(row.measured_fresh_minutes, 2)
            self.assertEqual(row.latest_call_seconds, 1)
            self.assertEqual(row.latest_activity, 'cache_load')
            self.assertTrue(np.isnan(table[table.stage=='Splat family'].iloc[0].measured_fresh_minutes))

    def test_failed_and_resumed_calls_are_not_fresh_fit_times(self):
        with tempfile.TemporaryDirectory() as folder, patch('paper.workflows.registration_timing.synchronize'):
            log = TimingLog(folder)
            with self.assertRaises(ValueError):
                with log.measure('case', 'PASTE2 full pipeline', 'fresh'):
                    raise ValueError('failed')
            with log.measure('case', 'MLP fields', 'cache_or_resumed'):
                pass
            table = log.summary([dict(name='case', section=19, shared_fraction=.6)])
            self.assertTrue(table.measured_fresh_minutes.isna().all())
            self.assertEqual(table[table.stage=='PASTE2 full pipeline'].iloc[0].latest_status, 'failed')


if __name__ == '__main__':
    unittest.main()
