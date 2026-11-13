import unittest

import numpy as np

from DigitalTwin.analysis import service_timing_budget_study as base
from scripts.compare_surface_alternatives import cutoff


class BufferedAvailabilityTests(unittest.TestCase):
    def test_original_timestamps_and_inclusive_availability(self):
        t = np.arange(20) / 10
        indices = base.delivered_indices(t, 10, .1)
        np.testing.assert_array_equal(indices, np.arange(-1, 19))
        valid = indices >= 0
        np.testing.assert_allclose(t[valid] - t[indices[valid]], .1)
        self.assertTrue(np.all(t[indices[valid]] + .1 <= t[valid] + 1e-12))

    def test_buffer_is_added_to_network_delay(self):
        t = np.arange(100) / 10
        indices = base.delivered_indices(t, 5, (.1 + .05))
        sources = np.arange(0, len(t), 2)
        expected = [max((k for k in sources if t[k] + .15 <= now + 1e-12), default=-1)
                    for now in t]
        np.testing.assert_array_equal(indices, expected)

    def test_buffer_changes_held_state_not_only_reported_age(self):
        t = np.arange(500) / 10
        data = dict(time_s=t, gt_east_m=t, gt_north_m=np.zeros(500),
                    gt_heading_rad=np.zeros(500), estimate_east_m=t.copy(),
                    estimate_north_m=np.zeros(500), estimate_heading_rad=np.zeros(500))
        service = dict(base.SERVICES[-1], pos_tol_m=.05)
        self.assertEqual(base.evaluate(data, service, 10, 0)["joint_satisfaction"], 1)
        buffered = base.evaluate(data, service, 10, 100)
        self.assertEqual(buffered["joint_satisfaction"], 0)
        self.assertAlmostEqual(buffered["mean_aoi_s"], .1)

    def test_tied_cutoffs_preserve_existing_rule(self):
        scores = np.r_[np.ones(3), np.zeros(7)]
        threshold, achieved = cutoff(scores, .1)
        self.assertEqual(threshold, np.inf)
        self.assertEqual(achieved, 0)
        self.assertEqual(cutoff(scores, .25), (1., .3))


if __name__ == "__main__":
    unittest.main()
