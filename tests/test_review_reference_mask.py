import unittest
import numpy as np
from DigitalTwin.analysis import service_timing_budget_study as base
from scripts.build_review_revision_20261109 import reference_mask_evaluation


class ReferenceMaskTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(1000)/10
        self.data = dict(time_s=t, gt_east_m=t, gt_north_m=np.zeros(1000),
                         gt_heading_rad=np.zeros(1000), estimate_east_m=t.copy(),
                         estimate_north_m=np.zeros(1000), estimate_heading_rad=np.zeros(1000))

    def test_local_gap_invalidates_both_reference_endpoints(self):
        mask = np.ones(1000, bool)
        mask[600:620] = False
        result = reference_mask_evaluation(self.data, base.SERVICES[0], mask)
        self.assertEqual(result['eligible_samples']-result['observable_samples'], 30)
        self.assertEqual(result['joint_satisfaction'], 1.)
        self.assertEqual(result['status'], 'qualified')

    def test_missing_reference_is_unknown_even_with_perfect_estimates(self):
        mask = np.ones(1000, bool)
        mask[600:700] = False
        result = reference_mask_evaluation(self.data, base.SERVICES[0], mask)
        self.assertEqual(result['status'], 'unobservable')
        self.assertEqual(result['joint_satisfaction'], 1.)

    def test_no_reference_has_undefined_satisfaction(self):
        result = reference_mask_evaluation(self.data, base.SERVICES[0], np.zeros(1000, bool))
        self.assertEqual(result['status'], 'unobservable')
        self.assertEqual(result['coverage'], 0.)
        self.assertTrue(np.isnan(result['joint_satisfaction']))


if __name__ == '__main__':
    unittest.main()
