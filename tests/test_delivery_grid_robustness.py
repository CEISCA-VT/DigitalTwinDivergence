"""Checks for the saved-evidence delivery-grid sensitivity analysis."""

import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.check_delivery_grid_robustness import (
    PREDICTION_KEY, build_predictions, duplicate_audit, paired_changes,
)
from scripts.compare_surface_alternatives import cutoff


def example_table():
    rows = []
    for sequence in range(10):
        for service in ("global", "local_1s", "local_5s", "local_10s"):
            for rate in (1., 2., 5., 10.):
                for delay in (0, 25, 50, 100, 200):
                    rows.append(dict(
                        outer_sequence="run00", sequence=f"run{sequence:02d}",
                        service=service, rate_hz=rate, delay_ms=delay,
                        qualified=int(rate >= 5 and service != "global"),
                        horizon_s=0 if service == "global" else 1,
                        pos_tol_m=.1, heading_tol_deg=2., aoi_limit_s=.6,
                        max_aoi_s=.2 if delay == 0 else .4,
                        speed_mean=1., speed_p95=1.5,
                        yaw_abs_mean=.01, yaw_abs_p95=.1,
                        accel_abs_p95=.2, turn_fraction=.05,
                    ))
    return pd.DataFrame(rows)


class DeliveryGridRobustnessTests(unittest.TestCase):
    def test_equal_delays_are_checked_and_measurement_changes_rejected(self):
        table = example_table()
        audit = duplicate_audit(table)
        self.assertTrue(all(row["max_absolute_difference"] == 0 for row in audit))
        table.loc[table.delay_ms.eq(100), "qualified"] = 1
        with self.assertRaises(AssertionError):
            duplicate_audit(table)

    def test_collapsing_duplicate_cells_can_change_the_selected_cutoff(self):
        full_threshold, _ = cutoff(np.array([1.] * 6 + [0.] * 7), .2)
        reduced_threshold, _ = cutoff(np.array([1.] * 3 + [0.] * 7), .2)
        self.assertTrue(np.isinf(full_threshold))
        self.assertEqual(reduced_threshold, 1.)

    def test_outer_outcomes_do_not_change_scores_or_calibration_cutoffs(self):
        table = example_table()
        table = table[table.delay_ms.isin([0, 50, 200])]
        changed = table.copy()
        outer = changed.sequence.eq("run00")
        changed.loc[outer, "qualified"] = 1 - changed.loc[outer, "qualified"]
        with threadpool_limits(limits=1):
            first = build_predictions(table).set_index(PREDICTION_KEY).sort_index()
            second = build_predictions(changed).set_index(PREDICTION_KEY).sort_index()
        np.testing.assert_array_equal(first.score, second.score)
        calibration = first.index.get_level_values("prediction_role") == "operating_point_calibration"
        np.testing.assert_array_equal(first.loc[calibration, "actual_qualified"],
                                      second.loc[calibration, "actual_qualified"])
        for _, block in first.reset_index().groupby("method"):
            scores = block.loc[block.prediction_role.eq("operating_point_calibration"), "score"].to_numpy()
            changed_block = second.reset_index()
            changed_scores = changed_block.loc[
                changed_block.method.eq(block.method.iloc[0])
                & changed_block.prediction_role.eq("operating_point_calibration"), "score"
            ].to_numpy()
            for target in (.1, .25, .5):
                self.assertEqual(cutoff(scores, target), cutoff(changed_scores, target))
        for row in first.reset_index().itertuples():
            self.assertFalse({row.outer_sequence, row.sequence} & set(row.training_sequences.split(";")))
            self.assertEqual(len(row.training_sequences.split(";")),
                             9 if row.prediction_role == "outer_test" else 8)

    def test_paired_bootstrap_preserves_zero_change_for_identical_results(self):
        rows = [dict(bank="test", method="service_empirical", target_acceptance=.25,
                     scope="pooled", heldout_sequence=f"run{i}", grid=grid,
                     accepted_count=i + 1, false_count=i % 2)
                for i in range(10) for grid in ("full", "unique_50ms")]
        result = paired_changes(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result.risk_difference_percentage_points, 0.)
        self.assertEqual(result.ci_low, 0.)
        self.assertEqual(result.ci_high, 0.)
        self.assertEqual(result.defined_bootstrap_samples, 5000)


if __name__ == "__main__":
    unittest.main()
