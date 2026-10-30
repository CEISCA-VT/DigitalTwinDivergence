"""Independent enumeration checks for the offline scheduling formulation."""
import itertools
import math
import unittest

import numpy as np

from DigitalTwin.analysis import delivery_feasibility as study
from DigitalTwin.analysis import service_timing_budget_study as base


def recordings(n=450):
    t = np.arange(n) / 10
    heading = .08 * np.sin(t / 3)
    x = np.cumsum(.04 * np.cos(heading))
    y = np.cumsum(.04 * np.sin(heading))
    return [{"time_s": t, "gt_east_m": x, "gt_north_m": y,
             "gt_heading_rad": heading,
             "estimate_east_m": x + scale * .02 * np.sin(t),
             "estimate_north_m": y + scale * .01 * np.cos(t),
             "estimate_heading_rad": heading + scale * .005 * np.sin(t / 2),
             "odo_speed_mps": np.full(n, .4), "imu_yaw_rate_radps": np.gradient(heading, .1)}
            for scale in [.5, 1., 1.5]]


class DeliveryFeasibilityTests(unittest.TestCase):
    def test_existing_evaluator_inclusive_ages_and_local_frame_are_reproduced(self):
        a = recordings()[0]
        for s in base.SERVICES:
            for rate in [1, 2, 5, 10]:
                for delay in [0, 25, 50, 100, 200]:
                    idx = base.delivered_indices(a["time_s"], rate, delay / 1000)
                    actual = study.replay(a, s, idx)
                    expected = base.evaluate(a, s, rate, delay)
                    for key in actual:
                        self.assertAlmostEqual(actual[key], expected[key], places=12)

    def test_all_short_schedules_match_factorization_and_upper_bound(self):
        arrays = recordings()
        for lag in [0, 1, 3]:
            edges, counts, costs, denom = study.build_factors(arrays, lag)
            upper = {s["service"]: study.optimistic_bound(arrays, s, lag)[0]
                     for s in base.SERVICES}
            for actions in itertools.product(range(4), repeat=2):
                idx = study.schedule_indices(450, actions, lag)
                selected = [i for i, edge in enumerate(edges)
                            if edge == (0, 0, actions[0]) or edge == (1, actions[0], actions[1])]
                self.assertEqual(len(selected), 2)
                for si, m in enumerate(study.summarize(arrays, idx)):
                    self.assertAlmostEqual(m["joint_satisfaction"], counts[selected, si].sum() / denom[si])
                    self.assertLessEqual(m["joint_satisfaction"], upper[m["service"]] + 1e-12)
                available = idx >= 0
                self.assertTrue(np.all(idx[available] <= np.arange(450)[available] - lag))

    def test_optimizer_matches_exhaustive_enumeration(self):
        arrays = recordings()
        edges, counts, costs, denom = study.build_factors(arrays, 1)
        for group in [("local_1s",), ("global",), ("local_1s", "local_5s", "local_10s")]:
            feasible = []
            for actions in itertools.product(range(4), repeat=2):
                metrics = study.summarize(arrays, study.schedule_indices(450, actions, 1))
                if all(next(m for m in metrics if m["service"] == name)["qualified"] for name in group):
                    feasible.append(sum(math.ceil(length / study.STRIDES[a])
                                        for length, a in zip([100, 50], actions)))
            actions, info = study.solve_schedule(edges, counts, costs, denom, group)
            self.assertEqual(actions is not None, bool(feasible))
            if feasible:
                self.assertEqual(info["solver_status"], 0)
                self.assertEqual(info["transmissions"], min(feasible))

    def test_source_inadequacy_and_seed_averaging_are_preserved(self):
        arrays = recordings()
        for a, offset in zip(arrays, [-100., 100., 0.]):
            a["gt_east_m"][:] = 0
            a["gt_north_m"][:] = 0
            a["estimate_east_m"][:] = offset
            a["estimate_north_m"][:] = 0
            a["estimate_heading_rad"][:] = a["gt_heading_rad"]
        global_service = next(s for s in base.SERVICES if s["family"] == "global")
        bound, _ = study.optimistic_bound(arrays, global_service, 0)
        self.assertAlmostEqual(bound, 1 / 3)
        edges, counts, costs, denom = study.build_factors(arrays, 0)
        actions, info = study.solve_schedule(edges, counts, costs, denom, ("global",))
        self.assertIsNone(actions)
        self.assertEqual(info["solver_status"], 2)


if __name__ == "__main__":
    unittest.main()
