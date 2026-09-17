import numpy as np
import pandas as pd

from DigitalTwin.analysis import service_timing_budget_study as base
from DigitalTwin.analysis.timing_reviewer_checks import discrepancy_trace, kinematic_scores, trace_summary


def synthetic(n=401):
    t = np.arange(n) * .1
    return {"time_s": t, "gt_east_m": t.copy(), "gt_north_m": np.zeros(n),
            "gt_heading_rad": np.zeros(n), "estimate_east_m": t.copy(),
            "estimate_north_m": np.zeros(n), "estimate_heading_rad": np.zeros(n),
            "odo_speed_mps": np.ones(n), "imu_yaw_rate_radps": np.zeros(n)}


def test_paired_global_ideal_zero_held_positive():
    a = synthetic()
    service = next(s for s in base.SERVICES if s["service"] == "global")
    tr = discrepancy_trace(a, service, 2.0, 200)
    assert np.allclose(tr["ideal_position"], 0)
    assert np.allclose(tr["held_position"], tr["hold_position"])
    assert np.all(tr["age"] >= .2 - 1e-10)
    summary = trace_summary(tr, service)
    assert summary["increment_position_mean"] > 0
    assert summary["increment_position_negative_fraction"] == 0


def test_ideal_condition_has_zero_hold_and_increment():
    a = synthetic()
    service = base.SERVICES[0]
    tr = discrepancy_trace(a, service, 10.0, 0)
    assert np.allclose(tr["hold_position"], 0)
    assert np.allclose(tr["held_position"], tr["ideal_position"])
    assert np.allclose(tr["held_heading"], tr["ideal_heading"])


def test_comparator_ignores_physical_outcomes():
    row = {"sequence": "x", "service": "local_1s", "rate_hz": 5., "delay_ms": 50,
           "max_aoi_s": .3, "aoi_limit_s": .6, "pos_tol_m": .1, "heading_tol_deg": 2.,
           "accel_abs_p95": .01, "yaw_abs_p95": .02, "horizon_s": 1., "qualified": 1}
    trace = {"sequence": "x", "rate_hz": 5., "delay_ms": 50,
             "hold_position_p95": .02, "hold_heading_p95": .5}
    one = kinematic_scores(pd.DataFrame([row]), pd.DataFrame([trace])).score.iloc[0]
    row["qualified"] = 0
    two = kinematic_scores(pd.DataFrame([row]), pd.DataFrame([trace])).score.iloc[0]
    assert one == two
