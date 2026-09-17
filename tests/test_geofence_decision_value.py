import numpy as np

from DigitalTwin.analysis.geofence_decision_value import (
    causal_stream,
    fence_membership,
    qualification,
    reference_updates,
    score,
)


def small_trace():
    t = np.arange(0, 2.1, 0.1)
    return {
        "time_s": t,
        "gt_east_m": t.copy(),
        "gt_north_m": np.zeros(len(t)),
        "estimate_east_m": t.copy(),
        "estimate_north_m": np.zeros(len(t)),
    }


def test_no_future_reference_and_repeated_cache():
    a = small_trace()
    _, arrival, stamp = causal_stream(a, "delay", 0, 0)
    update, err = reference_updates(a, arrival, stamp)
    assert np.flatnonzero(update).tolist() == [2, 12]
    assert np.isfinite(err).sum() == 2
    rule = qualification(update, err, a["time_s"], 1.0, 0.6)
    assert not rule["contract"][0]
    assert rule["error"][9] and not rule["error_fresh"][9]
    assert np.array_equal(rule["contract"], rule["error_fresh"])


def test_hidden_true_time_not_used_for_matching():
    a = small_trace()
    _, arrival, stamp = causal_stream(a, "jitter", 0, 0)
    stamp[0] = -1.0
    update, _ = reference_updates(a, arrival, stamp)
    assert not update[2]


def test_boundary_and_zero_release():
    assert fence_membership(np.array([1.0, 1.01]), np.array([0.0, 0.0]), 0, 0, 1).tolist() == [True, False]
    result = score(np.array([False, False]), np.array([True, False]),
                   np.array([True, True]), np.array([True, True]), np.array([0.1, 0.1]))
    assert np.isnan(result["released_error"])
    assert result["availability"] == 0
    assert abs(result["unnecessary_abstention"] - 0.5) < 1e-12
