import numpy as np

from DigitalTwin.alarm import (
    AlarmConfig,
    PersistentAlarm,
    motion_start_index,
    operational_run_statistic,
    robust_initial_state,
)


def test_persistent_alarm_requires_three_of_five_exceedances():
    alarm = PersistentAlarm(5.0, AlarmConfig(window_size=5, required_exceedances=3))
    outputs = [alarm.observe(score, enabled=True) for score in [6.0, 1.0, 7.0, 2.0, 8.0]]
    assert outputs == [False, False, False, False, True]


def test_disabled_alarm_clears_prior_evidence():
    alarm = PersistentAlarm(5.0)
    alarm.observe(8.0, enabled=True)
    alarm.observe(8.0, enabled=True)
    assert alarm.observe(0.0, enabled=False) is False
    assert alarm.observe(8.0, enabled=True) is False


def test_motion_start_requires_sustained_encoder_motion():
    controls = np.array([[0.0, 0.0], [0.03, 0.0], [0.0, 0.0], [0.04, 0.0], [0.05, 0.0]])
    assert motion_start_index(controls) == 3


def test_robust_initial_state_uses_recent_median_and_variance_floor():
    points = np.array([[100.0, 100.0], [1.0, 2.0], [1.1, 2.1], [0.9, 1.9], [1.0, 2.0]])
    state, covariance = robust_initial_state(points, 4, AlarmConfig(initialization_gps_samples=4))
    assert np.allclose(state[:2], [1.0, 2.0])
    assert covariance[0, 0] >= 0.25
    assert covariance[1, 1] >= 0.25


def test_operational_statistic_is_third_largest_in_five_sample_window():
    scores = np.array([1.0, 9.0, 2.0, 8.0, 7.0])
    enabled = np.ones(5, dtype=bool)
    assert operational_run_statistic(scores, enabled) == 7.0
