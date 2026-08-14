import numpy as np

from DigitalTwin.analysis.i2nav_uncertainty_study import (
    _body_q_to_world,
    _consistency_objective,
    _turn_slip_adjusted_body_q,
    estimate_motion_calibration,
    process_residuals,
    temporal_split_indices,
)
from DigitalTwin.kinematics import wrap_angle


def _synthetic_motion(rows: int = 800) -> dict[str, np.ndarray]:
    dt = np.full(rows, 0.1)
    true_speed = 1.2
    true_yaw_rate = 0.10 * np.sin(np.arange(rows) * 0.025)
    speed_scale = 0.96
    yaw_scale = 1.04
    yaw_bias = -0.012
    raw_speed = np.full(rows, true_speed / speed_scale)
    raw_yaw = true_yaw_rate / yaw_scale + yaw_bias
    east = np.zeros(rows)
    north = np.zeros(rows)
    heading = np.zeros(rows)
    for index in range(1, rows):
        midpoint = heading[index - 1] + 0.5 * true_yaw_rate[index] * dt[index]
        east[index] = east[index - 1] + true_speed * np.cos(midpoint) * dt[index]
        north[index] = north[index - 1] + true_speed * np.sin(midpoint) * dt[index]
        heading[index] = wrap_angle(
            heading[index - 1] + true_yaw_rate[index] * dt[index]
        )
    return {
        "dt_s": dt,
        "odo_forward_mps": raw_speed,
        "imu_yaw_rate_radps": raw_yaw,
        "gt_east_m": east,
        "gt_north_m": north,
        "gt_heading_rad": heading,
    }


def test_train_only_motion_calibration_removes_deterministic_error():
    data = _synthetic_motion()
    train = np.arange(10, 500)
    calibration = estimate_motion_calibration(data, train)
    residual = process_residuals(data, calibration)

    assert np.isclose(calibration["speed_scale"], 0.96, atol=0.01)
    assert np.isclose(calibration["yaw_scale"], 1.04, atol=0.01)
    assert np.isclose(calibration["yaw_bias_radps"], -0.012, atol=0.002)
    assert np.max(np.abs(residual[10:, :])) < 1e-5


def test_temporal_split_purges_boundaries():
    split = temporal_split_indices(1000, history_steps=10, horizon_steps=10)
    assert split["train"][-1] < 600
    assert split["validation"][0] >= 610
    assert split["validation"][-1] < 800
    assert split["test"][0] >= 810
    assert len(set(split["train"]) & set(split["validation"])) == 0


def test_body_covariance_rotation_stays_symmetric_positive_semidefinite():
    covariance = _body_q_to_world(np.array([0.04, 0.01, 0.002]), heading=1.2)
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= 0.0


def test_turn_slip_adjustment_only_inflates_lateral_and_heading_q():
    q = np.array([0.04, 0.01, 0.002])
    adjusted = _turn_slip_adjusted_body_q(
        q, speed_mps=1.5, yaw_rate_radps=0.4, dt_s=0.1, gain=2.0
    )

    assert adjusted[0] == q[0]
    assert adjusted[1] > q[1]
    assert adjusted[2] > q[2]


def test_consistency_objective_prefers_calibrated_coverage_and_means():
    calibrated = {
        "position_rmse_m": 1.0,
        "nis_mean": 2.0,
        "nees_mean": 3.0,
        "nis_95_coverage": 0.95,
        "nees_95_coverage": 0.95,
        "position_nees_mean": 2.0,
        "position_nees_95_coverage": 0.95,
        "heading_nees_mean": 1.0,
        "heading_nees_95_coverage": 0.95,
    }
    overconfident = {
        "position_rmse_m": 1.0,
        "nis_mean": 0.15,
        "nees_mean": 42.0,
        "nis_95_coverage": 1.0,
        "nees_95_coverage": 0.05,
        "position_nees_mean": 30.0,
        "position_nees_95_coverage": 0.10,
        "heading_nees_mean": 12.0,
        "heading_nees_95_coverage": 0.20,
    }
    assert _consistency_objective(calibrated, 1.0) < _consistency_objective(
        overconfident, 1.0
    )
