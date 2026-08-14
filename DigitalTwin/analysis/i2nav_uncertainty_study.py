"""Train and evaluate GPS-independent process uncertainty on i2Nav-Robot.

The split is chronological (60% train, 20% validation, 20% untouched test).
Ground truth creates process-error covariance labels and evaluation metrics, but
is never exposed to the model features or the EKF measurement update.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from DigitalTwin.ekf import RoverEKF
from DigitalTwin.kinematics import wrap_angle


DEFAULT_INPUT = Path("DigitalTwin/datasets/analysis/i2nav_playground00/aligned_samples.npz")
DEFAULT_OUTPUT = Path("DigitalTwin/datasets/analysis/i2nav_playground00/study")
TARGET_COLUMNS = ("q_forward_m2", "q_lateral_m2", "q_heading_rad2")
FEATURE_COLUMNS = (
    "abs_odo_forward_mps",
    "abs_odo_lateral_mps",
    "wheel_speed_std_mps",
    "steering_abs_mean_rad",
    "steering_std_rad",
    "abs_imu_yaw_rate_radps",
    "imu_yaw_rate_std_radps",
    "imu_accel_norm_std_mps2",
    "imu_accel_z_std_mps2",
    "rolling_speed_std_mps",
    "rolling_yaw_rate_std_radps",
    "rolling_accel_variation_mps2",
    "dt_s",
)
TARGET_FLOOR = np.asarray([1e-8, 1e-8, 1e-10], dtype=float)
CHI2_95_DOF1 = 3.841458820694124
CHI2_95_DOF2 = 5.991464547107979
CHI2_95_DOF3 = 7.814727903251179
EXPECTED_NIS_DOF = 2.0
EXPECTED_NEES_FULL_DOF = 3.0
EXPECTED_NEES_POSITION_DOF = 2.0
EXPECTED_NEES_HEADING_DOF = 1.0
MAX_GNSS_SIGMA_FOR_UPDATE_M = 10.0


def load_aligned(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"aligned dataset not found: {path}")
    with np.load(path) as archive:
        return {name: np.asarray(archive[name], dtype=float) for name in archive.files}


def _rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    cumulative_sq = np.concatenate([[0.0], np.cumsum(values * values)])
    index = np.arange(len(values))
    start = np.maximum(0, index + 1 - window)
    count = index + 1 - start
    total = cumulative[index + 1] - cumulative[start]
    total_sq = cumulative_sq[index + 1] - cumulative_sq[start]
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 0.0)
    return mean, np.sqrt(variance)


def build_features(data: dict[str, np.ndarray], history_steps: int) -> np.ndarray:
    _, speed_std = _rolling_mean_std(data["odo_forward_mps"], history_steps)
    _, yaw_std = _rolling_mean_std(data["imu_yaw_rate_radps"], history_steps)
    accel_mean, _ = _rolling_mean_std(data["imu_accel_norm_std_mps2"], history_steps)
    return np.column_stack(
        [
            np.abs(data["odo_forward_mps"]),
            np.abs(data["odo_lateral_mps"]),
            data["wheel_speed_std_mps"],
            data["steering_abs_mean_rad"],
            data["steering_std_rad"],
            np.abs(data["imu_yaw_rate_radps"]),
            data["imu_yaw_rate_std_radps"],
            data["imu_accel_norm_std_mps2"],
            data["imu_accel_z_std_mps2"],
            speed_std,
            yaw_std,
            accel_mean,
            data["dt_s"],
        ]
    )


def estimate_motion_calibration(
    data: dict[str, np.ndarray], train_indices: np.ndarray
) -> dict[str, float]:
    """Fit deterministic speed/yaw calibration on training ground truth only."""

    valid = train_indices[train_indices > 0]
    dt = data["dt_s"][valid]
    previous_heading = data["gt_heading_rad"][valid - 1]
    delta_east = data["gt_east_m"][valid] - data["gt_east_m"][valid - 1]
    delta_north = data["gt_north_m"][valid] - data["gt_north_m"][valid - 1]
    actual_forward = (
        delta_east * np.cos(previous_heading) + delta_north * np.sin(previous_heading)
    ) / dt
    raw_speed = data["odo_forward_mps"][valid]
    moving = np.abs(raw_speed) > 0.10
    speed_scale = float(
        np.dot(raw_speed[moving], actual_forward[moving])
        / max(np.dot(raw_speed[moving], raw_speed[moving]), 1e-12)
    )
    speed_scale = float(np.clip(speed_scale, 0.5, 1.5))

    actual_yaw_rate = np.asarray(
        [
            wrap_angle(current - previous) / delta_t
            for current, previous, delta_t in zip(
                data["gt_heading_rad"][valid], previous_heading, dt
            )
        ]
    )
    raw_yaw_rate = data["imu_yaw_rate_radps"][valid]
    nearly_straight = np.abs(actual_yaw_rate) < 0.005
    turning = np.abs(actual_yaw_rate) > 0.015
    if np.std(raw_yaw_rate[turning]) > 1e-3:
        design = np.column_stack(
            [raw_yaw_rate[turning], np.ones(np.sum(turning))]
        )
        yaw_scale, intercept = np.linalg.lstsq(
            design, actual_yaw_rate[turning], rcond=None
        )[0]
        yaw_bias = float(-intercept / yaw_scale)
    else:
        yaw_bias = (
            float(np.median(raw_yaw_rate[nearly_straight]))
            if np.sum(nearly_straight) >= 20
            else 0.0
        )
        centered_yaw = raw_yaw_rate[turning] - yaw_bias
        yaw_scale = float(
            np.dot(centered_yaw, actual_yaw_rate[turning])
            / max(np.dot(centered_yaw, centered_yaw), 1e-12)
        )
    yaw_scale = float(np.clip(yaw_scale, 0.5, 1.5))
    yaw_bias = float(np.clip(yaw_bias, -0.10, 0.10))
    return {
        "speed_scale": speed_scale,
        "yaw_scale": yaw_scale,
        "yaw_bias_radps": yaw_bias,
    }


def _calibrated_controls(
    data: dict[str, np.ndarray], calibration: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    speed = calibration["speed_scale"] * data["odo_forward_mps"]
    yaw_rate = calibration["yaw_scale"] * (
        data["imu_yaw_rate_radps"] - calibration["yaw_bias_radps"]
    )
    return speed, yaw_rate


def process_residuals(
    data: dict[str, np.ndarray], calibration: dict[str, float]
) -> np.ndarray:
    """Return one-step forward, lateral, and heading model residuals."""

    east = data["gt_east_m"]
    north = data["gt_north_m"]
    heading = data["gt_heading_rad"]
    dt = data["dt_s"]
    speed, yaw_rate = _calibrated_controls(data, calibration)
    residual = np.zeros((len(east), 3), dtype=float)
    for index in range(1, len(east)):
        theta_mid = heading[index - 1] + 0.5 * yaw_rate[index] * dt[index]
        predicted_east = east[index - 1] + speed[index] * math.cos(theta_mid) * dt[index]
        predicted_north = north[index - 1] + speed[index] * math.sin(theta_mid) * dt[index]
        error_east = east[index] - predicted_east
        error_north = north[index] - predicted_north
        cosine, sine = math.cos(theta_mid), math.sin(theta_mid)
        residual[index, 0] = cosine * error_east + sine * error_north
        residual[index, 1] = -sine * error_east + cosine * error_north
        residual[index, 2] = wrap_angle(
            heading[index] - (heading[index - 1] + yaw_rate[index] * dt[index])
        )
    return residual


def future_covariance_targets(residuals: np.ndarray, horizon_steps: int) -> np.ndarray:
    squared = residuals * residuals
    targets = np.full_like(squared, np.nan)
    for index in range(len(squared) - horizon_steps):
        targets[index] = np.mean(squared[index + 1 : index + 1 + horizon_steps], axis=0)
    return np.maximum(targets, TARGET_FLOOR)


def temporal_split_indices(
    row_count: int,
    *,
    history_steps: int,
    horizon_steps: int,
) -> dict[str, np.ndarray]:
    first = int(0.60 * row_count)
    second = int(0.80 * row_count)
    splits = {
        "train": np.arange(history_steps - 1, first - horizon_steps),
        "validation": np.arange(first + history_steps, second - horizon_steps),
        "test": np.arange(second + history_steps, row_count - horizon_steps),
    }
    if min(len(indices) for indices in splits.values()) < 100:
        raise RuntimeError("aligned sequence is too short for chronological 60/20/20 splits")
    return splits


def _make_mlp(seed: int):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    alpha=0.005,
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=seed,
                ),
            ),
        ]
    )


def _make_random_forest(seed: int):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=8,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )


def _fit_models(X: np.ndarray, y: np.ndarray, indices: np.ndarray, seed: int):
    low = np.maximum(np.quantile(y[indices], 0.005, axis=0), TARGET_FLOOR)
    high = np.maximum(np.quantile(y[indices], 0.995, axis=0), low * 1.01)
    clipped = np.clip(y[indices], low, high)
    mlp = _make_mlp(seed)
    mlp.fit(X[indices], np.log(clipped))
    forest = _make_random_forest(seed)
    forest.fit(X[indices], clipped)
    return {"mlp": mlp, "random_forest": forest}, low, high


def _predict(model: object, kind: str, X: np.ndarray, low: np.ndarray, high: np.ndarray):
    raw = model.predict(X)
    prediction = np.exp(raw) if kind == "mlp" else raw
    return np.clip(prediction, low, high)


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    from sklearn.metrics import r2_score

    absolute = np.abs(target - prediction)
    mae = np.mean(absolute, axis=0)
    return {
        "mae": mae,
        "normalized_mae": mae / np.maximum(np.mean(target, axis=0), TARGET_FLOOR),
        "r2": np.asarray(r2_score(target, prediction, multioutput="raw_values")),
    }


def _future_residual_stack(
    residuals: np.ndarray,
    sample_indices: np.ndarray,
    horizon_steps: int,
) -> np.ndarray:
    return np.vstack(
        [residuals[index + 1 : index + 1 + horizon_steps] for index in sample_indices]
    )


def _calibration_factors(
    residuals: np.ndarray,
    predictions: np.ndarray,
    indices: np.ndarray,
    horizon_steps: int,
) -> np.ndarray:
    observed = _future_residual_stack(residuals, indices, horizon_steps) ** 2
    repeated = np.repeat(predictions, horizon_steps, axis=0)
    ratios = observed / np.maximum(repeated, TARGET_FLOOR)
    factors = np.quantile(ratios, 0.95, axis=0) / CHI2_95_DOF1
    return np.clip(factors, 0.05, 100.0)


def _coverage_metrics(
    residuals: np.ndarray,
    prediction: np.ndarray,
    indices: np.ndarray,
    horizon_steps: int,
) -> dict[str, object]:
    observed = _future_residual_stack(residuals, indices, horizon_steps)
    repeated = np.repeat(prediction, horizon_steps, axis=0)
    normalized = observed * observed / np.maximum(repeated, TARGET_FLOOR)
    joint = np.sum(normalized, axis=1)
    return {
        "marginal_95_coverage": np.mean(normalized <= CHI2_95_DOF1, axis=0),
        "joint_95_coverage": float(np.mean(joint <= CHI2_95_DOF3)),
        "mean_normalized_error_squared": np.mean(normalized, axis=0),
        "mean_joint_normalized_error_squared": float(np.mean(joint)),
    }


def _body_q_to_world(q: np.ndarray, heading: float) -> np.ndarray:
    cosine, sine = math.cos(heading), math.sin(heading)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    planar = rotation @ np.diag(q[:2]) @ rotation.T
    result = np.zeros((3, 3), dtype=float)
    result[:2, :2] = planar
    result[2, 2] = q[2]
    return result


def _turn_slip_adjusted_body_q(
    q: np.ndarray,
    speed_mps: float,
    yaw_rate_radps: float,
    dt_s: float,
    gain: float,
) -> np.ndarray:
    """Inflate body-frame Q during turns using GPS-independent motion evidence."""

    adjusted = np.asarray(q, dtype=float).copy()
    if gain <= 0.0:
        return adjusted
    turn_increment = abs(float(yaw_rate_radps)) * max(float(dt_s), 0.0)
    speed = abs(float(speed_mps))
    slip_energy = turn_increment * (1.0 + min(speed, 5.0))
    multiplier = 1.0 + float(gain) * slip_energy
    adjusted[1] *= multiplier
    adjusted[2] *= multiplier
    return np.maximum(adjusted, TARGET_FLOOR)


def replay_ekf(
    data: dict[str, np.ndarray],
    q_by_row: np.ndarray,
    start: int,
    stop: int,
    motion_calibration: dict[str, float],
    *,
    q_scale: float = 1.0,
    r_scale: float = 1.0,
    p0_scale: float = 1.0,
    turn_slip_q_gain: float = 0.0,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    initial = np.array(
        [data["gt_east_m"][start], data["gt_north_m"][start], data["gt_heading_rad"][start]]
    )
    ekf = RoverEKF(
        initial_state=initial,
        initial_covariance=p0_scale * np.diag([0.25, 0.25, 0.03]),
    )
    states = [ekf.state.x.copy()]
    nis_values: list[float] = []
    skipped_quality = 0
    nees_values: list[float] = [0.0]
    position_nees_values: list[float] = [0.0]
    heading_nees_values: list[float] = [0.0]
    speed, yaw_rate = _calibrated_controls(data, motion_calibration)
    for index in range(start + 1, stop):
        body_q = _turn_slip_adjusted_body_q(
            q_by_row[index - 1],
            float(speed[index]),
            float(yaw_rate[index]),
            float(data["dt_s"][index]),
            turn_slip_q_gain,
        )
        Q = q_scale * _body_q_to_world(body_q, float(ekf.state.x[2]))
        ekf.predict(
            float(speed[index]),
            float(yaw_rate[index]),
            float(data["dt_s"][index]),
            Q,
        )
        if data["gps_available"][index] > 0.5:
            if (
                max(
                    float(data["gps_sigma_east_m"][index]),
                    float(data["gps_sigma_north_m"][index]),
                )
                > MAX_GNSS_SIGMA_FOR_UPDATE_M
            ):
                skipped_quality += 1
            else:
                measurement = np.array([data["gps_east_m"][index], data["gps_north_m"][index]])
                R = r_scale * np.diag(
                    [
                        max(data["gps_sigma_east_m"][index], 0.10) ** 2,
                        max(data["gps_sigma_north_m"][index], 0.10) ** 2,
                    ]
                )
                innovation, S = ekf.gps_innovation(measurement, R)
                nis_values.append(float(innovation.T @ np.linalg.solve(S, innovation)))
                ekf.update_gps(measurement, R)
        states.append(ekf.state.x.copy())
        truth = np.array(
            [data["gt_east_m"][index], data["gt_north_m"][index], data["gt_heading_rad"][index]]
        )
        error = ekf.state.x - truth
        error[2] = wrap_angle(float(error[2]))
        try:
            nees_values.append(float(error.T @ np.linalg.solve(ekf.state.P, error)))
        except np.linalg.LinAlgError:
            nees_values.append(float("nan"))
        try:
            position_nees_values.append(
                float(error[:2].T @ np.linalg.solve(ekf.state.P[:2, :2], error[:2]))
            )
        except np.linalg.LinAlgError:
            position_nees_values.append(float("nan"))
        heading_variance = max(float(ekf.state.P[2, 2]), 1e-12)
        heading_nees_values.append(float((error[2] * error[2]) / heading_variance))

    states_array = np.asarray(states)
    truth_xy = np.column_stack(
        [data["gt_east_m"][start:stop], data["gt_north_m"][start:stop]]
    )
    position_error = np.linalg.norm(states_array[:, :2] - truth_xy, axis=1)
    heading_error = np.abs(
        np.asarray(
            [
                wrap_angle(estimate - truth)
                for estimate, truth in zip(states_array[:, 2], data["gt_heading_rad"][start:stop])
            ]
        )
    )
    nees = np.asarray(nees_values)
    finite_nees = nees[np.isfinite(nees)]
    position_nees = np.asarray(position_nees_values)
    finite_position_nees = position_nees[np.isfinite(position_nees)]
    heading_nees = np.asarray(heading_nees_values)
    finite_heading_nees = heading_nees[np.isfinite(heading_nees)]
    metrics = {
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_median_m": float(np.median(position_error)),
        "position_p95_m": float(np.quantile(position_error, 0.95)),
        "heading_mae_deg": float(np.degrees(np.mean(heading_error))),
        "heading_p95_deg": float(np.degrees(np.quantile(heading_error, 0.95))),
        "nis_updates": len(nis_values),
        "gps_updates_skipped_quality": skipped_quality,
        "nis_mean": float(np.mean(nis_values)) if nis_values else float("nan"),
        "nis_95_coverage": float(np.mean(np.asarray(nis_values) <= CHI2_95_DOF2)) if nis_values else float("nan"),
        "nees_mean": float(np.mean(finite_nees)),
        "nees_95_coverage": float(np.mean(finite_nees <= CHI2_95_DOF3)),
        "position_nees_mean": float(np.mean(finite_position_nees)),
        "position_nees_95_coverage": float(
            np.mean(finite_position_nees <= CHI2_95_DOF2)
        ),
        "heading_nees_mean": float(np.mean(finite_heading_nees)),
        "heading_nees_95_coverage": float(
            np.mean(finite_heading_nees <= CHI2_95_DOF1)
        ),
    }
    return metrics, states_array, np.asarray(nis_values), nees


def replay_gps_bias_ekf(
    data: dict[str, np.ndarray],
    q_by_row: np.ndarray,
    start: int,
    stop: int,
    motion_calibration: dict[str, float],
    *,
    q_scale: float = 1.0,
    r_scale: float = 1.0,
    p0_scale: float = 1.0,
    turn_slip_q_gain: float = 0.0,
    bias_rw_sigma_mps: float = 0.001,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Replay an EKF with a slowly varying 2D GNSS bias state.

    State is [east, north, heading, gps_bias_east, gps_bias_north]. Metrics are
    computed on the physical rover state [east, north, heading].
    """

    state = np.array(
        [
            data["gt_east_m"][start],
            data["gt_north_m"][start],
            data["gt_heading_rad"][start],
            0.0,
            0.0,
        ],
        dtype=float,
    )
    covariance = p0_scale * np.diag([0.25, 0.25, 0.03, 4.0, 4.0])
    states = [state[:3].copy()]
    nis_values: list[float] = []
    skipped_quality = 0
    nees_values: list[float] = [0.0]
    position_nees_values: list[float] = [0.0]
    heading_nees_values: list[float] = [0.0]
    speed, yaw_rate = _calibrated_controls(data, motion_calibration)

    for index in range(start + 1, stop):
        dt_s = float(data["dt_s"][index])
        theta = float(state[2])
        theta_mid = theta + 0.5 * float(yaw_rate[index]) * dt_s

        F = np.eye(5)
        F[0, 2] = -float(speed[index]) * math.sin(theta_mid) * dt_s
        F[1, 2] = float(speed[index]) * math.cos(theta_mid) * dt_s

        state[0] += float(speed[index]) * math.cos(theta_mid) * dt_s
        state[1] += float(speed[index]) * math.sin(theta_mid) * dt_s
        state[2] = wrap_angle(state[2] + float(yaw_rate[index]) * dt_s)

        body_q = _turn_slip_adjusted_body_q(
            q_by_row[index - 1],
            float(speed[index]),
            float(yaw_rate[index]),
            dt_s,
            turn_slip_q_gain,
        )
        Q = np.zeros((5, 5), dtype=float)
        Q[:3, :3] = q_scale * _body_q_to_world(body_q, theta)
        Q[3, 3] = (float(bias_rw_sigma_mps) * dt_s) ** 2
        Q[4, 4] = (float(bias_rw_sigma_mps) * dt_s) ** 2
        covariance = F @ covariance @ F.T + Q
        covariance = 0.5 * (covariance + covariance.T)

        if data["gps_available"][index] > 0.5:
            if (
                max(
                    float(data["gps_sigma_east_m"][index]),
                    float(data["gps_sigma_north_m"][index]),
                )
                > MAX_GNSS_SIGMA_FOR_UPDATE_M
            ):
                skipped_quality += 1
            else:
                measurement = np.array([data["gps_east_m"][index], data["gps_north_m"][index]])
                R = r_scale * np.diag(
                    [
                        max(data["gps_sigma_east_m"][index], 0.10) ** 2,
                        max(data["gps_sigma_north_m"][index], 0.10) ** 2,
                    ]
                )
                H = np.array(
                    [[1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0]]
                )
                innovation = measurement - H @ state
                S = H @ covariance @ H.T + R
                nis_values.append(float(innovation.T @ np.linalg.solve(S, innovation)))
                K = covariance @ H.T @ np.linalg.inv(S)
                state = state + K @ innovation
                state[2] = wrap_angle(float(state[2]))
                I = np.eye(5)
                covariance = (I - K @ H) @ covariance @ (I - K @ H).T + K @ R @ K.T
                covariance = 0.5 * (covariance + covariance.T)

        states.append(state[:3].copy())
        truth = np.array(
            [data["gt_east_m"][index], data["gt_north_m"][index], data["gt_heading_rad"][index]]
        )
        error = state[:3] - truth
        error[2] = wrap_angle(float(error[2]))
        physical_covariance = covariance[:3, :3]
        try:
            nees_values.append(
                float(error.T @ np.linalg.solve(physical_covariance, error))
            )
        except np.linalg.LinAlgError:
            nees_values.append(float("nan"))
        try:
            position_nees_values.append(
                float(error[:2].T @ np.linalg.solve(physical_covariance[:2, :2], error[:2]))
            )
        except np.linalg.LinAlgError:
            position_nees_values.append(float("nan"))
        heading_variance = max(float(physical_covariance[2, 2]), 1e-12)
        heading_nees_values.append(float((error[2] * error[2]) / heading_variance))

    states_array = np.asarray(states)
    truth_xy = np.column_stack(
        [data["gt_east_m"][start:stop], data["gt_north_m"][start:stop]]
    )
    position_error = np.linalg.norm(states_array[:, :2] - truth_xy, axis=1)
    heading_error = np.abs(
        np.asarray(
            [
                wrap_angle(estimate - truth)
                for estimate, truth in zip(states_array[:, 2], data["gt_heading_rad"][start:stop])
            ]
        )
    )
    nees = np.asarray(nees_values)
    finite_nees = nees[np.isfinite(nees)]
    position_nees = np.asarray(position_nees_values)
    finite_position_nees = position_nees[np.isfinite(position_nees)]
    heading_nees = np.asarray(heading_nees_values)
    finite_heading_nees = heading_nees[np.isfinite(heading_nees)]
    metrics = {
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_median_m": float(np.median(position_error)),
        "position_p95_m": float(np.quantile(position_error, 0.95)),
        "heading_mae_deg": float(np.degrees(np.mean(heading_error))),
        "heading_p95_deg": float(np.degrees(np.quantile(heading_error, 0.95))),
        "nis_updates": len(nis_values),
        "gps_updates_skipped_quality": skipped_quality,
        "nis_mean": float(np.mean(nis_values)) if nis_values else float("nan"),
        "nis_95_coverage": float(np.mean(np.asarray(nis_values) <= CHI2_95_DOF2)) if nis_values else float("nan"),
        "nees_mean": float(np.mean(finite_nees)),
        "nees_95_coverage": float(np.mean(finite_nees <= CHI2_95_DOF3)),
        "position_nees_mean": float(np.mean(finite_position_nees)),
        "position_nees_95_coverage": float(
            np.mean(finite_position_nees <= CHI2_95_DOF2)
        ),
        "heading_nees_mean": float(np.mean(finite_heading_nees)),
        "heading_nees_95_coverage": float(
            np.mean(finite_heading_nees <= CHI2_95_DOF1)
        ),
    }
    return metrics, states_array, np.asarray(nis_values), nees


def _consistency_objective(
    metrics: dict[str, float],
    reference_rmse_m: float,
) -> float:
    """Score covariance honesty while discouraging accuracy degradation."""
    nis_mean = max(float(metrics["nis_mean"]), 1e-9)
    nees_mean = max(float(metrics["nees_mean"]), 1e-9)
    position_nees_mean = max(float(metrics.get("position_nees_mean", nees_mean)), 1e-9)
    heading_nees_mean = max(float(metrics.get("heading_nees_mean", nees_mean)), 1e-9)
    mean_error = (
        abs(math.log(nis_mean / EXPECTED_NIS_DOF))
        + abs(math.log(nees_mean / EXPECTED_NEES_FULL_DOF))
        + 0.5 * abs(math.log(position_nees_mean / EXPECTED_NEES_POSITION_DOF))
        + 0.5 * abs(math.log(heading_nees_mean / EXPECTED_NEES_HEADING_DOF))
    )
    coverage_error = (
        abs(float(metrics["nis_95_coverage"]) - 0.95)
        + 2.0 * abs(float(metrics["nees_95_coverage"]) - 0.95)
        + abs(float(metrics.get("position_nees_95_coverage", 0.95)) - 0.95)
        + abs(float(metrics.get("heading_nees_95_coverage", 0.95)) - 0.95)
    )
    relative_rmse = float(metrics["position_rmse_m"]) / max(reference_rmse_m, 1e-9)
    accuracy_penalty = 0.25 * max(relative_rmse - 1.0, 0.0)
    if relative_rmse > 1.10:
        accuracy_penalty += 10.0 * (relative_rmse - 1.10)
    return mean_error + 3.0 * coverage_error + accuracy_penalty


def calibrate_ekf_covariance_scales(
    data: dict[str, np.ndarray],
    q_by_row: np.ndarray,
    start: int,
    stop: int,
    motion_calibration: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float]]]:
    """Select global Q/R/P0 scales using validation data only."""
    reference, _, _, _ = replay_ekf(
        data, q_by_row, start, stop, motion_calibration
    )
    candidates: list[dict[str, float]] = []
    for q_scale in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        for r_scale in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            for p0_scale in (0.3, 1.0, 3.0, 10.0):
                for turn_slip_q_gain in (0.0, 0.5, 1.0, 2.0, 4.0):
                    metrics, _, _, _ = replay_ekf(
                        data,
                        q_by_row,
                        start,
                        stop,
                        motion_calibration,
                        q_scale=q_scale,
                        r_scale=r_scale,
                        p0_scale=p0_scale,
                        turn_slip_q_gain=turn_slip_q_gain,
                    )
                    score = _consistency_objective(
                        metrics, float(reference["position_rmse_m"])
                    )
                    candidates.append(
                        {
                            "q_scale": q_scale,
                            "r_scale": r_scale,
                            "p0_scale": p0_scale,
                            "turn_slip_q_gain": turn_slip_q_gain,
                            "objective": score,
                            **metrics,
                        }
                    )
    rmse_limit = 1.10 * float(reference["position_rmse_m"])
    eligible = [
        row for row in candidates if float(row["position_rmse_m"]) <= rmse_limit
    ]
    best = min(eligible, key=lambda row: row["objective"])
    scales = {
        name: float(best[name])
        for name in ("q_scale", "r_scale", "p0_scale", "turn_slip_q_gain")
    }
    metrics = {
        name: float(value)
        for name, value in best.items()
        if name not in {"q_scale", "r_scale", "p0_scale", "turn_slip_q_gain", "objective"}
    }
    metrics["calibration_objective"] = float(best["objective"])
    return scales, metrics, candidates


def calibrate_gps_bias_ekf_scales(
    data: dict[str, np.ndarray],
    q_by_row: np.ndarray,
    start: int,
    stop: int,
    motion_calibration: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float]]]:
    """Select GPS-bias EKF covariance scales using validation data only."""

    reference, _, _, _ = replay_gps_bias_ekf(
        data, q_by_row, start, stop, motion_calibration
    )
    candidates: list[dict[str, float]] = []
    for q_scale in (1.0, 10.0, 100.0, 300.0, 1000.0):
        for r_scale in (0.1, 0.3, 1.0, 3.0):
            for p0_scale in (0.3, 1.0, 3.0):
                for turn_slip_q_gain in (0.0, 1.0, 4.0):
                    for bias_rw_sigma_mps in (0.0001, 0.001, 0.01, 0.05):
                        metrics, _, _, _ = replay_gps_bias_ekf(
                            data,
                            q_by_row,
                            start,
                            stop,
                            motion_calibration,
                            q_scale=q_scale,
                            r_scale=r_scale,
                            p0_scale=p0_scale,
                            turn_slip_q_gain=turn_slip_q_gain,
                            bias_rw_sigma_mps=bias_rw_sigma_mps,
                        )
                        score = _consistency_objective(
                            metrics, float(reference["position_rmse_m"])
                        )
                        candidates.append(
                            {
                                "q_scale": q_scale,
                                "r_scale": r_scale,
                                "p0_scale": p0_scale,
                                "turn_slip_q_gain": turn_slip_q_gain,
                                "bias_rw_sigma_mps": bias_rw_sigma_mps,
                                "objective": score,
                                **metrics,
                            }
                        )
    rmse_limit = 1.10 * float(reference["position_rmse_m"])
    eligible = [
        row for row in candidates if float(row["position_rmse_m"]) <= rmse_limit
    ]
    best = min(eligible, key=lambda row: row["objective"])
    scales = {
        name: float(best[name])
        for name in (
            "q_scale",
            "r_scale",
            "p0_scale",
            "turn_slip_q_gain",
            "bias_rw_sigma_mps",
        )
    }
    metrics = {
        name: float(value)
        for name, value in best.items()
        if name
        not in {
            "q_scale",
            "r_scale",
            "p0_scale",
            "turn_slip_q_gain",
            "bias_rw_sigma_mps",
            "objective",
        }
    }
    metrics["calibration_objective"] = float(best["objective"])
    return scales, metrics, candidates


def _gps_metrics(data: dict[str, np.ndarray], start: int, stop: int) -> dict[str, float]:
    indices = np.flatnonzero(data["gps_available"][start:stop] > 0.5) + start
    errors = np.hypot(
        data["gps_east_m"][indices] - data["gt_east_m"][indices],
        data["gps_north_m"][indices] - data["gt_north_m"][indices],
    )
    return {
        "updates": len(indices),
        "position_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "position_median_m": float(np.median(errors)),
        "position_p95_m": float(np.quantile(errors, 0.95)),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _consistency_line(metrics: dict[str, float]) -> str:
    return (
        f"NIS mean {metrics['nis_mean']:.3f}, full NEES mean {metrics['nees_mean']:.3f}, "
        f"position NEES mean {metrics['position_nees_mean']:.3f}, "
        f"heading NEES mean {metrics['heading_nees_mean']:.3f}"
    )


def _plot_results(
    output_dir: Path,
    data: dict[str, np.ndarray],
    start: int,
    stop: int,
    fixed_states: np.ndarray,
    mlp_states: np.ndarray,
    model_metrics: dict[str, dict[str, np.ndarray]],
    baseline_mae: np.ndarray,
    target: np.ndarray,
    mlp_q: np.ndarray,
    test_indices: np.ndarray,
    consistency: dict[str, dict[str, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.plot(data["gt_east_m"][start:stop], data["gt_north_m"][start:stop], label="Ground truth", linewidth=2.0)
    gps = np.flatnonzero(data["gps_available"][start:stop] > 0.5) + start
    ax.scatter(data["gps_east_m"][gps], data["gps_north_m"][gps], s=8, alpha=0.45, label="F9P GNSS")
    ax.plot(
        fixed_states[:, 0],
        fixed_states[:, 1],
        label="Fixed-Q EKF (validation-calibrated)",
        linewidth=1.2,
    )
    ax.plot(
        mlp_states[:, 0],
        mlp_states[:, 1],
        label="MLP-Q EKF (validation-calibrated)",
        linewidth=1.2,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Untouched chronological test trajectory")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_comparison.png", dpi=180)
    plt.close(fig)

    labels = ["Forward", "Lateral", "Heading"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x - 0.25, np.ones(3), width=0.25, label="Median baseline")
    ax.bar(x, model_metrics["random_forest"]["mae"] / baseline_mae, width=0.25, label="Random forest")
    ax.bar(x + 0.25, model_metrics["mlp"]["mae"] / baseline_mae, width=0.25, label="MLP")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test MAE / median-baseline MAE")
    ax.set_title("Held-out covariance prediction (lower is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "covariance_model_comparison.png", dpi=180)
    plt.close(fig)

    view = test_indices[: min(len(test_indices), 1200)]
    elapsed = data["elapsed_s"][view] - data["elapsed_s"][view[0]]
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.0), sharex=True)
    for column, axis, label in zip(range(3), axes, labels):
        axis.plot(elapsed, target[view, column], label="Ground-truth target", alpha=0.75)
        axis.plot(elapsed, mlp_q[: len(view), column], label="MLP prediction", alpha=0.85)
        axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("Test time (s)")
    fig.suptitle("Ground-truth process variance and causal MLP estimate")
    fig.tight_layout()
    fig.savefig(output_dir / "q_prediction_trace.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    names = list(consistency)
    nis = [consistency[name]["nis_95_coverage"] * 100.0 for name in names]
    nees = [consistency[name]["nees_95_coverage"] * 100.0 for name in names]
    x = np.arange(len(names))
    ax.bar(x - 0.18, nis, width=0.36, label="NIS coverage")
    ax.bar(x + 0.18, nees, width=0.36, label="NEES coverage")
    ax.axhline(95.0, color="black", linestyle="--", linewidth=1.0, label="Ideal 95%")
    ax.set_xticks(x, [name.replace("_", " ").title() for name in names])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Held-out EKF consistency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ekf_consistency.png", dpi=180)
    plt.close(fig)


def run_study(input_path: Path, output_dir: Path, *, seed: int = 7) -> dict[str, object]:
    data = load_aligned(input_path)
    rate_hz = 1.0 / float(np.median(data["dt_s"]))
    history_steps = max(5, int(round(rate_hz)))
    horizon_steps = max(5, int(round(rate_hz)))
    X = build_features(data, history_steps)
    splits = temporal_split_indices(
        len(data["time_s"]), history_steps=history_steps, horizon_steps=horizon_steps
    )
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    motion_calibration = estimate_motion_calibration(data, train)
    residuals = process_residuals(data, motion_calibration)
    target = future_covariance_targets(residuals, horizon_steps)
    models, low, high = _fit_models(X, target, train, seed)

    train_median = np.median(target[train], axis=0)
    baseline_validation = np.repeat(train_median[None, :], len(validation), axis=0)
    baseline_test = np.repeat(train_median[None, :], len(test), axis=0)
    baseline_metrics = _regression_metrics(target[test], baseline_test)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    model_metrics: dict[str, dict[str, np.ndarray]] = {}
    calibration: dict[str, dict[str, object]] = {}
    for kind, model in models.items():
        validation_prediction = _predict(model, kind, X[validation], low, high)
        test_prediction = _predict(model, kind, X[test], low, high)
        factor = _calibration_factors(
            residuals, validation_prediction, validation, horizon_steps
        )
        calibrated_test = test_prediction * factor
        predictions[kind] = {
            "validation": validation_prediction,
            "test": test_prediction,
            "calibrated_test": calibrated_test,
            "factor": factor,
        }
        model_metrics[kind] = _regression_metrics(target[test], test_prediction)
        calibration[kind] = _coverage_metrics(
            residuals, calibrated_test, test, horizon_steps
        )

    start, stop = int(test[0]), int(test[-1] + 1)
    fixed_q = np.repeat(np.mean(target[train], axis=0)[None, :], len(data["time_s"]), axis=0)
    mlp_all = _predict(models["mlp"], "mlp", X, low, high) * predictions["mlp"]["factor"]
    validation_start, validation_stop = int(validation[0]), int(validation[-1] + 1)
    fixed_scales, fixed_validation, fixed_candidates = calibrate_ekf_covariance_scales(
        data, fixed_q, validation_start, validation_stop, motion_calibration
    )
    mlp_scales, mlp_validation, mlp_candidates = calibrate_ekf_covariance_scales(
        data, mlp_all, validation_start, validation_stop, motion_calibration
    )
    bias_mlp_scales, bias_mlp_validation, bias_mlp_candidates = (
        calibrate_gps_bias_ekf_scales(
            data, mlp_all, validation_start, validation_stop, motion_calibration
        )
    )
    fixed_ekf, fixed_states, _, _ = replay_ekf(
        data, fixed_q, start, stop, motion_calibration
    )
    mlp_ekf, mlp_states, _, _ = replay_ekf(
        data, mlp_all, start, stop, motion_calibration
    )
    fixed_calibrated_ekf, fixed_calibrated_states, _, _ = replay_ekf(
        data, fixed_q, start, stop, motion_calibration, **fixed_scales
    )
    mlp_calibrated_ekf, mlp_calibrated_states, _, _ = replay_ekf(
        data, mlp_all, start, stop, motion_calibration, **mlp_scales
    )
    bias_mlp_ekf, bias_mlp_states, _, _ = replay_gps_bias_ekf(
        data, mlp_all, start, stop, motion_calibration
    )
    bias_mlp_calibrated_ekf, bias_mlp_calibrated_states, _, _ = replay_gps_bias_ekf(
        data, mlp_all, start, stop, motion_calibration, **bias_mlp_scales
    )
    gps = _gps_metrics(data, start, stop)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "i2nav_uncertainty_model.pkl").open("wb") as file:
        pickle.dump(
            {
                "schema": "i2nav_gps_independent_uncertainty_v1",
                "model_type": "mlp",
                "model": models["mlp"],
                "feature_columns": FEATURE_COLUMNS,
                "target_columns": TARGET_COLUMNS,
                "target_low": low,
                "target_high": high,
                "calibration_factor": predictions["mlp"]["factor"],
                "history_steps": history_steps,
                "horizon_steps": horizon_steps,
                "motion_calibration": motion_calibration,
                "ekf_covariance_scales": mlp_scales,
                "gps_bias_ekf_covariance_scales": bias_mlp_scales,
            },
            file,
        )

    model_rows: list[dict[str, object]] = []
    for index, target_name in enumerate(TARGET_COLUMNS):
        for kind, values in [("median_baseline", baseline_metrics), *model_metrics.items()]:
            model_rows.append(
                {
                    "model": kind,
                    "target": target_name,
                    "test_mae": float(values["mae"][index]),
                    "test_normalized_mae": float(values["normalized_mae"][index]),
                    "test_r2": float(values["r2"][index]),
                    "improvement_over_median_fraction": (
                        0.0
                        if kind == "median_baseline"
                        else float(1.0 - values["mae"][index] / baseline_metrics["mae"][index])
                    ),
                }
            )
    _write_csv(output_dir / "covariance_model_metrics.csv", model_rows)
    ekf_rows = [
        {"method": "f9p_gnss", **gps},
        {"method": "fixed_q_ekf", **fixed_ekf},
        {"method": "mlp_q_ekf", **mlp_ekf},
        {"method": "fixed_q_ekf_calibrated", **fixed_calibrated_ekf},
        {"method": "mlp_q_ekf_calibrated", **mlp_calibrated_ekf},
        {"method": "gps_bias_mlp_q_ekf", **bias_mlp_ekf},
        {"method": "gps_bias_mlp_q_ekf_calibrated", **bias_mlp_calibrated_ekf},
    ]
    all_ekf_keys = sorted(set().union(*(row.keys() for row in ekf_rows)))
    with (output_dir / "ekf_test_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_ekf_keys)
        writer.writeheader()
        writer.writerows(ekf_rows)

    prediction_rows = []
    for local, index in enumerate(test):
        prediction_rows.append(
            {
                "time_s": float(data["time_s"][index]),
                **{f"target_{name}": float(target[index, column]) for column, name in enumerate(TARGET_COLUMNS)},
                **{f"mlp_{name}": float(predictions["mlp"]["test"][local, column]) for column, name in enumerate(TARGET_COLUMNS)},
                **{f"rf_{name}": float(predictions["random_forest"]["test"][local, column]) for column, name in enumerate(TARGET_COLUMNS)},
            }
        )
    _write_csv(output_dir / "test_covariance_predictions.csv", prediction_rows)

    calibration_rows = []
    for method, rows in (("fixed_q", fixed_candidates), ("mlp_q", mlp_candidates)):
        calibration_rows.extend({"method": method, **row} for row in rows)
    calibration_rows.extend(
        {"method": "gps_bias_mlp_q", **row} for row in bias_mlp_candidates
    )
    _write_csv(output_dir / "ekf_consistency_calibration.csv", calibration_rows)

    consistency = {
        "fixed_q_raw": fixed_ekf,
        "fixed_q_cal": fixed_calibrated_ekf,
        "mlp_q_raw": mlp_ekf,
        "mlp_q_cal": mlp_calibrated_ekf,
        "gps_bias_mlp_q_cal": bias_mlp_calibrated_ekf,
    }
    _plot_results(
        output_dir,
        data,
        start,
        stop,
        fixed_calibrated_states,
        mlp_calibrated_states,
        model_metrics,
        baseline_metrics["mae"],
        target,
        predictions["mlp"]["test"],
        test,
        consistency,
    )

    mlp_improvement = 1.0 - model_metrics["mlp"]["mae"] / baseline_metrics["mae"]
    accepted = bool(np.all(mlp_improvement > 0.0))
    result: dict[str, object] = {
        "schema": "i2nav_uncertainty_study_v1",
        "input": str(input_path),
        "rows": len(data["time_s"]),
        "duration_s": float(data["elapsed_s"][-1]),
        "rate_hz": rate_hz,
        "split": {name: len(indices) for name, indices in splits.items()},
        "split_policy": "chronological 60/20/20 with history/horizon purging",
        "ground_truth_feature_leakage": False,
        "gps_coordinate_feature_leakage": False,
        "motion_calibration_train_only": motion_calibration,
        "mlp_accepted_against_median_all_targets": accepted,
        "mlp_improvement_over_median": dict(zip(TARGET_COLUMNS, map(float, mlp_improvement))),
        "mlp_calibration_factor": dict(zip(TARGET_COLUMNS, map(float, predictions["mlp"]["factor"]))),
        "mlp_process_coverage": {
            "marginal_95": dict(zip(TARGET_COLUMNS, map(float, calibration["mlp"]["marginal_95_coverage"]))),
            "joint_95": float(calibration["mlp"]["joint_95_coverage"]),
        },
        "f9p_gnss_test": gps,
        "fixed_q_ekf_test": fixed_ekf,
        "mlp_q_ekf_test": mlp_ekf,
        "gps_bias_mlp_q_ekf_test": bias_mlp_ekf,
        "ekf_consistency_calibration": {
            "selection_split": "validation only",
            "objective": "NIS, full NEES, position NEES, and heading NEES mean/coverage with a 10% RMSE guard",
            "expected_means": {
                "nis": EXPECTED_NIS_DOF,
                "full_nees": EXPECTED_NEES_FULL_DOF,
                "position_nees": EXPECTED_NEES_POSITION_DOF,
                "heading_nees": EXPECTED_NEES_HEADING_DOF,
            },
            "turn_slip_q_gain": "selected on validation only from GPS-independent speed/yaw-rate evidence",
            "fixed_q": {
                "scales": fixed_scales,
                "validation_metrics": fixed_validation,
            },
            "mlp_q": {
                "scales": mlp_scales,
                "validation_metrics": mlp_validation,
            },
            "gps_bias_mlp_q": {
                "scales": bias_mlp_scales,
                "validation_metrics": bias_mlp_validation,
            },
        },
        "fixed_q_ekf_calibrated_test": fixed_calibrated_ekf,
        "mlp_q_ekf_calibrated_test": mlp_calibrated_ekf,
        "gps_bias_mlp_q_ekf_calibrated_test": bias_mlp_calibrated_ekf,
        "limitations": [
            "One public sequence is a preliminary external validation, not final UGV01 evidence.",
            "The MLP is trained on i2Nav vehicle dynamics and cannot be deployed on UGV01 without transfer validation.",
            "Ground-truth initialization is used to isolate propagation and covariance behavior.",
            "The GPS-bias EKF is an analysis baseline for public data; deployment needs independent validation of bias observability.",
        ],
    }
    (output_dir / "study_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    report = [
        "# i2Nav-Robot GPS-Independent Uncertainty Study",
        "",
        "## Design",
        "",
        f"- Aligned duration: **{result['duration_s'] / 60.0:.1f} min** at **{rate_hz:.1f} Hz**.",
        "- Split: chronological 60% training, 20% validation/calibration, and 20% untouched testing.",
        "- Model inputs use odometry and IMU behavior only. GNSS coordinates and ground truth are excluded.",
        "- Ground truth supplies process-covariance targets, ATE/heading error, and NEES evaluation.",
        f"- Training-only motion calibration: speed scale **{motion_calibration['speed_scale']:.4f}**, yaw scale **{motion_calibration['yaw_scale']:.4f}**, gyro bias **{motion_calibration['yaw_bias_radps']:.5f} rad/s**.",
        "- EKF consistency is decomposed into NIS, full-state NEES, position-only NEES, and heading-only NEES.",
        "- Turn/slip process-noise inflation is selected on validation only from speed and yaw-rate evidence; GNSS residuals are not used to choose it.",
        "",
        "## Held-Out Results",
        "",
        f"- MLP beats the temporal training-median baseline for all three covariance targets: **{accepted}**.",
        f"- Raw F9P test RMSE: **{gps['position_rmse_m']:.3f} m**.",
        f"- Fixed-Q EKF test RMSE: **{fixed_ekf['position_rmse_m']:.3f} m**; NIS/NEES 95% coverage: **{100*fixed_ekf['nis_95_coverage']:.1f}% / {100*fixed_ekf['nees_95_coverage']:.1f}%**.",
        f"- MLP-Q EKF test RMSE: **{mlp_ekf['position_rmse_m']:.3f} m**; NIS/NEES 95% coverage: **{100*mlp_ekf['nis_95_coverage']:.1f}% / {100*mlp_ekf['nees_95_coverage']:.1f}%**.",
        f"- Validation-calibrated fixed-Q EKF: **{fixed_calibrated_ekf['position_rmse_m']:.3f} m RMSE**; NIS/NEES coverage **{100*fixed_calibrated_ekf['nis_95_coverage']:.1f}% / {100*fixed_calibrated_ekf['nees_95_coverage']:.1f}%**.",
        f"- Validation-calibrated MLP-Q EKF: **{mlp_calibrated_ekf['position_rmse_m']:.3f} m RMSE**; NIS/NEES coverage **{100*mlp_calibrated_ekf['nis_95_coverage']:.1f}% / {100*mlp_calibrated_ekf['nees_95_coverage']:.1f}%**.",
        f"- Validation-calibrated GPS-bias MLP-Q EKF: **{bias_mlp_calibrated_ekf['position_rmse_m']:.3f} m RMSE**; NIS/NEES coverage **{100*bias_mlp_calibrated_ekf['nis_95_coverage']:.1f}% / {100*bias_mlp_calibrated_ekf['nees_95_coverage']:.1f}%**.",
        f"- MLP-Q calibrated heading MAE: **{mlp_calibrated_ekf['heading_mae_deg']:.2f} deg**.",
        f"- MLP-Q calibrated consistency detail: {_consistency_line(mlp_calibrated_ekf)}.",
        f"- GPS-bias MLP-Q calibrated consistency detail: {_consistency_line(bias_mlp_calibrated_ekf)}.",
        f"- Selected MLP-Q covariance scales from validation only: **Q x {mlp_scales['q_scale']:.3g}**, **R x {mlp_scales['r_scale']:.3g}**, **P0 x {mlp_scales['p0_scale']:.3g}**, **turn/slip gain {mlp_scales['turn_slip_q_gain']:.3g}**.",
        f"- Selected GPS-bias MLP-Q scales from validation only: **Q x {bias_mlp_scales['q_scale']:.3g}**, **R x {bias_mlp_scales['r_scale']:.3g}**, **P0 x {bias_mlp_scales['p0_scale']:.3g}**, **turn/slip gain {bias_mlp_scales['turn_slip_q_gain']:.3g}**, **bias random-walk sigma {bias_mlp_scales['bias_rw_sigma_mps']:.3g} m/s**.",
        "",
        "## Interpretation Rule",
        "",
        "A useful learned covariance model should beat the median predictor on the untouched test, improve EKF consistency toward 95%, and avoid degrading position error. Passing only one of these checks is not enough for acceptance.",
        "",
        "## Scope",
        "",
        "These results are public-dataset pre-validation. They show whether the method is technically plausible under precise independent ground truth; they do not replace UGV01 plus AprilTag validation.",
    ]
    (output_dir / "study_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run_study(args.input, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
