"""Telemetry-driven process and measurement uncertainty estimators."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import statistics

import numpy as np


@dataclass(frozen=True, slots=True)
class UncertaintyFeatures:
    dead_reckoning_residual_m: float
    imu_vertical_std: float
    imu_yaw_std: float
    velocity_variance: float
    packet_dt_s: float
    gps_hdop: float
    gps_satellites: int

    def model_vector(self) -> np.ndarray:
        return np.array(
            [
                self.dead_reckoning_residual_m,
                self.imu_vertical_std,
                self.imu_yaw_std,
                self.velocity_variance,
                self.packet_dt_s,
            ],
            dtype=float,
        )


class TelemetryStatisticsWindow:
    """Rolling statistics phi_k = [r_k, sigma_IMU, sigma_v, Delta t_k]."""

    def __init__(self, maxlen: int = 25) -> None:
        self.dead_reckoning_residuals: deque[float] = deque(maxlen=maxlen)
        self.accel_z_values: deque[float] = deque(maxlen=maxlen)
        self.gyro_z_values: deque[float] = deque(maxlen=maxlen)
        self.velocity_values: deque[float] = deque(maxlen=maxlen)
        self.packet_dt_values: deque[float] = deque(maxlen=maxlen)

    def observe(
        self,
        *,
        dead_reckoning_residual_m: float,
        accel_z: float,
        gyro_z: float,
        velocity_mps: float,
        packet_dt_s: float,
    ) -> None:
        self.dead_reckoning_residuals.append(abs(dead_reckoning_residual_m))
        self.accel_z_values.append(float(accel_z))
        self.gyro_z_values.append(float(gyro_z))
        self.velocity_values.append(float(velocity_mps))
        self.packet_dt_values.append(max(float(packet_dt_s), 0.0))

    def features(self, *, gps_hdop: float, gps_satellites: int, fallback_dt_s: float) -> UncertaintyFeatures:
        return UncertaintyFeatures(
            dead_reckoning_residual_m=_mean(self.dead_reckoning_residuals),
            imu_vertical_std=_pstdev(self.accel_z_values),
            imu_yaw_std=_pstdev(self.gyro_z_values),
            velocity_variance=_pvariance(self.velocity_values),
            packet_dt_s=_mean(self.packet_dt_values) or fallback_dt_s,
            gps_hdop=gps_hdop,
            gps_satellites=gps_satellites,
        )


def _mean(values: deque[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _pstdev(values: deque[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _pvariance(values: deque[float]) -> float:
    return float(statistics.pvariance(values)) if len(values) > 1 else 0.0


class TelemetryDrivenUncertaintyEstimator:
    """Deterministic g(phi_k) baseline with the proposal's feature contract.

    The coefficients are deliberately simple for Week 0 synthetic testing.  Real
    benign rover data can later train a Random Forest/MLP with the same
    `model_vector()` features and replace only this mapping.
    """

    def process_covariance(self, features: UncertaintyFeatures, dt_s: float) -> np.ndarray:
        residual_term = 0.010 + 0.035 * features.dead_reckoning_residual_m
        imu_term = 0.006 * features.imu_vertical_std + 0.020 * features.imu_yaw_std
        velocity_term = 0.030 * features.velocity_variance
        jitter_term = 0.350 * abs(features.packet_dt_s - dt_s)

        q_xy_sigma = residual_term + imu_term + velocity_term + jitter_term
        q_theta_sigma = 0.003 + 0.040 * features.imu_yaw_std + 0.010 * features.velocity_variance
        return np.diag([(q_xy_sigma * dt_s) ** 2, (q_xy_sigma * dt_s) ** 2, (q_theta_sigma * dt_s) ** 2])

    def measurement_covariance(self, features: UncertaintyFeatures) -> np.ndarray:
        hdop = max(features.gps_hdop, 0.8)
        sat_penalty = 1.0 + max(0, 8 - features.gps_satellites) * 0.18
        timing_penalty = 1.0 + max(0.0, features.packet_dt_s - 0.10)
        sigma = 0.55 * hdop * sat_penalty * timing_penalty
        return np.diag([sigma * sigma, sigma * sigma])


class LearnedUncertaintyModel:
    """Random Forest wrapper for the proposal's self-calibrating g(phi_k)."""

    def __init__(self) -> None:
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("scikit-learn is required for RandomForest training") from exc
        self.model = RandomForestRegressor(n_estimators=80, random_state=7, min_samples_leaf=3)
        self.model.fit(X, y)

    def predict_q_diagonal(self, features: UncertaintyFeatures) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        prediction = np.asarray(self.model.predict(features.model_vector().reshape(1, -1))[0], dtype=float)
        if prediction.shape[0] != 3:
            raise RuntimeError("learned uncertainty model must predict three Q diagonal values")
        return np.maximum(prediction, 0.0)
