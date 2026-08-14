"""Telemetry-driven process and measurement uncertainty estimators."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import statistics

import numpy as np


@dataclass(frozen=True, slots=True)
class AdaptiveUncertaintyPolicy:
    base_xy_sigma_mps: float = 0.010
    residual_gain: float = 0.035
    vertical_imu_gain: float = 0.006
    yaw_imu_gain: float = 0.020
    velocity_variance_gain: float = 0.030
    timing_mismatch_gain: float = 0.350
    base_heading_sigma_radps: float = 0.003
    heading_yaw_gain: float = 0.040
    heading_velocity_gain: float = 0.010
    gps_sigma_hdop_gain_m: float = 0.55
    minimum_hdop: float = 0.8
    nominal_satellites: int = 8
    satellite_penalty: float = 0.18
    nominal_packet_dt_s: float = 0.10


@dataclass(frozen=True, slots=True)
class FixedUncertaintyPolicy:
    gps_sigma_m: float = 1.75
    process_xy_sigma_mps: float = 0.05
    process_heading_sigma_radps: float = 0.01


@dataclass(frozen=True, slots=True)
class TurnSlipUncertaintyPolicy:
    """Empirical tracked-turn uncertainty from AprilTag turn-event residuals."""

    nominal_turn_fraction: float = 0.10
    robust_turn_fraction: float = 0.312
    calibration_events: int = 9
    median_absolute_angle_error_deg: float = 10.432206720810896


@dataclass(frozen=True, slots=True)
class EvidenceGatePolicy:
    soft_nis_threshold: float = 5.991464547107982
    persistent_bias_threshold: float = 90.54974331591768
    bias_memory: float = 0.90
    timing_mismatch_s: float = 0.20
    reject_stale_packets: bool = True
    reject_sequence_gaps: bool = True


DEFAULT_ADAPTIVE_POLICY = AdaptiveUncertaintyPolicy()
DEFAULT_FIXED_POLICY = FixedUncertaintyPolicy()
DEFAULT_TURN_SLIP_POLICY = TurnSlipUncertaintyPolicy()
DEFAULT_EVIDENCE_GATE_POLICY = EvidenceGatePolicy()

LEARNED_FEATURE_COLUMNS = (
    "imu_vertical_std",
    "imu_yaw_std",
    "velocity_variance",
    "packet_dt_s",
    "gps_hdop",
    "gps_satellites",
)


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

    def gps_independent_model_vector(self) -> np.ndarray:
        """Causal deployment features that exclude GPS coordinate residuals."""

        return np.array(
            [
                self.imu_vertical_std,
                self.imu_yaw_std,
                self.velocity_variance,
                self.packet_dt_s,
                self.gps_hdop,
                float(self.gps_satellites),
            ],
            dtype=float,
        )


def turn_slip_heading_sigma(
    omega_radps: float,
    dt_s: float,
    policy: TurnSlipUncertaintyPolicy = DEFAULT_TURN_SLIP_POLICY,
) -> float:
    """Return one-sigma heading uncertainty for an encoder-predicted turn."""

    turn_increment_rad = abs(float(omega_radps) * max(float(dt_s), 0.0))
    return policy.nominal_turn_fraction * turn_increment_rad


def add_turn_slip_uncertainty(
    process_covariance: np.ndarray,
    omega_radps: float,
    dt_s: float,
    policy: TurnSlipUncertaintyPolicy = DEFAULT_TURN_SLIP_POLICY,
) -> np.ndarray:
    """Add GPS-independent tracked-turn uncertainty without shifting the mean."""

    covariance = np.asarray(process_covariance, dtype=float)
    if covariance.shape != (3, 3):
        raise ValueError("process covariance must have shape (3, 3)")
    result = covariance.copy()
    sigma = turn_slip_heading_sigma(omega_radps, dt_s, policy)
    result[2, 2] += sigma * sigma
    return result


class TelemetryStatisticsWindow:
    """Rolling statistics phi_k = [r_k, sigma_IMU, sigma_v, Delta t_k]."""

    def __init__(self, maxlen: int = 25) -> None:
        self.dead_reckoning_residuals: deque[float] = deque(maxlen=maxlen)
        self.raw_dead_reckoning_residuals: deque[float] = deque(maxlen=maxlen)
        self.residual_admissions: deque[float] = deque(maxlen=maxlen)
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
        residual_admitted: bool = True,
    ) -> None:
        residual = abs(float(dead_reckoning_residual_m))
        admitted = bool(residual_admitted)
        self.raw_dead_reckoning_residuals.append(residual)
        self.residual_admissions.append(float(admitted))
        self.dead_reckoning_residuals.append(residual if admitted else 0.0)
        self.accel_z_values.append(float(accel_z))
        self.gyro_z_values.append(float(gyro_z))
        self.velocity_values.append(float(velocity_mps))
        self.packet_dt_values.append(max(float(packet_dt_s), 0.0))

    @property
    def residual_gate_pass_fraction(self) -> float:
        return _mean(self.residual_admissions)

    @property
    def residual_cover_bound_m(self) -> float:
        if not self.raw_dead_reckoning_residuals:
            return 0.0
        return self.residual_gate_pass_fraction * max(
            self.raw_dead_reckoning_residuals
        )

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
    """Frozen naive-adaptive mapping with GPS-residual feedback."""

    def __init__(self, policy: AdaptiveUncertaintyPolicy = DEFAULT_ADAPTIVE_POLICY) -> None:
        self.policy = policy

    def gps_independent_process_sigmas(
        self,
        features: UncertaintyFeatures,
        dt_s: float,
    ) -> tuple[float, float]:
        policy = self.policy
        imu_term = policy.vertical_imu_gain * features.imu_vertical_std + policy.yaw_imu_gain * features.imu_yaw_std
        velocity_term = policy.velocity_variance_gain * features.velocity_variance
        jitter_term = policy.timing_mismatch_gain * abs(features.packet_dt_s - dt_s)
        q_xy_sigma = policy.base_xy_sigma_mps + imu_term + velocity_term + jitter_term
        q_theta_sigma = (
            policy.base_heading_sigma_radps
            + policy.heading_yaw_gain * features.imu_yaw_std
            + policy.heading_velocity_gain * features.velocity_variance
        )
        return q_xy_sigma, q_theta_sigma

    def process_covariance(self, features: UncertaintyFeatures, dt_s: float) -> np.ndarray:
        policy = self.policy
        q_xy_independent, q_theta_sigma = self.gps_independent_process_sigmas(
            features, dt_s
        )
        q_xy_sigma = (
            q_xy_independent
            + policy.residual_gain * features.dead_reckoning_residual_m
        )
        return np.diag([(q_xy_sigma * dt_s) ** 2, (q_xy_sigma * dt_s) ** 2, (q_theta_sigma * dt_s) ** 2])

    def measurement_covariance(self, features: UncertaintyFeatures) -> np.ndarray:
        policy = self.policy
        hdop = max(features.gps_hdop, policy.minimum_hdop)
        sat_penalty = 1.0 + max(0, policy.nominal_satellites - features.gps_satellites) * policy.satellite_penalty
        timing_penalty = 1.0 + max(0.0, features.packet_dt_s - policy.nominal_packet_dt_s)
        sigma = policy.gps_sigma_hdop_gain_m * hdop * sat_penalty * timing_penalty
        return np.diag([sigma * sigma, sigma * sigma])


class GPSIndependentUncertaintyEstimator(TelemetryDrivenUncertaintyEstimator):
    """Adaptive process covariance that excludes GPS residual feedback."""

    def process_covariance(self, features: UncertaintyFeatures, dt_s: float) -> np.ndarray:
        q_xy_sigma, q_theta_sigma = self.gps_independent_process_sigmas(
            features, dt_s
        )
        return np.diag([(q_xy_sigma * dt_s) ** 2, (q_xy_sigma * dt_s) ** 2, (q_theta_sigma * dt_s) ** 2])


def residual_planar_variance_delta(
    *,
    gps_independent_sigma_mps: float,
    attacked_rolling_residual_m: float,
    reference_rolling_residual_m: float,
    residual_gain: float,
    dt_s: float,
) -> float:
    """Exact planar Q difference from the residual-coupled term."""

    c_k = float(gps_independent_sigma_mps)
    attacked_residual = float(attacked_rolling_residual_m)
    reference_residual = float(reference_rolling_residual_m)
    gain = float(residual_gain)
    dt = float(dt_s)
    if min(c_k, attacked_residual, reference_residual, gain, dt) < 0.0:
        raise ValueError("variance-delta inputs must be nonnegative")
    return float(
        dt**2
        * gain
        * (attacked_residual - reference_residual)
        * (2.0 * c_k + gain * (attacked_residual + reference_residual))
    )


# Preserve the original class name for compatibility while exposing the
# preregistered variant name used in reports and experiment manifests.
NaiveAdaptiveUncertaintyEstimator = TelemetryDrivenUncertaintyEstimator


class FixedUncertaintyEstimator:
    """Fixed Q/R baseline for matched replay comparisons."""

    def __init__(
        self,
        policy: FixedUncertaintyPolicy = DEFAULT_FIXED_POLICY,
    ) -> None:
        self.policy = policy

    def process_covariance(self, features: UncertaintyFeatures, dt_s: float) -> np.ndarray:
        del features
        return np.diag(
            [
                (self.policy.process_xy_sigma_mps * dt_s) ** 2,
                (self.policy.process_xy_sigma_mps * dt_s) ** 2,
                (self.policy.process_heading_sigma_radps * dt_s) ** 2,
            ]
        )

    def measurement_covariance(self, features: UncertaintyFeatures) -> np.ndarray:
        del features
        variance = self.policy.gps_sigma_m**2
        return np.diag([variance, variance])


class LearnedUncertaintyModel:
    """GPS-independent candidate model for the frozen offline target."""

    def __init__(self) -> None:
        self.model = None
        self.model_type = "raw"
        self.target_low: np.ndarray | None = None
        self.target_high: np.ndarray | None = None
        self.target_floor = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("scikit-learn is required for RandomForest training") from exc
        self.model = RandomForestRegressor(n_estimators=80, random_state=7, min_samples_leaf=3)
        self.model.fit(X, y)

    @classmethod
    def from_estimator(cls, estimator: object) -> "LearnedUncertaintyModel":
        instance = cls()
        if isinstance(estimator, dict) and "model" in estimator:
            instance.model = estimator["model"]
            instance.model_type = str(estimator.get("model_type", "raw"))
            if estimator.get("target_low") is not None:
                instance.target_low = np.asarray(estimator["target_low"], dtype=float)
            if estimator.get("target_high") is not None:
                instance.target_high = np.asarray(estimator["target_high"], dtype=float)
            instance.target_floor = float(estimator.get("target_floor", 0.0))
        else:
            instance.model = estimator
        return instance

    def predict_q_diagonal(self, features: UncertaintyFeatures) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        prediction = np.asarray(
            self.model.predict(features.gps_independent_model_vector().reshape(1, -1))[0], dtype=float
        )
        if self.model_type == "mlp":
            prediction = np.exp(prediction)
            if self.target_low is not None and self.target_high is not None:
                prediction = np.clip(
                    prediction,
                    np.maximum(self.target_low, self.target_floor),
                    np.maximum(self.target_high, self.target_floor),
                )
        if prediction.shape[0] != 3:
            raise RuntimeError("learned uncertainty model must predict three Q diagonal values")
        return np.maximum(prediction, 0.0)
