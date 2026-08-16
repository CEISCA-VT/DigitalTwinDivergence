"""Differential-drive kinematics used by simulation and EKF prediction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M = 0.192


@dataclass(frozen=True, slots=True)
class TrackedDriveCalibrationCandidate:
    """Development-only parameters awaiting an independent physical run."""

    surface: str
    distance_scale: float
    clockwise_effective_track_width_m: float
    counterclockwise_effective_track_width_m: float
    gyro_weight: float
    gyro_scale: float


UGV01_CARPET_DEVELOPMENT_CANDIDATE = TrackedDriveCalibrationCandidate(
    surface="carpet",
    distance_scale=0.975,
    clockwise_effective_track_width_m=0.20,
    counterclockwise_effective_track_width_m=0.19,
    gyro_weight=0.20,
    gyro_scale=1.0,
)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True, slots=True)
class DifferentialDriveGeometry:
    """Tracked-drive geometry using the UGV01 vendor motion-model values."""

    wheel_radius_m: float = 0.02615
    wheel_base_m: float = 0.141
    ticks_per_rev: int = 1092
    left_tick_sign: float = -1.0
    right_tick_sign: float = -1.0
    effective_track_width_m: float | None = None

    @property
    def meters_per_tick(self) -> float:
        return 2.0 * math.pi * self.wheel_radius_m / self.ticks_per_rev

    @property
    def turn_width_m(self) -> float:
        return (
            self.wheel_base_m
            if self.effective_track_width_m is None
            else self.effective_track_width_m
        )

    def ticks_to_control(self, delta_left: int, delta_right: int, dt_s: float) -> tuple[float, float]:
        if dt_s <= 0:
            return 0.0, 0.0
        dl = delta_left * self.meters_per_tick * self.left_tick_sign
        dr = delta_right * self.meters_per_tick * self.right_tick_sign
        v = (dr + dl) / (2.0 * dt_s)
        omega = (dr - dl) / (self.turn_width_m * dt_s)
        return v, omega

    def control_to_ticks(self, v_mps: float, omega_radps: float, dt_s: float) -> tuple[int, int]:
        dl = (v_mps - 0.5 * omega_radps * self.turn_width_m) * dt_s
        dr = (v_mps + 0.5 * omega_radps * self.turn_width_m) * dt_s
        return (
            round(dl / (self.meters_per_tick * self.left_tick_sign)),
            round(dr / (self.meters_per_tick * self.right_tick_sign)),
        )


def ugv01_calibrated_geometry() -> DifferentialDriveGeometry:
    """Use vendor linear scale with the AprilTag-calibrated tracked-turn width."""

    return DifferentialDriveGeometry(
        effective_track_width_m=UGV01_APRILTAG_EFFECTIVE_TRACK_WIDTH_M
    )


def integrate_unicycle(state: np.ndarray, v_mps: float, omega_radps: float, dt_s: float) -> np.ndarray:
    x, y, theta = map(float, state[:3])
    theta_mid = theta + 0.5 * omega_radps * dt_s
    x += v_mps * math.cos(theta_mid) * dt_s
    y += v_mps * math.sin(theta_mid) * dt_s
    theta = wrap_angle(theta + omega_radps * dt_s)
    return np.array([x, y, theta], dtype=float)


def trajectory_control(name: str, t_s: float, speed_mps: float) -> tuple[float, float]:
    name = name.lower()
    if name == "circle":
        return speed_mps, 0.45
    if name in {"figure8", "figure_8"}:
        return speed_mps, 0.8 * math.sin(0.45 * t_s)
    if name == "square":
        period = 8.0
        phase = t_s % period
        if phase < 5.5:
            return speed_mps, 0.0
        return 0.0, math.pi / 2.5
    raise ValueError(f"unknown trajectory {name!r}")
