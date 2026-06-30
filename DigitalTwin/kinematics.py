"""Differential-drive kinematics used by simulation and EKF prediction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True, slots=True)
class DifferentialDriveGeometry:
    wheel_radius_m: float = 0.033
    wheel_base_m: float = 0.175
    ticks_per_rev: int = 360

    @property
    def meters_per_tick(self) -> float:
        return 2.0 * math.pi * self.wheel_radius_m / self.ticks_per_rev

    def ticks_to_control(self, delta_left: int, delta_right: int, dt_s: float) -> tuple[float, float]:
        if dt_s <= 0:
            return 0.0, 0.0
        dl = delta_left * self.meters_per_tick
        dr = delta_right * self.meters_per_tick
        v = (dr + dl) / (2.0 * dt_s)
        omega = (dr - dl) / (self.wheel_base_m * dt_s)
        return v, omega

    def control_to_ticks(self, v_mps: float, omega_radps: float, dt_s: float) -> tuple[int, int]:
        dl = (v_mps - 0.5 * omega_radps * self.wheel_base_m) * dt_s
        dr = (v_mps + 0.5 * omega_radps * self.wheel_base_m) * dt_s
        return round(dl / self.meters_per_tick), round(dr / self.meters_per_tick)


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
