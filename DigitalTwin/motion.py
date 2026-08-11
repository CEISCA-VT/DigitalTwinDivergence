"""Encoder/IMU motion fusion and replay-only calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .kinematics import integrate_unicycle, wrap_angle


@dataclass(frozen=True, slots=True)
class MotionFusionPolicy:
    """Frozen conservative fusion policy selected on benign development runs."""

    gyro_weight: float = 0.0
    gyro_lowpass_alpha: float = 0.40
    gyro_clip_radps: float = 3.0
    stationary_speed_threshold_mps: float = 0.01
    stationary_yaw_threshold_radps: float = 0.05
    initialization_updates: int = 16


DEFAULT_MOTION_FUSION_POLICY = MotionFusionPolicy()


@dataclass(frozen=True, slots=True)
class MotionFusionResult:
    controls: np.ndarray
    encoder_controls: np.ndarray
    corrected_gyro_radps: np.ndarray
    gyro_bias_radps: float
    yaw_disagreement_radps: np.ndarray
    slip_indicator: np.ndarray


def estimate_stationary_gyro_bias(
    encoder_controls: np.ndarray,
    raw_gyro_radps: np.ndarray,
    mission_start_index: int,
    policy: MotionFusionPolicy = DEFAULT_MOTION_FUSION_POLICY,
) -> float:
    """Estimate run-specific z-gyro bias from the stationary pre-motion prefix."""

    controls = np.asarray(encoder_controls, dtype=float)
    gyro = np.asarray(raw_gyro_radps, dtype=float)
    prefix_end = max(int(mission_start_index), min(5, len(gyro)))
    indices = np.arange(len(gyro))
    stationary = (
        (indices < prefix_end)
        & (np.abs(controls[:, 0]) < policy.stationary_speed_threshold_mps)
        & (np.abs(controls[:, 1]) < policy.stationary_yaw_threshold_radps)
    )
    samples = gyro[stationary]
    if not len(samples):
        samples = gyro[:prefix_end]
    return float(np.median(samples)) if len(samples) else 0.0


def fuse_encoder_imu_motion(
    encoder_controls: np.ndarray,
    raw_gyro_radps: np.ndarray,
    mission_start_index: int,
    policy: MotionFusionPolicy = DEFAULT_MOTION_FUSION_POLICY,
) -> MotionFusionResult:
    """Fuse encoder yaw with a low-weight, bias-corrected IMU yaw rate."""

    encoder = np.asarray(encoder_controls, dtype=float)
    raw_gyro = np.asarray(raw_gyro_radps, dtype=float)
    if encoder.ndim != 2 or encoder.shape[1] != 2:
        raise ValueError("encoder controls must have shape (n, 2)")
    if len(raw_gyro) != len(encoder):
        raise ValueError("gyro and encoder controls must have equal length")

    bias = estimate_stationary_gyro_bias(
        encoder, raw_gyro, mission_start_index, policy
    )
    corrected = np.clip(
        raw_gyro - bias,
        -policy.gyro_clip_radps,
        policy.gyro_clip_radps,
    )
    filtered = np.zeros_like(corrected)
    if len(filtered):
        filtered[0] = corrected[0]
    alpha = policy.gyro_lowpass_alpha
    for index in range(1, len(filtered)):
        filtered[index] = (
            (1.0 - alpha) * filtered[index - 1] + alpha * corrected[index]
        )

    fused = encoder.copy()
    fused[:, 1] = (
        (1.0 - policy.gyro_weight) * encoder[:, 1]
        + policy.gyro_weight * filtered
    )
    disagreement = np.abs(encoder[:, 1] - filtered)
    denominator = np.abs(encoder[:, 1]) + np.abs(filtered) + 0.05
    slip = np.clip(disagreement / denominator, 0.0, 1.0)
    return MotionFusionResult(
        controls=fused,
        encoder_controls=encoder.copy(),
        corrected_gyro_radps=filtered,
        gyro_bias_radps=bias,
        yaw_disagreement_radps=disagreement,
        slip_indicator=slip,
    )


def estimate_aligned_initial_heading(
    gps_xy: np.ndarray,
    controls: np.ndarray,
    elapsed_s: np.ndarray,
    start_index: int,
    policy: MotionFusionPolicy = DEFAULT_MOTION_FUSION_POLICY,
) -> tuple[float, int]:
    """Align the early fused dead-reckoning shape to the clean GPS prefix."""

    gps = np.asarray(gps_xy, dtype=float)
    controls = np.asarray(controls, dtype=float)
    elapsed = np.asarray(elapsed_s, dtype=float)
    if not len(gps):
        return 0.0, 0
    start = min(max(int(start_index), 0), len(gps) - 1)
    end = min(start + policy.initialization_updates, len(gps) - 1)
    if end <= start:
        return 0.0, end

    predicted = np.zeros((end - start + 1, 2), dtype=float)
    state = np.zeros(3, dtype=float)
    for output_index, row_index in enumerate(range(start + 1, end + 1), start=1):
        dt_s = max(float(elapsed[row_index] - elapsed[row_index - 1]), 1e-3)
        state = integrate_unicycle(
            state,
            float(controls[row_index, 0]),
            float(controls[row_index, 1]),
            dt_s,
        )
        predicted[output_index] = state[:2]

    observed = gps[start : end + 1] - gps[start]
    cross = float(
        np.sum(predicted[:, 0] * observed[:, 1] - predicted[:, 1] * observed[:, 0])
    )
    dot = float(
        np.sum(predicted[:, 0] * observed[:, 0] + predicted[:, 1] * observed[:, 1])
    )
    if abs(cross) + abs(dot) <= 1e-12:
        displacement = observed[-1]
        heading = (
            math.atan2(float(displacement[1]), float(displacement[0]))
            if float(np.linalg.norm(displacement)) > 1e-6
            else 0.0
        )
    else:
        heading = math.atan2(cross, dot)
    return wrap_angle(heading), end
