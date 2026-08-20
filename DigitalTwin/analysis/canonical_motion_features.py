#!/usr/bin/env python3
"""
canonical_motion_features.py
============================

Deterministic platform-adapter utilities for the sensor-lightweight digital twin.

ZERO-SHOT DESIGN RULE
---------------------
The learned Twin must consume canonical physical quantities, not robot-specific
raw channels.

For i2Nav / Ranger MINI 3.0:
    raw four wheel speeds + four steering angles
        -> deterministic 4WS kinematic adapter
        -> canonical body-frame wheel velocity / wheel yaw rate

For a differential/skid-steer robot such as UGV01:
    left/right encoder velocity
        -> its own deterministic adapter
        -> the SAME canonical body quantities

For TerraSentia:
    its wheel encoder telemetry
        -> TerraSentia adapter
        -> the SAME canonical quantities

Nothing in this module trains a model.

IMPORTANT COORDINATE CONVENTION
-------------------------------
The Ranger raw adapter uses x-forward, y-right wheel coordinates internally.
The planar solve therefore produces a yaw sign opposite to the prepared i2Nav
IMU convention.  The returned canonical wheel yaw is explicitly negated so
that positive wheel yaw matches positive PreparedSequence.imu_yaw_rate.

This sign correction was empirically sanity-checked by the zero-shot observable
audit: wheel-yaw / IMU-yaw correlations become strongly positive across all
ten i2Nav sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class CanonicalMotionSignals:
    time_s: np.ndarray
    wheel_forward_mps: np.ndarray
    wheel_lateral_mps: np.ndarray
    wheel_yaw_radps: np.ndarray
    imu_yaw_radps: np.ndarray
    odo_forward_mps: np.ndarray
    yaw_disagreement_radps: np.ndarray
    yaw_disagreement_normalized: np.ndarray


def read_numeric_text(path: Path, min_cols: int = 9) -> np.ndarray:
    """Read whitespace/comma separated numeric text, ignoring headers/comments."""
    rows: list[np.ndarray] = []

    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("%"):
                continue

            arr = np.fromstring(s.replace(",", " "), sep=" ")
            if len(arr) >= min_cols and np.all(np.isfinite(arr[:min_cols])):
                rows.append(arr[:min_cols])

    if not rows:
        raise RuntimeError(
            f"No numeric rows with >= {min_cols} columns found in:\n{path}"
        )

    out = np.asarray(rows, dtype=np.float64)
    return out[np.argsort(out[:, 0], kind="stable")]


def ranger_wheel_positions(
    wheelbase_m: float = 0.494,
    track_m: float = 0.370,
) -> np.ndarray:
    """
    Ranger wheel positions in the internal x-forward, y-right frame.

    Official raw-text wheel order:
        1 RF
        2 LF
        3 RB
        4 LB
    """
    x = float(wheelbase_m) / 2.0
    y = float(track_m) / 2.0

    return np.asarray(
        [
            [ x,  y],   # RF
            [ x, -y],   # LF
            [-x,  y],   # RB
            [-x, -y],   # LB
        ],
        dtype=np.float64,
    )


def solve_planar_twist_batch(
    speeds_mps: np.ndarray,
    steering_angles_rad: np.ndarray,
    wheel_positions_xy_m: np.ndarray,
) -> np.ndarray:
    """
    Solve rolling constraints for [vx, vy, omega_internal].

    For wheel i:
        d_i^T ( [vx,vy] + omega[-y_i,x_i] ) = speed_i

    Uses a tiny ridge term and an explicit (N,3,1) RHS so NumPy 2.x batched
    solve semantics are unambiguous.
    """
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    angles = np.asarray(steering_angles_rad, dtype=np.float64)
    pos = np.asarray(wheel_positions_xy_m, dtype=np.float64)

    if speeds.ndim != 2 or speeds.shape[1] != 4:
        raise ValueError(f"Expected speeds shape (N,4), got {speeds.shape}")
    if angles.shape != speeds.shape:
        raise ValueError(
            f"angles must match speeds: {angles.shape} vs {speeds.shape}"
        )
    if pos.shape != (4, 2):
        raise ValueError(f"Expected wheel positions (4,2), got {pos.shape}")

    c = np.cos(angles)
    s = np.sin(angles)

    A = np.empty((len(speeds), 4, 3), dtype=np.float64)
    A[:, :, 0] = c
    A[:, :, 1] = s
    A[:, :, 2] = -c * pos[None, :, 1] + s * pos[None, :, 0]

    ATA = np.einsum("nij,nik->njk", A, A)
    ATb = np.einsum("nij,ni->nj", A, speeds)
    ATA[:, np.arange(3), np.arange(3)] += 1e-8

    twist = np.linalg.solve(ATA, ATb[..., None])[..., 0]

    if twist.shape != (len(speeds), 3):
        raise RuntimeError(f"Unexpected twist shape: {twist.shape}")
    if not np.all(np.isfinite(twist)):
        raise RuntimeError("Non-finite wheel kinematic solution.")

    return twist


def _interp_to_grid(
    source_t: np.ndarray,
    source_v: np.ndarray,
    grid_t: np.ndarray,
) -> np.ndarray:
    source_t = np.asarray(source_t, dtype=np.float64)
    source_v = np.asarray(source_v, dtype=np.float64)
    grid_t = np.asarray(grid_t, dtype=np.float64)

    if source_v.ndim == 1:
        return np.interp(grid_t, source_t, source_v)

    return np.column_stack(
        [
            np.interp(grid_t, source_t, source_v[:, j])
            for j in range(source_v.shape[1])
        ]
    )


def find_i2nav_ranger_odo(root: Path, sequence_name: str) -> Path:
    """Resolve exactly one sequence-specific *_RANGER_ODO.txt."""
    root = Path(root)
    candidates = list(
        (root / sequence_name).glob("*_RANGER_ODO.txt")
    )

    if not candidates:
        # Conservative fallback for alternate directory layouts.
        candidates = [
            p
            for p in root.rglob("*_RANGER_ODO.txt")
            if p.name.startswith(sequence_name)
        ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"{sequence_name}: expected exactly one *_RANGER_ODO.txt, "
            f"found {len(candidates)}: {candidates}"
        )

    return candidates[0]


def i2nav_ranger_to_canonical(
    sequence,
    root: Path,
    *,
    wheelbase_m: float = 0.494,
    track_m: float = 0.370,
    angle_sign: float = 1.0,
) -> CanonicalMotionSignals:
    """
    Build canonical encoder+IMU signals aligned to the exact V1 sequence grid.

    No GT is used here.
    """
    ranger_path = find_i2nav_ranger_odo(root, sequence.name)
    raw = read_numeric_text(ranger_path, min_cols=9)

    t_raw = raw[:, 0]
    speed_raw = raw[:, 1:5]
    angle_raw = float(angle_sign) * raw[:, 5:9]

    twist_internal = solve_planar_twist_batch(
        speed_raw,
        angle_raw,
        ranger_wheel_positions(wheelbase_m, track_m),
    )

    twist = _interp_to_grid(t_raw, twist_internal, sequence.grid)

    wheel_forward = twist[:, 0]
    wheel_lateral = twist[:, 1]

    # IMPORTANT:
    # Internal x-forward, y-right convention gives yaw sign opposite to the
    # PreparedSequence IMU convention. Return the canonical sign here.
    wheel_yaw = -twist[:, 2]

    imu_yaw = np.asarray(sequence.imu_yaw_rate, dtype=np.float64)
    odo_forward = np.asarray(sequence.odo_speed, dtype=np.float64)

    disagreement = imu_yaw - wheel_yaw
    disagreement_norm = disagreement / (
        np.abs(imu_yaw) + np.abs(wheel_yaw) + 0.02
    )

    return CanonicalMotionSignals(
        time_s=np.asarray(sequence.grid, dtype=np.float64),
        wheel_forward_mps=wheel_forward,
        wheel_lateral_mps=wheel_lateral,
        wheel_yaw_radps=wheel_yaw,
        imu_yaw_radps=imu_yaw,
        odo_forward_mps=odo_forward,
        yaw_disagreement_radps=disagreement,
        yaw_disagreement_normalized=disagreement_norm,
    )


def rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rolling(samples, min_periods=samples)
        .mean()
        .to_numpy()
    )


def rolling_std(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rolling(samples, min_periods=samples)
        .std(ddof=0)
        .to_numpy()
    )


def rolling_rms(values: np.ndarray, samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(
        np.maximum(
            rolling_mean(values * values, samples),
            0.0,
        )
    )


def build_slow_physical_features(
    signals: CanonicalMotionSignals,
    *,
    samples: int,
) -> np.ndarray:
    """
    Causal 30-s-style summary features.

    These use ONLY encoder/ODO/IMU information and therefore remain available
    at inference and on zero-shot target robots after their platform adapter.

    Columns:
      0 mean imu yaw
      1 std imu yaw
      2 rms imu yaw
      3 mean |imu yaw|
      4 mean wheel yaw
      5 std wheel yaw
      6 rms wheel yaw
      7 mean |wheel yaw|
      8 mean yaw disagreement
      9 std yaw disagreement
     10 rms yaw disagreement
     11 mean normalized disagreement
     12 std normalized disagreement
     13 mean odo forward speed
     14 std odo forward speed
     15 mean |odo forward speed|
    """
    imu = signals.imu_yaw_radps
    wheel = signals.wheel_yaw_radps
    diff = signals.yaw_disagreement_radps
    ndiff = signals.yaw_disagreement_normalized
    speed = signals.odo_forward_mps

    cols = [
        rolling_mean(imu, samples),
        rolling_std(imu, samples),
        rolling_rms(imu, samples),
        rolling_mean(np.abs(imu), samples),
        rolling_mean(wheel, samples),
        rolling_std(wheel, samples),
        rolling_rms(wheel, samples),
        rolling_mean(np.abs(wheel), samples),
        rolling_mean(diff, samples),
        rolling_std(diff, samples),
        rolling_rms(diff, samples),
        rolling_mean(ndiff, samples),
        rolling_std(ndiff, samples),
        rolling_mean(speed, samples),
        rolling_std(speed, samples),
        rolling_mean(np.abs(speed), samples),
    ]

    return np.column_stack(cols).astype(np.float32)


def build_affine_yaw_targets(
    imu_yaw_radps: np.ndarray,
    gt_yaw_radps: np.ndarray,
    *,
    samples: int,
    scale_delta_limit: float,
    bias_limit_radps: float,
    excitation_std_radps: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Training-only physical targets for:
        gt_yaw ~= (1 + delta_scale) * imu_yaw + bias

    The targets are fitted causally over the same slow window.

    Scale is poorly identifiable during near-straight motion.  A continuous
    excitation gate in [0,1] downweights scale supervision there.  Bias remains
    supervised.

    Returns:
        target_scale_delta  shape (N,)
        target_bias_radps   shape (N,)
        scale_gate          shape (N,)
    """
    x = np.asarray(imu_yaw_radps, dtype=np.float64)
    y = np.asarray(gt_yaw_radps, dtype=np.float64)

    # Fit the RESIDUAL directly:
    #
    #   r = y - x ~= delta_scale * x + bias
    #
    # This is preferable to ridge-fitting y ~= scale*x+bias because the prior
    # is naturally centered at delta_scale=0 (identity calibration).  When the
    # true relationship is y=x, ridge regularization therefore does NOT create
    # an artificial scale shrinkage.
    residual = y - x

    mx = rolling_mean(x, samples)
    mr = rolling_mean(residual, samples)
    mxx = rolling_mean(x * x, samples)
    mxr = rolling_mean(x * residual, samples)

    var_x = np.maximum(mxx - mx * mx, 0.0)
    cov_xr = mxr - mx * mr

    ridge = (0.01 ** 2)

    raw_delta_scale = cov_xr / (var_x + ridge)

    # Scale is weakly identifiable during near-straight motion.  Gate its
    # target continuously toward zero there; let the bias target explain the
    # local mean residual.
    gate = var_x / (var_x + float(excitation_std_radps) ** 2)

    scale_delta = gate * raw_delta_scale
    bias = mr - scale_delta * mx

    scale_delta = np.clip(
        scale_delta,
        -float(scale_delta_limit),
        float(scale_delta_limit),
    )
    bias = np.clip(
        bias,
        -float(bias_limit_radps),
        float(bias_limit_radps),
    )

    return (
        scale_delta.astype(np.float32),
        bias.astype(np.float32),
        gate.astype(np.float32),
    )
