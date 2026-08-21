#!/usr/bin/env python3
"""Robust planar wheel--IMU EKF baseline for the frozen i2Nav channel set.

Why this revision exists
------------------------
The first study implementation used a zero-yaw pseudo-measurement whenever
forward wheel speed was near zero. That is *not valid for a mobile robot that
can rotate in place or execute low-speed turns*: v≈0 does not imply omega≈0.
Those pseudo-updates can therefore absorb real turning into the estimated gyro
bias and catastrophically corrupt heading.

This revision removes that assumption. It also:
  * learns only training-fold affine sensor calibration (strict LOSO),
  * auto-checks wheel-yaw sign/scale against training-fold GT derivatives,
  * gates wheel-yaw bias updates with a normalized-innovation test,
  * falls back to calibrated IMU-only propagation if wheel-yaw is not useful,
  * records diagnostics needed to defend the baseline in a paper.

This remains a planar study baseline compatible with the saved wheel/IMU
channels. It is not claimed to be a bit-for-bit reproduction of WING's full
state EKF-IW.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple
import math
import numpy as np
import pandas as pd

from .common import (
    fit_speed_calibration,
    standardized_output,
    training_noise_statistics,
    wrap_angle,
)

METHOD_NAME = "Wheel-IMU EKF (planar EKF-IW, corrected)"


def _robust_sigma(a: np.ndarray, floor: float) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return float(floor)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return float(max(floor, 1.4826 * mad))


def _fit_affine_sensor_to_gt(
    train_frames: Sequence[pd.DataFrame],
    sensor_col: str,
    gt_col: str = "gt_yaw_rate_radps",
    max_abs_sensor: float = 20.0,
    max_abs_gt: float = 20.0,
) -> Tuple[float, float, float, float, int]:
    """Fit gt ~= scale*sensor + bias using training folds only.

    Returns (scale, bias, residual_sigma, correlation, n_samples).
    A light 1--99% residual trim protects against pose-derivative spikes.
    """
    xs, ys = [], []
    for d in train_frames:
        if sensor_col not in d.columns or gt_col not in d.columns:
            continue
        x = d[sensor_col].to_numpy(float)
        y = d[gt_col].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < max_abs_sensor) & (np.abs(y) < max_abs_gt)
        if np.any(ok):
            xs.append(x[ok]); ys.append(y[ok])
    if not xs:
        return 1.0, 0.0, math.radians(2.0), float("nan"), 0
    x = np.concatenate(xs); y = np.concatenate(ys)
    if len(x) < 50 or np.nanstd(x) < 1e-8:
        resid = y - x
        return 1.0, float(np.nanmedian(resid)), _robust_sigma(resid, math.radians(0.2)), float("nan"), len(x)

    # Initial ridge fit.
    X = np.column_stack([x, np.ones(len(x))])
    A = X.T @ X + np.diag([1e-8, 1e-8])
    b = X.T @ y
    scale, bias = np.linalg.solve(A, b)

    resid = y - (scale * x + bias)
    lo, hi = np.quantile(resid[np.isfinite(resid)], [0.01, 0.99])
    keep = np.isfinite(resid) & (resid >= lo) & (resid <= hi)
    if np.sum(keep) >= 50:
        X2 = np.column_stack([x[keep], np.ones(np.sum(keep))])
        A2 = X2.T @ X2 + np.diag([1e-8, 1e-8])
        b2 = X2.T @ y[keep]
        scale, bias = np.linalg.solve(A2, b2)
        resid = y[keep] - (scale * x[keep] + bias)
        xx = x[keep]; yy = y[keep]
    else:
        xx = x; yy = y

    corr = float(np.corrcoef(xx, yy)[0, 1]) if len(xx) > 2 and np.std(xx) > 0 and np.std(yy) > 0 else float("nan")
    sigma = _robust_sigma(resid, math.radians(0.2))
    return float(scale), float(bias), sigma, corr, int(len(xx))


@dataclass
class EKFIWConfig:
    speed_scale: float = 1.0
    speed_bias_mps: float = 0.0

    # Training-fold calibration maps raw yaw-rate channels into the GT yaw-rate convention.
    imu_yaw_scale: float = 1.0
    imu_yaw_bias_radps: float = 0.0
    wheel_yaw_scale: float = 1.0
    wheel_yaw_bias_radps: float = 0.0

    # Dynamic residual gyro bias state after the static affine calibration above.
    initial_gyro_bias_radps: float = 0.0

    wheel_speed_sigma_mps: float = 0.05
    imu_yaw_sigma_radps: float = math.radians(1.0)
    wheel_yaw_sigma_radps: float = math.radians(2.0)
    gyro_bias_rw_sigma_radps_sqrt_s: float = math.radians(0.03)

    use_wheel_yaw_updates: bool = True
    wheel_yaw_training_corr: float = float("nan")
    wheel_yaw_training_n: int = 0
    wheel_yaw_vs_gt_sigma_radps: float = math.radians(2.0)

    # Chi-square-like scalar innovation gate. 9 ~= 3-sigma for one dimension.
    yaw_update_nis_gate: float = 9.0
    max_abs_dynamic_bias_radps: float = math.radians(20.0)


def fit_config(train_frames: Sequence[pd.DataFrame]) -> EKFIWConfig:
    """Fit one fold-specific configuration using only the nine training sequences."""
    speed_scale, speed_bias = fit_speed_calibration(train_frames)
    stats = training_noise_statistics(train_frames)

    imu_scale, imu_bias, imu_sigma, imu_corr, imu_n = _fit_affine_sensor_to_gt(train_frames, "imu_yaw_rate_radps")
    wheel_scale, wheel_bias, wheel_sigma, wheel_corr, wheel_n = _fit_affine_sensor_to_gt(train_frames, "wheel_yaw_radps")

    # Wheel-yaw is useful only if it behaves as an independently informative rate signal
    # in the training folds. A sign inversion is handled by the affine scale; a weak or
    # almost-constant channel is rejected rather than allowed to corrupt the bias state.
    use_wheel = bool(
        wheel_n >= 100
        and np.isfinite(wheel_corr)
        and abs(wheel_corr) >= 0.25
        and np.isfinite(wheel_sigma)
        and wheel_sigma < math.radians(30.0)
        and abs(wheel_scale) > 0.05
        and abs(wheel_scale) < 20.0
    )

    # Residual initial bias after affine calibration. Median training residual is usually
    # near zero, but retaining it makes the convention explicit and reproducible.
    residuals = []
    for d in train_frames:
        raw = d["imu_yaw_rate_radps"].to_numpy(float)
        gt = d["gt_yaw_rate_radps"].to_numpy(float)
        rr = (imu_scale * raw + imu_bias) - gt
        rr = rr[np.isfinite(rr)]
        if len(rr):
            residuals.append(rr)
    initial_bg = float(np.median(np.concatenate(residuals))) if residuals else 0.0

    return EKFIWConfig(
        speed_scale=speed_scale,
        speed_bias_mps=speed_bias,
        imu_yaw_scale=imu_scale,
        imu_yaw_bias_radps=imu_bias,
        wheel_yaw_scale=wheel_scale,
        wheel_yaw_bias_radps=wheel_bias,
        initial_gyro_bias_radps=initial_bg,
        wheel_speed_sigma_mps=stats["wheel_speed_sigma"],
        imu_yaw_sigma_radps=max(stats["imu_yaw_sigma"], imu_sigma),
        wheel_yaw_sigma_radps=max(stats["wheel_yaw_sigma"], wheel_sigma),
        use_wheel_yaw_updates=use_wheel,
        wheel_yaw_training_corr=wheel_corr,
        wheel_yaw_training_n=wheel_n,
        wheel_yaw_vs_gt_sigma_radps=wheel_sigma,
    )


class PlanarEKFIW:
    def __init__(self, config: EKFIWConfig, initial_pose: tuple[float, float, float]):
        self.cfg = config
        self.x = np.array([
            initial_pose[0], initial_pose[1], initial_pose[2], config.initial_gyro_bias_radps
        ], dtype=float)
        self.P = np.diag([
            1e-8,
            1e-8,
            math.radians(0.05) ** 2,
            math.radians(1.0) ** 2,
        ]).astype(float)
        self.accepted_updates = 0
        self.rejected_updates = 0

    def calibrate_imu_yaw(self, raw: float) -> float:
        return self.cfg.imu_yaw_scale * float(raw) + self.cfg.imu_yaw_bias_radps

    def calibrate_wheel_yaw(self, raw: float) -> float:
        return self.cfg.wheel_yaw_scale * float(raw) + self.cfg.wheel_yaw_bias_radps

    def predict(self, dt: float, wheel_speed: float, imu_yaw_rate: float) -> tuple[float, float]:
        dt = max(0.0, float(dt))
        px, py, th, bg = self.x
        v = self.cfg.speed_scale * float(wheel_speed) + self.cfg.speed_bias_mps
        imu_corr = self.calibrate_imu_yaw(imu_yaw_rate)
        w = imu_corr - bg

        hm = th + 0.5 * w * dt
        self.x[0] = px + v * math.cos(hm) * dt
        self.x[1] = py + v * math.sin(hm) * dt
        self.x[2] = float(wrap_angle(th + w * dt))

        F = np.eye(4)
        F[0, 2] = -v * math.sin(hm) * dt
        F[1, 2] =  v * math.cos(hm) * dt
        F[0, 3] = 0.5 * v * math.sin(hm) * dt * dt
        F[1, 3] = -0.5 * v * math.cos(hm) * dt * dt
        F[2, 3] = -dt

        sv = self.cfg.wheel_speed_sigma_mps
        sw = self.cfg.imu_yaw_sigma_radps
        sb = self.cfg.gyro_bias_rw_sigma_radps_sqrt_s
        G = np.array([
            [math.cos(hm) * dt, -0.5 * v * math.sin(hm) * dt * dt, 0.0],
            [math.sin(hm) * dt,  0.5 * v * math.cos(hm) * dt * dt, 0.0],
            [0.0, dt, 0.0],
            [0.0, 0.0, math.sqrt(max(dt, 1e-9))],
        ])
        Qc = np.diag([sv * sv, sw * sw, sb * sb])
        self.P = F @ self.P @ F.T + G @ Qc @ G.T + np.eye(4) * 1e-12
        self.P = 0.5 * (self.P + self.P.T)
        return v, w

    def update_bias_from_yaw_rate(self, observed_yaw_rate: float, imu_yaw_rate: float, sigma: float) -> bool:
        """Use an independent yaw-rate observation to update only the residual IMU bias.

        Returns True if the update was accepted. The innovation gate prevents a bad wheel
        yaw sample from dragging the bias state through a large turn.
        """
        if not (np.isfinite(observed_yaw_rate) and np.isfinite(imu_yaw_rate)):
            return False

        H = np.array([[0.0, 0.0, 0.0, -1.0]])
        imu_corr = self.calibrate_imu_yaw(imu_yaw_rate)
        pred = imu_corr - self.x[3]
        innovation = float(observed_yaw_rate) - pred
        R = float(max(sigma, 1e-6) ** 2)
        S = float((H @ self.P @ H.T).item() + R)
        nis = (innovation * innovation) / max(S, 1e-12)
        if not np.isfinite(nis) or nis > self.cfg.yaw_update_nis_gate:
            self.rejected_updates += 1
            return False

        K = (self.P @ H.T)[:, 0] / S
        self.x = self.x + K * innovation
        self.x[3] = float(np.clip(
            self.x[3],
            -self.cfg.max_abs_dynamic_bias_radps,
            self.cfg.max_abs_dynamic_bias_radps,
        ))
        self.x[2] = float(wrap_angle(self.x[2]))
        I = np.eye(4)
        KH = np.outer(K, H[0])
        self.P = (I - KH) @ self.P @ (I - KH).T + np.outer(K, K) * R
        self.P = 0.5 * (self.P + self.P.T)
        self.accepted_updates += 1
        return True


def run_ekf_iw(test_df: pd.DataFrame, config: EKFIWConfig) -> pd.DataFrame:
    n = len(test_df)
    t = test_df["time_s"].to_numpy(float)
    raw_v = test_df["odo_speed_mps"].to_numpy(float)
    imu_w = test_df["imu_yaw_rate_radps"].to_numpy(float)
    wheel_w_raw = test_df["wheel_yaw_radps"].to_numpy(float) if "wheel_yaw_radps" in test_df.columns else np.full(n, np.nan)

    initial_pose = (
        float(test_df["gt_east_m"].iloc[0]),
        float(test_df["gt_north_m"].iloc[0]),
        float(test_df["gt_heading_rad"].iloc[0]),
    )
    f = PlanarEKFIW(config, initial_pose)

    x = np.empty(n); y = np.empty(n); h = np.empty(n); bg = np.empty(n)
    cv = np.empty(n); cw = np.empty(n); wheel_corr = np.full(n, np.nan)
    accepted = np.zeros(n, dtype=int)

    x[0], y[0], h[0], bg[0] = f.x
    cv[0] = config.speed_scale * raw_v[0] + config.speed_bias_mps
    cw[0] = f.calibrate_imu_yaw(imu_w[0]) - bg[0]
    if np.isfinite(wheel_w_raw[0]):
        wheel_corr[0] = f.calibrate_wheel_yaw(wheel_w_raw[0])

    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        v_used, _ = f.predict(dt, raw_v[k], imu_w[k])

        # IMPORTANT: no v≈0 -> omega=0 pseudo-measurement. A robot may rotate in place.
        if config.use_wheel_yaw_updates and np.isfinite(wheel_w_raw[k]):
            ww = f.calibrate_wheel_yaw(wheel_w_raw[k])
            wheel_corr[k] = ww
            accepted[k] = int(f.update_bias_from_yaw_rate(ww, imu_w[k], config.wheel_yaw_sigma_radps))

        x[k], y[k], h[k], bg[k] = f.x
        cv[k] = v_used
        cw[k] = f.calibrate_imu_yaw(imu_w[k]) - bg[k]

    update_count = max(f.accepted_updates + f.rejected_updates, 1)
    extra = {
        "estimated_gyro_bias_radps": bg,
        "calibrated_imu_yaw_rate_radps": config.imu_yaw_scale * imu_w + config.imu_yaw_bias_radps,
        "calibrated_wheel_yaw_rate_radps": wheel_corr,
        "yaw_update_accepted": accepted,
        "speed_scale": config.speed_scale,
        "speed_bias_mps": config.speed_bias_mps,
        "imu_yaw_scale": config.imu_yaw_scale,
        "imu_yaw_bias_radps": config.imu_yaw_bias_radps,
        "wheel_yaw_scale": config.wheel_yaw_scale,
        "wheel_yaw_bias_radps": config.wheel_yaw_bias_radps,
        "wheel_yaw_updates_enabled": int(config.use_wheel_yaw_updates),
        "wheel_yaw_update_acceptance_fraction": f.accepted_updates / update_count,
        "baseline_protocol": (
            "strict LOSO training-fold calibration; held-out propagation; GT only for training calibration "
            "and initial pose; no stationary-zero-yaw pseudo-update"
        ),
    }
    return standardized_output(test_df, x, y, h, METHOD_NAME, cv, cw, extra)
