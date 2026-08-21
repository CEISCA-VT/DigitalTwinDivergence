#!/usr/bin/env python3
"""LWOI-style reduced-input residual-learning baseline for i2Nav.

Published-method boundary
-------------------------
Brossard & Bonnabel (ICRA 2019) learn wheel-odometry / gyro model errors and
use learned corrections in localization. Their released implementation uses a
richer original sensor setup (including a FoG in key variants) and an older
Pyro Gaussian-process stack.

This file implements a *clearly labelled adaptation* to the frozen i2Nav
channels available in this project:
  - inputs: wheel forward speed, IMU yaw rate, optional wheel yaw rate, and
    causal temporal derivatives/statistics;
  - targets: held-in-sequence GT forward-speed and yaw-rate residuals;
  - learner: sparse RBF kernel residual regressor (posterior-mean-style
    approximation), trained strictly on the 9 LOSO training sequences;
  - held-out inference: corrected speed/yaw are integrated from the initial GT
    pose without any further GT correction.

Do not describe this as an exact reproduction of the official LWOI code.
For a paper table, use "LWOI-IMU adaptation (sparse RBF residual)".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Sequence, Tuple
import json
import math

import numpy as np
import pandas as pd

from .common import integrate_planar, standardized_output

METHOD_NAME = "LWOI-IMU adaptation (sparse RBF residual)"


FEATURE_NAMES = [
    "odo_speed_mps",
    "imu_yaw_rate_radps",
    "wheel_yaw_filled_radps",
    "wheel_imu_disagreement_filled_radps",
    "odo_accel_mps2",
    "imu_yaw_accel_radps2",
    "odo_speed_rollmean",
    "imu_yaw_rollmean",
    "odo_speed_rollstd",
    "imu_yaw_rollstd",
]


@dataclass
class LWOIConfig:
    n_centers: int = 256
    max_train_samples: int = 30000
    ridge: float = 1e-3
    length_scale: float = 2.0
    temporal_window_samples: int = 10
    max_abs_dv_mps: float = 0.50
    max_abs_dw_radps: float = math.radians(20.0)


def _causal_features(df: pd.DataFrame, window: int) -> np.ndarray:
    w = max(2, int(window))
    v = pd.Series(df["odo_speed_mps"].to_numpy(float))
    imu = pd.Series(df["imu_yaw_rate_radps"].to_numpy(float))
    if "wheel_yaw_radps" in df.columns:
        wy = df["wheel_yaw_radps"].to_numpy(float)
    else:
        wy = np.full(len(df), np.nan)
    wy_fill = np.where(np.isfinite(wy), wy, imu.to_numpy(float))
    if "wheel_imu_yaw_disagreement_radps" in df.columns:
        dis = df["wheel_imu_yaw_disagreement_radps"].to_numpy(float)
    else:
        dis = wy_fill - imu.to_numpy(float)
    dis = np.where(np.isfinite(dis), dis, 0.0)
    X = np.column_stack([
        v.to_numpy(float),
        imu.to_numpy(float),
        wy_fill,
        dis,
        df["odo_accel_mps2"].to_numpy(float),
        df["imu_yaw_accel_radps2"].to_numpy(float),
        v.rolling(w, min_periods=1).mean().to_numpy(float),
        imu.rolling(w, min_periods=1).mean().to_numpy(float),
        v.rolling(w, min_periods=2).std().fillna(0.0).to_numpy(float),
        imu.rolling(w, min_periods=2).std().fillna(0.0).to_numpy(float),
    ])
    return X


def _targets(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df["gt_forward_speed_mps"].to_numpy(float) - df["odo_speed_mps"].to_numpy(float),
        df["gt_yaw_rate_radps"].to_numpy(float) - df["imu_yaw_rate_radps"].to_numpy(float),
    ])


class SparseRBFResidual:
    def __init__(self, config: LWOIConfig, seed: int = 42):
        self.cfg = config
        self.seed = int(seed)
        self.x_mean = None
        self.x_std = None
        self.centers = None
        self.weights = None
        self.target_clip = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def _phi(self, Xn: np.ndarray, batch: int = 5000) -> np.ndarray:
        C = self.centers
        ls2 = float(self.cfg.length_scale) ** 2
        pieces = []
        for s in range(0, len(Xn), batch):
            A = Xn[s:s+batch]
            # squared Euclidean distance to inducing centers
            d2 = np.sum(A*A, axis=1, keepdims=True) + np.sum(C*C, axis=1)[None, :] - 2.0 * A @ C.T
            d2 = np.maximum(d2, 0.0)
            pieces.append(np.exp(-0.5 * d2 / max(ls2, 1e-9)))
        return np.vstack(pieces)

    def fit(self, frames: Sequence[pd.DataFrame]) -> "SparseRBFResidual":
        rng = np.random.default_rng(self.seed)
        X = np.vstack([_causal_features(d, self.cfg.temporal_window_samples) for d in frames])
        Y = np.vstack([_targets(d) for d in frames])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X = X[ok]; Y = Y[ok]
        if len(X) < 100:
            raise ValueError("Too few valid training samples for LWOI adaptation")

        # Trim the most extreme derivative-label artifacts before regression.
        lo = np.quantile(Y, 0.005, axis=0)
        hi = np.quantile(Y, 0.995, axis=0)
        keep = np.all((Y >= lo) & (Y <= hi), axis=1)
        X = X[keep]; Y = Y[keep]
        if len(X) > self.cfg.max_train_samples:
            idx = rng.choice(len(X), size=self.cfg.max_train_samples, replace=False)
            X = X[idx]; Y = Y[idx]

        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)
        self.x_std = np.where(self.x_std < 1e-6, 1.0, self.x_std)
        Xn = self._normalize(X)
        nc = min(int(self.cfg.n_centers), len(Xn))
        cidx = rng.choice(len(Xn), size=nc, replace=False)
        self.centers = Xn[cidx].copy()
        Phi = self._phi(Xn)
        # Add an intercept column by augmenting the RBF features.
        Phi = np.column_stack([Phi, np.ones(len(Phi))])
        A = Phi.T @ Phi
        A.flat[::A.shape[0]+1] += float(self.cfg.ridge)
        B = Phi.T @ Y
        self.weights = np.linalg.solve(A, B)
        empirical = np.quantile(np.abs(Y), 0.995, axis=0)
        self.target_clip = np.array([
            min(max(empirical[0], 0.05), self.cfg.max_abs_dv_mps),
            min(max(empirical[1], math.radians(1.0)), self.cfg.max_abs_dw_radps),
        ])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fit")
        X = _causal_features(df, self.cfg.temporal_window_samples)
        Xn = self._normalize(X)
        Phi = np.column_stack([self._phi(Xn), np.ones(len(Xn))])
        Y = Phi @ self.weights
        Y[:, 0] = np.clip(Y[:, 0], -self.target_clip[0], self.target_clip[0])
        Y[:, 1] = np.clip(Y[:, 1], -self.target_clip[1], self.target_clip[1])
        return Y

    def save(self, path: str | Path) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            x_mean=self.x_mean,
            x_std=self.x_std,
            centers=self.centers,
            weights=self.weights,
            target_clip=self.target_clip,
            config_json=json.dumps(asdict(self.cfg)),
            seed=np.array([self.seed], dtype=int),
        )


def run_lwoi_imu(test_df: pd.DataFrame, model: SparseRBFResidual) -> pd.DataFrame:
    corr = model.predict(test_df)
    v = test_df["odo_speed_mps"].to_numpy(float) + corr[:, 0]
    w = test_df["imu_yaw_rate_radps"].to_numpy(float) + corr[:, 1]
    t = test_df["time_s"].to_numpy(float)
    init = (
        float(test_df["gt_east_m"].iloc[0]),
        float(test_df["gt_north_m"].iloc[0]),
        float(test_df["gt_heading_rad"].iloc[0]),
    )
    x, y, h = integrate_planar(t, v, w, init)
    return standardized_output(
        test_df, x, y, h, METHOD_NAME, corrected_v=v, corrected_omega=w,
        extra={
            "pred_delta_v_mps": corr[:, 0],
            "pred_delta_omega_radps": corr[:, 1],
            "baseline_protocol": "9-sequence LOSO sparse-RBF residual learning; no held-out GT feedback",
        },
    )
