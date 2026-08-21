#!/usr/bin/env python3
"""Reduced-input YNet-style yaw-velocity baseline for the i2Nav study.

Published-method boundary
-------------------------
Zhou et al., "Learning Yaw Velocity for Inertial-Wheel Odometry on Autonomous
Vehicles" use an attention-based temporal convolutional network (YNet) over
multimodal IMU + wheel-encoder data and fuse learned yaw velocity with wheel
velocity in an invariant EKF.

The frozen V2 trajectory archive used by this project does not contain the
paper's full raw multimodal packet set. This implementation therefore provides
an explicitly labelled *reduced-input adaptation*:
  * causal TCN + attention predicts held-out yaw velocity from the available
    wheel-speed / IMU-yaw / optional wheel-yaw channels and derivatives;
  * training is strict 9-sequence LOSO;
  * the predicted yaw rate is used as a measurement aiding the same planar
    wheel-IMU EKF state used by the classical baseline.

Do not cite results from this script as an exact reproduction of the original
YNet paper unless you later validate the architecture/protocol against the
paper/authors' implementation and use equivalent raw sensor channels.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence, Tuple
import json
import math
import random

import numpy as np
import pandas as pd

from .common import fit_speed_calibration, standardized_output
from .ekf_iw import EKFIWConfig, PlanarEKFIW, fit_config as fit_ekf_config

METHOD_NAME = "YNet-style reduced-input yaw + EKF"


@dataclass
class YNetConfig:
    window_samples: int = 20
    train_stride: int = 2
    max_train_windows: int = 80000
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    channels: int = 48
    dropout: float = 0.10
    grad_clip: float = 5.0
    yaw_measurement_floor_degps: float = 0.5


FEATURE_NAMES = [
    "odo_speed_mps",
    "imu_yaw_rate_radps",
    "wheel_yaw_filled_radps",
    "wheel_imu_disagreement_filled_radps",
    "odo_accel_mps2",
    "imu_yaw_accel_radps2",
]


def _features(df: pd.DataFrame) -> np.ndarray:
    imu = df["imu_yaw_rate_radps"].to_numpy(float)
    if "wheel_yaw_radps" in df.columns:
        wy = df["wheel_yaw_radps"].to_numpy(float)
    else:
        wy = np.full(len(df), np.nan)
    wyf = np.where(np.isfinite(wy), wy, imu)
    if "wheel_imu_yaw_disagreement_radps" in df.columns:
        dis = df["wheel_imu_yaw_disagreement_radps"].to_numpy(float)
    else:
        dis = wyf - imu
    dis = np.where(np.isfinite(dis), dis, 0.0)
    return np.column_stack([
        df["odo_speed_mps"].to_numpy(float),
        imu,
        wyf,
        dis,
        df["odo_accel_mps2"].to_numpy(float),
        df["imu_yaw_accel_radps2"].to_numpy(float),
    ]).astype(np.float32)


def _causal_windows(X: np.ndarray, window: int) -> np.ndarray:
    """Return one left-padded causal window ending at every sample."""
    window = int(window)
    if window <= 1:
        return X[:, None, :]
    pad = np.repeat(X[:1], window - 1, axis=0)
    xp = np.vstack([pad, X])
    # shape from sliding_window_view: [n, features, window] when axis=0? Avoid
    # version-specific surprises by explicit stride construction.
    n, f = X.shape
    shape = (n, window, f)
    strides = (xp.strides[0], xp.strides[0], xp.strides[1])
    return np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides, writeable=False)


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except Exception as exc:
        raise RuntimeError("YNet-style baseline requires PyTorch. Install torch or omit ynet_reduced.") from exc


class YNetModelFactory:
    @staticmethod
    def build(n_features: int, cfg: YNetConfig):
        torch, nn, F = _import_torch()

        class ResidualTCNBlock(nn.Module):
            def __init__(self, c: int, dilation: int):
                super().__init__()
                pad = dilation
                self.conv1 = nn.Conv1d(c, c, 3, padding=pad, dilation=dilation)
                self.conv2 = nn.Conv1d(c, c, 3, padding=pad, dilation=dilation)
                self.norm1 = nn.GroupNorm(4 if c % 4 == 0 else 1, c)
                self.norm2 = nn.GroupNorm(4 if c % 4 == 0 else 1, c)
                self.drop = nn.Dropout(cfg.dropout)

            def forward(self, x):
                y = self.drop(F.gelu(self.norm1(self.conv1(x))))
                y = self.drop(F.gelu(self.norm2(self.conv2(y))))
                return x + y

        class ReducedYNet(nn.Module):
            def __init__(self):
                super().__init__()
                c = int(cfg.channels)
                self.stem = nn.Conv1d(n_features, c, 3, padding=1)
                self.blocks = nn.Sequential(
                    ResidualTCNBlock(c, 1),
                    ResidualTCNBlock(c, 2),
                    ResidualTCNBlock(c, 4),
                )
                self.attn = nn.Conv1d(c, 1, 1)
                self.head = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(c, 1))

            def forward(self, x):
                # input [B,T,F] -> [B,F,T]
                z = F.gelu(self.stem(x.transpose(1, 2)))
                z = self.blocks(z)
                a = torch.softmax(self.attn(z), dim=-1)
                pooled = torch.sum(z * a, dim=-1)
                return self.head(pooled).squeeze(-1)

        return ReducedYNet()


@dataclass
class TrainedYNet:
    model: object
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    yaw_sigma_radps: float
    config: YNetConfig
    seed: int

    def predict(self, df: pd.DataFrame, batch_size: int = 4096) -> np.ndarray:
        torch, _, _ = _import_torch()
        X = _features(df)
        X = (X - self.x_mean) / self.x_std
        W = _causal_windows(X.astype(np.float32), self.config.window_samples)
        device = next(self.model.parameters()).device
        out = []
        self.model.eval()
        with torch.inference_mode():
            for s in range(0, len(W), batch_size):
                xb = torch.from_numpy(np.ascontiguousarray(W[s:s+batch_size])).to(device)
                yp = self.model(xb).detach().cpu().numpy()
                out.append(yp)
        y = np.concatenate(out) * self.y_std + self.y_mean
        return y.astype(float)

    def save(self, path: str | Path) -> None:
        torch, _, _ = _import_torch()
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "yaw_sigma_radps": self.yaw_sigma_radps,
            "config": asdict(self.config),
            "seed": self.seed,
            "feature_names": FEATURE_NAMES,
            "method_name": METHOD_NAME,
        }, p)


def fit_ynet(train_frames: Sequence[pd.DataFrame], cfg: YNetConfig, seed: int = 42) -> TrainedYNet:
    torch, nn, F = _import_torch()
    seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    raw_X = np.vstack([_features(d) for d in train_frames])
    x_mean = np.nanmean(raw_X, axis=0).astype(np.float32)
    x_std = np.nanstd(raw_X, axis=0).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)

    windows = []
    targets = []
    stride = max(1, int(cfg.train_stride))
    for d in train_frames:
        X = (_features(d) - x_mean) / x_std
        W = _causal_windows(X.astype(np.float32), cfg.window_samples)
        y = d["gt_yaw_rate_radps"].to_numpy(np.float32)
        idx = np.arange(cfg.window_samples - 1, len(d), stride)
        if not len(idx):
            continue
        w = W[idx]; yy = y[idx]
        ok = np.isfinite(w).all(axis=(1, 2)) & np.isfinite(yy)
        windows.append(np.ascontiguousarray(w[ok])); targets.append(yy[ok])
    if not windows:
        raise ValueError("No valid YNet training windows")
    W = np.concatenate(windows, axis=0)
    y = np.concatenate(targets, axis=0)
    rng = np.random.default_rng(seed)
    if len(W) > cfg.max_train_windows:
        idx = rng.choice(len(W), size=cfg.max_train_windows, replace=False)
        W = W[idx]; y = y[idx]
    # Trim derivative-induced yaw-rate spikes.
    qlo, qhi = np.quantile(y, [0.002, 0.998])
    keep = (y >= qlo) & (y <= qhi)
    W = W[keep]; y = y[keep]

    y_mean = float(np.mean(y)); y_std = float(max(np.std(y), 1e-4))
    yn = ((y - y_mean) / y_std).astype(np.float32)
    model = YNetModelFactory.build(W.shape[2], cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    order = np.arange(len(W))
    bs = max(16, int(cfg.batch_size))
    for epoch in range(int(cfg.epochs)):
        rng.shuffle(order)
        model.train()
        running = 0.0; count = 0
        for s in range(0, len(order), bs):
            idx = order[s:s+bs]
            xb = torch.from_numpy(W[idx]).to(device)
            yb = torch.from_numpy(yn[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss.detach().cpu()) * len(idx); count += len(idx)
        print(f"    YNet seed={seed} epoch {epoch+1:02d}/{cfg.epochs}: loss={running/max(count,1):.6f}")

    trained = TrainedYNet(model, x_mean, x_std, y_mean, y_std, math.radians(cfg.yaw_measurement_floor_degps), cfg, seed)
    # Training-residual scale is used only to set held-out EKF measurement noise.
    preds = []
    model.eval()
    with torch.inference_mode():
        sample_idx = np.arange(0, len(W), max(1, len(W)//20000))
        for s in range(0, len(sample_idx), 4096):
            idx = sample_idx[s:s+4096]
            xb = torch.from_numpy(W[idx]).to(device)
            pp = model(xb).cpu().numpy() * y_std + y_mean
            preds.append(pp)
    pp = np.concatenate(preds)
    yy = y[sample_idx]
    resid = pp - yy
    med = np.median(resid); mad = np.median(np.abs(resid - med))
    trained.yaw_sigma_radps = float(max(trained.yaw_sigma_radps, 1.4826 * mad))
    return trained


def run_ynet(test_df: pd.DataFrame, trained: TrainedYNet, ekf_cfg: EKFIWConfig) -> pd.DataFrame:
    ynet_w = trained.predict(test_df)
    n = len(test_df)
    t = test_df["time_s"].to_numpy(float)
    raw_v = test_df["odo_speed_mps"].to_numpy(float)
    imu_w = test_df["imu_yaw_rate_radps"].to_numpy(float)
    initial_pose = (
        float(test_df["gt_east_m"].iloc[0]),
        float(test_df["gt_north_m"].iloc[0]),
        float(test_df["gt_heading_rad"].iloc[0]),
    )
    f = PlanarEKFIW(ekf_cfg, initial_pose)
    x = np.empty(n); y = np.empty(n); h = np.empty(n); bg = np.empty(n)
    cv = np.empty(n); cw = np.empty(n)
    x[0], y[0], h[0], bg[0] = f.x
    cv[0] = ekf_cfg.speed_scale * raw_v[0] + ekf_cfg.speed_bias_mps
    cw[0] = imu_w[0] - bg[0]
    for k in range(1, n):
        dt = float(t[k] - t[k-1])
        v_used, _ = f.predict(dt, raw_v[k], imu_w[k])
        f.update_bias_from_yaw_rate(ynet_w[k], imu_w[k], trained.yaw_sigma_radps)
        x[k], y[k], h[k], bg[k] = f.x
        cv[k] = v_used
        cw[k] = imu_w[k] - bg[k]
    return standardized_output(
        test_df, x, y, h, METHOD_NAME, corrected_v=cv, corrected_omega=cw,
        extra={
            "ynet_pred_yaw_rate_radps": ynet_w,
            "estimated_gyro_bias_radps": bg,
            "ynet_measurement_sigma_radps": trained.yaw_sigma_radps,
            "baseline_protocol": "reduced-input LOSO YNet-style TCN+attention aiding planar EKF; no held-out GT feedback",
        },
    )
