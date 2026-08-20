#!/usr/bin/env python3
"""
i2nav_v2_physical_yaw_pilot.py
==============================

Twin V2 pilot with a two-timescale, zero-shot-compatible physical yaw model.

Why this exists
---------------
The previous V2:
  * improved short-horizon RPE,
  * strongly improved parking01,
  * reduced parking02 seed instability,
  * but did NOT remove parking02's persistent signed yaw error.

Diagnostics then showed:
  * remaining signed yaw bias strongly tracks ATE,
  * a free "bias head" did not actually track the true persistent bias,
  * merely increasing ODO+IMU history length did not resolve ambiguity,
  * canonical wheel/IMU information helped only modestly at the slow timescale.

This pilot therefore makes ONE physically motivated architectural change:

FAST BRANCH (2 s)
    exact original six V1 features
        -> GRU
        -> delta_v_fast
        -> delta_omega_fast

SLOW PHYSICAL BRANCH (30 s causal summaries)
    canonical encoder+IMU physical summaries
        -> small MLP
        -> delta_scale_omega
        -> bias_omega

Yaw correction:
    omega_corrected
      = (1 + delta_scale_omega) * omega_imu
        + bias_omega
        + delta_omega_fast

Equivalently:
    delta_omega_total
      = delta_scale_omega * omega_imu
        + bias_omega
        + delta_omega_fast

Training-only physical supervision
----------------------------------
Ground truth is used to fit a causal 30 s local affine target:

    omega_GT ~= (1 + delta_scale_GT) * omega_IMU + bias_GT

The slow branch is supervised toward those physical calibration targets.
The fast branch is separately supervised toward the remaining residual and is
forced toward zero mean over long windows.

ZERO-SHOT RULE
--------------
The learned network NEVER sees Ranger wheel IDs, steering angles, track width,
or wheelbase.

Those remain inside a deterministic platform adapter.  The learned slow branch
receives only canonical SI-unit body-motion summaries that can be recreated on
TerraSentia and UGV01.

External zero-shot evaluation must freeze:
    architecture
    model weights
    i2Nav normalization
    thresholds/hyperparameters
before looking at target-domain labels.

Frozen V1 remains read-only.

Pilot:
    parking02, parking01, street00
    x seeds 42, 1042, 2042
"""

from __future__ import annotations

import argparse
import copy
import importlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from DigitalTwin.analysis import i2nav_v2_yaw_bias_pilot as base
from DigitalTwin.analysis.canonical_motion_features import (
    build_affine_yaw_targets,
    build_slow_physical_features,
    i2nav_ranger_to_canonical,
)


PILOT_FOLDS = (
    "parking02",
    "parking01",
    "street00",
)

BASE_SEEDS = (
    42,
    1042,
    2042,
)


@dataclass
class PhysicalCache:
    name: str
    fast_windows: np.ndarray
    slow_features: np.ndarray
    target: np.ndarray

    odo_speed: np.ndarray
    imu_yaw_rate: np.ndarray

    gt_x: np.ndarray
    gt_y: np.ndarray
    gt_heading: np.ndarray

    target_scale_delta: np.ndarray
    target_bias_radps: np.ndarray
    scale_gate: np.ndarray

    wheel_yaw_radps: np.ndarray
    yaw_disagreement_radps: np.ndarray

    grid: np.ndarray


def slow_feature_normalization(
    slow_by_name: dict[str, np.ndarray],
    training_names: list[str],
    *,
    valid_from: int,
) -> tuple[np.ndarray, np.ndarray]:
    blocks = []

    for name in training_names:
        arr = np.asarray(slow_by_name[name], dtype=np.float32)
        arr = arr[valid_from:]
        arr = arr[np.all(np.isfinite(arr), axis=1)]

        if len(arr):
            blocks.append(arr)

    if not blocks:
        raise RuntimeError("No valid slow features for train-only normalization.")

    x = np.concatenate(blocks, axis=0)

    mean = np.mean(x, axis=0).astype(np.float32)
    std = np.std(x, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4)

    return mean, std


def build_cache(
    sequence,
    *,
    fast_feature_mean: np.ndarray,
    fast_feature_std: np.ndarray,
    slow_features: np.ndarray,
    slow_feature_mean: np.ndarray,
    slow_feature_std: np.ndarray,
    target_scale_delta: np.ndarray,
    target_bias_radps: np.ndarray,
    scale_gate: np.ndarray,
    wheel_yaw_radps: np.ndarray,
    yaw_disagreement_radps: np.ndarray,
    fast_window: int,
) -> PhysicalCache:
    fast_normalized = (
        np.asarray(sequence.features, dtype=np.float32)
        - fast_feature_mean[None, :]
    ) / fast_feature_std[None, :]

    slow_normalized = (
        np.asarray(slow_features, dtype=np.float32)
        - slow_feature_mean[None, :]
    ) / slow_feature_std[None, :]

    # Keep pre-warmup rows finite. They are never sampled for training and the
    # slow physical correction is explicitly held at zero before warmup.
    slow_normalized = np.nan_to_num(
        slow_normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    return PhysicalCache(
        name=sequence.name,
        fast_windows=base.sliding_windows(
            fast_normalized,
            fast_window,
        ),
        slow_features=slow_normalized,
        target=np.asarray(sequence.target_corrections, dtype=np.float32),
        odo_speed=np.asarray(sequence.odo_speed, dtype=np.float32),
        imu_yaw_rate=np.asarray(sequence.imu_yaw_rate, dtype=np.float32),
        gt_x=np.asarray(sequence.gt_x, dtype=np.float32),
        gt_y=np.asarray(sequence.gt_y, dtype=np.float32),
        gt_heading=np.asarray(sequence.gt_heading, dtype=np.float32),
        target_scale_delta=np.asarray(target_scale_delta, dtype=np.float32),
        target_bias_radps=np.asarray(target_bias_radps, dtype=np.float32),
        scale_gate=np.asarray(scale_gate, dtype=np.float32),
        wheel_yaw_radps=np.asarray(wheel_yaw_radps, dtype=np.float32),
        yaw_disagreement_radps=np.asarray(
            yaw_disagreement_radps,
            dtype=np.float32,
        ),
        grid=np.asarray(sequence.grid, dtype=np.float64),
    )


class PhysicalChunkDataset(Dataset):
    def __init__(
        self,
        caches: dict[str, PhysicalCache],
        names: list[str],
        *,
        fast_window: int,
        slow_window_samples: int,
        chunk_steps: int,
        stride: int,
    ):
        self.caches = caches
        self.fast_window = int(fast_window)
        self.slow_window_samples = int(slow_window_samples)
        self.chunk_steps = int(chunk_steps)

        # Both branches must be fully causal/valid.
        self.first_valid = max(
            self.fast_window - 1,
            self.slow_window_samples - 1,
        )

        self.items: list[tuple[str, int]] = []

        for name in names:
            cache = caches[name]
            n = len(cache.grid)
            last_start_exclusive = n - self.chunk_steps + 1

            for start in range(
                self.first_valid,
                last_start_exclusive,
                int(stride),
            ):
                self.items.append((name, start))

        if not self.items:
            raise RuntimeError("No PhysicalChunkDataset items were generated.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        name, start = self.items[index]
        cache = self.caches[name]

        fast_window_start = start - (self.fast_window - 1)
        end = start + self.chunk_steps

        return {
            "fast_windows": torch.from_numpy(
                cache.fast_windows[
                    fast_window_start:
                    fast_window_start + self.chunk_steps
                ]
            ),
            "slow_features": torch.from_numpy(
                cache.slow_features[start:end]
            ),
            "target": torch.from_numpy(
                cache.target[start:end]
            ),
            "odo_speed": torch.from_numpy(
                cache.odo_speed[start:end]
            ),
            "imu_yaw_rate": torch.from_numpy(
                cache.imu_yaw_rate[start:end]
            ),
            "gt_x": torch.from_numpy(
                cache.gt_x[start:end]
            ),
            "gt_y": torch.from_numpy(
                cache.gt_y[start:end]
            ),
            "gt_heading": torch.from_numpy(
                cache.gt_heading[start:end]
            ),
            "target_scale_delta": torch.from_numpy(
                cache.target_scale_delta[start:end]
            ),
            "target_bias_radps": torch.from_numpy(
                cache.target_bias_radps[start:end]
            ),
            "scale_gate": torch.from_numpy(
                cache.scale_gate[start:end]
            ),
        }


class V2PhysicalYaw(nn.Module):
    """
    Two-timescale model.

    fast GRU:
        exact six V1 features over 2 s

    slow MLP:
        deterministic 30 s causal canonical physical summaries

    corrected yaw:
        (1 + scale_delta) * imu_yaw + yaw_bias + dw_fast
    """

    def __init__(
        self,
        fast_input_dim: int,
        slow_input_dim: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.10,
        slow_hidden_size: int = 32,
        dv_limit: float = 0.15,
        dw_fast_limit: float = 0.020,
        scale_delta_limit: float = 0.15,
        yaw_bias_limit: float = 0.005,
    ):
        super().__init__()

        self.dv_limit = float(dv_limit)
        self.dw_fast_limit = float(dw_fast_limit)
        self.scale_delta_limit = float(scale_delta_limit)
        self.yaw_bias_limit = float(yaw_bias_limit)

        self.fast_gru = nn.GRU(
            input_size=int(fast_input_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fast_norm = nn.LayerNorm(hidden_size)

        self.dv_head = nn.Linear(hidden_size, 1)
        self.dw_fast_head = nn.Linear(hidden_size, 1)

        self.slow_mlp = nn.Sequential(
            nn.Linear(slow_input_dim, slow_hidden_size),
            nn.Tanh(),
            nn.Linear(slow_hidden_size, slow_hidden_size),
            nn.Tanh(),
        )
        self.scale_head = nn.Linear(slow_hidden_size, 1)
        self.bias_head = nn.Linear(slow_hidden_size, 1)

        nn.init.zeros_(self.dv_head.bias)
        nn.init.zeros_(self.dw_fast_head.bias)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.bias_head.bias)

        # Begin very close to identity yaw calibration.
        nn.init.normal_(self.scale_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.bias_head.weight, mean=0.0, std=1e-3)

    def forward(
        self,
        fast_windows: torch.Tensor,
        slow_features: torch.Tensor,
        imu_yaw_rate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if fast_windows.ndim == 4:
            batch, horizon, window, features = fast_windows.shape

            fast_x = fast_windows.reshape(
                batch * horizon,
                window,
                features,
            )
            slow_x = slow_features.reshape(
                batch * horizon,
                slow_features.shape[-1],
            )
            imu = imu_yaw_rate.reshape(batch * horizon)
            restore = (batch, horizon)

        elif fast_windows.ndim == 3:
            fast_x = fast_windows
            slow_x = slow_features
            imu = imu_yaw_rate.reshape(-1)
            restore = None

        else:
            raise RuntimeError(
                f"Unexpected fast-window shape {fast_windows.shape}"
            )

        _, hidden = self.fast_gru(fast_x)
        fast_latent = self.fast_norm(hidden[-1])

        dv = self.dv_limit * torch.tanh(
            self.dv_head(fast_latent)
        ).squeeze(-1)

        dw_fast = self.dw_fast_limit * torch.tanh(
            self.dw_fast_head(fast_latent)
        ).squeeze(-1)

        slow_latent = self.slow_mlp(slow_x)

        scale_delta = self.scale_delta_limit * torch.tanh(
            self.scale_head(slow_latent)
        ).squeeze(-1)

        yaw_bias = self.yaw_bias_limit * torch.tanh(
            self.bias_head(slow_latent)
        ).squeeze(-1)

        dw_slow = (
            scale_delta * imu
            + yaw_bias
        )

        dw = (
            dw_slow
            + dw_fast
        )

        if restore is not None:
            batch, horizon = restore

            def reshape(z):
                return z.reshape(batch, horizon)

            dv = reshape(dv)
            dw_fast = reshape(dw_fast)
            scale_delta = reshape(scale_delta)
            yaw_bias = reshape(yaw_bias)
            dw_slow = reshape(dw_slow)
            dw = reshape(dw)

        return {
            "dv": dv,
            "dw_fast": dw_fast,
            "scale_delta": scale_delta,
            "yaw_bias": yaw_bias,
            "dw_slow": dw_slow,
            "dw": dw,
        }


def compute_loss(
    model: V2PhysicalYaw,
    batch: dict[str, torch.Tensor],
    *,
    dt: float,
    point_weight: float,
    trajectory_weight: float,
    physical_weight: float,
    fast_target_weight: float,
    fast_mean_weight: float,
    slow_smooth_weight: float,
    long_heading_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = model(
        batch["fast_windows"],
        batch["slow_features"],
        batch["imu_yaw_rate"],
    )

    dv = pred["dv"]
    dw = pred["dw"]
    dw_fast = pred["dw_fast"]
    dw_slow = pred["dw_slow"]
    scale_delta = pred["scale_delta"]
    yaw_bias = pred["yaw_bias"]

    true_dv = batch["target"][:, :, 0]
    true_dw = batch["target"][:, :, 1]

    # ------------------------------------------------------------------
    # 1. Point correction accuracy
    # ------------------------------------------------------------------
    point_dv = torch.mean(
        ((dv - true_dv) / 0.05) ** 2
    )
    point_dw = torch.mean(
        ((dw - true_dw) / 0.03) ** 2
    )
    point_loss = point_dv + point_dw

    # ------------------------------------------------------------------
    # 2. Explicit physical target supervision
    # ------------------------------------------------------------------
    target_scale = batch["target_scale_delta"]
    target_bias = batch["target_bias_radps"]
    scale_gate = batch["scale_gate"]

    scale_error = (
        (scale_delta - target_scale) / 0.03
    ) ** 2

    # Scale is only strongly supervised where the 30 s window contains
    # sufficient yaw excitation.
    scale_loss = torch.sum(
        scale_gate * scale_error
    ) / torch.clamp(
        torch.sum(scale_gate),
        min=1.0,
    )

    bias_loss = torch.mean(
        ((yaw_bias - target_bias) / 0.001) ** 2
    )

    target_slow_dw = (
        target_scale * batch["imu_yaw_rate"]
        + target_bias
    )

    slow_correction_loss = torch.mean(
        ((dw_slow - target_slow_dw) / 0.005) ** 2
    )

    physical_loss = (
        scale_loss
        + bias_loss
        + slow_correction_loss
    ) / 3.0

    # ------------------------------------------------------------------
    # 3. Explicit fast residual target
    # ------------------------------------------------------------------
    target_fast_dw = (
        true_dw
        - target_slow_dw
    )

    fast_target_loss = torch.mean(
        ((dw_fast - target_fast_dw) / 0.02) ** 2
    )

    # ------------------------------------------------------------------
    # 4. Differentiable trajectory fidelity
    # ------------------------------------------------------------------
    corrected_v = (
        batch["odo_speed"]
        + dv
    )
    corrected_w = (
        batch["imu_yaw_rate"]
        + dw
    )

    pred_x, pred_y, pred_heading = base.propagate_chunk(
        corrected_v,
        corrected_w,
        batch["gt_x"],
        batch["gt_y"],
        batch["gt_heading"],
        dt,
    )

    trajectory_terms = []

    for seconds in (1.0, 5.0, 10.0):
        index = int(round(seconds / dt))

        if index >= pred_x.shape[1]:
            continue

        pos_error = torch.sqrt(
            (
                pred_x[:, index]
                - batch["gt_x"][:, index]
            ) ** 2
            +
            (
                pred_y[:, index]
                - batch["gt_y"][:, index]
            ) ** 2
            + 1e-12
        )

        heading_error = base.wrap_tensor(
            pred_heading[:, index]
            - batch["gt_heading"][:, index]
        )

        # Keep same broad scaling logic as the previous V2 trajectory loss.
        trajectory_terms.append(
            torch.mean(
                (pos_error / max(0.10 * seconds, 0.10)) ** 2
                +
                (heading_error / math.radians(3.0)) ** 2
            )
        )

    trajectory_loss = torch.stack(
        trajectory_terms
    ).mean()

    # ------------------------------------------------------------------
    # 5. 30 s heading fidelity: directly targets the failure mode
    # ------------------------------------------------------------------
    index30 = min(
        int(round(30.0 / dt)),
        pred_heading.shape[1] - 1,
    )

    heading30 = base.wrap_tensor(
        pred_heading[:, index30]
        - batch["gt_heading"][:, index30]
    )

    long_heading_loss = torch.mean(
        (heading30 / math.radians(4.0)) ** 2
    )

    # ------------------------------------------------------------------
    # 6. Fast residual must remain mean-neutral
    # ------------------------------------------------------------------
    fast_mean_terms = []

    for seconds in (5.0, 10.0, 30.0):
        steps = min(
            int(round(seconds / dt)),
            dw_fast.shape[1],
        )

        mean_fast = torch.mean(
            dw_fast[:, :steps],
            dim=1,
        )

        fast_mean_terms.append(
            torch.mean(
                (mean_fast / 0.001) ** 2
            )
        )

    fast_mean_loss = torch.stack(
        fast_mean_terms
    ).mean()

    # ------------------------------------------------------------------
    # 7. Slowly varying calibration parameters
    # ------------------------------------------------------------------
    if scale_delta.shape[1] > 1:
        scale_smooth = torch.mean(
            (
                (
                    scale_delta[:, 1:]
                    - scale_delta[:, :-1]
                )
                / 0.005
            ) ** 2
        )

        bias_smooth = torch.mean(
            (
                (
                    yaw_bias[:, 1:]
                    - yaw_bias[:, :-1]
                )
                / 0.0002
            ) ** 2
        )

        slow_smooth_loss = (
            scale_smooth
            + bias_smooth
        ) / 2.0

    else:
        slow_smooth_loss = torch.zeros(
            (),
            device=dw.device,
        )

    total = (
        point_weight * point_loss
        + trajectory_weight * trajectory_loss
        + physical_weight * physical_loss
        + fast_target_weight * fast_target_loss
        + fast_mean_weight * fast_mean_loss
        + slow_smooth_weight * slow_smooth_loss
        + long_heading_weight * long_heading_loss
    )

    parts = {
        "total": float(total.detach().cpu()),
        "point": float(point_loss.detach().cpu()),
        "trajectory": float(trajectory_loss.detach().cpu()),
        "physical": float(physical_loss.detach().cpu()),
        "scale": float(scale_loss.detach().cpu()),
        "bias": float(bias_loss.detach().cpu()),
        "slow_correction": float(slow_correction_loss.detach().cpu()),
        "fast_target": float(fast_target_loss.detach().cpu()),
        "fast_mean": float(fast_mean_loss.detach().cpu()),
        "slow_smooth": float(slow_smooth_loss.detach().cpu()),
        "heading30": float(long_heading_loss.detach().cpu()),
    }

    return total, parts


def move_batch(batch, device):
    return {
        key: value.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        for key, value in batch.items()
    }


def validation_loss(
    model,
    loader,
    device,
    loss_kwargs,
):
    model.eval()
    sums: dict[str, float] = {}
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            _, parts = compute_loss(
                model,
                batch,
                **loss_kwargs,
            )

            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value

            count += 1

    return {
        key: value / max(count, 1)
        for key, value in sums.items()
    }


def train_model(
    model,
    train_loader,
    validation_loader,
    *,
    device,
    epochs,
    patience,
    learning_rate,
    weight_decay,
    loss_kwargs,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state = None
    best_validation_loss = float("inf")
    history = []
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running: dict[str, float] = {}
        batch_count = 0

        for batch in train_loader:
            batch = move_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)

            loss, parts = compute_loss(
                model,
                batch,
                **loss_kwargs,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value

            batch_count += 1

        train_mean = {
            key: value / max(batch_count, 1)
            for key, value in running.items()
        }

        val_mean = validation_loss(
            model,
            validation_loader,
            device,
            loss_kwargs,
        )

        row = {
            "epoch": epoch,
            **{
                f"train_{key}": value
                for key, value in train_mean.items()
            },
            **{
                f"val_{key}": value
                for key, value in val_mean.items()
            },
        }

        history.append(row)

        print(
            f"      epoch {epoch:02d}  "
            f"train={train_mean['total']:.4f}  "
            f"val={val_mean['total']:.4f}  "
            f"physical={val_mean['physical']:.4f}  "
            f"h30={val_mean['heading30']:.4f}  "
            f"fastmean={val_mean['fast_mean']:.4f}"
        )

        if val_mean["total"] < best_validation_loss - 1e-6:
            best_validation_loss = float(val_mean["total"])

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            bad_epochs = 0

        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(
                f"      early stopping: "
                f"best val={best_validation_loss:.4f}"
            )
            break

    if best_state is None:
        raise RuntimeError("No valid V2 checkpoint was produced.")

    return best_state, history, best_validation_loss


def predict_sequence(
    model: V2PhysicalYaw,
    cache: PhysicalCache,
    *,
    fast_window: int,
    slow_window_samples: int,
    batch_size: int,
    device,
) -> dict[str, np.ndarray]:
    model.eval()

    n = len(cache.grid)

    # Fast branch can begin after 2 s. Slow branch is held at identity/zero
    # until a complete slow context exists.
    fast_offset = fast_window - 1
    slow_offset = slow_window_samples - 1

    full = {
        key: np.zeros(n, dtype=np.float32)
        for key in (
            "dv",
            "dw_fast",
            "scale_delta",
            "yaw_bias",
            "dw_slow",
            "dw",
        )
    }

    # --------------------------------------------------------------
    # Fast-only warm-up: 2 s -> 30 s
    # --------------------------------------------------------------
    if slow_offset > fast_offset:
        fast_windows = cache.fast_windows[
            0:
            slow_offset - fast_offset
        ]

        if len(fast_windows):
            with torch.no_grad():
                for start in range(0, len(fast_windows), batch_size):
                    xb = torch.from_numpy(
                        fast_windows[start:start + batch_size]
                    ).to(
                        device=device,
                        dtype=torch.float32,
                    )

                    # Slow branch is forced to identity/zero during warmup by
                    # setting scale/bias outputs aside; only fast heads are used.
                    _, hidden = model.fast_gru(xb)
                    latent = model.fast_norm(hidden[-1])

                    dv = model.dv_limit * torch.tanh(
                        model.dv_head(latent)
                    ).squeeze(-1)

                    dw_fast = model.dw_fast_limit * torch.tanh(
                        model.dw_fast_head(latent)
                    ).squeeze(-1)

                    dst0 = fast_offset + start
                    dst1 = dst0 + len(dv)

                    full["dv"][dst0:dst1] = (
                        dv.detach().cpu().numpy()
                    )
                    full["dw_fast"][dst0:dst1] = (
                        dw_fast.detach().cpu().numpy()
                    )
                    full["dw"][dst0:dst1] = (
                        dw_fast.detach().cpu().numpy()
                    )

    # --------------------------------------------------------------
    # Full physical model after 30 s context is available
    # --------------------------------------------------------------
    valid_indices = np.arange(slow_offset, n, dtype=int)

    for start in range(0, len(valid_indices), batch_size):
        indices = valid_indices[start:start + batch_size]

        # fast_windows index corresponding to original sample i is
        # i - (fast_window - 1)
        fw_idx = indices - fast_offset

        fast_x = torch.from_numpy(
            cache.fast_windows[fw_idx]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        slow_x = torch.from_numpy(
            cache.slow_features[indices]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        imu = torch.from_numpy(
            cache.imu_yaw_rate[indices]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            output = model(
                fast_x,
                slow_x,
                imu,
            )

        for key in full:
            full[key][indices] = (
                output[key]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    full["corrections"] = np.column_stack(
        [
            full["dv"],
            full["dw"],
        ]
    ).astype(np.float32)

    return full


def save_prediction_trace(
    path: Path,
    sequence,
    cache: PhysicalCache,
    prediction: dict[str, np.ndarray],
) -> None:
    rows = []

    for i in range(len(sequence.grid)):
        true_dv = float(sequence.target_corrections[i, 0])
        true_dw = float(sequence.target_corrections[i, 1])
        pred_dw = float(prediction["dw"][i])

        rows.append(
            {
                "time_s": float(sequence.grid[i]),
                "true_delta_v_mps": true_dv,
                "pred_delta_v_mps": float(prediction["dv"][i]),
                "true_delta_omega_radps": true_dw,
                "pred_total_delta_omega_radps": pred_dw,
                "pred_fast_yaw_residual_radps": float(
                    prediction["dw_fast"][i]
                ),
                "pred_slow_yaw_correction_radps": float(
                    prediction["dw_slow"][i]
                ),
                "pred_yaw_scale_delta": float(
                    prediction["scale_delta"][i]
                ),
                "pred_yaw_bias_radps": float(
                    prediction["yaw_bias"][i]
                ),
                "wheel_yaw_radps": float(
                    cache.wheel_yaw_radps[i]
                ),
                "imu_yaw_radps": float(
                    cache.imu_yaw_rate[i]
                ),
                "wheel_imu_yaw_disagreement_radps": float(
                    cache.yaw_disagreement_radps[i]
                ),
                "remaining_yaw_error_radps": (
                    true_dw - pred_dw
                ),
            }
        )

    base.write_csv(path, rows)


def yaw_diagnostics(
    sequence,
    prediction,
    *,
    slow_window_samples: int,
) -> dict[str, float]:
    start = slow_window_samples - 1

    true_dw = np.asarray(
        sequence.target_corrections[start:, 1],
        dtype=float,
    )
    pred_dw = np.asarray(
        prediction["dw"][start:],
        dtype=float,
    )

    residual = true_dw - pred_dw

    mean_residual = float(np.mean(residual))

    return {
        "remaining_yaw_bias_radps": mean_residual,
        "remaining_yaw_bias_deg_per_min": (
            mean_residual
            * 180.0
            / math.pi
            * 60.0
        ),
        "remaining_yaw_rmse_radps": float(
            np.sqrt(np.mean(residual ** 2))
        ),
        "mean_fast_yaw_residual_radps": float(
            np.mean(prediction["dw_fast"][start:])
        ),
        "mean_fast_yaw_residual_deg_per_min": float(
            np.mean(prediction["dw_fast"][start:])
            * 180.0
            / math.pi
            * 60.0
        ),
        "mean_slow_yaw_correction_radps": float(
            np.mean(prediction["dw_slow"][start:])
        ),
        "mean_scale_delta": float(
            np.mean(prediction["scale_delta"][start:])
        ),
        "mean_yaw_bias_radps": float(
            np.mean(prediction["yaw_bias"][start:])
        ),
        "mean_yaw_bias_deg_per_min": float(
            np.mean(prediction["yaw_bias"][start:])
            * 180.0
            / math.pi
            * 60.0
        ),
    }



def aggregate_results(
    rows: list[dict],
    pilot_folds: tuple[str, ...],
) -> list[dict]:
    output = []

    for sequence in pilot_folds:
        subset = [
            row
            for row in rows
            if row["test_sequence"] == sequence
        ]

        if not subset:
            continue

        def values(field):
            return np.asarray(
                [float(row[field]) for row in subset],
                dtype=float,
            )

        v1_ate = values("v1_ate_rmse_m")
        v2_ate = values("v2_ate_rmse_m")

        row = {
            "test_sequence": sequence,
            "n_seeds": len(subset),

            "v1_ate_mean_m": float(np.mean(v1_ate)),
            "v1_ate_std_m": (
                float(np.std(v1_ate, ddof=1))
                if len(v1_ate) > 1 else 0.0
            ),
            "v1_ate_range_m": float(np.ptp(v1_ate)),

            "v2_ate_mean_m": float(np.mean(v2_ate)),
            "v2_ate_std_m": (
                float(np.std(v2_ate, ddof=1))
                if len(v2_ate) > 1 else 0.0
            ),
            "v2_ate_range_m": float(np.ptp(v2_ate)),

            "mean_ate_change_pct": float(
                100.0
                * (np.mean(v2_ate) - np.mean(v1_ate))
                / np.mean(v1_ate)
            ),

            "mean_abs_remaining_yaw_bias_deg_min": float(
                np.mean(
                    np.abs(
                        values(
                            "v2_remaining_yaw_bias_deg_per_min"
                        )
                    )
                )
            ),

            "mean_abs_fast_component_deg_min": float(
                np.mean(
                    np.abs(
                        values(
                            "v2_mean_fast_yaw_residual_deg_per_min"
                        )
                    )
                )
            ),

            "mean_abs_physical_bias_deg_min": float(
                np.mean(
                    np.abs(
                        values(
                            "v2_mean_yaw_bias_deg_per_min"
                        )
                    )
                )
            ),

            "mean_scale_delta_pct": float(
                100.0
                * np.mean(
                    values("v2_mean_scale_delta")
                )
            ),
        }

        for horizon in (1, 5, 10):
            v1 = values(f"v1_rpe_{horizon}s_m")
            v2 = values(f"v2_rpe_{horizon}s_m")

            row[f"v1_rpe_{horizon}s_mean_m"] = float(
                np.mean(v1)
            )
            row[f"v2_rpe_{horizon}s_mean_m"] = float(
                np.mean(v2)
            )
            row[f"rpe_{horizon}s_change_pct"] = float(
                100.0
                * (np.mean(v2) - np.mean(v1))
                / np.mean(v1)
            )

        output.append(row)

    return output

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("public_datasets/im2nav"),
    )
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=Path("results/i2nav_v1_frozen"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/i2nav_v2_physical_yaw_pilot"),
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--folds",
        default="parking02,parking01,street00",
        help=(
            "Comma-separated pilot folds. "
            "Use parking02 for a one-fold smoke."
        ),
    )
    parser.add_argument(
        "--base-seeds",
        default="42,1042,2042",
        help=(
            "Comma-separated subset of frozen replicate base seeds. "
            "Allowed: 42,1042,2042."
        ),
    )

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument(
        "--fast-window",
        type=int,
        default=20,
        help="2 s at 10 Hz.",
    )
    parser.add_argument(
        "--slow-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--train-stride",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--validation-stride",
        type=int,
        default=100,
    )

    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--slow-hidden-size", type=int, default=32)

    parser.add_argument("--dv-limit", type=float, default=0.15)
    parser.add_argument("--dw-fast-limit", type=float, default=0.020)
    parser.add_argument(
        "--scale-delta-limit",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--yaw-bias-limit",
        type=float,
        default=0.005,
    )

    # One fixed, physics-motivated loss configuration for the pilot.
    parser.add_argument("--point-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-weight", type=float, default=0.50)
    parser.add_argument("--physical-weight", type=float, default=0.75)
    parser.add_argument("--fast-target-weight", type=float, default=0.25)
    parser.add_argument("--fast-mean-weight", type=float, default=0.25)
    parser.add_argument("--slow-smooth-weight", type=float, default=0.05)
    parser.add_argument("--long-heading-weight", type=float, default=0.50)

    parser.add_argument(
        "--ranger-wheelbase-m",
        type=float,
        default=0.494,
    )
    parser.add_argument(
        "--ranger-track-m",
        type=float,
        default=0.370,
    )

    args = parser.parse_args()

    pilot_folds = tuple(
        x.strip()
        for x in args.folds.split(",")
        if x.strip()
    )
    base_seeds = tuple(
        int(x.strip())
        for x in args.base_seeds.split(",")
        if x.strip()
    )

    allowed_folds = set(PILOT_FOLDS)
    bad_folds = [
        x for x in pilot_folds
        if x not in allowed_folds
    ]
    if bad_folds:
        raise ValueError(
            f"Unsupported pilot folds: {bad_folds}. "
            f"Allowed: {PILOT_FOLDS}"
        )

    seed_to_replicate_number = {
        42: 1,
        1042: 2,
        2042: 3,
    }
    bad_seeds = [
        x for x in base_seeds
        if x not in seed_to_replicate_number
    ]
    if bad_seeds:
        raise ValueError(
            f"Unsupported base seeds: {bad_seeds}. "
            f"Allowed: {tuple(seed_to_replicate_number)}"
        )

    root = args.root.resolve()
    frozen_dir = args.frozen_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = base.resolve_device(args.device)

    original = importlib.import_module(
        "DigitalTwin.analysis.i2nav_loso_ablation"
    )
    defaults = base.original_default_args(original)

    rate = float(defaults.rate_hz)
    dt = 1.0 / rate

    slow_window_samples = int(
        round(args.slow_seconds * rate)
    )
    chunk_steps = int(
        round(args.chunk_seconds * rate)
    ) + 1

    if args.chunk_seconds < 30.0:
        raise RuntimeError("chunk-seconds must be >= 30.")

    if slow_window_samples <= args.fast_window:
        raise RuntimeError(
            "Slow physical context must be longer than fast context."
        )

    print()
    print("=" * 100)
    print("TWIN V2 PHYSICAL-YAW PILOT — TWO TIMESCALES / ZERO-SHOT-SAFE INPUTS")
    print("=" * 100)
    print(f"Dataset          : {root}")
    print(f"Frozen V1        : {frozen_dir}")
    print(f"Output           : {output_dir}")
    print(f"Device           : {device}")
    print(f"Rate             : {rate:.3f} Hz")
    print(f"Fast context     : {args.fast_window / rate:.1f} s")
    print(f"Slow context     : {args.slow_seconds:.1f} s")
    print(f"Pilot folds      : {pilot_folds}")
    print(f"Base seeds       : {base_seeds}")

    discovered_list = original.discover_files(root)
    discovered = {
        item.name: item
        for item in discovered_list
    }

    prepared = {}

    print()
    print("Preparing exact V1 sequences...")

    for name in original.SEQUENCES:
        if name not in discovered:
            raise RuntimeError(
                f"Dataset discovery did not find {name}."
            )

        sequence = original.prepare_sequence(
            discovered[name],
            hz=defaults.rate_hz,
            imu_yaw_sign=defaults.imu_yaw_sign,
            gnss_sigma_max_m=defaults.gnss_sigma_max_m,
            gnss_anchor_count=defaults.gnss_anchor_count,
        )

        base.validate_sequence(sequence)
        prepared[name] = sequence

        print(
            f"  {name:<12} {len(sequence.grid):6d} samples"
        )

    # ------------------------------------------------------------------
    # Build canonical physical signals and training-only physical targets.
    # ------------------------------------------------------------------
    canonical = {}
    slow_features_raw = {}
    target_scale = {}
    target_bias = {}
    target_gate = {}

    print()
    print("Building canonical wheel/IMU signals and 30 s physical summaries...")

    for name in original.SEQUENCES:
        seq = prepared[name]

        sig = i2nav_ranger_to_canonical(
            seq,
            root,
            wheelbase_m=args.ranger_wheelbase_m,
            track_m=args.ranger_track_m,
            angle_sign=1.0,
        )

        corr = np.corrcoef(
            sig.wheel_yaw_radps,
            sig.imu_yaw_radps,
        )[0, 1]

        if not np.isfinite(corr) or corr < 0.50:
            raise RuntimeError(
                f"{name}: canonical wheel-yaw sanity check failed; "
                f"corr(wheel_yaw, imu_yaw)={corr:.3f}. "
                f"Do not train with a malformed adapter."
            )

        canonical[name] = sig

        slow_features_raw[name] = build_slow_physical_features(
            sig,
            samples=slow_window_samples,
        )

        (
            target_scale[name],
            target_bias[name],
            target_gate[name],
        ) = build_affine_yaw_targets(
            seq.imu_yaw_rate,
            seq.gt_yaw_rate,
            samples=slow_window_samples,
            scale_delta_limit=args.scale_delta_limit,
            bias_limit_radps=args.yaw_bias_limit,
        )

        print(
            f"  {name:<12} "
            f"wheel/IMU yaw corr={corr:+.3f}"
        )

    manifest_lookup = base.frozen_manifest_lookup(
        frozen_dir
    )
    metric_lookup = base.frozen_metric_lookup(
        frozen_dir
    )

    results = []

    for test_name in pilot_folds:
        print()
        print("=" * 100)
        print(f"FOLD {test_name}")
        print("=" * 100)

        training_names, validation_names = original.build_fold_split(
            test_name,
            int(defaults.validation_count),
        )

        print(f"Train      : {training_names}")
        print(f"Validation : {validation_names}")
        print(f"Test       : {test_name}")

        fast_feature_mean, fast_feature_std = base.feature_normalization(
            prepared,
            training_names,
        )

        slow_feature_mean, slow_feature_std = slow_feature_normalization(
            slow_features_raw,
            training_names,
            valid_from=slow_window_samples - 1,
        )

        required_names = set(
            training_names
            + validation_names
            + [test_name]
        )

        caches = {
            name: build_cache(
                prepared[name],
                fast_feature_mean=fast_feature_mean,
                fast_feature_std=fast_feature_std,
                slow_features=slow_features_raw[name],
                slow_feature_mean=slow_feature_mean,
                slow_feature_std=slow_feature_std,
                target_scale_delta=target_scale[name],
                target_bias_radps=target_bias[name],
                scale_gate=target_gate[name],
                wheel_yaw_radps=canonical[name].wheel_yaw_radps,
                yaw_disagreement_radps=(
                    canonical[name].yaw_disagreement_radps
                ),
                fast_window=args.fast_window,
            )
            for name in required_names
        }

        train_dataset = PhysicalChunkDataset(
            caches,
            training_names,
            fast_window=args.fast_window,
            slow_window_samples=slow_window_samples,
            chunk_steps=chunk_steps,
            stride=args.train_stride,
        )

        validation_dataset = PhysicalChunkDataset(
            caches,
            validation_names,
            fast_window=args.fast_window,
            slow_window_samples=slow_window_samples,
            chunk_steps=chunk_steps,
            stride=args.validation_stride,
        )

        print(
            f"Training chunks   : {len(train_dataset)}"
        )
        print(
            f"Validation chunks : {len(validation_dataset)}"
        )

        for base_seed in base_seeds:
            replicate_number = seed_to_replicate_number[
                base_seed
            ]

            replicate = (
                f"replicate_{replicate_number:02d}_"
                f"base{base_seed}"
            )

            actual_seed = (
                base_seed
                + original.SEQUENCES.index(test_name) * 100
                + 23
            )

            base.seed_everything(actual_seed)

            print()
            print(
                f"  [replicate {replicate_number}; "
                f"{base_seeds.index(base_seed) + 1}/"
                f"{len(base_seeds)} selected] "
                f"{replicate} seed={actual_seed}"
            )

            generator = torch.Generator()
            generator.manual_seed(actual_seed)

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
                generator=generator,
            )

            validation_loader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

            model = V2PhysicalYaw(
                fast_input_dim=int(
                    prepared[test_name].features.shape[1]
                ),
                slow_input_dim=int(
                    slow_features_raw[test_name].shape[1]
                ),
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                slow_hidden_size=args.slow_hidden_size,
                dv_limit=args.dv_limit,
                dw_fast_limit=args.dw_fast_limit,
                scale_delta_limit=args.scale_delta_limit,
                yaw_bias_limit=args.yaw_bias_limit,
            ).to(device)

            loss_kwargs = {
                "dt": dt,
                "point_weight": args.point_weight,
                "trajectory_weight": args.trajectory_weight,
                "physical_weight": args.physical_weight,
                "fast_target_weight": args.fast_target_weight,
                "fast_mean_weight": args.fast_mean_weight,
                "slow_smooth_weight": args.slow_smooth_weight,
                "long_heading_weight": args.long_heading_weight,
            }

            start_time = time.time()

            (
                best_state,
                history,
                best_validation_loss,
            ) = train_model(
                model,
                train_loader,
                validation_loader,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                loss_kwargs=loss_kwargs,
            )

            elapsed = time.time() - start_time

            model.load_state_dict(
                best_state,
                strict=True,
            )
            model.eval()

            fold_number = (
                original.SEQUENCES.index(test_name)
                + 1
            )

            run_dir = (
                output_dir
                / replicate
                / f"fold_{fold_number:02d}_{test_name}"
            )
            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            base.write_csv(
                run_dir / "training_history.csv",
                history,
            )

            torch.save(
                {
                    "schema": "i2nav_twin_v2_physical_yaw_v1",
                    "state_dict": best_state,
                    "fast_feature_mean": fast_feature_mean,
                    "fast_feature_std": fast_feature_std,
                    "slow_feature_mean": slow_feature_mean,
                    "slow_feature_std": slow_feature_std,
                    "fast_window": args.fast_window,
                    "slow_seconds": args.slow_seconds,
                    "slow_window_samples": slow_window_samples,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "slow_hidden_size": args.slow_hidden_size,
                    "dv_limit": args.dv_limit,
                    "dw_fast_limit": args.dw_fast_limit,
                    "scale_delta_limit": args.scale_delta_limit,
                    "yaw_bias_limit": args.yaw_bias_limit,
                    "best_validation_loss": best_validation_loss,
                    "base_seed": base_seed,
                    "actual_seed": actual_seed,
                    "test_sequence": test_name,
                    "training_names": training_names,
                    "validation_names": validation_names,
                    "zero_shot_policy": {
                        "canonical_encoder_imu_only": True,
                        "target_domain_normalization_fitting": False,
                        "target_domain_finetuning": False,
                    },
                },
                run_dir / "v2_physical_yaw.pt",
            )

            test_sequence = prepared[test_name]
            test_cache = caches[test_name]

            prediction = predict_sequence(
                model,
                test_cache,
                fast_window=args.fast_window,
                slow_window_samples=slow_window_samples,
                batch_size=int(defaults.eval_batch_size),
                device=device,
            )

            save_prediction_trace(
                run_dir / "v2_prediction_trace.csv",
                test_sequence,
                test_cache,
                prediction,
            )

            frozen_key = (
                replicate,
                test_name,
            )

            if frozen_key not in manifest_lookup:
                raise RuntimeError(
                    f"Frozen manifest missing {frozen_key}."
                )

            alphas = base.frozen_v1_alphas(
                original,
                test_sequence,
                frozen_dir,
                manifest_lookup[frozen_key],
                device,
                int(defaults.eval_batch_size),
            )

            evaluation = original.evaluate_predictions(
                fold=fold_number,
                method="v2_physical_yaw",
                sequence=test_sequence,
                training_names=training_names,
                validation_names=validation_names,
                corrections=prediction["corrections"],
                alphas=alphas,
                args=defaults,
                trajectory_path=(
                    run_dir / "v2_evaluated_trajectory.csv"
                ),
            )

            metrics = vars(evaluation)

            diagnostic = yaw_diagnostics(
                test_sequence,
                prediction,
                slow_window_samples=slow_window_samples,
            )

            if frozen_key not in metric_lookup:
                raise RuntimeError(
                    f"Frozen metric table missing {frozen_key}."
                )

            v1 = metric_lookup[frozen_key]

            result = {
                "replicate": replicate,
                "base_seed": base_seed,
                "actual_v2_seed": actual_seed,
                "test_sequence": test_name,
                "training_seconds": elapsed,
                "best_validation_loss": best_validation_loss,

                "v1_ate_rmse_m": float(v1["ate_rmse_m"]),
                "v1_heading_mae_deg": float(v1["heading_mae_deg"]),
                "v1_rpe_1s_m": float(v1["rpe_1s_trans_rmse_m"]),
                "v1_rpe_5s_m": float(v1["rpe_5s_trans_rmse_m"]),
                "v1_rpe_10s_m": float(v1["rpe_10s_trans_rmse_m"]),

                "v2_ate_rmse_m": float(metrics["ate_rmse_m"]),
                "v2_heading_mae_deg": float(metrics["heading_mae_deg"]),
                "v2_rpe_1s_m": float(
                    metrics["rpe_1s_trans_rmse_m"]
                ),
                "v2_rpe_5s_m": float(
                    metrics["rpe_5s_trans_rmse_m"]
                ),
                "v2_rpe_10s_m": float(
                    metrics["rpe_10s_trans_rmse_m"]
                ),

                "v2_remaining_yaw_bias_radps": (
                    diagnostic["remaining_yaw_bias_radps"]
                ),
                "v2_remaining_yaw_bias_deg_per_min": (
                    diagnostic["remaining_yaw_bias_deg_per_min"]
                ),
                "v2_remaining_yaw_rmse_radps": (
                    diagnostic["remaining_yaw_rmse_radps"]
                ),

                "v2_mean_fast_yaw_residual_radps": (
                    diagnostic["mean_fast_yaw_residual_radps"]
                ),
                "v2_mean_fast_yaw_residual_deg_per_min": (
                    diagnostic[
                        "mean_fast_yaw_residual_deg_per_min"
                    ]
                ),

                "v2_mean_slow_yaw_correction_radps": (
                    diagnostic[
                        "mean_slow_yaw_correction_radps"
                    ]
                ),
                "v2_mean_scale_delta": (
                    diagnostic["mean_scale_delta"]
                ),
                "v2_mean_yaw_bias_radps": (
                    diagnostic["mean_yaw_bias_radps"]
                ),
                "v2_mean_yaw_bias_deg_per_min": (
                    diagnostic["mean_yaw_bias_deg_per_min"]
                ),
            }

            result["ate_change_pct"] = (
                100.0
                * (
                    result["v2_ate_rmse_m"]
                    - result["v1_ate_rmse_m"]
                )
                / result["v1_ate_rmse_m"]
            )

            results.append(result)

            base.write_json(
                run_dir / "run_summary.json",
                result,
            )

            # Partial results survive interruption.
            base.write_csv(
                output_dir / "pilot_run_results.csv",
                results,
            )

            print()
            print("      RESULT")
            print(
                f"      V1 ATE = "
                f"{result['v1_ate_rmse_m']:.4f} m"
            )
            print(
                f"      V2 ATE = "
                f"{result['v2_ate_rmse_m']:.4f} m "
                f"({result['ate_change_pct']:+.1f}%)"
            )
            print(
                f"      V2 remaining yaw bias = "
                f"{result['v2_remaining_yaw_bias_deg_per_min']:+.3f} "
                f"deg/min"
            )
            print(
                f"      fast residual mean = "
                f"{result['v2_mean_fast_yaw_residual_deg_per_min']:+.3f} "
                f"deg/min"
            )
            print(
                f"      physical scale delta = "
                f"{100.0 * result['v2_mean_scale_delta']:+.3f}%"
            )
            print(
                f"      physical bias = "
                f"{result['v2_mean_yaw_bias_deg_per_min']:+.3f} "
                f"deg/min"
            )

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate = aggregate_results(results, pilot_folds)

    base.write_csv(
        output_dir / "pilot_aggregate_results.csv",
        aggregate,
    )

    base.write_json(
        output_dir / "pilot_configuration.json",
        {
            "schema": "i2nav_twin_v2_physical_yaw_v1",
            "pilot_folds": pilot_folds,
            "base_seeds": base_seeds,
            "rate_hz": rate,
            "fast_window": args.fast_window,
            "fast_seconds": args.fast_window / rate,
            "slow_seconds": args.slow_seconds,
            "chunk_seconds": args.chunk_seconds,
            "epochs": args.epochs,
            "patience": args.patience,
            "model": {
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "slow_hidden_size": args.slow_hidden_size,
                "dv_limit": args.dv_limit,
                "dw_fast_limit": args.dw_fast_limit,
                "scale_delta_limit": args.scale_delta_limit,
                "yaw_bias_limit": args.yaw_bias_limit,
            },
            "loss": {
                "point_weight": args.point_weight,
                "trajectory_weight": args.trajectory_weight,
                "physical_weight": args.physical_weight,
                "fast_target_weight": args.fast_target_weight,
                "fast_mean_weight": args.fast_mean_weight,
                "slow_smooth_weight": args.slow_smooth_weight,
                "long_heading_weight": args.long_heading_weight,
            },
            "zero_shot_contract": {
                "learned_inputs": (
                    "original six V1 ODO+IMU fast features plus "
                    "canonical encoder+IMU slow physical summaries"
                ),
                "ranger_specific_raw_channels_enter_learned_core": False,
                "external_normalization_refit_allowed": False,
                "external_finetuning_allowed_for_zero_shot_claim": False,
            },
        },
    )

    print()
    print("=" * 100)
    print("PILOT COMPLETE")
    print("=" * 100)

    for row in aggregate:
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
