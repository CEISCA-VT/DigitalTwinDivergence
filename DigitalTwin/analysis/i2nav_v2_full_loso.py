"""Frozen Twin V2 one-run launcher for the full i2Nav LOSO study.

This file is a direct extraction of the successful
DigitalTwin_i2Nav_V2_SlowAdditive_SensorConsistency_Kaggle_NO_COMMIT notebook.
It intentionally preserves the pilot architecture, losses, contexts, split
logic, seed logic, normalization policy, and original i2Nav evaluation path.

One invocation trains/evaluates exactly ONE held-out sequence and ONE base
seed.  The intended HPC execution is therefore a 30-task SLURM array:
10 held-out sequences x 3 base seeds.

Frozen V2 specification from the successful pilot:
  * 10 Hz
  * 2 s / 20-sample fast history using the exact six V1 features
  * 30 s canonical wheel/IMU/ODO sensor-consistency slow history
  * 2-layer GRU, hidden 64, dropout 0.10
  * slow MLP hidden 32
  * dv limit 0.15 m/s
  * fast yaw correction limit 0.020 rad/s
  * slow additive yaw state limit 0.005 rad/s
  * NO learned yaw scale
  * 25 epochs, patience 5, batch size 8
  * train stride 50, validation stride 100
  * AdamW(lr=1e-3, weight_decay=1e-5)
  * V2 actual seed = base_seed + 100*fold_index + 23

The frozen Twin V1 directory is treated as read-only.  A temporary per-job
compatibility copy is made only to normalize Windows separators in the old
manifest before the original frozen-V1 alpha/Q replay helper is called.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from DigitalTwin.analysis import i2nav_v2_yaw_bias_pilot as base
from DigitalTwin.analysis.i2nav_fidelity_evaluator import (
    evaluate_trajectory_files,
    write_json as write_fidelity_json,
    write_timeseries as write_fidelity_timeseries,
)


original = importlib.import_module("DigitalTwin.analysis.i2nav_loso_ablation")

# ---------------------------------------------------------------------------
# Frozen experiment specification -- copied from the successful pilot.
# ---------------------------------------------------------------------------
PILOT_REFERENCE_COMMIT = "2e8710f405cdd63fc0fd7960950d038077696eb9"
SCHEMA = "i2nav_twin_v2_slow_additive_sensor_consistency_v1"

FAST_WINDOW = 20
SLOW_SECONDS = 30.0
CHUNK_SECONDS = 30.0

EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 8
TRAIN_STRIDE = 50
VALIDATION_STRIDE = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.10
SLOW_HIDDEN_SIZE = 32
DV_LIMIT = 0.15
DW_FAST_LIMIT = 0.020
SLOW_BIAS_LIMIT = 0.005

BASE_SEEDS = (42, 1042, 2042)
SEED_TO_REPLICATE = {
    42: "replicate_01_base42",
    1042: "replicate_02_base1042",
    2042: "replicate_03_base2042",
}

LOSS_WEIGHTS = {
    "point": 1.0,
    "trajectory": 0.50,
    "persistent": 1.0,
    "slow_target": 0.75,
    "fast_target": 0.25,
    "fast_magnitude": 0.01,
    "fast_mean": 0.25,
    "slow_smooth": 0.05,
}


# These globals are set exactly once in main() after reading the repository's
# authoritative default args.  Keeping the constants here makes the extracted
# training path match the notebook while still supporting command-line paths.
RATE: float
DT: float
SLOW_SAMPLES: int
CHUNK_STEPS: int
DEVICE: torch.device


# ---------------------------------------------------------------------------
# Small I/O / provenance helpers.
# ---------------------------------------------------------------------------
def write_json(path: Path | str, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def repo_root() -> Path:
    # .../DigitalTwin/analysis/i2nav_v2_full_loso.py -> repository root
    return Path(__file__).resolve().parents[2]


def git_state(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", True


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    data: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "hostname": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    if device.type == "cuda":
        data["gpu_name"] = torch.cuda.get_device_name(device)
        data["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    else:
        data["gpu_name"] = None
        data["gpu_capability"] = None
    return data


@contextmanager
def frozen_v1_runtime_copy(frozen_source: Path):
    """Create a per-process read/write compatibility copy of frozen V1.

    The source directory is never modified.  This preserves the exact notebook
    workaround for old Windows path separators while being safe for concurrent
    SLURM array jobs.
    """
    frozen_source = Path(frozen_source).resolve()
    if not frozen_source.exists():
        raise FileNotFoundError(f"Frozen V1 directory not found: {frozen_source}")

    temp_parent = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
    kwargs: dict[str, Any] = {"prefix": "i2nav_v1_runtime_"}
    if temp_parent and Path(temp_parent).exists():
        kwargs["dir"] = temp_parent

    with tempfile.TemporaryDirectory(**kwargs) as tmp:
        runtime = Path(tmp) / "i2nav_v1_frozen"
        shutil.copytree(frozen_source, runtime)

        manifest_path = runtime / "FROZEN_MANIFEST.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing frozen manifest: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs = manifest.get("runs", [])
        if len(runs) != 30:
            raise RuntimeError(f"Expected 30 frozen V1 runs; found {len(runs)}")

        for run in runs:
            run["frozen_checkpoint"] = str(run["frozen_checkpoint"]).replace("\\", "/")

        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        checkpoint_paths = [runtime / run["frozen_checkpoint"] for run in runs]
        missing = [str(path) for path in checkpoint_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Frozen checkpoint(s) missing:\n" + "\n".join(missing))
        if len({str(path.resolve()) for path in checkpoint_paths}) != 30:
            raise RuntimeError("Frozen manifest does not resolve to 30 unique checkpoints")

        yield runtime


# ---------------------------------------------------------------------------
# Deterministic Ranger -> canonical sensor adapter (exact notebook logic).
# ---------------------------------------------------------------------------
def read_numeric_text(path: Path, min_cols: int = 9) -> np.ndarray:
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#") or text.startswith("%"):
                continue
            values = np.fromstring(text.replace(",", " "), sep=" ")
            if len(values) >= min_cols and np.all(np.isfinite(values[:min_cols])):
                rows.append(values[:min_cols])
    if not rows:
        raise RuntimeError(f"No usable numeric rows in {path}")
    data = np.asarray(rows, dtype=np.float64)
    return data[np.argsort(data[:, 0], kind="stable")]


def find_ranger_odo(data_root: Path, sequence_name: str) -> Path:
    candidates = list((data_root / sequence_name).glob("*_RANGER_ODO.txt"))
    if not candidates:
        candidates = [
            path
            for path in data_root.rglob("*_RANGER_ODO.txt")
            if path.name.startswith(sequence_name)
        ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"{sequence_name}: expected exactly one *_RANGER_ODO.txt; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def ranger_wheel_positions(wheelbase_m: float = 0.494, track_m: float = 0.370) -> np.ndarray:
    x = float(wheelbase_m) / 2.0
    y = float(track_m) / 2.0
    # Internal frame is x-forward, y-right. Raw order: RF, LF, RB, LB.
    return np.asarray([[x, y], [x, -y], [-x, y], [-x, -y]], dtype=np.float64)


def solve_planar_twist_batch(
    speeds_mps: np.ndarray,
    steering_rad: np.ndarray,
    positions_xy_m: np.ndarray,
) -> np.ndarray:
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    angles = np.asarray(steering_rad, dtype=np.float64)
    pos = np.asarray(positions_xy_m, dtype=np.float64)
    if speeds.ndim != 2 or speeds.shape[1] != 4:
        raise ValueError(f"Expected speeds (N,4); got {speeds.shape}")
    if angles.shape != speeds.shape:
        raise ValueError(f"Steering/speed mismatch: {angles.shape} vs {speeds.shape}")
    if pos.shape != (4, 2):
        raise ValueError(f"Expected wheel positions (4,2); got {pos.shape}")

    c = np.cos(angles)
    s = np.sin(angles)
    matrix = np.empty((len(speeds), 4, 3), dtype=np.float64)
    matrix[:, :, 0] = c
    matrix[:, :, 1] = s
    matrix[:, :, 2] = -c * pos[None, :, 1] + s * pos[None, :, 0]
    ata = np.einsum("nij,nik->njk", matrix, matrix)
    atb = np.einsum("nij,ni->nj", matrix, speeds)
    ata[:, np.arange(3), np.arange(3)] += 1e-8
    twist = np.linalg.solve(ata, atb[..., None])[..., 0]
    if twist.shape != (len(speeds), 3) or not np.all(np.isfinite(twist)):
        raise RuntimeError(f"Invalid wheel-kinematic solution shape={twist.shape}")
    return twist


def interp_to_grid(source_t: np.ndarray, source_v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    source_t = np.asarray(source_t, dtype=np.float64)
    source_v = np.asarray(source_v, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if source_v.ndim == 1:
        return np.interp(grid, source_t, source_v)
    return np.column_stack(
        [np.interp(grid, source_t, source_v[:, index]) for index in range(source_v.shape[1])]
    )


def build_canonical_signals(sequence: Any, data_root: Path) -> dict[str, Any]:
    ranger_path = find_ranger_odo(data_root, sequence.name)
    raw = read_numeric_text(ranger_path, min_cols=9)
    t_raw = raw[:, 0]
    wheel_speeds = raw[:, 1:5]
    steering_angles = raw[:, 5:9]

    twist_internal = solve_planar_twist_batch(
        wheel_speeds,
        steering_angles,
        ranger_wheel_positions(),
    )
    twist = interp_to_grid(t_raw, twist_internal, sequence.grid)
    wheel_forward = twist[:, 0]
    # x-forward/y-right internal solve has the opposite positive-yaw convention
    # from PreparedSequence.imu_yaw_rate. This is the audited pilot sign fix.
    wheel_yaw = -twist[:, 2]

    imu_yaw = np.asarray(sequence.imu_yaw_rate, dtype=np.float64)
    odo_forward = np.asarray(sequence.odo_speed, dtype=np.float64)
    disagreement = imu_yaw - wheel_yaw
    normalized_disagreement = disagreement / (
        np.abs(imu_yaw) + np.abs(wheel_yaw) + 0.02
    )

    return {
        "time_s": np.asarray(sequence.grid, dtype=np.float64),
        "wheel_forward_mps": wheel_forward,
        "wheel_yaw_radps": wheel_yaw,
        "imu_yaw_radps": imu_yaw,
        "odo_forward_mps": odo_forward,
        "yaw_disagreement_radps": disagreement,
        "yaw_disagreement_normalized": normalized_disagreement,
        "ranger_path": str(ranger_path),
    }


def rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rolling(int(samples), min_periods=int(samples))
        .mean()
        .to_numpy()
    )


def rolling_std(values: np.ndarray, samples: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rolling(int(samples), min_periods=int(samples))
        .std(ddof=0)
        .to_numpy()
    )


def rolling_rms(values: np.ndarray, samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.maximum(rolling_mean(values * values, samples), 0.0))


def build_slow_sensor_features(signals: dict[str, Any], samples: int) -> np.ndarray:
    """Inference-available canonical sensor/odometry summaries only."""
    imu = signals["imu_yaw_radps"]
    wheel = signals["wheel_yaw_radps"]
    diff = signals["yaw_disagreement_radps"]
    ndiff = signals["yaw_disagreement_normalized"]
    speed = signals["odo_forward_mps"]
    columns = [
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
    return np.column_stack(columns).astype(np.float32)


def build_slow_bias_target(sequence: Any, samples: int) -> np.ndarray:
    """TRAINING ONLY: causal 30-s mean of exact V1 yaw residual, clipped."""
    true_dw = np.asarray(sequence.target_corrections[:, 1], dtype=np.float64)
    target = rolling_mean(true_dw, samples)
    return np.clip(target, -SLOW_BIAS_LIMIT, SLOW_BIAS_LIMIT).astype(np.float32)


# ---------------------------------------------------------------------------
# Cache, dataset, and frozen V2 model.
# ---------------------------------------------------------------------------
@dataclass
class SlowBiasCache:
    name: str
    fast_windows: np.ndarray
    slow_features: np.ndarray
    target: np.ndarray
    slow_bias_target: np.ndarray
    odo_speed: np.ndarray
    imu_yaw_rate: np.ndarray
    gt_x: np.ndarray
    gt_y: np.ndarray
    gt_heading: np.ndarray
    wheel_yaw_radps: np.ndarray
    yaw_disagreement_radps: np.ndarray
    grid: np.ndarray


def slow_feature_normalization(
    slow_features_raw: dict[str, np.ndarray], training_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    blocks = []
    for name in training_names:
        values = np.asarray(slow_features_raw[name], dtype=np.float32)[SLOW_SAMPLES - 1 :]
        values = values[np.all(np.isfinite(values), axis=1)]
        if len(values):
            blocks.append(values)
    if not blocks:
        raise RuntimeError("No valid slow sensor features in training split")
    merged = np.concatenate(blocks, axis=0)
    mean = np.mean(merged, axis=0).astype(np.float32)
    std = np.maximum(np.std(merged, axis=0).astype(np.float32), 1e-4)
    return mean, std


def build_cache(
    sequence: Any,
    fast_mean: np.ndarray,
    fast_std: np.ndarray,
    slow_mean: np.ndarray,
    slow_std: np.ndarray,
    slow_features_raw: dict[str, np.ndarray],
    slow_bias_targets: dict[str, np.ndarray],
    canonical: dict[str, dict[str, Any]],
) -> SlowBiasCache:
    fast_normalized = (
        np.asarray(sequence.features, dtype=np.float32) - fast_mean[None, :]
    ) / fast_std[None, :]
    slow_normalized = (
        np.asarray(slow_features_raw[sequence.name], dtype=np.float32) - slow_mean[None, :]
    ) / slow_std[None, :]
    slow_normalized = np.nan_to_num(
        slow_normalized, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    return SlowBiasCache(
        name=sequence.name,
        fast_windows=base.sliding_windows(fast_normalized, FAST_WINDOW),
        slow_features=slow_normalized,
        target=np.asarray(sequence.target_corrections, dtype=np.float32),
        slow_bias_target=np.asarray(slow_bias_targets[sequence.name], dtype=np.float32),
        odo_speed=np.asarray(sequence.odo_speed, dtype=np.float32),
        imu_yaw_rate=np.asarray(sequence.imu_yaw_rate, dtype=np.float32),
        gt_x=np.asarray(sequence.gt_x, dtype=np.float32),
        gt_y=np.asarray(sequence.gt_y, dtype=np.float32),
        gt_heading=np.asarray(sequence.gt_heading, dtype=np.float32),
        wheel_yaw_radps=np.asarray(canonical[sequence.name]["wheel_yaw_radps"], dtype=np.float32),
        yaw_disagreement_radps=np.asarray(
            canonical[sequence.name]["yaw_disagreement_radps"], dtype=np.float32
        ),
        grid=np.asarray(sequence.grid, dtype=np.float64),
    )


class SlowBiasChunkDataset(Dataset):
    def __init__(self, caches: dict[str, SlowBiasCache], names: list[str], stride: int):
        self.caches = caches
        self.items: list[tuple[str, int]] = []
        self.first_valid = max(FAST_WINDOW - 1, SLOW_SAMPLES - 1)
        for name in names:
            cache = caches[name]
            last_start_exclusive = len(cache.grid) - CHUNK_STEPS + 1
            for start in range(self.first_valid, last_start_exclusive, int(stride)):
                self.items.append((name, start))
        if not self.items:
            raise RuntimeError("No SlowBiasChunkDataset samples were generated")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        name, start = self.items[index]
        cache = self.caches[name]
        end = start + CHUNK_STEPS
        fast_window_start = start - (FAST_WINDOW - 1)
        return {
            "fast_windows": torch.from_numpy(
                cache.fast_windows[fast_window_start : fast_window_start + CHUNK_STEPS]
            ),
            "slow_features": torch.from_numpy(cache.slow_features[start:end]),
            "target": torch.from_numpy(cache.target[start:end]),
            "slow_bias_target": torch.from_numpy(cache.slow_bias_target[start:end]),
            "odo_speed": torch.from_numpy(cache.odo_speed[start:end]),
            "imu_yaw_rate": torch.from_numpy(cache.imu_yaw_rate[start:end]),
            "gt_x": torch.from_numpy(cache.gt_x[start:end]),
            "gt_y": torch.from_numpy(cache.gt_y[start:end]),
            "gt_heading": torch.from_numpy(cache.gt_heading[start:end]),
        }


class V2SlowAdditiveYaw(nn.Module):
    """2-s fast GRU + 30-s sensor-consistency slow additive yaw state."""

    def __init__(self, fast_input_dim: int, slow_input_dim: int):
        super().__init__()
        self.dv_limit = float(DV_LIMIT)
        self.dw_fast_limit = float(DW_FAST_LIMIT)
        self.slow_bias_limit = float(SLOW_BIAS_LIMIT)

        self.fast_gru = nn.GRU(
            input_size=int(fast_input_dim),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.fast_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.dv_head = nn.Linear(HIDDEN_SIZE, 1)
        self.dw_fast_head = nn.Linear(HIDDEN_SIZE, 1)
        self.slow_mlp = nn.Sequential(
            nn.Linear(int(slow_input_dim), SLOW_HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(SLOW_HIDDEN_SIZE, SLOW_HIDDEN_SIZE),
            nn.Tanh(),
        )
        self.slow_bias_head = nn.Linear(SLOW_HIDDEN_SIZE, 1)
        nn.init.zeros_(self.dv_head.bias)
        nn.init.zeros_(self.dw_fast_head.bias)
        nn.init.zeros_(self.slow_bias_head.bias)
        nn.init.normal_(self.slow_bias_head.weight, mean=0.0, std=1e-3)

    def forward(self, fast_windows: torch.Tensor, slow_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if fast_windows.ndim == 4:
            batch, horizon, window, features = fast_windows.shape
            fast_x = fast_windows.reshape(batch * horizon, window, features)
            slow_x = slow_features.reshape(batch * horizon, slow_features.shape[-1])
            restore = (batch, horizon)
        elif fast_windows.ndim == 3:
            fast_x = fast_windows
            slow_x = slow_features
            restore = None
        else:
            raise RuntimeError(f"Unexpected fast-window shape {fast_windows.shape}")

        _, hidden = self.fast_gru(fast_x)
        fast_latent = self.fast_norm(hidden[-1])
        dv = self.dv_limit * torch.tanh(self.dv_head(fast_latent)).squeeze(-1)
        dw_fast = self.dw_fast_limit * torch.tanh(self.dw_fast_head(fast_latent)).squeeze(-1)
        slow_latent = self.slow_mlp(slow_x)
        b_slow = self.slow_bias_limit * torch.tanh(self.slow_bias_head(slow_latent)).squeeze(-1)
        dw = b_slow + dw_fast

        if restore is not None:
            batch, horizon = restore
            dv = dv.reshape(batch, horizon)
            dw_fast = dw_fast.reshape(batch, horizon)
            b_slow = b_slow.reshape(batch, horizon)
            dw = dw.reshape(batch, horizon)
        return {"dv": dv, "dw_fast": dw_fast, "b_slow": b_slow, "dw": dw}


def move_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
        for key, value in batch.items()
    }


# ---------------------------------------------------------------------------
# Exact frozen pilot losses and training loop.
# ---------------------------------------------------------------------------
def compute_loss(model: V2SlowAdditiveYaw, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    pred = model(batch["fast_windows"], batch["slow_features"])
    dv = pred["dv"]
    dw_fast = pred["dw_fast"]
    b_slow = pred["b_slow"]
    dw = pred["dw"]
    true_dv = batch["target"][:, :, 0]
    true_dw = batch["target"][:, :, 1]
    target_slow = batch["slow_bias_target"]

    point_dv = torch.mean(((dv - true_dv) / 0.05) ** 2)
    point_dw = torch.mean(((dw - true_dw) / 0.03) ** 2)
    point_loss = point_dv + point_dw

    slow_target_loss = torch.mean(((b_slow - target_slow) / 0.001) ** 2)
    target_fast = true_dw - target_slow
    fast_target_loss = torch.mean(((dw_fast - target_fast) / 0.02) ** 2)

    corrected_v = batch["odo_speed"] + dv
    corrected_w = batch["imu_yaw_rate"] + dw
    pred_x, pred_y, pred_heading = base.propagate_chunk(
        corrected_v,
        corrected_w,
        batch["gt_x"],
        batch["gt_y"],
        batch["gt_heading"],
        DT,
    )

    trajectory_terms = []
    for seconds in (1.0, 5.0, 10.0):
        index = int(round(seconds / DT))
        if index >= pred_x.shape[1]:
            continue
        position_sq = (
            (pred_x[:, index] - batch["gt_x"][:, index]) ** 2
            + (pred_y[:, index] - batch["gt_y"][:, index]) ** 2
        )
        heading_error = base.wrap_tensor(
            pred_heading[:, index] - batch["gt_heading"][:, index]
        )
        trajectory_terms.append(
            torch.mean(position_sq)
            + torch.mean((heading_error / math.radians(5.0)) ** 2)
        )
    trajectory_loss = torch.stack(trajectory_terms).mean()

    remaining_dw = true_dw - dw
    persistent_terms = []
    for seconds in (5.0, 10.0, 30.0):
        steps = min(int(round(seconds / DT)), remaining_dw.shape[1])
        mean_signed = torch.mean(remaining_dw[:, :steps], dim=1)
        persistent_terms.append(torch.mean((mean_signed / 0.001) ** 2))
    persistent_loss = torch.stack(persistent_terms).mean()

    fast_mean_terms = []
    for seconds in (5.0, 10.0, 30.0):
        steps = min(int(round(seconds / DT)), dw_fast.shape[1])
        mean_fast = torch.mean(dw_fast[:, :steps], dim=1)
        fast_mean_terms.append(torch.mean((mean_fast / 0.001) ** 2))
    fast_mean_loss = torch.stack(fast_mean_terms).mean()
    fast_magnitude_loss = torch.mean((dw_fast / 0.01) ** 2)

    if b_slow.shape[1] > 1:
        slow_smooth_loss = torch.mean(
            ((b_slow[:, 1:] - b_slow[:, :-1]) / 0.0002) ** 2
        )
    else:
        slow_smooth_loss = torch.zeros((), dtype=b_slow.dtype, device=b_slow.device)

    total = (
        LOSS_WEIGHTS["point"] * point_loss
        + LOSS_WEIGHTS["trajectory"] * trajectory_loss
        + LOSS_WEIGHTS["persistent"] * persistent_loss
        + LOSS_WEIGHTS["slow_target"] * slow_target_loss
        + LOSS_WEIGHTS["fast_target"] * fast_target_loss
        + LOSS_WEIGHTS["fast_magnitude"] * fast_magnitude_loss
        + LOSS_WEIGHTS["fast_mean"] * fast_mean_loss
        + LOSS_WEIGHTS["slow_smooth"] * slow_smooth_loss
    )
    parts = {
        "total": float(total.detach().cpu()),
        "point": float(point_loss.detach().cpu()),
        "trajectory": float(trajectory_loss.detach().cpu()),
        "persistent": float(persistent_loss.detach().cpu()),
        "slow_target": float(slow_target_loss.detach().cpu()),
        "fast_target": float(fast_target_loss.detach().cpu()),
        "fast_magnitude": float(fast_magnitude_loss.detach().cpu()),
        "fast_mean": float(fast_mean_loss.detach().cpu()),
        "slow_smooth": float(slow_smooth_loss.detach().cpu()),
    }
    return total, parts


def validation_loss(model: V2SlowAdditiveYaw, loader: DataLoader) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch)
            _, parts = compute_loss(model, batch)
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
            count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


def train_model(
    model: V2SlowAdditiveYaw,
    train_loader: DataLoader,
    validation_loader: DataLoader,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        sums: dict[str, float] = {}
        count = 0
        for batch in train_loader:
            batch = move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = compute_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
            count += 1

        train_mean = {key: value / max(count, 1) for key, value in sums.items()}
        val_mean = validation_loss(model, validation_loader)
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_mean.items()},
                **{f"val_{key}": value for key, value in val_mean.items()},
            }
        )
        print(
            f"epoch {epoch:02d} train={train_mean['total']:.4f} "
            f"val={val_mean['total']:.4f} slow={val_mean['slow_target']:.4f} "
            f"persist={val_mean['persistent']:.4f} fastmean={val_mean['fast_mean']:.4f}",
            flush=True,
        )

        if val_mean["total"] < best_val - 1e-6:
            best_val = float(val_mean["total"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"early stopping: best val={best_val:.4f}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")
    return best_state, history, best_val


# ---------------------------------------------------------------------------
# Prediction and diagnostics -- exact pilot inference schedule.
# ---------------------------------------------------------------------------
def predict_sequence(
    model: V2SlowAdditiveYaw,
    cache: SlowBiasCache,
    eval_batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    n = len(cache.grid)
    fast_offset = FAST_WINDOW - 1
    slow_offset = SLOW_SAMPLES - 1
    full = {
        key: np.zeros(n, dtype=np.float32)
        for key in ("dv", "dw_fast", "b_slow", "dw")
    }

    # 2 s -> 30 s: fast correction can run, persistent state remains neutral.
    if slow_offset > fast_offset:
        fast_windows = cache.fast_windows[0 : slow_offset - fast_offset]
        for start in range(0, len(fast_windows), int(eval_batch_size)):
            xb = torch.from_numpy(
                fast_windows[start : start + int(eval_batch_size)]
            ).to(device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                _, hidden = model.fast_gru(xb)
                latent = model.fast_norm(hidden[-1])
                dv = model.dv_limit * torch.tanh(model.dv_head(latent)).squeeze(-1)
                dw_fast = model.dw_fast_limit * torch.tanh(
                    model.dw_fast_head(latent)
                ).squeeze(-1)
            dst0 = fast_offset + start
            dst1 = dst0 + len(dv)
            full["dv"][dst0:dst1] = dv.detach().cpu().numpy()
            full["dw_fast"][dst0:dst1] = dw_fast.detach().cpu().numpy()
            full["dw"][dst0:dst1] = dw_fast.detach().cpu().numpy()

    indices_all = np.arange(slow_offset, n, dtype=int)
    for start in range(0, len(indices_all), int(eval_batch_size)):
        indices = indices_all[start : start + int(eval_batch_size)]
        fast_indices = indices - fast_offset
        fast_x = torch.from_numpy(cache.fast_windows[fast_indices]).to(
            device=DEVICE, dtype=torch.float32
        )
        slow_x = torch.from_numpy(cache.slow_features[indices]).to(
            device=DEVICE, dtype=torch.float32
        )
        with torch.no_grad():
            out = model(fast_x, slow_x)
        for key in full:
            full[key][indices] = out[key].detach().cpu().numpy().astype(np.float32)

    full["corrections"] = np.column_stack([full["dv"], full["dw"]]).astype(np.float32)
    return full


def post30_yaw_diagnostics(sequence: Any, cache: SlowBiasCache, prediction: dict[str, np.ndarray]) -> dict[str, float]:
    """Pilot-compatible mechanistic diagnostics after full 30-s slow context.

    These are NOT the DT heading envelope.  Accumulated yaw-rate residual is
    explicitly named Iomega-like diagnostic rather than heading divergence.
    """
    start = SLOW_SAMPLES - 1
    true_dw = np.asarray(sequence.target_corrections[start:, 1], dtype=np.float64)
    pred_dw = np.asarray(prediction["dw"][start:], dtype=np.float64)
    residual = true_dw - pred_dw
    fast = np.asarray(prediction["dw_fast"][start:], dtype=np.float64)
    slow = np.asarray(prediction["b_slow"][start:], dtype=np.float64)
    sensor_disagreement = np.asarray(cache.yaw_disagreement_radps[start:], dtype=np.float64)
    accumulated_deg = np.degrees(np.cumsum(residual * DT))
    deg_per_min = 180.0 / math.pi * 60.0
    return {
        "post30_remaining_yaw_bias_radps": float(np.mean(residual)),
        "post30_remaining_yaw_bias_deg_per_min": float(np.mean(residual) * deg_per_min),
        "post30_remaining_yaw_rmse_radps": float(np.sqrt(np.mean(residual ** 2))),
        "post30_mean_fast_yaw_residual_deg_per_min": float(np.mean(fast) * deg_per_min),
        "post30_mean_slow_bias_deg_per_min": float(np.mean(slow) * deg_per_min),
        "post30_sensor_disagreement_mean_deg_per_min": float(
            np.mean(sensor_disagreement) * deg_per_min
        ),
        "post30_sensor_disagreement_rms_radps": float(
            np.sqrt(np.mean(sensor_disagreement ** 2))
        ),
        "post30_Iomega_final_deg": float(accumulated_deg[-1]),
        "post30_Iomega_p95_abs_deg": float(np.percentile(np.abs(accumulated_deg), 95.0)),
        "post30_Iomega_max_abs_deg": float(np.max(np.abs(accumulated_deg))),
    }


def save_prediction_trace(
    path: Path,
    sequence: Any,
    cache: SlowBiasCache,
    prediction: dict[str, np.ndarray],
) -> None:
    rows: list[dict[str, Any]] = []
    true_target = np.asarray(sequence.target_corrections, dtype=np.float64)
    accumulated_yaw_residual = 0.0
    for index in range(len(sequence.grid)):
        remaining_dw = float(true_target[index, 1] - prediction["dw"][index])
        accumulated_yaw_residual += remaining_dw * DT
        rows.append(
            {
                "time_s": float(sequence.grid[index]),
                "true_delta_v_mps": float(true_target[index, 0]),
                "pred_delta_v_mps": float(prediction["dv"][index]),
                "true_delta_omega_radps": float(true_target[index, 1]),
                "pred_total_delta_omega_radps": float(prediction["dw"][index]),
                "pred_slow_bias_radps": float(prediction["b_slow"][index]),
                "pred_fast_yaw_residual_radps": float(prediction["dw_fast"][index]),
                "remaining_yaw_error_radps": remaining_dw,
                "wheel_yaw_radps": float(cache.wheel_yaw_radps[index]),
                "imu_yaw_radps": float(cache.imu_yaw_rate[index]),
                "wheel_imu_yaw_disagreement_radps": float(
                    cache.yaw_disagreement_radps[index]
                ),
                "accumulated_yaw_residual_deg": float(
                    math.degrees(accumulated_yaw_residual)
                ),
            }
        )
    write_csv(path, rows)


def assert_fidelity_matches_original(
    profile: dict[str, Any], metrics: dict[str, Any], atol: float = 1e-8
) -> None:
    checks = {
        "ATE_m": float(metrics["ate_rmse_m"]),
        "heading_MAE_deg": float(metrics["heading_mae_deg"]),
        "RPEp_1s_m": float(metrics["rpe_1s_trans_rmse_m"]),
        "RPEp_5s_m": float(metrics["rpe_5s_trans_rmse_m"]),
        "RPEp_10s_m": float(metrics["rpe_10s_trans_rmse_m"]),
    }
    failed = []
    for key, expected in checks.items():
        actual = float(profile[key])
        if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=atol):
            failed.append((key, actual, expected))
    if failed:
        raise RuntimeError(
            "Independent fidelity evaluator does not match original evaluator: "
            + "; ".join(f"{key}: {actual} vs {expected}" for key, actual, expected in failed)
        )


# ---------------------------------------------------------------------------
# One fold x one seed execution.
# ---------------------------------------------------------------------------
def prepare_all_sequences(data_root: Path, defaults: Any) -> dict[str, Any]:
    discovered_list = original.discover_files(data_root)
    discovered = {item.name: item for item in discovered_list}
    prepared: dict[str, Any] = {}
    print("Preparing exact V1 sequences from current repository...", flush=True)
    for name in original.SEQUENCES:
        if name not in discovered:
            raise RuntimeError(f"Dataset discovery did not find {name}")
        sequence = original.prepare_sequence(
            discovered[name],
            hz=defaults.rate_hz,
            imu_yaw_sign=defaults.imu_yaw_sign,
            gnss_sigma_max_m=defaults.gnss_sigma_max_m,
            gnss_anchor_count=defaults.gnss_anchor_count,
        )
        base.validate_sequence(sequence)
        prepared[name] = sequence
        print(f"  {name:<12} {len(sequence.grid):6d} samples", flush=True)

    # Hard target-definition replay from the successful notebook.
    for name in original.SEQUENCES:
        sequence = prepared[name]
        ev = np.asarray(sequence.gt_forward_speed) - np.asarray(sequence.odo_speed)
        ew = np.asarray(sequence.gt_yaw_rate) - np.asarray(sequence.imu_yaw_rate)
        if np.max(np.abs(ev - sequence.target_corrections[:, 0])) >= 1e-6:
            raise RuntimeError(f"{name}: velocity target identity failed")
        if np.max(np.abs(ew - sequence.target_corrections[:, 1])) >= 1e-6:
            raise RuntimeError(f"{name}: yaw target identity failed")
    print("PreparedSequence target identities: PASS", flush=True)
    return prepared


def build_all_canonical(
    prepared: dict[str, Any], data_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    canonical: dict[str, dict[str, Any]] = {}
    slow_features_raw: dict[str, np.ndarray] = {}
    slow_bias_targets: dict[str, np.ndarray] = {}
    for name in original.SEQUENCES:
        sequence = prepared[name]
        signals = build_canonical_signals(sequence, data_root)
        canonical[name] = signals
        yaw_corr = float(np.corrcoef(signals["wheel_yaw_radps"], signals["imu_yaw_radps"])[0, 1])
        vx_corr = float(
            np.corrcoef(signals["wheel_forward_mps"], signals["odo_forward_mps"])[0, 1]
        )
        if not np.isfinite(yaw_corr) or yaw_corr < 0.80:
            raise RuntimeError(f"{name}: wheel-yaw sanity failed: corr={yaw_corr:.3f}")
        if not np.isfinite(vx_corr) or vx_corr < 0.99:
            raise RuntimeError(f"{name}: wheel-forward sanity failed: corr={vx_corr:.3f}")
        slow_features_raw[name] = build_slow_sensor_features(signals, SLOW_SAMPLES)
        slow_bias_targets[name] = build_slow_bias_target(sequence, SLOW_SAMPLES)
    print("Canonical sensor adapter sanity: PASS", flush=True)
    return canonical, slow_features_raw, slow_bias_targets


def run_one(args: argparse.Namespace) -> Path:
    global RATE, DT, SLOW_SAMPLES, CHUNK_STEPS, DEVICE

    data_root = args.root.resolve()
    frozen_source = args.frozen_v1_dir.resolve()
    output_root = args.output_dir.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"i2Nav root not found: {data_root}")
    if not frozen_source.exists():
        raise FileNotFoundError(f"Frozen V1 directory not found: {frozen_source}")

    defaults = base.original_default_args(original)
    RATE = float(defaults.rate_hz)
    DT = 1.0 / RATE
    SLOW_SAMPLES = int(round(SLOW_SECONDS * RATE))
    CHUNK_STEPS = int(round(CHUNK_SECONDS * RATE)) + 1

    if args.test_sequence not in original.SEQUENCES:
        raise ValueError(
            f"Unknown --test-sequence {args.test_sequence!r}; expected one of {list(original.SEQUENCES)}"
        )
    if args.base_seed not in BASE_SEEDS:
        raise ValueError(f"Unsupported base seed {args.base_seed}; allowed {BASE_SEEDS}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    DEVICE = torch.device(args.device)

    test_name = args.test_sequence
    base_seed = int(args.base_seed)
    replicate = SEED_TO_REPLICATE[base_seed]
    fold_index = original.SEQUENCES.index(test_name)
    fold_number = fold_index + 1
    actual_seed = base_seed + fold_index * 100 + 23

    run_dir = output_root / replicate / f"fold_{fold_number:02d}_{test_name}"
    complete_marker = run_dir / "RUN_COMPLETE.json"
    if complete_marker.exists() and not args.overwrite:
        print(f"Completed run already exists; skipping: {run_dir}")
        return run_dir
    if run_dir.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Partial/existing run directory found: {run_dir}. "
                "Use --overwrite to replace it after checking the prior SLURM log."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    root = repo_root()
    commit, dirty = git_state(root)
    manifest = {
        "schema": SCHEMA,
        "status": "running",
        "pilot_reference_commit": PILOT_REFERENCE_COMMIT,
        "repo_commit": commit,
        "repo_dirty": dirty,
        "test_sequence": test_name,
        "fold": fold_number,
        "replicate": replicate,
        "base_seed": base_seed,
        "actual_v2_seed": actual_seed,
        "frozen_specification": {
            "rate_hz": RATE,
            "fast_window_samples": FAST_WINDOW,
            "fast_seconds": FAST_WINDOW / RATE,
            "slow_seconds": SLOW_SECONDS,
            "slow_samples": SLOW_SAMPLES,
            "chunk_seconds": CHUNK_SECONDS,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "train_stride": TRAIN_STRIDE,
            "validation_stride": VALIDATION_STRIDE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "slow_hidden_size": SLOW_HIDDEN_SIZE,
            "dv_limit": DV_LIMIT,
            "dw_fast_limit": DW_FAST_LIMIT,
            "slow_bias_limit": SLOW_BIAS_LIMIT,
            "loss_weights": LOSS_WEIGHTS,
            "persistent_model": "single bounded additive slow yaw state; no learned yaw scale",
        },
        "runtime": runtime_metadata(DEVICE),
    }
    write_json(run_dir / "run_manifest.json", manifest)

    print("=" * 88)
    print(f"Twin V2 frozen LOSO run: fold={fold_number:02d} sequence={test_name}")
    print(f"replicate={replicate} base_seed={base_seed} actual_seed={actual_seed}")
    print(f"device={DEVICE}")
    if DEVICE.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(DEVICE)}")
    print(f"repo_commit={commit} dirty={dirty}")
    print("=" * 88, flush=True)

    prepared = prepare_all_sequences(data_root, defaults)
    canonical, slow_features_raw, slow_bias_targets = build_all_canonical(prepared, data_root)

    training_names, validation_names = original.build_fold_split(
        test_name, int(defaults.validation_count)
    )
    print("Train      :", training_names)
    print("Validation :", validation_names)
    print("Test       :", test_name)

    fast_mean, fast_std = base.feature_normalization(prepared, training_names)
    slow_mean, slow_std = slow_feature_normalization(slow_features_raw, training_names)
    required_names = set(training_names + validation_names + [test_name])
    caches = {
        name: build_cache(
            prepared[name],
            fast_mean,
            fast_std,
            slow_mean,
            slow_std,
            slow_features_raw,
            slow_bias_targets,
            canonical,
        )
        for name in required_names
    }
    train_dataset = SlowBiasChunkDataset(caches, training_names, stride=TRAIN_STRIDE)
    validation_dataset = SlowBiasChunkDataset(
        caches, validation_names, stride=VALIDATION_STRIDE
    )
    print("Training chunks   :", len(train_dataset))
    print("Validation chunks :", len(validation_dataset), flush=True)

    base.seed_everything(actual_seed)
    generator = torch.Generator()
    generator.manual_seed(actual_seed)
    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = V2SlowAdditiveYaw(
        fast_input_dim=prepared[test_name].features.shape[1],
        slow_input_dim=slow_features_raw[test_name].shape[1],
    ).to(DEVICE)

    start_time = time.time()
    best_state, history, best_val = train_model(model, train_loader, validation_loader)
    training_seconds = float(time.time() - start_time)
    model.load_state_dict(best_state, strict=True)
    model.eval()
    write_csv(run_dir / "training_history.csv", history)

    checkpoint = {
        "schema": SCHEMA,
        "pilot_reference_commit": PILOT_REFERENCE_COMMIT,
        "repo_commit": commit,
        "state_dict": best_state,
        "fast_feature_mean": fast_mean,
        "fast_feature_std": fast_std,
        "slow_feature_mean": slow_mean,
        "slow_feature_std": slow_std,
        "fast_window": FAST_WINDOW,
        "slow_seconds": SLOW_SECONDS,
        "slow_samples": SLOW_SAMPLES,
        "slow_feature_dim": slow_features_raw[test_name].shape[1],
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "slow_hidden_size": SLOW_HIDDEN_SIZE,
        "dv_limit": DV_LIMIT,
        "dw_fast_limit": DW_FAST_LIMIT,
        "slow_bias_limit": SLOW_BIAS_LIMIT,
        "loss_weights": LOSS_WEIGHTS,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "train_stride": TRAIN_STRIDE,
        "validation_stride": VALIDATION_STRIDE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_validation_loss": best_val,
        "base_seed": base_seed,
        "actual_seed": actual_seed,
        "test_sequence": test_name,
        "training_names": training_names,
        "validation_names": validation_names,
        "slow_target_definition": "causal 30-second mean of training-only true yaw correction",
        "zero_shot_contract": {
            "canonical_encoder_imu_inputs_only": True,
            "ranger_raw_channels_in_learned_core": False,
            "target_domain_normalization_refit": False,
            "target_domain_finetuning": False,
        },
    }
    torch.save(checkpoint, run_dir / "v2_slow_additive_yaw.pt")

    test_sequence = prepared[test_name]
    test_cache = caches[test_name]
    prediction = predict_sequence(model, test_cache, int(defaults.eval_batch_size))
    prediction_trace_path = run_dir / "v2_prediction_trace.csv"
    save_prediction_trace(prediction_trace_path, test_sequence, test_cache, prediction)

    with frozen_v1_runtime_copy(frozen_source) as frozen_runtime:
        manifest_lookup = base.frozen_manifest_lookup(frozen_runtime)
        metric_lookup = base.frozen_metric_lookup(frozen_runtime)
        if len(metric_lookup) != 30:
            raise RuntimeError(f"Expected 30 frozen V1 metric rows; found {len(metric_lookup)}")
        frozen_key = (replicate, test_name)
        if frozen_key not in manifest_lookup or frozen_key not in metric_lookup:
            raise RuntimeError(f"Frozen V1 evidence missing {frozen_key}")

        alphas = base.frozen_v1_alphas(
            original,
            test_sequence,
            frozen_runtime,
            manifest_lookup[frozen_key],
            DEVICE,
            int(defaults.eval_batch_size),
        )
        trajectory_path = run_dir / "v2_evaluated_trajectory.csv"
        evaluation = original.evaluate_predictions(
            fold=fold_number,
            method="v2_slow_additive",
            sequence=test_sequence,
            training_names=training_names,
            validation_names=validation_names,
            corrections=prediction["corrections"],
            alphas=alphas,
            args=defaults,
            trajectory_path=trajectory_path,
        )
        metrics = vars(evaluation)
        v1 = metric_lookup[frozen_key]

    diagnostic = post30_yaw_diagnostics(test_sequence, test_cache, prediction)

    fidelity_profile, fidelity_timeseries = evaluate_trajectory_files(
        trajectory_path,
        prediction_trace_path,
        model="V2",
        sequence=test_name,
        seed=actual_seed,
        replicate=replicate,
    )
    assert_fidelity_matches_original(fidelity_profile, metrics)
    write_fidelity_json(run_dir / "fidelity_profile.json", fidelity_profile)
    write_fidelity_timeseries(run_dir / "fidelity_timeseries.csv", fidelity_timeseries)

    result = {
        "schema": SCHEMA,
        "replicate": replicate,
        "base_seed": base_seed,
        "actual_v2_seed": actual_seed,
        "test_sequence": test_name,
        "fold": fold_number,
        "training_seconds": training_seconds,
        "best_validation_loss": float(best_val),
        "v1_ate_rmse_m": float(v1["ate_rmse_m"]),
        "v1_heading_mae_deg": float(v1["heading_mae_deg"]),
        "v1_rpe_1s_m": float(v1["rpe_1s_trans_rmse_m"]),
        "v1_rpe_5s_m": float(v1["rpe_5s_trans_rmse_m"]),
        "v1_rpe_10s_m": float(v1["rpe_10s_trans_rmse_m"]),
        "v2_ate_rmse_m": float(metrics["ate_rmse_m"]),
        "v2_heading_mae_deg": float(metrics["heading_mae_deg"]),
        "v2_rpe_1s_m": float(metrics["rpe_1s_trans_rmse_m"]),
        "v2_rpe_5s_m": float(metrics["rpe_5s_trans_rmse_m"]),
        "v2_rpe_10s_m": float(metrics["rpe_10s_trans_rmse_m"]),
        **diagnostic,
        "fidelity_Dp_p95_m": float(fidelity_profile["Dp_p95_m"]),
        "fidelity_Dp_max_m": float(fidelity_profile["Dp_max_m"]),
        "fidelity_Dtheta_p95_deg": float(fidelity_profile["Dtheta_p95_deg"]),
        "fidelity_Dtheta_max_deg": float(fidelity_profile["Dtheta_max_deg"]),
        "fidelity_abs_yaw_bias_deg_per_min": fidelity_profile[
            "abs_yaw_bias_deg_per_min"
        ],
        "fidelity_Iomega_max_abs_deg": fidelity_profile["Iomega_max_abs_deg"],
    }
    result["ate_change_pct"] = float(
        100.0
        * (result["v2_ate_rmse_m"] - result["v1_ate_rmse_m"])
        / result["v1_ate_rmse_m"]
    )
    write_json(run_dir / "run_summary.json", result)

    manifest["status"] = "complete"
    manifest["training_names"] = list(training_names)
    manifest["validation_names"] = list(validation_names)
    manifest["training_seconds"] = training_seconds
    manifest["best_validation_loss"] = float(best_val)
    manifest["artifacts"] = {
        "checkpoint": "v2_slow_additive_yaw.pt",
        "training_history": "training_history.csv",
        "prediction_trace": "v2_prediction_trace.csv",
        "evaluated_trajectory": "v2_evaluated_trajectory.csv",
        "fidelity_profile": "fidelity_profile.json",
        "fidelity_timeseries": "fidelity_timeseries.csv",
        "run_summary": "run_summary.json",
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(
        complete_marker,
        {
            "status": "complete",
            "repo_commit": commit,
            "test_sequence": test_name,
            "replicate": replicate,
            "base_seed": base_seed,
            "actual_v2_seed": actual_seed,
        },
    )

    print("\nRESULT")
    print(f"V1 ATE = {result['v1_ate_rmse_m']:.6f} m")
    print(
        f"V2 ATE = {result['v2_ate_rmse_m']:.6f} m "
        f"({result['ate_change_pct']:+.2f}%)"
    )
    print(f"V2 heading MAE = {result['v2_heading_mae_deg']:.6f} deg")
    print(f"V2 Dp p95 = {result['fidelity_Dp_p95_m']:.6f} m")
    print(f"V2 Dtheta p95 = {result['fidelity_Dtheta_p95_deg']:.6f} deg")
    print(f"Independent fidelity replay: PASS")
    print(f"Run directory: {run_dir}", flush=True)

    del model, train_loader, validation_loader
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one frozen Twin V2 i2Nav LOSO fold/seed for SLURM-array execution."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("public_datasets/im2nav"),
        help="i2Nav dataset root.",
    )
    parser.add_argument(
        "--frozen-v1-dir",
        type=Path,
        default=Path("results/i2nav_v1_frozen"),
        help="Read-only frozen Twin V1 evidence directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/i2nav_v2_full_loso"),
        help="Parent directory shared by the 30 independent runs.",
    )
    parser.add_argument("--test-sequence", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing partial/completed run directory. Use deliberately.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
