#!/usr/bin/env python3
"""
i2nav_predictive_nll_pilot_v3.py
==============================

One reproducible master study for the next three i2Nav steps:

PHASE 1 -- Official i2Nav/evo evaluation mechanics
    * 5 ms timestamp association
    * SE(3) trajectory alignment, no scale correction
    * APE translation + rotation RMSE
    * distance-based RPE at 50/100/150/200/250/300 m, all pairs,
      relative-delta tolerance 0.002
    * retains the project's own 1/5/10 s RPE separately

PHASE 2 -- Predictive dual-GRU dynamics model (V3)
    * trusted ODO/IMU features only
    * dual heads: bounded dynamics correction + heteroscedastic residual uncertainty
    * horizon-weighted differentiable rollout loss at 1 s / 5 s / 10 s
    * uncertainty head predicts physical sigma_v / sigma_omega and is trained by Gaussian NLL
    * outer leave-one-sequence-out (LOSO), using the same frozen folds when available
    * train-only normalization, target scaling, and correction limits
    * validation-only checkpoint selection; outer test sequence never used for tuning

PHASE 3 -- Residual-uncertainty/Q isolation and calibration
    * SAME refined dual checkpoint evaluated with:
          (A) learned Q from predicted sigma_v / sigma_omega propagated through the motion Jacobian
          (B) frozen V5 fixed Q
      No retraining between A and B.
    * NIS coverage at 95%/99%
    * posterior 3-state NEES coverage
    * 2-D position covariance ellipse coverage
    * sigma_v / sigma_omega distribution/saturation
    * GNSS gate/reacquisition statistics

The completed prior LOSO experiment is never overwritten. This script reads it as a
frozen baseline and writes everything to a new directory.

Recommended project location:
    DigitalTwin/analysis/i2nav_predictive_nll_pilot_v3.py

Typical run:
    python -m DigitalTwin.analysis.i2nav_final_model_study \
        --root public_datasets/im2nav \
        --frozen-loso-dir results/i2nav_loso_ablation \
        --device cuda

Fast smoke test:
    python -m DigitalTwin.analysis.i2nav_final_model_study \
        --root public_datasets/im2nav \
        --folds parking02 \
        --epochs 3 \
        --train-stride 20 \
        --device cuda

Official i2Nav evaluator source replicated here:
    https://github.com/i2Nav-WHU/evaluate_odometry

Security rule:
    GNSS position, innovation, NIS, covariance, residuals, and reported GNSS
    uncertainty NEVER enter the neural network. Ground truth is used only for
    supervised training targets on train/validation sequences and for evaluation.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Install a CUDA build matching your NVIDIA driver, "
        "then rerun this script."
    ) from exc

# evo is optional at import time so the training/calibration study can still run,
# but Phase 1 will explicitly report unavailable if evo is missing.
try:
    from evo.core import metrics as evo_metrics
    from evo.core import sync as evo_sync
    from evo.core import trajectory as evo_trajectory

    EVO_AVAILABLE = True
except Exception:
    evo_metrics = None
    evo_sync = None
    evo_trajectory = None
    EVO_AVAILABLE = False

try:
    from scipy.stats import chi2 as scipy_chi2
except Exception:
    scipy_chi2 = None


# =============================================================================
# Constants
# =============================================================================

KNOWN_SEQUENCES = (
    "building00",
    "building01",
    "building02",
    "parking00",
    "parking01",
    "parking02",
    "playground00",
    "street00",
    "street01",
    "street02",
)

FEATURE_NAMES = (
    "odo_speed_mps",
    "imu_yaw_rate_radps",
    "odo_accel_mps2",
    "imu_yaw_accel_radps2",
    "abs_yaw_rate_radps",
    "abs_odo_accel_mps2",
)

# Exact constants from the public i2Nav-WHU/evaluate_odometry evaluate.py.
OFFICIAL_MAX_TIME_SYNC_DIFF_S = 0.005
OFFICIAL_RPE_DELTAS_M = (50, 100, 150, 200, 250, 300)
OFFICIAL_RPE_REL_DELTA_TOL = 0.002
OFFICIAL_RPE_ALL_PAIRS = True

# Chi-square thresholds used for calibration diagnostics.
CHI2_2_95 = 5.991464547107979
CHI2_2_99 = 9.210340371976184
CHI2_3_95 = 7.814727903251179
CHI2_3_025 = 0.21579528262389797
CHI2_3_975 = 9.348403604496148

EARTH_RADIUS_M = 6_378_137.0

DEFAULT_FOLDS = (
    (1, "building00", ("building01", "building02")),
    (2, "building01", ("building02", "parking00")),
    (3, "building02", ("parking00", "parking01")),
    (4, "parking00", ("parking01", "parking02")),
    (5, "parking01", ("parking02", "playground00")),
    (6, "parking02", ("playground00", "street00")),
    (7, "playground00", ("street00", "street01")),
    (8, "street00", ("street01", "street02")),
    (9, "street01", ("street02", "building00")),
    (10, "street02", ("building00", "building01")),
)


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class SequenceFiles:
    name: str
    groundtruth: Path
    imu: Path
    odo_speed: Path | None
    ranger_odo: Path | None
    gnss: Path | None
    gnss_source: str
    official_truth_trajectory: Path | None


@dataclass
class PreparedSequence:
    name: str
    grid: np.ndarray
    gt_xyz: np.ndarray
    gt_heading: np.ndarray
    gt_forward_speed: np.ndarray
    gt_yaw_rate: np.ndarray
    speed: np.ndarray
    omega: np.ndarray
    features: np.ndarray
    target_delta_v: np.ndarray
    target_delta_omega: np.ndarray
    gnss: dict[str, np.ndarray] | None
    gnss_source: str
    odo_source: str
    official_truth: np.ndarray
    official_truth_source: str


@dataclass
class FoldDefinition:
    fold: int
    test: str
    validation: list[str]
    train: list[str]


@dataclass
class StandardMetrics:
    ate_rmse_m: float
    ate_median_m: float
    ate_p95_m: float
    ate_max_m: float
    ate_se2_rmse_m: float
    heading_mae_deg: float
    heading_p95_deg: float
    rpe_1s_trans_rmse_m: float
    rpe_5s_trans_rmse_m: float
    rpe_10s_trans_rmse_m: float
    final_error_m: float
    final_error_se2_m: float
    final_drift_per_m: float
    path_length_ratio: float


# =============================================================================
# Generic utilities
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def wrap_array(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def wrap_tensor(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def nan() -> float:
    return float("nan")


def safe_float(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else nan()
    except Exception:
        return nan()


def mean_finite(values: Iterable[float]) -> float | None:
    a = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    return float(np.mean(a)) if len(a) else None


def percentile_finite(values: Iterable[float], p: float) -> float | None:
    a = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    return float(np.percentile(a, p)) if len(a) else None


def path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig", errors="ignore") as f:
        return list(csv.DictReader(f))


def read_numeric_table(path: Path, min_cols: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            line = line.replace(",", " ")
            toks = line.split()
            if len(toks) < min_cols:
                continue
            try:
                vals = [float(x) for x in toks]
            except ValueError:
                continue
            if len(vals) >= min_cols and np.all(np.isfinite(vals[:min_cols])):
                rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric rows with >= {min_cols} columns in {path}")
    width = min(len(r) for r in rows)
    return np.asarray([r[:width] for r in rows], dtype=float)


def sorted_unique_by_time(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a[:, 0])
    a = a[order]
    _, rev_idx = np.unique(a[::-1, 0], return_index=True)
    keep = len(a) - 1 - rev_idx
    keep.sort()
    return a[keep]


def interp_angle(t_src: np.ndarray, angle_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return wrap_array(np.interp(t_dst, t_src, np.unwrap(angle_src)))


def make_grid(gt_t: np.ndarray, odo_t: np.ndarray, imu_t: np.ndarray, hz: float) -> np.ndarray:
    start = max(float(gt_t[0]), float(odo_t[0]), float(imu_t[0]))
    end = min(float(gt_t[-1]), float(odo_t[-1]), float(imu_t[-1]))
    if end <= start:
        raise ValueError(f"No common GT/ODO/IMU time interval: {start}..{end}")
    dt = 1.0 / hz
    start = math.ceil(start / dt) * dt
    end = math.floor(end / dt) * dt
    n = int(round((end - start) / dt)) + 1
    return start + np.arange(n, dtype=float) * dt


def geodetic_to_local_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    east = EARTH_RADIUS_M * math.cos(lat0) * (lon - lon0)
    north = EARTH_RADIUS_M * (lat - lat0)
    return east, north


def yaw_to_xyzw(yaw: np.ndarray | float) -> np.ndarray:
    y = np.asarray(yaw, dtype=float)
    q = np.zeros(y.shape + (4,), dtype=float)
    q[..., 2] = np.sin(0.5 * y)
    q[..., 3] = np.cos(0.5 * y)
    return q


def euler_ned_to_xyzw(
    roll: np.ndarray | float,
    pitch: np.ndarray | float,
    yaw: np.ndarray | float,
) -> np.ndarray:
    """Match i2Nav-WHU/evaluate_odometry nav2traj.py Euler->quaternion convention.

    Inputs are NED/FRD roll, pitch, yaw in radians. Output order is TUM xyzw.
    """
    phi, theta, psi = np.broadcast_arrays(
        np.asarray(roll, dtype=float),
        np.asarray(pitch, dtype=float),
        np.asarray(yaw, dtype=float),
    )
    sphi, cphi = np.sin(0.5 * phi), np.cos(0.5 * phi)
    stheta, ctheta = np.sin(0.5 * theta), np.cos(0.5 * theta)
    spsi, cpsi = np.sin(0.5 * psi), np.cos(0.5 * psi)

    qw = cphi * ctheta * cpsi + sphi * stheta * spsi
    qx = sphi * ctheta * cpsi - cphi * stheta * spsi
    qy = cphi * stheta * cpsi + sphi * ctheta * spsi
    qz = cphi * ctheta * spsi - sphi * stheta * cpsi
    q = np.stack([qx, qy, qz, qw], axis=-1)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-12)


# =============================================================================
# Dataset discovery/loading
# =============================================================================

def _closest_match(root: Path, patterns: Sequence[str]) -> Path | None:
    """Return the best match while PRESERVING caller pattern priority.

    The v1 final-study helper pooled matches from every pattern and then globally
    sorted them. That accidentally defeated the documented OEM719-before-F9P
    preference because the shorter F9P path could win. Search one pattern at a
    time instead; only tie-break among matches for the same priority pattern.
    """
    for pattern in patterns:
        matches = sorted(
            set(root.rglob(pattern)),
            key=lambda p: (len(p.parts), len(str(p)), str(p)),
        )
        if matches:
            return matches[0]
    return None


def discover_files(root: Path) -> list[SequenceFiles]:
    gt_files = list(root.rglob("*_groundtruth.nav"))
    out: list[SequenceFiles] = []

    for gt in sorted(gt_files):
        name = gt.name[: -len("_groundtruth.nav")]
        imu = _closest_match(root, [f"{name}_ADIS16465_IMU.txt"])
        odo_speed = _closest_match(root, [f"{name}_ODO_SPEED.txt"])
        ranger = _closest_match(root, [f"{name}_RANGER_ODO.txt"])

        if imu is None or (odo_speed is None and ranger is None):
            continue

        # Prefer the high-grade OEM719 stream used by the later fidelity study.
        gnss = _closest_match(
            root,
            [
                f"{name}_OEM7_GNSS.pos",
                f"{name}_OEM719*.pos",
                f"{name}*OEM7*.pos",
                f"{name}*OEM719*.pos",
                f"{name}_F9P_GNSS.pos",
                f"{name}*F9P*.pos",
            ],
        )
        if gnss is None:
            gnss_source = "NONE"
        elif "OEM7" in gnss.name.upper() or "OEM719" in gnss.name.upper():
            gnss_source = "OEM719_RTK"
        elif "F9P" in gnss.name.upper():
            gnss_source = "F9P"
        else:
            gnss_source = gnss.stem

        official_truth = _closest_match(
            root,
            [
                f"{name}_trajectory.csv",
                f"{name}_groundtruth_trajectory.csv",
                f"{name}_truth_trajectory.csv",
            ],
        )

        out.append(
            SequenceFiles(
                name=name,
                groundtruth=gt,
                imu=imu,
                odo_speed=odo_speed,
                ranger_odo=ranger,
                gnss=gnss,
                gnss_source=gnss_source,
                official_truth_trajectory=official_truth,
            )
        )

    rank = {name: i for i, name in enumerate(KNOWN_SEQUENCES)}
    out.sort(key=lambda x: (rank.get(x.name, 999), x.name))
    return out


def load_groundtruth_full(path: Path) -> dict[str, np.ndarray]:
    a = sorted_unique_by_time(read_numeric_table(path, min_cols=10))
    t = a[:, 0]
    north = a[:, 1]
    east = a[:, 2]
    down = a[:, 3]
    v_north = a[:, 4]
    v_east = a[:, 5]
    yaw_ned = np.deg2rad(a[:, 9])
    heading_enu = wrap_array(np.pi / 2.0 - yaw_ned)
    return {
        "t": t,
        "x": east,
        "y": north,
        "z": -down,
        "v_east": v_east,
        "v_north": v_north,
        "heading": heading_enu,
        "raw": a,
    }


def load_odo(files: SequenceFiles) -> tuple[np.ndarray, np.ndarray, str]:
    if files.odo_speed is not None:
        a = sorted_unique_by_time(read_numeric_table(files.odo_speed, min_cols=2))
        return a[:, 0], a[:, 1], "ODO_SPEED"

    assert files.ranger_odo is not None
    a = sorted_unique_by_time(read_numeric_table(files.ranger_odo, min_cols=9))
    speeds = a[:, 1:5]
    angles = a[:, 5:9]
    forward = np.mean(speeds * np.cos(angles), axis=1)
    return a[:, 0], forward, "RANGER_ODO_forward_component"


def load_imu_yaw(path: Path, yaw_sign: float) -> tuple[np.ndarray, np.ndarray]:
    a = sorted_unique_by_time(read_numeric_table(path, min_cols=7))
    t = a[:, 0]
    dtheta_z = a[:, 3]
    cumulative = np.cumsum(yaw_sign * dtheta_z)
    cumulative -= cumulative[0]
    return t, cumulative


def sample_yaw_rate(imu_t: np.ndarray, imu_cum_yaw: np.ndarray, grid: np.ndarray) -> np.ndarray:
    cum = np.interp(grid, imu_t, imu_cum_yaw)
    omega = np.zeros_like(cum)
    if len(grid) > 1:
        dt = np.diff(grid)
        omega[1:] = np.diff(cum) / dt
        omega[0] = omega[1]
    return omega


def stationary_gyro_bias(
    grid: np.ndarray,
    speed: np.ndarray,
    omega: np.ndarray,
    max_seconds: float = 5.0,
) -> float:
    if len(grid) < 2:
        return 0.0
    mask = (grid - grid[0] <= max_seconds) & (np.abs(speed) < 0.05)
    if np.count_nonzero(mask) >= 5:
        return float(np.median(omega[mask]))
    return 0.0


def load_gnss(
    path: Path | None,
    gt: dict[str, np.ndarray],
    sigma_max_m: float,
    anchor_count: int,
) -> dict[str, np.ndarray] | None:
    if path is None or not path.exists():
        return None

    a = sorted_unique_by_time(read_numeric_table(path, min_cols=7))
    t = a[:, 0]
    lat = a[:, 1]
    lon = a[:, 2]
    sigma_n = np.abs(a[:, 4])
    sigma_e = np.abs(a[:, 5])
    sigma_h = np.hypot(sigma_n, sigma_e)

    finite = (
        np.isfinite(t)
        & np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(sigma_n)
        & np.isfinite(sigma_e)
    )
    if not np.any(finite):
        return None

    t = t[finite]
    lat = lat[finite]
    lon = lon[finite]
    sigma_n = sigma_n[finite]
    sigma_e = sigma_e[finite]
    sigma_h = sigma_h[finite]

    east_rel, north_rel = geodetic_to_local_enu(lat, lon, lat[0], lon[0])
    valid = np.flatnonzero(sigma_h <= sigma_max_m)
    if not len(valid):
        return None
    idx = valid[: max(1, anchor_count)]
    gt_e = np.interp(t[idx], gt["t"], gt["x"])
    gt_n = np.interp(t[idx], gt["t"], gt["y"])
    offset_e = float(np.median(gt_e - east_rel[idx]))
    offset_n = float(np.median(gt_n - north_rel[idx]))

    return {
        "t": t,
        "x": east_rel + offset_e,
        "y": north_rel + offset_n,
        "sigma_n": sigma_n,
        "sigma_e": sigma_e,
        "sigma_h": sigma_h,
    }


def fallback_official_truth_from_gt(gt: dict[str, np.ndarray]) -> np.ndarray:
    """Build an i2Nav/TUM trajectory directly in the official NED/FRD convention.

    groundtruth.nav is t, local NED position, NED velocity, roll/pitch/yaw (deg).
    Using the raw columns avoids silently mixing the project's internal ENU/FLU
    state convention with the official evaluator's NED/FRD trajectory convention.
    """
    raw = np.asarray(gt["raw"], dtype=float)
    q = euler_ned_to_xyzw(
        np.deg2rad(raw[:, 7]),
        np.deg2rad(raw[:, 8]),
        np.deg2rad(raw[:, 9]),
    )
    return np.column_stack([raw[:, 0], raw[:, 1:4], q]).astype(float)


def load_official_truth(files: SequenceFiles, gt: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    p = files.official_truth_trajectory
    if p is not None and p.exists():
        try:
            a = sorted_unique_by_time(read_numeric_table(p, min_cols=8))
            if a.shape[1] >= 8:
                return a[:, :8], str(p)
        except Exception:
            pass
    return fallback_official_truth_from_gt(gt), "fallback_from_groundtruth.nav_yaw_only"


def prepare_sequence(
    files: SequenceFiles,
    *,
    hz: float,
    imu_yaw_sign: float,
    gnss_sigma_max_m: float,
    gnss_anchor_count: int,
) -> PreparedSequence:
    gt = load_groundtruth_full(files.groundtruth)
    odo_t, odo_speed, odo_source = load_odo(files)
    imu_t, imu_cum = load_imu_yaw(files.imu, imu_yaw_sign)
    grid = make_grid(gt["t"], odo_t, imu_t, hz)
    dt = 1.0 / hz

    gt_x = np.interp(grid, gt["t"], gt["x"])
    gt_y = np.interp(grid, gt["t"], gt["y"])
    gt_z = np.interp(grid, gt["t"], gt["z"])
    gt_heading = interp_angle(gt["t"], gt["heading"], grid)
    gt_v_e = np.interp(grid, gt["t"], gt["v_east"])
    gt_v_n = np.interp(grid, gt["t"], gt["v_north"])

    # Signed forward body velocity from GT velocity projected onto GT heading.
    gt_forward = gt_v_e * np.cos(gt_heading) + gt_v_n * np.sin(gt_heading)
    gt_yaw_unwrapped = np.unwrap(gt_heading)
    gt_yaw_rate = np.gradient(gt_yaw_unwrapped, dt)

    speed = np.interp(grid, odo_t, odo_speed)
    omega = sample_yaw_rate(imu_t, imu_cum, grid)
    omega -= stationary_gyro_bias(grid, speed, omega)

    accel = np.gradient(speed, dt)
    yaw_accel = np.gradient(omega, dt)
    features = np.column_stack(
        [speed, omega, accel, yaw_accel, np.abs(omega), np.abs(accel)]
    ).astype(np.float32)

    target_delta_v = (gt_forward - speed).astype(np.float32)
    target_delta_omega = (gt_yaw_rate - omega).astype(np.float32)

    gnss = load_gnss(
        files.gnss,
        gt,
        sigma_max_m=gnss_sigma_max_m,
        anchor_count=gnss_anchor_count,
    )
    official_truth, official_truth_source = load_official_truth(files, gt)

    return PreparedSequence(
        name=files.name,
        grid=grid.astype(np.float64),
        gt_xyz=np.column_stack([gt_x, gt_y, gt_z]).astype(np.float64),
        gt_heading=gt_heading.astype(np.float64),
        gt_forward_speed=gt_forward.astype(np.float32),
        gt_yaw_rate=gt_yaw_rate.astype(np.float32),
        speed=speed.astype(np.float32),
        omega=omega.astype(np.float32),
        features=features,
        target_delta_v=target_delta_v,
        target_delta_omega=target_delta_omega,
        gnss=gnss,
        gnss_source=files.gnss_source if gnss is not None else "NONE",
        odo_source=odo_source,
        official_truth=official_truth,
        official_truth_source=official_truth_source,
    )


# =============================================================================
# Fold loading / frozen LOSO preservation
# =============================================================================

def load_folds(frozen_loso_dir: Path, available_names: Sequence[str]) -> list[FoldDefinition]:
    candidates = [
        frozen_loso_dir / "fold_splits.json",
        frozen_loso_dir / "fold_splits(1).json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            folds: list[FoldDefinition] = []
            for item in data:
                folds.append(
                    FoldDefinition(
                        fold=int(item["fold"]),
                        test=str(item["test"]),
                        validation=[str(x) for x in item["validation"]],
                        train=[str(x) for x in item["train"]],
                    )
                )
            return folds
        except Exception as exc:
            print(f"[warn] Could not parse {p}: {exc}")

    names = [n for n in KNOWN_SEQUENCES if n in set(available_names)]
    by_name = set(names)
    folds = []
    for fold_id, test, val in DEFAULT_FOLDS:
        if test not in by_name or not all(v in by_name for v in val):
            continue
        train = [n for n in names if n != test and n not in set(val)]
        folds.append(FoldDefinition(fold_id, test, list(val), train))
    return folds


def preserve_frozen_results(frozen_loso_dir: Path, output_dir: Path) -> dict[str, Any]:
    summary_candidates = [
        frozen_loso_dir / "loso_summary.json",
        frozen_loso_dir / "loso_summary(1).json",
    ]
    csv_candidates = [
        frozen_loso_dir / "loso_results.csv",
        frozen_loso_dir / "loso_results(1).csv",
    ]

    summary_path = next((p for p in summary_candidates if p.exists()), None)
    csv_path = next((p for p in csv_candidates if p.exists()), None)

    frozen: dict[str, Any] = {
        "summary_path": str(summary_path) if summary_path else None,
        "results_csv_path": str(csv_path) if csv_path else None,
        "summary": None,
        "rows": [],
    }
    if summary_path:
        try:
            frozen["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            write_json(output_dir / "frozen_loso_summary_copy.json", frozen["summary"])
        except Exception as exc:
            frozen["summary_error"] = str(exc)
    if csv_path:
        frozen["rows"] = read_dict_rows(csv_path)
        write_dict_rows(output_dir / "frozen_loso_results_copy.csv", frozen["rows"])
    return frozen


# =============================================================================
# Standard project metrics (anchored + SE2 + time-RPE)
# =============================================================================

def se2_align(estimate_xy: np.ndarray, truth_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_mean = estimate_xy.mean(axis=0)
    b_mean = truth_xy.mean(axis=0)
    a = estimate_xy - a_mean
    b = truth_xy - b_mean
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    translation = b_mean - r @ a_mean
    aligned = (r @ estimate_xy.T).T + translation
    return aligned, r, translation


def summarize_standard_errors(
    estimate_xy: np.ndarray,
    estimate_heading: np.ndarray,
    truth_xy: np.ndarray,
    truth_heading: np.ndarray,
    hz: float,
) -> StandardMetrics:
    err = np.linalg.norm(estimate_xy - truth_xy, axis=1)
    aligned_xy, r, _ = se2_align(estimate_xy, truth_xy)
    align_angle = math.atan2(float(r[1, 0]), float(r[0, 0]))
    aligned_heading = wrap_array(estimate_heading + align_angle)
    err_aligned = np.linalg.norm(aligned_xy - truth_xy, axis=1)
    heading_err = np.abs(wrap_array(estimate_heading - truth_heading))

    def rpe(horizon_s: float) -> float:
        lag = max(1, int(round(horizon_s * hz)))
        if len(truth_xy) <= lag:
            return nan()
        trans_errors = []
        for i in range(len(truth_xy) - lag):
            j = i + lag
            d_gt_world = truth_xy[j] - truth_xy[i]
            d_est_world = estimate_xy[j] - estimate_xy[i]
            c_gt, s_gt = math.cos(-truth_heading[i]), math.sin(-truth_heading[i])
            c_es, s_es = math.cos(-estimate_heading[i]), math.sin(-estimate_heading[i])
            r_gt = np.array([[c_gt, -s_gt], [s_gt, c_gt]])
            r_es = np.array([[c_es, -s_es], [s_es, c_es]])
            trans_errors.append(float(np.linalg.norm(r_es @ d_est_world - r_gt @ d_gt_world)))
        te = np.asarray(trans_errors, dtype=float)
        return float(np.sqrt(np.mean(te * te)))

    gt_len = path_length(truth_xy)
    est_len = path_length(estimate_xy)
    final_error = float(np.linalg.norm(estimate_xy[-1] - truth_xy[-1]))
    final_error_se2 = float(np.linalg.norm(aligned_xy[-1] - truth_xy[-1]))

    return StandardMetrics(
        ate_rmse_m=float(np.sqrt(np.mean(err * err))),
        ate_median_m=float(np.median(err)),
        ate_p95_m=float(np.percentile(err, 95)),
        ate_max_m=float(np.max(err)),
        ate_se2_rmse_m=float(np.sqrt(np.mean(err_aligned * err_aligned))),
        heading_mae_deg=math.degrees(float(np.mean(heading_err))),
        heading_p95_deg=math.degrees(float(np.percentile(heading_err, 95))),
        rpe_1s_trans_rmse_m=rpe(1.0),
        rpe_5s_trans_rmse_m=rpe(5.0),
        rpe_10s_trans_rmse_m=rpe(10.0),
        final_error_m=final_error,
        final_error_se2_m=final_error_se2,
        final_drift_per_m=final_error_se2 / gt_len if gt_len > 0 else nan(),
        path_length_ratio=est_len / gt_len if gt_len > 0 else nan(),
    )


# =============================================================================
# Official i2Nav/evo evaluator mechanics
# =============================================================================

def numpy_to_evo_trajectory(traj: np.ndarray):
    if not EVO_AVAILABLE:
        raise RuntimeError("evo is not installed")
    stamps = traj[:, 0]
    xyz = traj[:, 1:4]
    quat_xyzw = traj[:, 4:8]
    quat_wxyz = np.roll(quat_xyzw, 1, axis=1)
    return evo_trajectory.PoseTrajectory3D(xyz, quat_wxyz, stamps)


def official_i2nav_evaluate(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> dict[str, Any]:
    """Replicate the metric mechanics in i2Nav-WHU/evaluate_odometry/evaluate.py."""
    if not EVO_AVAILABLE:
        return {
            "available": False,
            "error": "evo is not installed; run: python -m pip install evo",
        }

    try:
        ref_raw = numpy_to_evo_trajectory(reference)
        est_raw = numpy_to_evo_trajectory(estimate)
        ref, est = evo_sync.associate_trajectories(
            ref_raw,
            est_raw,
            OFFICIAL_MAX_TIME_SYNC_DIFF_S,
        )
        est_aligned = copy.deepcopy(est)
        est_aligned.align(ref, correct_scale=False, correct_only_scale=False)

        ape_t = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)
        ape_t.process_data((ref, est_aligned))
        ape_t_stats = ape_t.get_all_statistics()

        ape_r = evo_metrics.APE(evo_metrics.PoseRelation.rotation_angle_deg)
        ape_r.process_data((ref, est_aligned))
        ape_r_stats = ape_r.get_all_statistics()

        out: dict[str, Any] = {
            "available": True,
            "associated_poses": int(len(est_aligned.timestamps)),
            "max_time_sync_diff_s": OFFICIAL_MAX_TIME_SYNC_DIFF_S,
            "alignment": "SE3_no_scale",
            "ape_translation_rmse_m": float(ape_t_stats["rmse"]),
            "ape_rotation_rmse_deg": float(ape_r_stats["rmse"]),
            "rpe": {},
        }

        for delta in OFFICIAL_RPE_DELTAS_M:
            try:
                rpe_t = evo_metrics.RPE(
                    evo_metrics.PoseRelation.translation_part,
                    delta,
                    evo_metrics.Unit.meters,
                    OFFICIAL_RPE_REL_DELTA_TOL,
                    OFFICIAL_RPE_ALL_PAIRS,
                )
                rpe_t.process_data((ref, est_aligned))
                t_rmse = float(rpe_t.get_all_statistics()["rmse"])

                rpe_r = evo_metrics.RPE(
                    evo_metrics.PoseRelation.rotation_angle_deg,
                    delta,
                    evo_metrics.Unit.meters,
                    OFFICIAL_RPE_REL_DELTA_TOL,
                    OFFICIAL_RPE_ALL_PAIRS,
                )
                rpe_r.process_data((ref, est_aligned))
                r_rmse = float(rpe_r.get_all_statistics()["rmse"])

                out["rpe"][str(delta)] = {
                    "translation_rmse_m": t_rmse,
                    "translation_percent": 100.0 * t_rmse / float(delta),
                    "rotation_rmse_deg": r_rmse,
                }
            except Exception as exc:
                out["rpe"][str(delta)] = {
                    "translation_rmse_m": None,
                    "translation_percent": None,
                    "rotation_rmse_deg": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return out
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def internal_enu_planar_to_official_ned(
    seq: PreparedSequence,
    states: np.ndarray,
    *,
    z_enu: np.ndarray | None = None,
) -> np.ndarray:
    """Convert project ENU/FLU planar poses to i2Nav official NED/FRD TUM poses.

    Internal position is [east, north, up]. Official position is [north, east, down].
    Internal heading is CCW from east with a FLU body convention. Official navigation
    yaw is clockwise from north in NED/FRD, hence yaw_ned = pi/2 - heading_enu.

    For real estimates z_enu is omitted and the initial height is held constant so no
    future GT height leaks into the twin. For the frame self-check only, interpolated
    GT z may be supplied to verify the coordinate conversion itself.
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError("states must have shape (N, >=3) containing x_enu,y_enu,heading_enu")
    if len(states) != len(seq.grid):
        raise ValueError("states length must match seq.grid")

    if z_enu is None:
        z_up = np.full(len(states), float(seq.gt_xyz[0, 2]), dtype=float)
    else:
        z_up = np.asarray(z_enu, dtype=float)
        if z_up.shape != (len(states),):
            raise ValueError("z_enu must have shape (N,)")

    north = states[:, 1]
    east = states[:, 0]
    down = -z_up
    yaw_ned = wrap_array(np.pi / 2.0 - states[:, 2])
    q = yaw_to_xyzw(yaw_ned)
    return np.column_stack([seq.grid, north, east, down, q]).astype(float)


def estimate_to_official_trajectory(seq: PreparedSequence, states: np.ndarray) -> np.ndarray:
    return internal_enu_planar_to_official_ned(seq, states)


def official_gt_frame_selfcheck(seq: PreparedSequence) -> dict[str, Any]:
    """Check the internal ENU -> official NED frame conversion using GT as input.

    This is a diagnostic only and never contributes to model performance. Position
    should be essentially zero-error after association/alignment. Rotation retains
    real GT roll/pitch while the diagnostic estimate is yaw-only, so a small nonzero
    rotation error is expected; a ~180 deg result indicates a frame/body convention bug.
    """
    gt_states = np.column_stack([seq.gt_xyz[:, 0], seq.gt_xyz[:, 1], seq.gt_heading])
    converted = internal_enu_planar_to_official_ned(
        seq, gt_states, z_enu=seq.gt_xyz[:, 2]
    )
    official = official_i2nav_evaluate(seq.official_truth, converted)
    out = {
        "sequence": seq.name,
        "official_truth_source": seq.official_truth_source,
        **flatten_official_metrics(official),
    }
    ape_t = safe_float(out.get("official_ape_translation_rmse_m"))
    ape_r = safe_float(out.get("official_ape_rotation_rmse_deg"))
    out["position_frame_check_pass"] = bool(np.isfinite(ape_t) and ape_t <= 0.05)
    out["no_180deg_body_frame_bug"] = bool(np.isfinite(ape_r) and ape_r < 30.0)
    return out


def save_official_trajectory(path: Path, traj: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, traj, fmt="%.9f %.9f %.9f %.9f %.10f %.10f %.10f %.10f")


# =============================================================================
# Multi-step dual GRU
# =============================================================================

class RefinedDualGRU(nn.Module):
    """Physics-guided GRU with mean-correction and calibrated residual-uncertainty heads.

    The first head predicts deterministic corrections (delta_v, delta_omega).
    The second predicts conditional standard deviations (sigma_v, sigma_omega)
    for the remaining correction error.  This follows the probabilistic idea used
    by wheel-inertial neural odometry work such as WING: learn the mean error and
    the uncertainty of that error as separate quantities rather than asking a
    network to emit an arbitrary EKF-Q multiplier.
    """

    def __init__(
        self,
        *,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        delta_v_limit: float,
        delta_omega_limit: float,
        sigma_v_min: float = 0.005,
        sigma_v_max: float = 0.5,
        sigma_omega_min: float = 0.002,
        sigma_omega_max: float = 0.5,
        sigma_v_init: float = 0.05,
        sigma_omega_init: float = 0.02,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.delta_v_limit = float(delta_v_limit)
        self.delta_omega_limit = float(delta_omega_limit)
        self.sigma_v_min = float(sigma_v_min)
        self.sigma_v_max = float(sigma_v_max)
        self.sigma_omega_min = float(sigma_omega_min)
        self.sigma_omega_max = float(sigma_omega_max)

        if not (0 < self.sigma_v_min < self.sigma_v_max):
            raise ValueError("Require 0 < sigma_v_min < sigma_v_max")
        if not (0 < self.sigma_omega_min < self.sigma_omega_max):
            raise ValueError("Require 0 < sigma_omega_min < sigma_omega_max")

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.dynamics_head = nn.Linear(hidden_size, 2)
        self.uncertainty_head = nn.Linear(hidden_size, 2)

        # Start the uncertainty head near the train-fold residual scale rather
        # than near either bound.  Zero weights make the initial uncertainty
        # constant, preventing random Q fluctuations at epoch 0.
        nn.init.zeros_(self.uncertainty_head.weight)

        def inv_sigmoid_for_sigma(value: float, lo: float, hi: float) -> float:
            value = min(max(float(value), lo + 1e-6), hi - 1e-6)
            p = (value - lo) / (hi - lo)
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            return math.log(p / (1.0 - p))

        with torch.no_grad():
            self.uncertainty_head.bias[0] = inv_sigmoid_for_sigma(
                sigma_v_init, self.sigma_v_min, self.sigma_v_max
            )
            self.uncertainty_head.bias[1] = inv_sigmoid_for_sigma(
                sigma_omega_init, self.sigma_omega_min, self.sigma_omega_max
            )

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Supports [B,W,F] and [B,H,W,F].
        prefix = windows.shape[:-2]
        w = windows.shape[-2]
        f = windows.shape[-1]
        flat = windows.reshape(-1, w, f)
        y, _ = self.gru(flat)
        h = self.trunk(y[:, -1, :])

        dyn_raw = self.dynamics_head(h)
        dv = self.delta_v_limit * torch.tanh(dyn_raw[:, 0])
        dw = self.delta_omega_limit * torch.tanh(dyn_raw[:, 1])
        dynamics = torch.stack([dv, dw], dim=-1)

        u_raw = self.uncertainty_head(h)
        sigma_v = self.sigma_v_min + (self.sigma_v_max - self.sigma_v_min) * torch.sigmoid(u_raw[:, 0])
        sigma_w = self.sigma_omega_min + (self.sigma_omega_max - self.sigma_omega_min) * torch.sigmoid(u_raw[:, 1])
        uncertainty = torch.stack([sigma_v, sigma_w], dim=-1)

        dynamics = dynamics.reshape(*prefix, 2)
        uncertainty = uncertainty.reshape(*prefix, 2)
        return dynamics, uncertainty


class RolloutDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[PreparedSequence],
        *,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        window: int,
        max_horizon_steps: int,
        horizons_steps: Sequence[int],
        stride: int,
    ) -> None:
        self.sequences = list(sequences)
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.window = int(window)
        self.max_horizon_steps = int(max_horizon_steps)
        self.horizons_steps = tuple(int(h) for h in horizons_steps)
        self.index: list[tuple[int, int]] = []
        self.windows: list[np.ndarray] = []

        for si, seq in enumerate(self.sequences):
            normalized = (seq.features - self.feature_mean) / self.feature_std
            sw = np.lib.stride_tricks.sliding_window_view(
                normalized,
                window_shape=self.window,
                axis=0,
            ).transpose(0, 2, 1)
            self.windows.append(sw)

            # Start pose is sample `a`; first predicted control is a+1, whose
            # rolling window must exist. First valid a is window-2.
            first_anchor = self.window - 2
            last_anchor = len(seq.grid) - self.max_horizon_steps - 1
            if last_anchor < first_anchor:
                continue
            for a in range(first_anchor, last_anchor + 1, max(1, stride)):
                self.index.append((si, a))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        si, a = self.index[idx]
        seq = self.sequences[si]
        sw = self.windows[si]
        h = self.max_horizon_steps

        # Window ending at sample a+1 has sliding-window row a-window+2.
        row0 = a - self.window + 2
        windows = np.array(sw[row0 : row0 + h], dtype=np.float32, copy=True)
        base_v = np.asarray(seq.speed[a + 1 : a + h + 1], dtype=np.float32)
        base_w = np.asarray(seq.omega[a + 1 : a + h + 1], dtype=np.float32)
        target = np.column_stack(
            [
                seq.target_delta_v[a + 1 : a + h + 1],
                seq.target_delta_omega[a + 1 : a + h + 1],
            ]
        ).astype(np.float32)

        gt_segment = np.column_stack(
            [
                seq.gt_xyz[a : a + h + 1, 0],
                seq.gt_xyz[a : a + h + 1, 1],
                seq.gt_heading[a : a + h + 1],
            ]
        ).astype(np.float32)

        return (
            torch.from_numpy(windows),
            torch.from_numpy(base_v),
            torch.from_numpy(base_w),
            torch.from_numpy(target),
            torch.from_numpy(gt_segment),
        )


def derive_train_statistics(
    sequences: Sequence[PreparedSequence],
) -> dict[str, np.ndarray | float]:
    feat = np.concatenate([s.features for s in sequences], axis=0).astype(np.float64)
    dv = np.concatenate([s.target_delta_v for s in sequences]).astype(np.float64)
    dw = np.concatenate([s.target_delta_omega for s in sequences]).astype(np.float64)

    feature_mean = np.mean(feat, axis=0)
    feature_std = np.std(feat, axis=0)
    feature_std = np.maximum(feature_std, 1e-5)

    target_scale = np.array(
        [max(np.std(dv), 0.02), max(np.std(dw), 0.01)],
        dtype=np.float64,
    )

    # Robust train-only correction bounds; no validation/test information used.
    dv_limit = float(np.clip(1.25 * np.percentile(np.abs(dv), 99.0), 0.05, 1.0))
    dw_limit = float(np.clip(1.25 * np.percentile(np.abs(dw), 99.0), 0.05, 1.0))

    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "target_scale": target_scale.astype(np.float32),
        "delta_v_limit": dv_limit,
        "delta_omega_limit": dw_limit,
    }


def integrate_rollout_torch(
    start_pose: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    dt: float,
    horizons_steps: Sequence[int],
) -> dict[int, torch.Tensor]:
    px = start_pose[:, 0]
    py = start_pose[:, 1]
    th = start_pose[:, 2]
    wanted = set(int(h) for h in horizons_steps)
    out: dict[int, torch.Tensor] = {}

    for j in range(v.shape[1]):
        dtheta = w[:, j] * dt
        mid = th + 0.5 * dtheta
        ds = v[:, j] * dt
        px = px + ds * torch.cos(mid)
        py = py + ds * torch.sin(mid)
        th = wrap_tensor(th + dtheta)
        step = j + 1
        if step in wanted:
            out[step] = torch.stack([px, py, th], dim=-1)
    return out


def batch_loss(
    model: RefinedDualGRU,
    batch,
    *,
    device: torch.device,
    dt: float,
    horizons_steps: Sequence[int],
    horizon_weights: Sequence[float],
    target_scale: torch.Tensor,
    dyn_weight: float,
    rollout_weight: float,
    heading_rollout_weight: float,
    nll_weight: float,
    uncertainty_enabled: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    windows, base_v, base_w, target, gt_segment = [x.to(device, non_blocking=True) for x in batch]
    dynamics, sigma = model(windows)

    # Mean correction objective.
    dyn_error = dynamics - target
    dyn_norm = dyn_error / target_scale.view(1, 1, 2)
    dyn_loss = torch.nn.functional.smooth_l1_loss(
        dyn_norm,
        torch.zeros_like(dyn_norm),
        beta=1.0,
    )

    corrected_v = base_v + dynamics[..., 0]
    corrected_w = base_w + dynamics[..., 1]
    rollout_pred = integrate_rollout_torch(
        gt_segment[:, 0, :], corrected_v, corrected_w, dt, horizons_steps
    )

    if len(horizon_weights) != len(horizons_steps):
        raise ValueError("horizon_weights must have the same length as horizons_steps")
    hw = torch.tensor(horizon_weights, dtype=windows.dtype, device=device)
    hw = hw / torch.clamp(torch.sum(hw), min=1e-8)

    rollout_terms: list[torch.Tensor] = []
    heading_terms: list[torch.Tensor] = []
    parts: dict[str, float] = {}
    for idx, h in enumerate(horizons_steps):
        pred = rollout_pred[int(h)]
        truth = gt_segment[:, int(h), :]
        pos_error = pred[:, :2] - truth[:, :2]
        heading_error = wrap_tensor(pred[:, 2] - truth[:, 2])

        pos_term = torch.nn.functional.smooth_l1_loss(
            pos_error, torch.zeros_like(pos_error), beta=1.0
        )
        # Normalize angle so 10 deg roughly corresponds to unit error before
        # Smooth-L1. The explicit horizon weights then determine which future
        # horizons matter most.
        heading_term = torch.nn.functional.smooth_l1_loss(
            heading_error / math.radians(10.0),
            torch.zeros_like(heading_error),
            beta=1.0,
        )
        rollout_terms.append(pos_term)
        heading_terms.append(heading_term)
        parts[f"rollout_pos_h{int(h)}"] = float(pos_term.detach().cpu())
        parts[f"rollout_heading_h{int(h)}"] = float(heading_term.detach().cpu())

    rollout_pos_loss = torch.sum(hw * torch.stack(rollout_terms))
    rollout_heading_loss = torch.sum(hw * torch.stack(heading_terms))

    # Heteroscedastic Gaussian NLL on the *physical correction residuals*.
    # sigma_v and sigma_omega therefore have direct units (m/s and rad/s) and
    # can later be propagated into state-space Q with the motion Jacobian.
    sigma_safe = torch.clamp(sigma, min=1e-6)
    sigma_norm = sigma_safe / target_scale.view(1, 1, 2)
    nll = 0.5 * torch.mean(
        (dyn_norm / sigma_norm) ** 2 + 2.0 * torch.log(sigma_norm)
    )

    effective_nll_weight = nll_weight if uncertainty_enabled else 0.0
    total = (
        dyn_weight * dyn_loss
        + rollout_weight * rollout_pos_loss
        + heading_rollout_weight * rollout_heading_loss
        + effective_nll_weight * nll
    )

    parts.update(
        {
            "total": float(total.detach().cpu()),
            "dyn": float(dyn_loss.detach().cpu()),
            "rollout_pos": float(rollout_pos_loss.detach().cpu()),
            "rollout_heading": float(rollout_heading_loss.detach().cpu()),
            "nll": float(nll.detach().cpu()),
            "sigma_v_mean": float(sigma[..., 0].detach().mean().cpu()),
            "sigma_omega_mean": float(sigma[..., 1].detach().mean().cpu()),
        }
    )
    return total, parts


def evaluate_loader_loss(
    model: RefinedDualGRU,
    loader: DataLoader,
    **loss_kwargs,
) -> tuple[float, dict[str, float]]:
    model.eval()
    sums: dict[str, float] = {}
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            loss, parts = batch_loss(model, batch, **loss_kwargs)
            b = int(batch[0].shape[0])
            total_n += b
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + b * float(v)
    if total_n == 0:
        return float("inf"), {}
    avg = {k: v / total_n for k, v in sums.items()}
    return float(avg["total"]), avg


def train_refined_dual(
    *,
    fold: FoldDefinition,
    train_sequences: Sequence[PreparedSequence],
    val_sequences: Sequence[PreparedSequence],
    args: argparse.Namespace,
    device: torch.device,
    fold_dir: Path,
) -> tuple[RefinedDualGRU, dict[str, Any]]:
    stats = derive_train_statistics(train_sequences)
    feature_mean = stats["feature_mean"]
    feature_std = stats["feature_std"]
    target_scale_np = stats["target_scale"]
    dv_limit = float(stats["delta_v_limit"])
    dw_limit = float(stats["delta_omega_limit"])

    horizons_s = tuple(float(x) for x in args.rollout_horizons_s)
    horizons_steps = tuple(int(round(x * args.rate_hz)) for x in horizons_s)
    if any(h <= 0 for h in horizons_steps):
        raise ValueError("All rollout horizons must be positive")
    if len(args.rollout_horizon_weights) != len(horizons_steps):
        raise ValueError("--rollout-horizon-weights must match --rollout-horizons-s")
    if any(float(w) <= 0 for w in args.rollout_horizon_weights):
        raise ValueError("All rollout horizon weights must be > 0")
    max_h = max(horizons_steps)

    train_ds = RolloutDataset(
        train_sequences,
        feature_mean=feature_mean,
        feature_std=feature_std,
        window=args.window,
        max_horizon_steps=max_h,
        horizons_steps=horizons_steps,
        stride=args.train_stride,
    )
    val_ds = RolloutDataset(
        val_sequences,
        feature_mean=feature_mean,
        feature_std=feature_std,
        window=args.window,
        max_horizon_steps=max_h,
        horizons_steps=horizons_steps,
        stride=args.val_stride,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Fold {fold.fold}: no rollout anchors. Need sequences longer than "
            f"window + {max_h} samples."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # Start uncertainty near the empirical train-only correction scale.
    sigma_v_init = float(np.clip(target_scale_np[0], args.sigma_v_min, args.sigma_v_max))
    sigma_w_init = float(np.clip(target_scale_np[1], args.sigma_omega_min, args.sigma_omega_max))
    model = RefinedDualGRU(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        delta_v_limit=dv_limit,
        delta_omega_limit=dw_limit,
        sigma_v_min=args.sigma_v_min,
        sigma_v_max=args.sigma_v_max,
        sigma_omega_min=args.sigma_omega_min,
        sigma_omega_max=args.sigma_omega_max,
        sigma_v_init=sigma_v_init,
        sigma_omega_init=sigma_w_init,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    target_scale = torch.tensor(target_scale_np, dtype=torch.float32, device=device)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    loss_common = dict(
        device=device,
        dt=1.0 / args.rate_hz,
        horizons_steps=horizons_steps,
        horizon_weights=tuple(float(x) for x in args.rollout_horizon_weights),
        target_scale=target_scale,
        dyn_weight=args.dyn_loss_weight,
        rollout_weight=args.rollout_loss_weight,
        heading_rollout_weight=args.heading_rollout_weight,
        nll_weight=args.nll_weight,
    )

    print(f"      train rollout anchors = {len(train_ds):,}")
    print(f"      val rollout anchors   = {len(val_ds):,}")
    print(f"      dv limit              = {dv_limit:.4f} m/s")
    print(f"      domega limit          = {dw_limit:.4f} rad/s")
    print(f"      horizons (s)          = {list(horizons_s)}")
    print(f"      horizon weights       = {list(args.rollout_horizon_weights)}")
    print(f"      sigma_v bounds        = [{args.sigma_v_min:.4f}, {args.sigma_v_max:.4f}] m/s")
    print(f"      sigma_w bounds        = [{args.sigma_omega_min:.4f}, {args.sigma_omega_max:.4f}] rad/s")

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        total_n = 0
        uncertainty_enabled = epoch > args.uncertainty_warmup_epochs

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else contextlib.nullcontext()
            with amp_ctx:
                loss, parts = batch_loss(
                    model,
                    batch,
                    uncertainty_enabled=uncertainty_enabled,
                    **loss_common,
                )
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            b = int(batch[0].shape[0])
            total_n += b
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + b * float(v)

        train_avg = {k: v / max(total_n, 1) for k, v in sums.items()}
        val_total, val_avg = evaluate_loader_loss(
            model,
            val_loader,
            uncertainty_enabled=uncertainty_enabled,
            **loss_common,
        )

        row = {
            "epoch": epoch,
            "uncertainty_enabled": uncertainty_enabled,
            **{f"train_{k}": v for k, v in train_avg.items()},
            **{f"val_{k}": v for k, v in val_avg.items()},
        }
        history.append(row)

        print(
            f"      v3 epoch={epoch:03d} "
            f"train={train_avg.get('total', nan()):.5f} "
            f"val={val_total:.5f} "
            f"roll={val_avg.get('rollout_pos', nan()):.5f} "
            f"head={val_avg.get('rollout_heading', nan()):.5f} "
            f"sv={val_avg.get('sigma_v_mean', nan()):.4f} "
            f"sw={val_avg.get('sigma_omega_mean', nan()):.4f}"
        )

        if val_total < best_val - args.min_delta:
            best_val = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"      early stop at epoch {epoch}; best val={best_val:.6f}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    checkpoint = {
        "schema": "i2nav_predictive_nll_dual_v3",
        "state_dict": best_state,
        "feature_mean": np.asarray(feature_mean),
        "feature_std": np.asarray(feature_std),
        "target_scale": np.asarray(target_scale_np),
        "feature_names": list(FEATURE_NAMES),
        "window": args.window,
        "rate_hz": args.rate_hz,
        "rollout_horizons_s": list(horizons_s),
        "rollout_horizon_weights": [float(x) for x in args.rollout_horizon_weights],
        "delta_v_limit": dv_limit,
        "delta_omega_limit": dw_limit,
        "sigma_v_min": args.sigma_v_min,
        "sigma_v_max": args.sigma_v_max,
        "sigma_omega_min": args.sigma_omega_min,
        "sigma_omega_max": args.sigma_omega_max,
        "sigma_v_init": sigma_v_init,
        "sigma_omega_init": sigma_w_init,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "fold": asdict(fold),
        "best_validation_loss": best_val,
        "loss_weights": {
            "dynamics": args.dyn_loss_weight,
            "rollout_position": args.rollout_loss_weight,
            "rollout_heading": args.heading_rollout_weight,
            "heteroscedastic_nll": args.nll_weight,
        },
        "security_constraint": "Network inputs are ODO/IMU-derived only; no GNSS-derived feature is allowed.",
    }
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, fold_dir / "predictive_nll_best.pt")
    write_dict_rows(fold_dir / "training_history.csv", history)
    write_json(
        fold_dir / "training_metadata.json",
        {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in checkpoint.items()
            if k not in {"state_dict"}
        },
    )
    return model, checkpoint


def predict_model_outputs(
    model: RefinedDualGRU,
    seq: PreparedSequence,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    window: int,
    device: torch.device,
    eval_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(seq.grid)
    dynamics = np.zeros((n, 2), dtype=np.float64)
    uncertainty = np.zeros((n, 2), dtype=np.float64)
    # Before the first complete temporal window, use conservative/fixed-like
    # residual sigmas; these samples are a tiny prefix and are not NN outputs.
    uncertainty[:, 0] = 0.05
    uncertainty[:, 1] = 0.01
    if n < window:
        return dynamics, uncertainty

    normalized = (seq.features - feature_mean.astype(np.float32)) / feature_std.astype(np.float32)
    sw = np.lib.stride_tricks.sliding_window_view(
        normalized, window_shape=window, axis=0
    ).transpose(0, 2, 1)

    model.eval()
    out_dyn = []
    out_sigma = []
    with torch.no_grad():
        for start in range(0, len(sw), eval_batch_size):
            batch_np = np.array(sw[start : start + eval_batch_size], dtype=np.float32, copy=True)
            batch = torch.from_numpy(batch_np).to(device)
            d, s = model(batch)
            out_dyn.append(d.detach().cpu().numpy())
            out_sigma.append(s.detach().cpu().numpy())
    d_all = np.concatenate(out_dyn, axis=0)
    s_all = np.concatenate(out_sigma, axis=0)
    dynamics[window - 1 :] = d_all
    uncertainty[window - 1 :] = s_all
    return dynamics, uncertainty


# =============================================================================
# EKF replay + Q isolation/calibration
# =============================================================================

def predict_state_and_covariance(
    x: np.ndarray,
    p: np.ndarray,
    v: float,
    w: float,
    dt: float,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    theta = float(x[2])
    dtheta = w * dt
    mid = theta + 0.5 * dtheta
    ds = v * dt
    x_new = x.copy()
    x_new[0] += ds * math.cos(mid)
    x_new[1] += ds * math.sin(mid)
    x_new[2] = float(wrap_array(np.array([theta + dtheta]))[0])

    f = np.eye(3, dtype=float)
    f[0, 2] = -ds * math.sin(mid)
    f[1, 2] = ds * math.cos(mid)
    p_new = f @ p @ f.T + q
    p_new = 0.5 * (p_new + p_new.T)
    return x_new, p_new


def gnss_innovation(x: np.ndarray, p: np.ndarray, z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    nu = z - h @ x
    s = h @ p @ h.T + r
    nis = float(nu.T @ np.linalg.pinv(s) @ nu)
    return nu, s, nis


def gnss_update(x: np.ndarray, p: np.ndarray, z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    nu = z - h @ x
    s = h @ p @ h.T + r
    k = p @ h.T @ np.linalg.pinv(s)
    x_new = x + k @ nu
    x_new[2] = float(wrap_array(np.array([x_new[2]]))[0])
    i = np.eye(3)
    # Joseph form.
    p_new = (i - k @ h) @ p @ (i - k @ h).T + k @ r @ k.T
    p_new = 0.5 * (p_new + p_new.T)
    return x_new, p_new


def posterior_consistency(x: np.ndarray, p: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    e = np.array(
        [
            x[0] - truth[0],
            x[1] - truth[1],
            float(wrap_array(np.array([x[2] - truth[2]]))[0]),
        ],
        dtype=float,
    )
    nees = float(e.T @ np.linalg.pinv(p) @ e)
    e_xy = e[:2]
    p_xy = p[:2, :2]
    pos_mahal = float(e_xy.T @ np.linalg.pinv(p_xy) @ e_xy)
    return nees, pos_mahal


def learned_motion_q(
    x: np.ndarray,
    corrected_v: float,
    corrected_w: float,
    dt: float,
    sigma_v: float,
    sigma_w: float,
    *,
    q_floor_xy_sigma_mps: float,
    q_floor_heading_sigma_radps: float,
) -> np.ndarray:
    """Propagate velocity/yaw-rate residual uncertainty into planar state Q.

    G is the Jacobian of midpoint unicycle propagation with respect to [v, w].
    This yields an SPD, generally non-diagonal covariance with direct physical
    meaning. A small fixed floor prevents pathological covariance collapse.
    """
    theta = float(x[2])
    mid = theta + 0.5 * corrected_w * dt
    g = np.array(
        [
            [dt * math.cos(mid), -0.5 * corrected_v * dt * dt * math.sin(mid)],
            [dt * math.sin(mid),  0.5 * corrected_v * dt * dt * math.cos(mid)],
            [0.0,                 dt],
        ],
        dtype=float,
    )
    u_cov = np.diag([max(float(sigma_v), 1e-6) ** 2, max(float(sigma_w), 1e-6) ** 2])
    q = g @ u_cov @ g.T
    q += np.diag(
        [
            (q_floor_xy_sigma_mps * dt) ** 2,
            (q_floor_xy_sigma_mps * dt) ** 2,
            (q_floor_heading_sigma_radps * dt) ** 2,
        ]
    )
    return 0.5 * (q + q.T)


def run_filter_variant(
    seq: PreparedSequence,
    dynamics: np.ndarray,
    uncertainty_pred: np.ndarray,
    *,
    use_learned_q: bool,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    n = len(seq.grid)
    dt_nominal = 1.0 / args.rate_hz

    x = np.array([seq.gt_xyz[0, 0], seq.gt_xyz[0, 1], seq.gt_heading[0]], dtype=float)
    p = np.diag([0.25**2, 0.25**2, math.radians(5.0) ** 2]).astype(float)
    states = np.zeros((n, 3), dtype=float)
    covariances = np.zeros((n, 3, 3), dtype=float)
    states[0] = x
    covariances[0] = p

    gnss_index = 0
    if seq.gnss is not None:
        while gnss_index < len(seq.gnss["t"]) and seq.gnss["t"][gnss_index] < seq.grid[0]:
            gnss_index += 1

    gnss_seen = gnss_normal = gnss_reacquired = gnss_rejected = gnss_quality_skipped = 0
    last_accept_t = float(seq.grid[0])
    reacq_streak = 0
    nis_values: list[float] = []
    nees_values: list[float] = []
    pos_mahal_values: list[float] = []
    calibration_trace: list[dict[str, Any]] = []

    sigma_v_used = np.asarray(uncertainty_pred[:, 0], dtype=float)
    sigma_w_used = np.asarray(uncertainty_pred[:, 1], dtype=float)

    for k in range(1, n):
        dt = float(seq.grid[k] - seq.grid[k - 1])
        if not np.isfinite(dt) or dt <= 0:
            dt = dt_nominal

        corrected_v = float(seq.speed[k] + dynamics[k, 0])
        corrected_w = float(seq.omega[k] + dynamics[k, 1])
        sv = float(np.clip(sigma_v_used[k], args.sigma_v_min, args.sigma_v_max))
        sw = float(np.clip(sigma_w_used[k], args.sigma_omega_min, args.sigma_omega_max))

        if use_learned_q:
            q = learned_motion_q(
                x, corrected_v, corrected_w, dt, sv, sw,
                q_floor_xy_sigma_mps=args.q_floor_xy_sigma_mps,
                q_floor_heading_sigma_radps=args.q_floor_heading_sigma_radps,
            )
        else:
            # Exact frozen V5 Q for a clean same-dynamics isolation test.
            q = np.diag(
                [
                    (args.q_xy_sigma_mps * dt) ** 2,
                    (args.q_xy_sigma_mps * dt) ** 2,
                    (args.q_heading_sigma_radps * dt) ** 2,
                ]
            )
        x, p = predict_state_and_covariance(x, p, corrected_v, corrected_w, dt, q)

        gnss_event = "none"
        nis_this = nan()
        reacq_nis_this = nan()
        if seq.gnss is not None:
            latest = None
            while gnss_index < len(seq.gnss["t"]) and seq.gnss["t"][gnss_index] <= seq.grid[k] + 1e-9:
                latest = gnss_index
                gnss_index += 1
            if latest is not None:
                gnss_seen += 1
                sigma_h = float(seq.gnss["sigma_h"][latest])
                if not np.isfinite(sigma_h) or sigma_h > args.gnss_sigma_max_m:
                    gnss_quality_skipped += 1
                    gnss_event = "quality_skip"
                    reacq_streak = 0
                else:
                    sigma_e = max(float(seq.gnss["sigma_e"][latest]), args.gnss_sigma_floor_m)
                    sigma_n = max(float(seq.gnss["sigma_n"][latest]), args.gnss_sigma_floor_m)
                    r = np.diag([sigma_e**2, sigma_n**2])
                    z = np.array([seq.gnss["x"][latest], seq.gnss["y"][latest]], dtype=float)
                    _, _, nis_this = gnss_innovation(x, p, z, r)
                    nis_values.append(nis_this)

                    if nis_this <= args.gnss_nis_gate:
                        x, p = gnss_update(x, p, z, r)
                        gnss_normal += 1
                        gnss_event = "normal"
                        last_accept_t = float(seq.grid[k])
                        reacq_streak = 0
                    else:
                        coast = max(0.0, float(seq.grid[k] - last_accept_t))
                        if coast >= args.reacq_start_s:
                            gate_sigma = min(args.reacq_sigma_max_m, args.reacq_sigma_growth_mps * coast)
                            r_gate = r + np.eye(2) * gate_sigma**2
                            _, _, reacq_nis_this = gnss_innovation(x, p, z, r_gate)
                            if reacq_nis_this <= args.gnss_nis_gate:
                                reacq_streak += 1
                                gnss_event = f"reacq_candidate_{reacq_streak}"
                                if reacq_streak >= args.reacq_consecutive:
                                    x, p = gnss_update(x, p, z, r_gate)
                                    gnss_reacquired += 1
                                    gnss_event = "reacquired"
                                    last_accept_t = float(seq.grid[k])
                                    reacq_streak = 0
                            else:
                                gnss_rejected += 1
                                gnss_event = "rejected"
                                reacq_streak = 0
                        else:
                            gnss_rejected += 1
                            gnss_event = "rejected"
                            reacq_streak = 0

        truth_state = np.array([seq.gt_xyz[k, 0], seq.gt_xyz[k, 1], seq.gt_heading[k]], dtype=float)
        nees, pos_mahal = posterior_consistency(x, p, truth_state)
        nees_values.append(nees)
        pos_mahal_values.append(pos_mahal)
        states[k] = x
        covariances[k] = p
        calibration_trace.append(
            {
                "time_s": float(seq.grid[k]),
                "sigma_v_mps": sv,
                "sigma_omega_radps": sw,
                "nis": nis_this,
                "reacq_nis": reacq_nis_this,
                "nees": nees,
                "position_mahalanobis2": pos_mahal,
                "gnss_event": gnss_event,
            }
        )

    nis_arr = np.asarray(nis_values, dtype=float)
    nees_arr = np.asarray(nees_values, dtype=float)
    pos_arr = np.asarray(pos_mahal_values, dtype=float)
    sv_arr = np.asarray(sigma_v_used, dtype=float)
    sw_arr = np.asarray(sigma_w_used, dtype=float)

    def frac(a: np.ndarray, predicate) -> float:
        if len(a) == 0:
            return nan()
        return 100.0 * float(np.mean(predicate(a)))

    calibration = {
        "use_learned_q": bool(use_learned_q),
        "nis_count": int(len(nis_arr)),
        "nis_mean": float(np.mean(nis_arr)) if len(nis_arr) else nan(),
        "nis_median": float(np.median(nis_arr)) if len(nis_arr) else nan(),
        "nis_p95": float(np.percentile(nis_arr, 95)) if len(nis_arr) else nan(),
        "nis_coverage_95_pct": frac(nis_arr, lambda a: a <= CHI2_2_95),
        "nis_coverage_99_pct": frac(nis_arr, lambda a: a <= CHI2_2_99),
        "nees_count": int(len(nees_arr)),
        "nees_mean": float(np.mean(nees_arr)) if len(nees_arr) else nan(),
        "nees_median": float(np.median(nees_arr)) if len(nees_arr) else nan(),
        "nees_p95": float(np.percentile(nees_arr, 95)) if len(nees_arr) else nan(),
        "nees_upper95_coverage_pct": frac(nees_arr, lambda a: a <= CHI2_3_95),
        "nees_two_sided_95_pct": frac(nees_arr, lambda a: (a >= CHI2_3_025) & (a <= CHI2_3_975)),
        "position_ellipse_95_coverage_pct": frac(pos_arr, lambda a: a <= CHI2_2_95),
        "sigma_v_mean_mps": float(np.mean(sv_arr)),
        "sigma_v_p95_mps": float(np.percentile(sv_arr, 95)),
        "sigma_v_max_mps": float(np.max(sv_arr)),
        "sigma_v_upper_saturation_pct": frac(sv_arr, lambda a: a >= args.sigma_v_max - 0.01 * (args.sigma_v_max - args.sigma_v_min)),
        "sigma_omega_mean_radps": float(np.mean(sw_arr)),
        "sigma_omega_p95_radps": float(np.percentile(sw_arr, 95)),
        "sigma_omega_max_radps": float(np.max(sw_arr)),
        "sigma_omega_upper_saturation_pct": frac(sw_arr, lambda a: a >= args.sigma_omega_max - 0.01 * (args.sigma_omega_max - args.sigma_omega_min)),
        "gnss_seen": gnss_seen,
        "gnss_normal": gnss_normal,
        "gnss_reacquired": gnss_reacquired,
        "gnss_rejected": gnss_rejected,
        "gnss_quality_skipped": gnss_quality_skipped,
        "gnss_rejection_rate_pct": 100.0 * gnss_rejected / gnss_seen if gnss_seen else nan(),
    }
    return states, covariances, calibration, calibration_trace


# =============================================================================
# Phase-1 baseline official replay
# =============================================================================

def run_fixed_v5_replay(
    seq: PreparedSequence,
    *,
    args: argparse.Namespace,
    seq_dir: Path,
) -> dict[str, Any]:
    dynamics = np.zeros((len(seq.grid), 2), dtype=float)
    uncertainty = np.column_stack([
        np.full(len(seq.grid), 0.05, dtype=float),
        np.full(len(seq.grid), 0.01, dtype=float),
    ])
    states, _, calibration, trace = run_filter_variant(
        seq,
        dynamics,
        uncertainty,
        use_learned_q=False,
        args=args,
    )
    standard = summarize_standard_errors(
        states[:, :2],
        states[:, 2],
        seq.gt_xyz[:, :2],
        seq.gt_heading,
        args.rate_hz,
    )
    est_official = estimate_to_official_trajectory(seq, states)
    official = official_i2nav_evaluate(seq.official_truth, est_official)
    save_official_trajectory(seq_dir / "fixed_v5_estimate_traj.txt", est_official)
    save_official_trajectory(seq_dir / "official_reference_traj.txt", seq.official_truth)
    write_dict_rows(seq_dir / "fixed_v5_calibration_trace.csv", trace)

    return {
        "sequence": seq.name,
        "method": "fixed_v5_replay",
        "mode": "GNSS+ODO+IMU" if seq.gnss is not None else "ODO+IMU",
        "gnss_source": seq.gnss_source,
        "official_truth_source": seq.official_truth_source,
        **asdict(standard),
        **{f"cal_{k}": v for k, v in calibration.items()},
        **flatten_official_metrics(official),
    }


def compare_phase1_to_frozen_v5(
    phase1_rows: list[dict[str, Any]],
    frozen_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Check that this self-contained V5 replay matches the frozen LOSO V5 scalar ATE.

    The original LOSO script remains the source of truth for the frozen ablation. If
    this replay differs materially, the script flags it rather than silently treating
    a reimplemented reacquisition detail as identical.
    """
    frozen_by_seq: dict[str, float] = {}
    for r in frozen_rows:
        if r.get("method") != "fixed_v5":
            continue
        seq = r.get("test_sequence") or r.get("sequence")
        if not seq:
            continue
        frozen_by_seq[str(seq)] = safe_float(r.get("ate_rmse_m"))

    out: list[dict[str, Any]] = []
    for r in phase1_rows:
        seq = str(r.get("sequence", ""))
        if seq not in frozen_by_seq or r.get("status", "ok") == "failed":
            continue
        frozen_ate = frozen_by_seq[seq]
        replay_ate = safe_float(r.get("ate_rmse_m"))
        abs_diff = abs(replay_ate - frozen_ate)
        rel_pct = 100.0 * abs_diff / max(abs(frozen_ate), 1e-9)
        out.append(
            {
                "sequence": seq,
                "frozen_v5_ate_m": frozen_ate,
                "replay_v5_ate_m": replay_ate,
                "absolute_difference_m": abs_diff,
                "relative_difference_pct": rel_pct,
                "within_1pct": rel_pct <= 1.0,
                "within_5pct": rel_pct <= 5.0,
            }
        )
    return out


def flatten_official_metrics(official: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "official_available": official.get("available", False),
        "official_error": official.get("error", ""),
        "official_associated_poses": official.get("associated_poses"),
        "official_ape_translation_rmse_m": official.get("ape_translation_rmse_m"),
        "official_ape_rotation_rmse_deg": official.get("ape_rotation_rmse_deg"),
    }
    rpe = official.get("rpe", {}) if isinstance(official.get("rpe", {}), dict) else {}
    for delta in OFFICIAL_RPE_DELTAS_M:
        d = rpe.get(str(delta), {}) if isinstance(rpe, dict) else {}
        row[f"official_rpe_{delta}m_translation_rmse_m"] = d.get("translation_rmse_m")
        row[f"official_rpe_{delta}m_translation_pct"] = d.get("translation_percent")
        row[f"official_rpe_{delta}m_rotation_rmse_deg"] = d.get("rotation_rmse_deg")
    return row


# =============================================================================
# Refined fold evaluation
# =============================================================================

def evaluate_refined_fold(
    *,
    fold: FoldDefinition,
    seq: PreparedSequence,
    model: RefinedDualGRU,
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    fold_dir: Path,
) -> list[dict[str, Any]]:
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    dynamics, uncertainty = predict_model_outputs(
        model,
        seq,
        feature_mean=feature_mean,
        feature_std=feature_std,
        window=int(checkpoint["window"]),
        device=device,
        eval_batch_size=args.eval_batch_size,
    )

    pred_rows = []
    for k in range(len(seq.grid)):
        pred_rows.append(
            {
                "time_s": float(seq.grid[k]),
                "delta_v_mps": float(dynamics[k, 0]),
                "delta_omega_radps": float(dynamics[k, 1]),
                "sigma_v_mps": float(uncertainty[k, 0]),
                "sigma_omega_radps": float(uncertainty[k, 1]),
            }
        )
    write_dict_rows(fold_dir / "test_model_outputs.csv", pred_rows)

    rows: list[dict[str, Any]] = []
    for variant, use_learned_q in (
        ("predictive_nll_learned_q", True),
        ("predictive_nll_fixed_q", False),
    ):
        states, covariances, calibration, trace = run_filter_variant(
            seq, dynamics, uncertainty, use_learned_q=use_learned_q, args=args
        )
        standard = summarize_standard_errors(
            states[:, :2], states[:, 2], seq.gt_xyz[:, :2], seq.gt_heading, args.rate_hz
        )
        est_official = estimate_to_official_trajectory(seq, states)
        official = official_i2nav_evaluate(seq.official_truth, est_official)

        save_official_trajectory(fold_dir / f"{variant}_estimate_traj.txt", est_official)
        write_dict_rows(fold_dir / f"{variant}_calibration_trace.csv", trace)
        np.savez_compressed(
            fold_dir / f"{variant}_state_covariance.npz",
            time_s=seq.grid,
            state=states,
            covariance=covariances,
        )

        row = {
            "fold": fold.fold,
            "test_sequence": fold.test,
            "method": variant,
            "status": "ok",
            "train_sequences": ";".join(fold.train),
            "validation_sequences": ";".join(fold.validation),
            "samples": len(seq.grid),
            "duration_s": float(seq.grid[-1] - seq.grid[0]),
            "mode": "GNSS+ODO+IMU" if seq.gnss is not None else "ODO+IMU",
            "gnss_source": seq.gnss_source,
            "odo_source": seq.odo_source,
            "official_truth_source": seq.official_truth_source,
            **asdict(standard),
            **{f"cal_{k}": v for k, v in calibration.items()},
            **flatten_official_metrics(official),
        }
        rows.append(row)
        nis_text = f"{calibration['nis_coverage_95_pct']:.1f}%" if np.isfinite(calibration['nis_coverage_95_pct']) else "n/a"
        print(
            f"      {variant:<26} ATE={standard.ate_rmse_m:.3f} m  "
            f"RPE1={standard.rpe_1s_trans_rmse_m:.3f}  "
            f"RPE5={standard.rpe_5s_trans_rmse_m:.3f}  "
            f"RPE10={standard.rpe_10s_trans_rmse_m:.3f}  "
            f"head={standard.heading_mae_deg:.2f} deg  NIS95={nis_text}"
        )
    return rows


# =============================================================================
# Aggregation/statistics
# =============================================================================

def bootstrap_mean_ci(values: Sequence[float], seed: int, iterations: int = 10000) -> list[float] | None:
    a = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if len(a) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        means[i] = np.mean(rng.choice(a, size=len(a), replace=True))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize_method_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        by_method.setdefault(str(row["method"]), []).append(row)

    summary: dict[str, Any] = {}
    for method, mrows in by_method.items():
        ate = [safe_float(r.get("ate_rmse_m")) for r in mrows]
        rpe1 = [safe_float(r.get("rpe_1s_trans_rmse_m")) for r in mrows]
        rpe5 = [safe_float(r.get("rpe_5s_trans_rmse_m")) for r in mrows]
        rpe10 = [safe_float(r.get("rpe_10s_trans_rmse_m")) for r in mrows]
        heading = [safe_float(r.get("heading_mae_deg")) for r in mrows]
        gnss_free = [safe_float(r.get("ate_rmse_m")) for r in mrows if r.get("mode") == "ODO+IMU"]
        fused = [safe_float(r.get("ate_rmse_m")) for r in mrows if r.get("mode") != "ODO+IMU"]
        official_ape = [safe_float(r.get("official_ape_translation_rmse_m")) for r in mrows]
        nis95 = [safe_float(r.get("cal_nis_coverage_95_pct")) for r in mrows]
        nees95 = [safe_float(r.get("cal_nees_upper95_coverage_pct")) for r in mrows]
        pos95 = [safe_float(r.get("cal_position_ellipse_95_coverage_pct")) for r in mrows]

        summary[method] = {
            "held_out_sequences": len(mrows),
            "ate_macro_mean_m": mean_finite(ate),
            "ate_median_m": percentile_finite(ate, 50),
            "ate_sequence_rms_m": (
                float(np.sqrt(np.mean(np.asarray(ate) ** 2)))
                if len(ate) and np.all(np.isfinite(ate))
                else None
            ),
            "ate_bootstrap_95ci_m": bootstrap_mean_ci(ate, seed),
            "rpe_1s_macro_mean_m": mean_finite(rpe1),
            "rpe_5s_macro_mean_m": mean_finite(rpe5),
            "rpe_10s_macro_mean_m": mean_finite(rpe10),
            "heading_mae_macro_mean_deg": mean_finite(heading),
            "gnss_free_ate_mean_m": mean_finite(gnss_free),
            "gnss_fused_ate_mean_m": mean_finite(fused),
            "official_ape_translation_macro_mean_m": mean_finite(official_ape),
            "nis_95_coverage_macro_mean_pct": mean_finite(nis95),
            "nees_upper95_coverage_macro_mean_pct": mean_finite(nees95),
            "position_ellipse_95_coverage_macro_mean_pct": mean_finite(pos95),
        }
    return summary


def build_q_isolation_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seq: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_seq.setdefault(str(row["test_sequence"]), {})[str(row["method"])] = row
    out = []
    for seq, methods in by_seq.items():
        l = methods.get("predictive_nll_learned_q")
        f = methods.get("predictive_nll_fixed_q")
        if not l or not f:
            continue
        learned = safe_float(l.get("ate_rmse_m"))
        fixed = safe_float(f.get("ate_rmse_m"))
        out.append(
            {
                "sequence": seq,
                "learned_q_ate_m": learned,
                "fixed_q_ate_m": fixed,
                "learned_minus_fixed_ate_m": learned - fixed,
                "learned_q_better": learned < fixed,
                "learned_q_nis95_pct": safe_float(l.get("cal_nis_coverage_95_pct")),
                "fixed_q_nis95_pct": safe_float(f.get("cal_nis_coverage_95_pct")),
                "learned_q_nees95_pct": safe_float(l.get("cal_nees_upper95_coverage_pct")),
                "fixed_q_nees95_pct": safe_float(f.get("cal_nees_upper95_coverage_pct")),
                "sigma_v_mean_mps": safe_float(l.get("cal_sigma_v_mean_mps")),
                "sigma_omega_mean_radps": safe_float(l.get("cal_sigma_omega_mean_radps")),
            }
        )
    return out


def build_pilot_vs_original_dual(
    new_rows: list[dict[str, Any]],
    frozen_rows: list[dict[str, str]],
    *,
    max_regression_pct: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old: dict[str, dict[str, float]] = {}
    for r in frozen_rows:
        if r.get("method") != "gru_dual":
            continue
        seq = str(r.get("test_sequence") or r.get("sequence") or "")
        if not seq:
            continue
        old[seq] = {
            "ate": safe_float(r.get("ate_rmse_m")),
            "rpe1": safe_float(r.get("rpe_1s_trans_rmse_m")),
            "rpe5": safe_float(r.get("rpe_5s_trans_rmse_m")),
            "rpe10": safe_float(r.get("rpe_10s_trans_rmse_m")),
            "heading": safe_float(r.get("heading_mae_deg")),
        }

    candidate = {
        str(r.get("test_sequence")): r
        for r in new_rows
        if r.get("status") == "ok" and r.get("method") == "predictive_nll_learned_q"
    }
    rows: list[dict[str, Any]] = []
    wins = 0
    regressions: list[float] = []
    old_rpe5: list[float] = []
    new_rpe5: list[float] = []
    old_rpe10: list[float] = []
    new_rpe10: list[float] = []

    for seq, nr in candidate.items():
        if seq not in old:
            continue
        o = old[seq]
        n_ate = safe_float(nr.get("ate_rmse_m"))
        pct = 100.0 * (n_ate - o["ate"]) / max(abs(o["ate"]), 1e-9)
        win = n_ate < o["ate"]
        wins += int(win)
        regressions.append(pct)
        old_rpe5.append(o["rpe5"]); new_rpe5.append(safe_float(nr.get("rpe_5s_trans_rmse_m")))
        old_rpe10.append(o["rpe10"]); new_rpe10.append(safe_float(nr.get("rpe_10s_trans_rmse_m")))
        rows.append(
            {
                "sequence": seq,
                "original_dual_ate_m": o["ate"],
                "v3_ate_m": n_ate,
                "ate_change_pct": pct,
                "v3_ate_better": win,
                "original_dual_rpe1_m": o["rpe1"],
                "v3_rpe1_m": safe_float(nr.get("rpe_1s_trans_rmse_m")),
                "original_dual_rpe5_m": o["rpe5"],
                "v3_rpe5_m": safe_float(nr.get("rpe_5s_trans_rmse_m")),
                "original_dual_rpe10_m": o["rpe10"],
                "v3_rpe10_m": safe_float(nr.get("rpe_10s_trans_rmse_m")),
                "original_dual_heading_deg": o["heading"],
                "v3_heading_deg": safe_float(nr.get("heading_mae_deg")),
            }
        )

    completed = len(rows)
    max_reg = max(regressions) if regressions else None
    mean_old_rpe5 = mean_finite(old_rpe5)
    mean_new_rpe5 = mean_finite(new_rpe5)
    mean_old_rpe10 = mean_finite(old_rpe10)
    mean_new_rpe10 = mean_finite(new_rpe10)
    required_wins = 2 if completed >= 3 else completed
    go = bool(
        completed >= 3
        and wins >= required_wins
        and max_reg is not None and max_reg <= max_regression_pct
        and mean_old_rpe5 is not None and mean_new_rpe5 is not None and mean_new_rpe5 <= mean_old_rpe5
        and mean_old_rpe10 is not None and mean_new_rpe10 is not None and mean_new_rpe10 <= mean_old_rpe10
    )
    decision = {
        "completed_pilot_sequences": completed,
        "ate_wins_vs_original_dual": wins,
        "required_ate_wins": required_wins,
        "max_ate_regression_pct": max_reg,
        "allowed_max_ate_regression_pct": max_regression_pct,
        "original_dual_rpe5_mean_m": mean_old_rpe5,
        "v3_rpe5_mean_m": mean_new_rpe5,
        "original_dual_rpe10_mean_m": mean_old_rpe10,
        "v3_rpe10_mean_m": mean_new_rpe10,
        "go_to_full_loso": go,
        "rule": "GO only if V3 wins ATE on >=2/3 pilot folds, no fold regresses ATE by more than the configured tolerance, and mean 5 s and 10 s RPE do not worsen.",
    }
    return rows, decision


def load_refined_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[RefinedDualGRU, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if checkpoint.get("schema") != "i2nav_predictive_nll_dual_v3":
        raise ValueError(
            f"Checkpoint {checkpoint_path} is schema={checkpoint.get('schema')!r}; "
            "V3 evaluation requires an i2nav_predictive_nll_dual_v3 checkpoint."
        )
    model = RefinedDualGRU(
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        delta_v_limit=float(checkpoint["delta_v_limit"]),
        delta_omega_limit=float(checkpoint["delta_omega_limit"]),
        sigma_v_min=float(checkpoint["sigma_v_min"]),
        sigma_v_max=float(checkpoint["sigma_v_max"]),
        sigma_omega_min=float(checkpoint["sigma_omega_min"]),
        sigma_omega_max=float(checkpoint["sigma_omega_max"]),
        sigma_v_init=float(checkpoint.get("sigma_v_init", 0.05)),
        sigma_omega_init=float(checkpoint.get("sigma_omega_init", 0.02)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="i2Nav V3 pilot: horizon-weighted predictive GRU + WING-style heteroscedastic residual uncertainty."
    )
    p.add_argument("--root", type=Path, default=Path("public_datasets/im2nav"))
    p.add_argument(
        "--frozen-loso-dir",
        type=Path,
        default=Path("results/i2nav_loso_ablation"),
        help="Completed LOSO result directory. Read-only; never overwritten.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/i2nav_predictive_nll_pilot_v3"),
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing existing fold_XX_<sequence>/predictive_nll_best.pt "
            "checkpoints. Used with --eval-existing. Defaults to --output-dir."
        ),
    )
    p.add_argument(
        "--eval-existing",
        action="store_true",
        help="Re-evaluate existing refined checkpoints; do not retrain them.",
    )
    p.add_argument(
        "--allow-v5-mismatch",
        action="store_true",
        help="Continue even if repaired V5 replay differs from frozen V5 by >1%%.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument(
        "--folds", nargs="*", default=["building00", "building02", "parking02"],
        help="Pilot defaults to building00 building02 parking02. Pass explicit names/numbers to override."
    )
    p.add_argument("--full-loso", action="store_true", help="Ignore pilot default and run all available LOSO folds.")

    p.add_argument("--rate-hz", type=float, default=10.0)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--sigma-v-min", type=float, default=0.005)
    p.add_argument("--sigma-v-max", type=float, default=0.5)
    p.add_argument("--sigma-omega-min", type=float, default=0.002)
    p.add_argument("--sigma-omega-max", type=float, default=0.5)
    p.add_argument("--q-floor-xy-sigma-mps", type=float, default=0.005)
    p.add_argument("--q-floor-heading-sigma-radps", type=float, default=0.001)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=2048)
    p.add_argument("--train-stride", type=int, default=5)
    p.add_argument("--val-stride", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--uncertainty-warmup-epochs", type=int, default=5)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument(
        "--rollout-horizons-s",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0],
    )
    p.add_argument("--dyn-loss-weight", type=float, default=1.0)
    p.add_argument(
        "--rollout-horizon-weights", type=float, nargs="+", default=[0.5, 1.0, 2.0],
        help="Relative weights corresponding to --rollout-horizons-s. Default emphasizes 10 s."
    )
    p.add_argument("--rollout-loss-weight", type=float, default=1.0)
    p.add_argument("--heading-rollout-weight", type=float, default=1.0)
    p.add_argument("--nll-weight", type=float, default=0.10)
    p.add_argument("--pilot-max-regression-pct", type=float, default=15.0)

    # Frozen V5-safe filter constants.
    p.add_argument("--q-xy-sigma-mps", type=float, default=0.05)
    p.add_argument("--q-heading-sigma-radps", type=float, default=0.01)
    p.add_argument("--gnss-sigma-max-m", type=float, default=10.0)
    p.add_argument("--gnss-sigma-floor-m", type=float, default=0.05)
    p.add_argument("--gnss-anchor-count", type=int, default=1)
    p.add_argument("--gnss-nis-gate", type=float, default=CHI2_2_99)
    p.add_argument("--reacq-start-s", type=float, default=10.0)
    p.add_argument("--reacq-sigma-growth-mps", type=float, default=0.05)
    p.add_argument("--reacq-sigma-max-m", type=float, default=5.0)
    p.add_argument("--reacq-consecutive", type=int, default=3)
    p.add_argument("--imu-yaw-sign", type=float, choices=(-1.0, 1.0), default=-1.0)

    p.add_argument(
        "--skip-fixed-v5-official-replay",
        action="store_true",
        help="Skip Phase-1 fixed-V5 replay/evo evaluation.",
    )
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Run Phase 1 only; do not train refined folds.",
    )
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    req = requested.lower()
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but torch.cuda.is_available() is False; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def filter_requested_folds(folds: list[FoldDefinition], requested: list[str] | None) -> list[FoldDefinition]:
    if not requested:
        return folds
    wanted = set(str(x) for x in requested)
    return [f for f in folds if str(f.fold) in wanted or f.test in wanted]


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    if args.full_loso:
        args.folds = None
    seed_everything(args.seed)

    root = args.root.resolve()
    frozen_dir = args.frozen_loso_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_dir = (args.checkpoint_dir or args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"ERROR: i2Nav root does not exist: {root}")
        return 2

    device = resolve_device(args.device)
    print("=" * 100)
    print("i2Nav PREDICTIVE NLL PILOT V3")
    print("=" * 100)
    print(f"Dataset root       : {root}")
    print(f"Frozen LOSO dir    : {frozen_dir}")
    print(f"New output dir     : {output_dir}")
    if args.eval_existing:
        print(f"Checkpoint dir     : {checkpoint_dir}")
        print("Mode               : EVALUATE EXISTING CHECKPOINTS (no retraining)")
    print(f"Device             : {device}")
    if device.type == "cuda":
        print(f"GPU                : {torch.cuda.get_device_name(device)}")
    print(f"evo available      : {EVO_AVAILABLE}")
    if not EVO_AVAILABLE:
        print("  -> Official Phase-1 metrics need evo: python -m pip install evo")
    print()

    # -------------------------------------------------------------------------
    # Freeze/copy prior LOSO evidence.
    # -------------------------------------------------------------------------
    frozen = preserve_frozen_results(frozen_dir, output_dir)
    if frozen.get("summary"):
        methods = frozen["summary"].get("method_summaries", {})
        print("Frozen completed LOSO headline (copied, never overwritten):")
        for method in ("fixed_v5", "heuristic_v6", "gru_dynamics", "gru_q", "gru_dual"):
            d = methods.get(method)
            if d:
                print(f"  {method:<16} mean ATE = {d.get('ate_macro_mean_m', nan()):.3f} m")
        print()
    else:
        print("[warn] No frozen LOSO summary found; continuing with new study.")

    # -------------------------------------------------------------------------
    # Prepare each sequence ONCE.
    # -------------------------------------------------------------------------
    files = discover_files(root)
    if not files:
        print("ERROR: no usable i2Nav sequences found.")
        return 2
    print("Preparing i2Nav sequences...")
    prepared: dict[str, PreparedSequence] = {}
    for sf in files:
        try:
            seq = prepare_sequence(
                sf,
                hz=args.rate_hz,
                imu_yaw_sign=args.imu_yaw_sign,
                gnss_sigma_max_m=args.gnss_sigma_max_m,
                gnss_anchor_count=args.gnss_anchor_count,
            )
            prepared[seq.name] = seq
            print(
                f"  {seq.name:<12} samples={len(seq.grid):6d} "
                f"GNSS={seq.gnss_source:<12} official_ref={Path(seq.official_truth_source).name}"
            )
        except Exception as exc:
            print(f"  [FAILED] {sf.name}: {type(exc).__name__}: {exc}")

    if not prepared:
        print("ERROR: all sequence preparation failed.")
        return 2
    print()

    folds = load_folds(frozen_dir, list(prepared.keys()))
    folds = filter_requested_folds(folds, args.folds)
    folds = [
        f
        for f in folds
        if f.test in prepared
        and all(x in prepared for x in f.validation)
        and all(x in prepared for x in f.train)
    ]
    if not folds:
        print("ERROR: no requested complete folds are available.")
        return 2

    # -------------------------------------------------------------------------
    # PRE-FLIGHT: verify ENU/FLU -> official NED/FRD trajectory conversion.
    # -------------------------------------------------------------------------
    print("=" * 100)
    print("PRE-FLIGHT: OFFICIAL FRAME / BODY-CONVENTION SELF-CHECK")
    print("=" * 100)
    selfcheck_rows: list[dict[str, Any]] = []
    test_names = []
    for f in folds:
        if f.test not in test_names:
            test_names.append(f.test)
    for name in test_names:
        row = official_gt_frame_selfcheck(prepared[name])
        selfcheck_rows.append(row)
        print(
            f"  {name:<12} GT-roundtrip APE_t="
            f"{safe_float(row.get('official_ape_translation_rmse_m')):.4f} m  "
            f"APE_R={safe_float(row.get('official_ape_rotation_rmse_deg')):.2f} deg  "
            f"pos_ok={row.get('position_frame_check_pass')}  "
            f"body_ok={row.get('no_180deg_body_frame_bug')}"
        )
    write_dict_rows(output_dir / "official_gt_frame_selfcheck.csv", selfcheck_rows)
    if any(not bool(r.get("position_frame_check_pass")) or not bool(r.get("no_180deg_body_frame_bug")) for r in selfcheck_rows):
        print("  [IMPORTANT] Official frame self-check failed. Do not use official SOTA metrics yet.")
    print()

    # -------------------------------------------------------------------------
    # PHASE 1: official protocol replay of fixed V5-like robust filter.
    # -------------------------------------------------------------------------
    phase1_rows: list[dict[str, Any]] = []
    if not args.skip_fixed_v5_official_replay:
        print("=" * 100)
        print("PHASE 1: OFFICIAL i2Nav/evo METRIC PROTOCOL ON FIXED-Q ROBUST BASELINE")
        print("=" * 100)
        phase1_dir = output_dir / "phase1_official_fixed_v5"
        # Only test sequences requested by the selected folds; each once.
        for name in test_names:
            print(f"  [official baseline] {name}")
            try:
                row = run_fixed_v5_replay(
                    prepared[name],
                    args=args,
                    seq_dir=phase1_dir / name,
                )
                phase1_rows.append(row)
                print(
                    f"      anchored ATE={row['ate_rmse_m']:.3f} m  "
                    f"official APE={safe_float(row.get('official_ape_translation_rmse_m')):.3f} m"
                )
            except Exception as exc:
                phase1_rows.append(
                    {
                        "sequence": name,
                        "method": "fixed_v5_replay",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"      FAILED: {type(exc).__name__}: {exc}")
        write_dict_rows(output_dir / "phase1_official_fixed_v5.csv", phase1_rows)
        replay_check = compare_phase1_to_frozen_v5(phase1_rows, frozen.get("rows", []))
        write_dict_rows(output_dir / "phase1_v5_replay_consistency.csv", replay_check)
        if replay_check:
            worst = max(replay_check, key=lambda r: r["relative_difference_pct"])
            print(
                f"  V5 replay consistency: worst frozen-vs-replay ATE difference = "
                f"{worst['relative_difference_pct']:.2f}% on {worst['sequence']}"
            )
            if worst["relative_difference_pct"] > 5.0:
                print(
                    "  [IMPORTANT] Replay differs >5% from frozen V5. Treat the original LOSO "
                    "V5 rows as the ablation source of truth and reconcile gate/reacquisition "
                    "implementation before using this replay as a like-for-like V5 comparison."
                )
            if worst["relative_difference_pct"] > 1.0 and not args.allow_v5_mismatch:
                print(
                    "ERROR: repaired V5 replay still differs from the frozen V5 by >1%. "
                    "Stopping before refined-model evaluation/training. Use --allow-v5-mismatch "
                    "only for diagnostics, not final science."
                )
                return 3
        print()

    protocol_manifest = {
        "schema": "i2nav_official_protocol_manifest_v1",
        "source_repository": "i2Nav-WHU/evaluate_odometry",
        "max_time_sync_diff_s": OFFICIAL_MAX_TIME_SYNC_DIFF_S,
        "alignment": "SE3, correct_scale=False, correct_only_scale=False",
        "ape": ["translation_part RMSE", "rotation_angle_deg RMSE"],
        "rpe_deltas_m": list(OFFICIAL_RPE_DELTAS_M),
        "rpe_unit": "meters",
        "rpe_relative_delta_tolerance": OFFICIAL_RPE_REL_DELTA_TOL,
        "rpe_all_pairs": OFFICIAL_RPE_ALL_PAIRS,
        "important_separation": (
            "Official i2Nav RPE is distance-based. Project 1/5/10 s RPE is retained "
            "separately and is not presented as the official benchmark metric."
        ),
        "planar_model_note": (
            "The twin estimates x/y/yaw internally in ENU/FLU. Before official evo evaluation, "
            "estimates are explicitly converted to i2Nav NED/FRD; estimated height is held at "
            "its initial value so future GT z is not copied into the estimate."
        ),
    }
    write_json(output_dir / "official_protocol_manifest.json", protocol_manifest)

    if args.skip_training:
        print("--skip-training supplied; Phase 1 complete.")
        return 0

    # -------------------------------------------------------------------------
    # PHASE 2 + 3: refined multi-step dual + same-checkpoint Q isolation.
    # -------------------------------------------------------------------------
    all_new_rows: list[dict[str, Any]] = []
    print("=" * 100)
    print("PHASE 2+3: HORIZON-WEIGHTED GRU + HETEROSCEDASTIC NLL + Q ISOLATION")
    print("=" * 100)

    for outer_i, fold in enumerate(folds, start=1):
        print()
        print("=" * 100)
        print(f"OUTER FOLD {fold.fold}  ({outer_i}/{len(folds)})")
        print(f"TEST : {fold.test}")
        print(f"VAL  : {', '.join(fold.validation)}")
        print(f"TRAIN: {', '.join(fold.train)}")
        print("=" * 100)

        fold_seed = args.seed + fold.fold * 1009
        seed_everything(fold_seed)
        fold_dir = output_dir / f"fold_{fold.fold:02d}_{fold.test}"

        try:
            train_seqs = [prepared[n] for n in fold.train]
            val_seqs = [prepared[n] for n in fold.validation]
            test_seq = prepared[fold.test]

            if args.eval_existing:
                source_fold_dir = checkpoint_dir / f"fold_{fold.fold:02d}_{fold.test}"
                checkpoint_path = source_fold_dir / "predictive_nll_best.pt"
                print(f"      loading existing checkpoint: {checkpoint_path}")
                model, checkpoint = load_refined_checkpoint(
                    checkpoint_path,
                    device=device,
                )
            else:
                model, checkpoint = train_refined_dual(
                    fold=fold,
                    train_sequences=train_seqs,
                    val_sequences=val_seqs,
                    args=args,
                    device=device,
                    fold_dir=fold_dir,
                )
            fold_rows = evaluate_refined_fold(
                fold=fold,
                seq=test_seq,
                model=model,
                checkpoint=checkpoint,
                args=args,
                device=device,
                fold_dir=fold_dir,
            )
            all_new_rows.extend(fold_rows)
        except RuntimeError as exc:
            # Give a helpful CUDA OOM recovery path without silently changing science.
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  FOLD FAILED: {msg}")
            if "out of memory" in str(exc).lower():
                print("  CUDA OOM: rerun with --batch-size 8 (or 4); do not change folds.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for method in ("predictive_nll_learned_q", "predictive_nll_fixed_q"):
                all_new_rows.append(
                    {
                        "fold": fold.fold,
                        "test_sequence": fold.test,
                        "method": method,
                        "status": "failed",
                        "error": msg,
                    }
                )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  FOLD FAILED: {msg}")
            for method in ("predictive_nll_learned_q", "predictive_nll_fixed_q"):
                all_new_rows.append(
                    {
                        "fold": fold.fold,
                        "test_sequence": fold.test,
                        "method": method,
                        "status": "failed",
                        "error": msg,
                    }
                )

        # Partial results after EVERY fold.
        write_dict_rows(output_dir / "refined_loso_results.csv", all_new_rows)
        write_json(
            output_dir / "refined_loso_partial_summary.json",
            {
                "completed_rows": len(all_new_rows),
                "method_summaries": summarize_method_rows(all_new_rows, args.seed),
            },
        )

    # -------------------------------------------------------------------------
    # Final aggregation and decision evidence.
    # -------------------------------------------------------------------------
    method_summaries = summarize_method_rows(all_new_rows, args.seed)
    q_compare = build_q_isolation_comparison(all_new_rows)
    write_dict_rows(output_dir / "q_isolation_comparison.csv", q_compare)

    learned_mean = method_summaries.get("predictive_nll_learned_q", {}).get("ate_macro_mean_m")
    fixed_mean = method_summaries.get("predictive_nll_fixed_q", {}).get("ate_macro_mean_m")
    learned_nis = method_summaries.get("predictive_nll_learned_q", {}).get("nis_95_coverage_macro_mean_pct")
    fixed_nis = method_summaries.get("predictive_nll_fixed_q", {}).get("nis_95_coverage_macro_mean_pct")

    pilot_rows, pilot_decision = build_pilot_vs_original_dual(
        all_new_rows, frozen.get("rows", []), max_regression_pct=args.pilot_max_regression_pct
    )
    write_dict_rows(output_dir / "pilot_vs_original_dual.csv", pilot_rows)
    write_json(output_dir / "pilot_go_no_go.json", pilot_decision)

    if learned_mean is not None and fixed_mean is not None:
        if abs(learned_mean - fixed_mean) <= 0.05 * max(fixed_mean, 1e-9):
            q_interpretation = (
                "Residual-uncertainty Q and fixed-Q ATE are within 5%. Prefer fixed/tightly constrained Q "
                "for deployment unless learned Q has materially better calibration; the Q head "
                "may be serving mainly as an auxiliary multi-task training signal."
            )
        elif learned_mean < fixed_mean:
            q_interpretation = (
                "Residual-uncertainty Q materially improves ATE. Keep it only if NIS/NEES/coverage also show "
                "credible calibration and alpha saturation is controlled."
            )
        else:
            q_interpretation = (
                "Fixed Q beats residual-uncertainty Q with the exact same dynamics checkpoint. Keep the uncertainty head "
                "as an ablation/auxiliary training signal, but deploy fixed or tightly constrained Q."
            )
    else:
        q_interpretation = "Insufficient completed folds to choose learned versus fixed Q."

    final_summary = {
        "schema": "i2nav_predictive_nll_pilot_v3",
        "device": str(device),
        "evo_available": EVO_AVAILABLE,
        "frozen_loso_summary": frozen.get("summary"),
        "new_method_summaries": method_summaries,
        "q_isolation_interpretation": q_interpretation,
        "pilot_go_no_go": pilot_decision,
        "q_isolation_headline": {
            "learned_q_ate_macro_mean_m": learned_mean,
            "fixed_q_ate_macro_mean_m": fixed_mean,
            "learned_q_nis95_macro_mean_pct": learned_nis,
            "fixed_q_nis95_macro_mean_pct": fixed_nis,
        },
        "official_protocol": protocol_manifest,
        "model_configuration": {
            "rate_hz": args.rate_hz,
            "window_samples": args.window,
            "window_seconds": args.window / args.rate_hz,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "rollout_horizons_s": args.rollout_horizons_s,
            "rollout_horizon_weights": args.rollout_horizon_weights,
            "sigma_v_bounds_mps": [args.sigma_v_min, args.sigma_v_max],
            "sigma_omega_bounds_radps": [args.sigma_omega_min, args.sigma_omega_max],
            "q_floor_xy_sigma_mps": args.q_floor_xy_sigma_mps,
            "q_floor_heading_sigma_radps": args.q_floor_heading_sigma_radps,
            "q_xy_sigma_mps": args.q_xy_sigma_mps,
            "q_heading_sigma_radps": args.q_heading_sigma_radps,
            "gnss_nis_gate": args.gnss_nis_gate,
        },
        "security_constraint": (
            "No neural/adaptive input contains GNSS position, GNSS innovation/residual, NIS, "
            "GNSS covariance, or reported GNSS uncertainty. GNSS is external to the learned twin."
        ),
        "reporting_rule": (
            "Only each outer fold's held-out test sequence contributes to headline LOSO metrics. "
            "Validation chooses checkpoints only. Official distance-RPE and project time-RPE are "
            "reported as distinct metric families."
        ),
    }
    write_json(output_dir / "final_summary.json", final_summary)

    print()
    print("=" * 100)
    print("V3 PILOT SUMMARY")
    print("=" * 100)
    for method, d in method_summaries.items():
        print(
            f"  {method:<24} "
            f"ATE={d.get('ate_macro_mean_m')}  "
            f"RPE1={d.get('rpe_1s_macro_mean_m')}  "
            f"head={d.get('heading_mae_macro_mean_deg')}"
        )
    print()
    print("PILOT GO/NO-GO against frozen original dual GRU:")
    print(f"  ATE wins: {pilot_decision.get('ate_wins_vs_original_dual')}/{pilot_decision.get('completed_pilot_sequences')}")
    print(f"  max ATE regression: {pilot_decision.get('max_ate_regression_pct')}")
    print(f"  mean RPE5 old/new: {pilot_decision.get('original_dual_rpe5_mean_m')} / {pilot_decision.get('v3_rpe5_mean_m')}")
    print(f"  mean RPE10 old/new: {pilot_decision.get('original_dual_rpe10_mean_m')} / {pilot_decision.get('v3_rpe10_mean_m')}")
    print(f"  GO TO FULL LOSO: {pilot_decision.get('go_to_full_loso')}")
    print()
    print("Q isolation interpretation:")
    print(f"  {q_interpretation}")
    print()
    print(f"Results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
