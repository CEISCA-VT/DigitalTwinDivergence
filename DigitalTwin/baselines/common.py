#!/usr/bin/env python3
"""Common utilities for the i2Nav external-baseline suite.

The suite deliberately uses a canonical planar trajectory schema so the same
fidelity evaluators can be applied to Fixed Physics, classical filters,
learned baselines, Twin V1/V2, and externally generated methods.

Important protocol rule
-----------------------
Ground truth may be used for *training-fold calibration/labels* and for the
initial pose of a held-out trajectory. It must never be used to correct a
held-out trajectory after initialization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json
import math
import re
import warnings

import numpy as np
import pandas as pd

SEQUENCE_RE = re.compile(r"(building\d+|parking\d+|playground\d+|street\d+)", re.I)
SEED_PATTERNS = [
    re.compile(r"replicate_(\d+)_base(\d+)", re.I),
    re.compile(r"seed[_-]?(\d+)", re.I),
    re.compile(r"base[_-]?(\d+)", re.I),
]

# Aliases make the evaluators usable on official/recomputed/external outputs.
ALIASES: Dict[str, Sequence[str]] = {
    "time_s": ("time_s", "time", "timestamp_s", "timestamp", "t"),
    "gt_east_m": ("gt_east_m", "gt_x_m", "gt_x", "ground_truth_x", "physical_x", "x_gt"),
    "gt_north_m": ("gt_north_m", "gt_y_m", "gt_y", "ground_truth_y", "physical_y", "y_gt"),
    "gt_heading_rad": ("gt_heading_rad", "gt_yaw_rad", "gt_theta_rad", "ground_truth_heading_rad", "physical_heading_rad", "heading_gt_rad"),
    "estimate_east_m": ("estimate_east_m", "est_east_m", "estimate_x_m", "est_x", "twin_x", "virtual_x", "x_est"),
    "estimate_north_m": ("estimate_north_m", "est_north_m", "estimate_y_m", "est_y", "twin_y", "virtual_y", "y_est"),
    "estimate_heading_rad": ("estimate_heading_rad", "est_heading_rad", "estimate_yaw_rad", "est_yaw_rad", "twin_heading_rad", "virtual_heading_rad", "heading_est_rad"),
    "odo_speed_mps": ("odo_speed_mps", "wheel_speed_mps", "wheel_velocity_mps", "forward_speed_mps", "odom_speed_mps"),
    "imu_yaw_rate_radps": ("imu_yaw_rate_radps", "imu_yaw_radps", "gyro_z_radps", "yaw_rate_radps"),
    "wheel_yaw_radps": ("wheel_yaw_radps", "wheel_yaw_rate_radps", "odo_yaw_rate_radps"),
    "wheel_imu_yaw_disagreement_radps": ("wheel_imu_yaw_disagreement_radps", "wheel_imu_yaw_diff_radps"),
}

REQUIRED_POSE = [
    "time_s",
    "gt_east_m",
    "gt_north_m",
    "gt_heading_rad",
    "estimate_east_m",
    "estimate_north_m",
    "estimate_heading_rad",
]
REQUIRED_RAW = [
    "time_s",
    "gt_east_m",
    "gt_north_m",
    "gt_heading_rad",
    "odo_speed_mps",
    "imu_yaw_rate_radps",
]


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def sequence_id(path: str | Path) -> str:
    p = str(path)
    m = SEQUENCE_RE.search(p)
    return m.group(1).lower() if m else Path(p).parent.name.lower()


def seed_id(path: str | Path) -> str:
    text = str(path)
    m = SEED_PATTERNS[0].search(text)
    if m:
        return f"rep{m.group(1)}_base{m.group(2)}"
    for pat in SEED_PATTERNS[1:]:
        m = pat.search(text)
        if m:
            return f"seed_{m.group(1)}"
    return "deterministic"


def _resolve_alias(columns: Iterable[str], canonical: str) -> Optional[str]:
    lookup = {c.lower(): c for c in columns}
    for a in ALIASES.get(canonical, (canonical,)):
        if a.lower() in lookup:
            return lookup[a.lower()]
    return None


def canonicalize_columns(df: pd.DataFrame, required: Sequence[str] = ()) -> pd.DataFrame:
    """Rename recognized aliases to the canonical schema without overwriting existing names."""
    ren = {}
    for canonical in ALIASES:
        if canonical in df.columns:
            continue
        found = _resolve_alias(df.columns, canonical)
        if found is not None and found != canonical:
            ren[found] = canonical
    out = df.rename(columns=ren).copy()
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}; available columns: {list(df.columns)}")
    return out


def clean_numeric_time(df: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    out = canonicalize_columns(df, required)
    num_cols = [c for c in set(required).union(ALIASES.keys()) if c in out.columns]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=list(required)).sort_values("time_s", kind="mergesort")
    out = out.loc[~out["time_s"].duplicated(keep="first")].reset_index(drop=True)
    if len(out) < 3:
        raise ValueError("Trajectory has fewer than 3 valid unique timestamps")
    return out


def _merge_prediction_trace(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge optional V2 prediction-trace wheel-yaw channels by timestamp."""
    sibling_names = [
        "v2_prediction_trace.csv",
        "prediction_trace.csv",
    ]
    pred = None
    for name in sibling_names:
        q = path.with_name(name)
        if q.exists():
            pred = q
            break
    if pred is None:
        return df
    try:
        p = pd.read_csv(pred)
        p = canonicalize_columns(p)
        if "time_s" not in p.columns:
            return df
        keep = [c for c in [
            "time_s", "wheel_yaw_radps", "imu_yaw_rate_radps",
            "wheel_imu_yaw_disagreement_radps",
        ] if c in p.columns]
        p = p[keep].copy()
        p["time_s"] = pd.to_numeric(p["time_s"], errors="coerce")
        p = p.dropna(subset=["time_s"]).sort_values("time_s")
        for c in keep:
            if c != "time_s":
                p[c] = pd.to_numeric(p[c], errors="coerce")
        # Preserve evaluated-trajectory values when they already exist.
        add = [c for c in keep if c == "time_s" or c not in df.columns]
        if len(add) <= 1:
            return df
        med_dt = np.nanmedian(np.diff(df["time_s"].to_numpy(float)))
        tol = max(0.02, 0.6 * med_dt) if np.isfinite(med_dt) and med_dt > 0 else 0.06
        return pd.merge_asof(
            df.sort_values("time_s"), p[add].sort_values("time_s"),
            on="time_s", direction="nearest", tolerance=tol,
        )
    except Exception as exc:
        warnings.warn(f"Could not merge optional prediction trace next to {path}: {exc}")
        return df


def load_raw_sequence(path: str | Path, merge_prediction_trace: bool = True) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df = clean_numeric_time(df, REQUIRED_RAW)
    if merge_prediction_trace:
        df = _merge_prediction_trace(df, path)
    return df


def load_pose_trajectory(path: str | Path) -> pd.DataFrame:
    return clean_numeric_time(pd.read_csv(path), REQUIRED_POSE)


def robust_dt(t: np.ndarray) -> float:
    dt = np.diff(np.asarray(t, float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt):
        raise ValueError("No positive timestamp increments")
    return float(np.median(dt))


def angular_rate(t: np.ndarray, heading_rad: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    h = np.unwrap(np.asarray(heading_rad, float))
    if len(t) < 3:
        return np.gradient(h, t, edge_order=1)
    return np.gradient(h, t, edge_order=2)


def forward_speed(t: np.ndarray, x: np.ndarray, y: np.ndarray, heading_rad: np.ndarray) -> np.ndarray:
    """Ground-truth body-forward speed from differentiated world-frame pose."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    h = np.asarray(heading_rad, float)
    eo = 2 if len(t) >= 3 else 1
    vx = np.gradient(x, t, edge_order=eo)
    vy = np.gradient(y, t, edge_order=eo)
    return vx * np.cos(h) + vy * np.sin(h)


def add_ground_truth_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = out["time_s"].to_numpy(float)
    out["gt_forward_speed_mps"] = forward_speed(
        t,
        out["gt_east_m"].to_numpy(float),
        out["gt_north_m"].to_numpy(float),
        out["gt_heading_rad"].to_numpy(float),
    )
    out["gt_yaw_rate_radps"] = angular_rate(t, out["gt_heading_rad"].to_numpy(float))
    return out


def add_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create reduced-input features shared by the learned baseline adaptations."""
    out = add_ground_truth_kinematics(df)
    t = out["time_s"].to_numpy(float)
    v = out["odo_speed_mps"].to_numpy(float)
    w = out["imu_yaw_rate_radps"].to_numpy(float)
    eo = 2 if len(t) >= 3 else 1
    out["odo_accel_mps2"] = np.gradient(v, t, edge_order=eo)
    out["imu_yaw_accel_radps2"] = np.gradient(w, t, edge_order=eo)
    if "wheel_yaw_radps" not in out.columns:
        out["wheel_yaw_radps"] = np.nan
    if "wheel_imu_yaw_disagreement_radps" not in out.columns:
        out["wheel_imu_yaw_disagreement_radps"] = out["wheel_yaw_radps"] - out["imu_yaw_rate_radps"]
    return out


def integrate_planar(
    time_s: np.ndarray,
    speed_mps: np.ndarray,
    yaw_rate_radps: np.ndarray,
    initial_pose: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint integration of planar forward speed and yaw rate."""
    t = np.asarray(time_s, float)
    v = np.asarray(speed_mps, float)
    w = np.asarray(yaw_rate_radps, float)
    n = len(t)
    if not (len(v) == len(w) == n):
        raise ValueError("time, speed and yaw-rate arrays must have identical length")
    x = np.empty(n, float); y = np.empty(n, float); h = np.empty(n, float)
    x[0], y[0], h[0] = map(float, initial_pose)
    for k in range(1, n):
        dt = max(0.0, float(t[k] - t[k - 1]))
        vk = 0.5 * (v[k - 1] + v[k])
        wk = 0.5 * (w[k - 1] + w[k])
        hm = h[k - 1] + 0.5 * wk * dt
        x[k] = x[k - 1] + vk * math.cos(hm) * dt
        y[k] = y[k - 1] + vk * math.sin(hm) * dt
        h[k] = float(wrap_angle(h[k - 1] + wk * dt))
    return x, y, h


def standardized_output(
    source: pd.DataFrame,
    est_x: np.ndarray,
    est_y: np.ndarray,
    est_heading: np.ndarray,
    method: str,
    corrected_v: Optional[np.ndarray] = None,
    corrected_omega: Optional[np.ndarray] = None,
    extra: Optional[Mapping[str, np.ndarray | float | str]] = None,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "time_s": source["time_s"].to_numpy(float),
        "gt_east_m": source["gt_east_m"].to_numpy(float),
        "gt_north_m": source["gt_north_m"].to_numpy(float),
        "gt_heading_rad": source["gt_heading_rad"].to_numpy(float),
        "estimate_east_m": np.asarray(est_x, float),
        "estimate_north_m": np.asarray(est_y, float),
        "estimate_heading_rad": np.asarray(est_heading, float),
        "odo_speed_mps": source["odo_speed_mps"].to_numpy(float),
        "imu_yaw_rate_radps": source["imu_yaw_rate_radps"].to_numpy(float),
        "method": method,
    })
    for c in ["wheel_yaw_radps", "wheel_imu_yaw_disagreement_radps"]:
        if c in source.columns:
            out[c] = source[c].to_numpy(float)
    if corrected_v is not None:
        out["corrected_v_mps"] = np.asarray(corrected_v, float)
    if corrected_omega is not None:
        out["corrected_omega_radps"] = np.asarray(corrected_omega, float)
    if extra:
        for k, v in extra.items():
            if np.isscalar(v):
                out[k] = v
            else:
                arr = np.asarray(v)
                if len(arr) != len(out):
                    raise ValueError(f"Extra column {k} length {len(arr)} != trajectory length {len(out)}")
                out[k] = arr
    return out


def _raw_fingerprint(df: pd.DataFrame) -> str:
    cols = [c for c in [
        "time_s", "gt_east_m", "gt_north_m", "gt_heading_rad",
        "odo_speed_mps", "imu_yaw_rate_radps", "wheel_yaw_radps",
    ] if c in df.columns]
    a = df[cols].to_numpy(float)
    # Rounding removes irrelevant CSV representation differences while catching actual data mismatch.
    a = np.round(a, decimals=8)
    return hashlib.sha256(a.tobytes()).hexdigest()


@dataclass
class CorpusSequence:
    sequence: str
    path: Path
    data: pd.DataFrame
    duplicates: List[Path]
    raw_fingerprint: str


def discover_i2nav_corpus(
    input_root: str | Path,
    glob_pattern: str = "**/v2_evaluated_trajectory.csv",
    verify_duplicates: bool = True,
) -> Dict[str, CorpusSequence]:
    """Deduplicate the 3-seed V2 archive into one raw/GT source per physical sequence.

    The evaluated V2 files are used only as a convenient container for raw wheel/IMU
    channels and ground truth. V2 estimates are not used by the maintenance baselines.
    """
    root = Path(input_root)
    files = sorted(root.glob(glob_pattern), key=lambda p: str(p).lower())
    if not files:
        raise FileNotFoundError(f"No source files matched {root / glob_pattern}")
    groups: Dict[str, List[Path]] = {}
    for p in files:
        groups.setdefault(sequence_id(p), []).append(p)
    corpus: Dict[str, CorpusSequence] = {}
    for seq, paths in sorted(groups.items()):
        loaded = [(p, load_raw_sequence(p, merge_prediction_trace=True)) for p in paths]
        p0, d0 = loaded[0]
        fp0 = _raw_fingerprint(d0)
        if verify_duplicates and len(loaded) > 1:
            for p, d in loaded[1:]:
                fp = _raw_fingerprint(d)
                if fp != fp0:
                    # Wheel-yaw merge can differ if prediction trace is seed-dependent. Recheck the
                    # truly raw/GT channels before declaring the physical sequence inconsistent.
                    base_cols = ["time_s", "gt_east_m", "gt_north_m", "gt_heading_rad", "odo_speed_mps", "imu_yaw_rate_radps"]
                    h0 = hashlib.sha256(np.round(d0[base_cols].to_numpy(float), 8).tobytes()).hexdigest()
                    h1 = hashlib.sha256(np.round(d[base_cols].to_numpy(float), 8).tobytes()).hexdigest()
                    if h0 != h1:
                        raise ValueError(
                            f"Raw/GT data for {seq} differs across replicate files:\n  {p0}\n  {p}\n"
                            "Do not train baselines until this provenance mismatch is understood."
                        )
                    warnings.warn(
                        f"Optional derived wheel-yaw channels differ across replicates for {seq}; "
                        f"using {p0} as the canonical source."
                    )
        corpus[seq] = CorpusSequence(seq, p0, add_sensor_features(d0), list(paths), fp0)
    return corpus


def fit_speed_calibration(train_frames: Sequence[pd.DataFrame]) -> Tuple[float, float]:
    """Robust-ish affine mapping GT forward speed ~= scale*wheel_speed + bias.

    Fitted only on training sequences. A small ridge term prevents singular fits.
    """
    x = np.concatenate([d["odo_speed_mps"].to_numpy(float) for d in train_frames])
    y = np.concatenate([d["gt_forward_speed_mps"].to_numpy(float) for d in train_frames])
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 20) & (np.abs(y) < 20)
    x = x[ok]; y = y[ok]
    if len(x) < 20:
        return 1.0, 0.0
    # Trim extreme derivative artifacts.
    r = y - x
    lo, hi = np.quantile(r, [0.01, 0.99])
    keep = (r >= lo) & (r <= hi)
    X = np.column_stack([x[keep], np.ones(np.sum(keep))])
    A = X.T @ X + np.diag([1e-6, 1e-6])
    b = X.T @ y[keep]
    scale, bias = np.linalg.solve(A, b)
    return float(scale), float(bias)


def training_noise_statistics(train_frames: Sequence[pd.DataFrame]) -> Dict[str, float]:
    dv = []
    dw = []
    wheel_dw = []
    for d in train_frames:
        dv.append(d["gt_forward_speed_mps"].to_numpy(float) - d["odo_speed_mps"].to_numpy(float))
        dw.append(d["gt_yaw_rate_radps"].to_numpy(float) - d["imu_yaw_rate_radps"].to_numpy(float))
        if "wheel_yaw_radps" in d.columns:
            a = d["wheel_yaw_radps"].to_numpy(float) - d["gt_yaw_rate_radps"].to_numpy(float)
            wheel_dw.append(a[np.isfinite(a)])
    dv = np.concatenate(dv); dw = np.concatenate(dw)
    def rs(a: np.ndarray, floor: float) -> float:
        a = a[np.isfinite(a)]
        if len(a) < 10:
            return floor
        med = np.median(a); mad = np.median(np.abs(a - med))
        return float(max(floor, 1.4826 * mad))
    out = {
        "wheel_speed_sigma": rs(dv, 0.02),
        "imu_yaw_sigma": rs(dw, math.radians(0.2)),
    }
    if wheel_dw:
        out["wheel_yaw_sigma"] = rs(np.concatenate(wheel_dw), math.radians(0.5))
    else:
        out["wheel_yaw_sigma"] = math.radians(2.0)
    return out


def save_json(path: str | Path, obj: Mapping) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)
