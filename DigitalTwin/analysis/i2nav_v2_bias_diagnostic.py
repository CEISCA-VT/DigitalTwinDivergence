#!/usr/bin/env python3
"""
i2nav_v2_bias_diagnostic.py
===========================

Targeted post-hoc diagnostic for the Twin V2 yaw-bias pilot.

Purpose
-------
This script does NOT train or modify V1/V2.

It diagnoses why a V2 run can:
  * work strongly on parking01,
  * preserve street00,
  * yet still fail on parking02.

It uses the exact artifacts written by i2nav_v2_yaw_bias_pilot.py:
  * v2_prediction_trace.csv
  * run_summary.json
  * pilot_run_results.csv

Primary questions
-----------------
1. Does the explicit bias head track the true persistent yaw residual?
2. Does it track the shape but have the wrong DC level?
3. Is the explicit bias nearly constant, suggesting sequence-prior behavior?
4. Are the explicit and fast heads carrying opposing DC components?
5. Does post-V2 persistent yaw bias correlate with ATE?
6. Do parking01 and parking02 contain similar ODO+IMU histories that require
   meaningfully different persistent corrections?  This is an identifiability
   diagnostic, not a formal proof.

The script excludes the first window-1 samples because the V2 pilot writes
zero corrections during GRU warmup.

Expected repository layout
--------------------------
DigitalTwinDivergence/
    DigitalTwin/analysis/i2nav_loso_ablation.py
    public_datasets/im2nav/
    results/i2nav_v2_yaw_bias_pilot/

Example
-------
python -u -m DigitalTwin.analysis.i2nav_v2_bias_diagnostic `
    --root ./public_datasets/im2nav `
    --pilot-dir ./results/i2nav_v2_yaw_bias_pilot `
    --output-dir ./results/i2nav_v2_bias_diagnostic
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RADPS_TO_DEG_PER_MIN = 180.0 / math.pi * 60.0
EPS = 1e-12

TRACE_REQUIRED = {
    "time_s",
    "true_delta_v_mps",
    "pred_delta_v_mps",
    "true_delta_omega_radps",
    "pred_total_delta_omega_radps",
    "pred_explicit_yaw_bias_radps",
    "pred_fast_yaw_residual_radps",
    "remaining_yaw_error_radps",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Targeted Twin V2 persistent-yaw diagnostic."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("public_datasets/im2nav"),
        help="i2Nav dataset root.",
    )
    p.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("results/i2nav_v2_yaw_bias_pilot"),
        help="Completed V2 pilot output directory.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/i2nav_v2_bias_diagnostic"),
        help="Diagnostic output directory.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=20,
        help="V2 GRU history length. Default: 20 samples.",
    )
    p.add_argument(
        "--persistent-windows-s",
        type=str,
        default="5,10,30",
        help="Comma-separated persistent-bias windows in seconds.",
    )
    p.add_argument(
        "--identifiability-step",
        type=int,
        default=10,
        help="Subsample step for cross-sequence history-neighbor analysis.",
    )
    p.add_argument(
        "--identifiability-target-s",
        type=float,
        default=30.0,
        help="Persistent target window for identifiability analysis.",
    )
    p.add_argument(
        "--skip-identifiability",
        action="store_true",
        help="Skip the parking01-vs-parking02 ODO+IMU history analysis.",
    )
    return p.parse_args()


def safe_float(value, default=float("nan")) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if np.isfinite(x) else default


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    xr = x[mask].rank(method="average").to_numpy()
    yr = y[mask].rank(method="average").to_numpy()
    return pearson(xr, yr)


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x * x)))


def cumulative_trapezoid(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros(len(y), dtype=float)
    if len(y) <= 1:
        return out
    dt = np.diff(t)
    increments = 0.5 * (y[:-1] + y[1:]) * dt
    out[1:] = np.cumsum(increments)
    return out


def rolling_mean_samples(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=samples, min_periods=samples)
        .mean()
        .to_numpy()
    )


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < EPS:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def find_run_artifacts(pilot_dir: Path) -> list[Path]:
    traces = sorted(pilot_dir.rglob("v2_prediction_trace.csv"))
    if not traces:
        raise FileNotFoundError(
            f"No v2_prediction_trace.csv files found under:\n{pilot_dir}"
        )
    return traces


def load_summary(run_dir: Path, pilot_dir: Path) -> dict:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    # Fallback to pilot_run_results.csv if a summary file is absent.
    table_path = pilot_dir / "pilot_run_results.csv"
    if not table_path.exists():
        return {}

    table = pd.read_csv(table_path)
    replicate = run_dir.parent.name
    fold_name = run_dir.name
    test_sequence = fold_name.split("_", 2)[-1]

    mask = (
        table["replicate"].astype(str).eq(replicate)
        & table["test_sequence"].astype(str).eq(test_sequence)
    )
    if int(mask.sum()) != 1:
        return {}
    return table.loc[mask].iloc[0].to_dict()


def infer_run_identity(run_dir: Path, summary: dict) -> tuple[str, str, int]:
    replicate = str(summary.get("replicate", run_dir.parent.name))
    test_sequence = str(
        summary.get("test_sequence", run_dir.name.split("_", 2)[-1])
    )
    base_seed = int(safe_float(summary.get("base_seed", -1), -1))
    return replicate, test_sequence, base_seed


def plot_bias_timeseries(
    out_path: Path,
    time_s: np.ndarray,
    true_rolling: np.ndarray,
    explicit_bias: np.ndarray,
    total_pred: np.ndarray,
    title: str,
) -> None:
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)
    ax.plot(time_s, true_rolling * RADPS_TO_DEG_PER_MIN, label="True rolling yaw residual")
    ax.plot(time_s, explicit_bias * RADPS_TO_DEG_PER_MIN, label="Explicit bias head")
    ax.plot(time_s, total_pred * RADPS_TO_DEG_PER_MIN, label="Total predicted yaw correction")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Yaw correction (deg/min)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_integrated_remaining(
    out_path: Path,
    time_s: np.ndarray,
    integrated_deg: np.ndarray,
    title: str,
) -> None:
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)
    ax.plot(time_s, integrated_deg)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Integrated remaining yaw error (deg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_explicit_vs_true(
    out_path: Path,
    true_bias: np.ndarray,
    explicit_bias: np.ndarray,
    title: str,
) -> None:
    mask = np.isfinite(true_bias) & np.isfinite(explicit_bias)
    x = true_bias[mask] * RADPS_TO_DEG_PER_MIN
    y = explicit_bias[mask] * RADPS_TO_DEG_PER_MIN

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=8, alpha=0.35)

    if len(x):
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)

    ax.set_xlabel("True rolling yaw residual (deg/min)")
    ax.set_ylabel("Explicit bias prediction (deg/min)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def analyze_one_run(
    trace_path: Path,
    pilot_dir: Path,
    output_dir: Path,
    window: int,
    persistent_windows_s: list[float],
) -> dict:
    run_dir = trace_path.parent
    summary = load_summary(run_dir, pilot_dir)
    replicate, sequence, base_seed = infer_run_identity(run_dir, summary)

    df = pd.read_csv(trace_path)
    missing = TRACE_REQUIRED - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{trace_path}: missing expected columns {sorted(missing)}"
        )

    time_all = df["time_s"].to_numpy(dtype=float)
    if len(time_all) < max(window + 2, 10):
        raise RuntimeError(f"{trace_path}: trace is unexpectedly short.")

    dts = np.diff(time_all)
    dt = float(np.median(dts[np.isfinite(dts) & (dts > 0)]))
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError(f"{trace_path}: invalid time grid.")

    hz = 1.0 / dt
    start = int(window) - 1

    t = time_all[start:]
    true_dw = df["true_delta_omega_radps"].to_numpy(dtype=float)[start:]
    total_pred = df["pred_total_delta_omega_radps"].to_numpy(dtype=float)[start:]
    explicit = df["pred_explicit_yaw_bias_radps"].to_numpy(dtype=float)[start:]
    fast = df["pred_fast_yaw_residual_radps"].to_numpy(dtype=float)[start:]
    remaining = df["remaining_yaw_error_radps"].to_numpy(dtype=float)[start:]

    # Recompute rather than trusting the saved column blindly.
    recomputed_remaining = true_dw - total_pred
    max_trace_mismatch = float(np.max(np.abs(remaining - recomputed_remaining)))
    if max_trace_mismatch > 1e-7:
        raise RuntimeError(
            f"{trace_path}: remaining-yaw column does not match true-total "
            f"(max mismatch={max_trace_mismatch:.3e})."
        )

    integrated_rad = cumulative_trapezoid(remaining, t)
    integrated_deg = np.rad2deg(integrated_rad)

    mean_true = float(np.mean(true_dw))
    mean_total = float(np.mean(total_pred))
    mean_explicit = float(np.mean(explicit))
    mean_fast = float(np.mean(fast))
    mean_remaining = float(np.mean(remaining))

    denom = abs(mean_explicit) + abs(mean_fast)
    dc_cancellation_fraction = (
        1.0 - abs(mean_total) / denom if denom > EPS else 0.0
    )
    dc_cancellation_fraction = float(
        np.clip(dc_cancellation_fraction, 0.0, 1.0)
    )

    result = {
        "replicate": replicate,
        "base_seed": base_seed,
        "test_sequence": sequence,
        "run_dir": str(run_dir),
        "n_samples_after_warmup": len(t),
        "estimated_rate_hz": hz,
        "duration_after_warmup_s": float(t[-1] - t[0]),
        "v1_ate_rmse_m": safe_float(summary.get("v1_ate_rmse_m")),
        "v2_ate_rmse_m": safe_float(summary.get("v2_ate_rmse_m")),
        "ate_change_pct": safe_float(summary.get("ate_change_pct")),
        "v1_heading_mae_deg": safe_float(summary.get("v1_heading_mae_deg")),
        "v2_heading_mae_deg": safe_float(summary.get("v2_heading_mae_deg")),
        "v1_rpe_1s_m": safe_float(summary.get("v1_rpe_1s_m")),
        "v1_rpe_5s_m": safe_float(summary.get("v1_rpe_5s_m")),
        "v1_rpe_10s_m": safe_float(summary.get("v1_rpe_10s_m")),
        "v2_rpe_1s_m": safe_float(summary.get("v2_rpe_1s_m")),
        "v2_rpe_5s_m": safe_float(summary.get("v2_rpe_5s_m")),
        "v2_rpe_10s_m": safe_float(summary.get("v2_rpe_10s_m")),
        "true_yaw_residual_mean_radps": mean_true,
        "true_yaw_residual_mean_deg_per_min": mean_true * RADPS_TO_DEG_PER_MIN,
        "pred_total_yaw_mean_radps": mean_total,
        "pred_total_yaw_mean_deg_per_min": mean_total * RADPS_TO_DEG_PER_MIN,
        "explicit_bias_mean_radps": mean_explicit,
        "explicit_bias_mean_deg_per_min": mean_explicit * RADPS_TO_DEG_PER_MIN,
        "fast_residual_mean_radps": mean_fast,
        "fast_residual_mean_deg_per_min": mean_fast * RADPS_TO_DEG_PER_MIN,
        "remaining_yaw_mean_radps": mean_remaining,
        "remaining_yaw_mean_deg_per_min": mean_remaining * RADPS_TO_DEG_PER_MIN,
        "remaining_yaw_abs_mean_deg_per_min": abs(mean_remaining) * RADPS_TO_DEG_PER_MIN,
        "remaining_yaw_rmse_radps": rmse(remaining),
        "true_vs_total_yaw_corr": pearson(true_dw, total_pred),
        "true_vs_explicit_corr": pearson(true_dw, explicit),
        "explicit_bias_std_radps": float(np.std(explicit)),
        "fast_residual_std_radps": float(np.std(fast)),
        "head_dc_cancellation_fraction": dc_cancellation_fraction,
        "final_integrated_remaining_yaw_deg": float(integrated_deg[-1]),
        "abs_final_integrated_remaining_yaw_deg": float(abs(integrated_deg[-1])),
        "max_abs_integrated_remaining_yaw_deg": float(np.max(np.abs(integrated_deg))),
    }

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sequence}_{replicate}"

    rolling_cache = {}
    for T in persistent_windows_s:
        samples = max(1, int(round(T / dt)))
        true_roll = rolling_mean_samples(true_dw, samples)
        rolling_cache[T] = true_roll

        mask = np.isfinite(true_roll)
        if int(mask.sum()) < 3:
            continue

        true_v = true_roll[mask]
        explicit_v = explicit[mask]
        diff = explicit_v - true_v
        slope, intercept = linear_fit(true_v, explicit_v)

        key = f"{int(T)}s"
        result[f"explicit_vs_true_{key}_corr"] = pearson(true_v, explicit_v)
        result[f"explicit_vs_true_{key}_rmse_radps"] = rmse(diff)
        result[f"explicit_vs_true_{key}_mae_radps"] = float(np.mean(np.abs(diff)))
        result[f"explicit_minus_true_{key}_mean_radps"] = float(np.mean(diff))
        result[f"explicit_minus_true_{key}_mean_deg_per_min"] = (
            float(np.mean(diff)) * RADPS_TO_DEG_PER_MIN
        )
        result[f"true_{key}_mean_radps"] = float(np.mean(true_v))
        result[f"true_{key}_std_radps"] = float(np.std(true_v))
        result[f"explicit_{key}_std_radps"] = float(np.std(explicit_v))
        result[f"explicit_to_true_{key}_std_ratio"] = float(
            np.std(explicit_v) / max(np.std(true_v), EPS)
        )
        result[f"explicit_vs_true_{key}_slope"] = slope
        result[f"explicit_vs_true_{key}_intercept_radps"] = intercept

    # 30 s is the primary visual if present, otherwise use the longest requested.
    visual_T = 30.0 if 30.0 in rolling_cache else max(rolling_cache)
    true_roll = rolling_cache[visual_T]

    plot_bias_timeseries(
        plots_dir / f"{stem}_bias_{int(visual_T)}s.png",
        t,
        true_roll,
        explicit,
        total_pred,
        f"{sequence} {replicate}: persistent yaw diagnostic ({visual_T:g} s target)",
    )
    plot_integrated_remaining(
        plots_dir / f"{stem}_integrated_remaining_yaw.png",
        t,
        integrated_deg,
        f"{sequence} {replicate}: integrated post-V2 yaw residual",
    )
    plot_explicit_vs_true(
        plots_dir / f"{stem}_explicit_vs_true_{int(visual_T)}s.png",
        true_roll,
        explicit,
        f"{sequence} {replicate}: explicit bias vs true {visual_T:g} s residual",
    )

    # Heuristic flags. These are diagnostic labels, not statistical claims.
    corr30 = result.get("explicit_vs_true_30s_corr", float("nan"))
    offset30 = result.get(
        "explicit_minus_true_30s_mean_deg_per_min",
        float("nan"),
    )
    std_ratio30 = result.get(
        "explicit_to_true_30s_std_ratio",
        float("nan"),
    )

    flags = []
    if np.isfinite(corr30) and corr30 >= 0.70 and np.isfinite(offset30) and abs(offset30) >= 0.50:
        flags.append("shape_tracks_but_DC_offset_large")
    if np.isfinite(std_ratio30) and std_ratio30 <= 0.25:
        flags.append("explicit_bias_nearly_constant_vs_true_30s")
    if (
        dc_cancellation_fraction >= 0.50
        and abs(mean_explicit * RADPS_TO_DEG_PER_MIN) >= 0.50
        and abs(mean_fast * RADPS_TO_DEG_PER_MIN) >= 0.50
    ):
        flags.append("large_opposing_DC_between_heads")
    if abs(mean_remaining * RADPS_TO_DEG_PER_MIN) >= 1.0:
        flags.append("large_persistent_post_V2_yaw_bias")
    if not flags:
        flags.append("no_major_heuristic_flag")

    result["diagnostic_flags"] = ";".join(flags)
    return result


def make_sequence_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "v1_ate_rmse_m",
        "v2_ate_rmse_m",
        "ate_change_pct",
        "remaining_yaw_abs_mean_deg_per_min",
        "abs_final_integrated_remaining_yaw_deg",
        "max_abs_integrated_remaining_yaw_deg",
        "head_dc_cancellation_fraction",
        "explicit_vs_true_30s_corr",
        "explicit_minus_true_30s_mean_deg_per_min",
        "explicit_to_true_30s_std_ratio",
        "v2_rpe_1s_m",
        "v2_rpe_5s_m",
        "v2_rpe_10s_m",
    ]
    cols = [c for c in numeric if c in per_run.columns]

    rows = []
    for seq, g in per_run.groupby("test_sequence", sort=False):
        row = {
            "test_sequence": seq,
            "n_runs": len(g),
        }
        for c in cols:
            vals = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                row[f"{c}_mean"] = float(np.mean(vals))
                row[f"{c}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def make_ate_correlations(per_run: pd.DataFrame) -> pd.DataFrame:
    target = pd.to_numeric(per_run["v2_ate_rmse_m"], errors="coerce").to_numpy(dtype=float)
    candidate_cols = [
        "remaining_yaw_abs_mean_deg_per_min",
        "abs_final_integrated_remaining_yaw_deg",
        "max_abs_integrated_remaining_yaw_deg",
        "head_dc_cancellation_fraction",
        "explicit_vs_true_30s_corr",
        "explicit_minus_true_30s_mean_deg_per_min",
        "explicit_to_true_30s_std_ratio",
        "remaining_yaw_rmse_radps",
    ]

    rows = []
    for c in candidate_cols:
        if c not in per_run.columns:
            continue
        x = pd.to_numeric(per_run[c], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(target)
        rows.append(
            {
                "predictor": c,
                "n": int(mask.sum()),
                "pearson_with_v2_ate": pearson(x[mask], target[mask]),
                "spearman_with_v2_ate": spearman(x[mask], target[mask]),
            }
        )
    return pd.DataFrame(rows)


def plot_global_scatter(
    per_run: pd.DataFrame,
    x_col: str,
    x_label: str,
    out_path: Path,
) -> None:
    if x_col not in per_run.columns:
        return
    x = pd.to_numeric(per_run[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(per_run["v2_ate_rmse_m"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.scatter(x[mask], y[mask], s=35)

    rows = per_run.loc[mask].reset_index(drop=True)
    for i, row in rows.iterrows():
        label = f"{row['test_sequence']}:{row['replicate']}"
        ax.annotate(label, (x[mask][i], y[mask][i]), fontsize=7)

    ax.set_xlabel(x_label)
    ax.set_ylabel("V2 ATE RMSE (m)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def original_default_args(original):
    old_argv = sys.argv[:]
    try:
        sys.argv = ["i2nav_loso_ablation.py"]
        return original.parse_args()
    finally:
        sys.argv = old_argv


def build_exact_sequences(root: Path, names: list[str]):
    original = importlib.import_module(
        "DigitalTwin.analysis.i2nav_loso_ablation"
    )
    defaults = original_default_args(original)
    files_by_name = {x.name: x for x in original.discover_files(root)}

    prepared = {}
    for name in names:
        if name not in files_by_name:
            raise RuntimeError(f"Could not discover i2Nav sequence {name}.")
        prepared[name] = original.prepare_sequence(
            files_by_name[name],
            hz=defaults.rate_hz,
            imu_yaw_sign=defaults.imu_yaw_sign,
            gnss_sigma_max_m=defaults.gnss_sigma_max_m,
            gnss_anchor_count=defaults.gnss_anchor_count,
        )
    return prepared


def history_matrix(features: np.ndarray, window: int, indices: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    blocks = []
    for idx in indices:
        start = int(idx) - window + 1
        blocks.append(features[start : int(idx) + 1].reshape(-1))
    return np.asarray(blocks, dtype=np.float32)


def nearest_cross_sequence(
    query: np.ndarray,
    reference: np.ndarray,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    # Exact Euclidean NN in standardized history space, chunked to avoid a
    # large NxM distance matrix.
    q = np.asarray(query, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    r_norm = np.sum(r * r, axis=1)

    best_d2 = np.full(len(q), np.inf, dtype=np.float64)
    best_idx = np.full(len(q), -1, dtype=np.int64)

    for start in range(0, len(q), chunk):
        stop = min(start + chunk, len(q))
        qc = q[start:stop]
        q_norm = np.sum(qc * qc, axis=1)[:, None]
        d2 = q_norm + r_norm[None, :] - 2.0 * (qc @ r.T)
        d2 = np.maximum(d2, 0.0)
        idx = np.argmin(d2, axis=1)
        vals = d2[np.arange(len(idx)), idx]
        best_d2[start:stop] = vals
        best_idx[start:stop] = idx

    # RMS standardized distance per history dimension is easier to read than
    # raw sqrt(sum(square)).
    rms_distance = np.sqrt(best_d2 / max(q.shape[1], 1))
    return rms_distance, best_idx


def identifiability_analysis(
    root: Path,
    output_dir: Path,
    window: int,
    target_s: float,
    step: int,
) -> pd.DataFrame:
    pair = ["parking01", "parking02"]
    prepared = build_exact_sequences(root, pair)

    payload = {}
    for name in pair:
        seq = prepared[name]
        t = np.asarray(seq.grid, dtype=float)
        dt = float(np.median(np.diff(t)))
        target = np.asarray(seq.target_corrections[:, 1], dtype=float)
        samples_target = max(1, int(round(target_s / dt)))
        persistent = rolling_mean_samples(target, samples_target)

        first = max(window - 1, samples_target - 1)
        indices = np.arange(first, len(t), max(1, int(step)), dtype=int)
        valid = np.isfinite(persistent[indices])
        indices = indices[valid]

        X = history_matrix(seq.features, window, indices)
        y = persistent[indices]
        payload[name] = {
            "X": X,
            "y": y,
            "indices": indices,
            "time": t[indices],
        }

    # Diagnostic-only standardization across the two held-out sequences.
    combined = np.vstack([payload["parking01"]["X"], payload["parking02"]["X"]])
    mean = np.mean(combined, axis=0)
    std = np.maximum(np.std(combined, axis=0), 1e-5)

    for name in pair:
        payload[name]["Xz"] = (payload[name]["X"] - mean) / std

    rows = []
    directions = [
        ("parking02", "parking01"),
        ("parking01", "parking02"),
    ]

    raw_rows = []
    for q_name, r_name in directions:
        q = payload[q_name]
        r = payload[r_name]

        distance, idx = nearest_cross_sequence(q["Xz"], r["Xz"])
        q_bias = q["y"]
        r_bias = r["y"][idx]
        gap = (q_bias - r_bias) * RADPS_TO_DEG_PER_MIN

        for i in range(len(distance)):
            raw_rows.append(
                {
                    "query_sequence": q_name,
                    "reference_sequence": r_name,
                    "query_time_s": float(q["time"][i]),
                    "reference_time_s": float(r["time"][idx[i]]),
                    "history_rms_standardized_distance": float(distance[i]),
                    "query_true_persistent_bias_deg_per_min": float(
                        q_bias[i] * RADPS_TO_DEG_PER_MIN
                    ),
                    "nearest_reference_true_persistent_bias_deg_per_min": float(
                        r_bias[i] * RADPS_TO_DEG_PER_MIN
                    ),
                    "signed_bias_gap_deg_per_min": float(gap[i]),
                    "abs_bias_gap_deg_per_min": float(abs(gap[i])),
                }
            )

        median_distance = float(np.median(distance))
        close = distance <= median_distance
        abs_gap = np.abs(gap)

        rows.append(
            {
                "query_sequence": q_name,
                "reference_sequence": r_name,
                "n_query_histories": len(distance),
                "median_nearest_history_distance": median_distance,
                "p90_nearest_history_distance": float(np.percentile(distance, 90)),
                "median_abs_persistent_bias_gap_deg_per_min": float(
                    np.median(abs_gap)
                ),
                "p90_abs_persistent_bias_gap_deg_per_min": float(
                    np.percentile(abs_gap, 90)
                ),
                "close_half_fraction_bias_gap_gt_0p5_deg_min": float(
                    np.mean(abs_gap[close] > 0.5)
                ),
                "close_half_fraction_bias_gap_gt_1p0_deg_min": float(
                    np.mean(abs_gap[close] > 1.0)
                ),
            }
        )

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(
        output_dir / "parking01_parking02_history_neighbor_pairs.csv",
        index=False,
    )

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    for name, g in raw.groupby("query_sequence"):
        ax.scatter(
            g["history_rms_standardized_distance"],
            g["abs_bias_gap_deg_per_min"],
            s=10,
            alpha=0.30,
            label=f"{name} query",
        )
    ax.set_xlabel("Nearest cross-sequence ODO+IMU history distance")
    ax.set_ylabel(f"|true {target_s:g}s persistent-bias gap| (deg/min)")
    ax.set_title("Cross-sequence history similarity vs required yaw correction")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "plots" / "history_similarity_vs_bias_gap.png",
        dpi=160,
    )
    plt.close(fig)

    return pd.DataFrame(rows)


def write_findings(
    output_dir: Path,
    per_run: pd.DataFrame,
    correlations: pd.DataFrame,
    ident: pd.DataFrame | None,
) -> None:
    lines = []
    lines.append("Twin V2 yaw-bias diagnostic")
    lines.append("=" * 72)
    lines.append("")
    lines.append("These are diagnostic observations. Heuristic labels are not formal proofs.")
    lines.append("")

    for seq, g in per_run.groupby("test_sequence", sort=False):
        ate = pd.to_numeric(g["v2_ate_rmse_m"], errors="coerce")
        rem = pd.to_numeric(
            g["remaining_yaw_abs_mean_deg_per_min"],
            errors="coerce",
        )
        corr30 = (
            pd.to_numeric(g["explicit_vs_true_30s_corr"], errors="coerce")
            if "explicit_vs_true_30s_corr" in g
            else pd.Series(dtype=float)
        )
        offset30 = (
            pd.to_numeric(
                g["explicit_minus_true_30s_mean_deg_per_min"],
                errors="coerce",
            )
            if "explicit_minus_true_30s_mean_deg_per_min" in g
            else pd.Series(dtype=float)
        )

        lines.append(f"{seq}")
        lines.append(
            f"  V2 ATE mean: {ate.mean():.4f} m"
            if ate.notna().any()
            else "  V2 ATE mean: unavailable"
        )
        lines.append(
            f"  |remaining yaw bias| mean: {rem.mean():.3f} deg/min"
            if rem.notna().any()
            else "  |remaining yaw bias| mean: unavailable"
        )
        if corr30.notna().any():
            lines.append(
                f"  explicit-vs-true 30s correlation mean: {corr30.mean():.3f}"
            )
        if offset30.notna().any():
            lines.append(
                f"  explicit-minus-true 30s DC offset mean: "
                f"{offset30.mean():+.3f} deg/min"
            )

        flag_counts = {}
        for flags in g["diagnostic_flags"].astype(str):
            for flag in flags.split(";"):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        for flag, count in sorted(flag_counts.items()):
            lines.append(f"  flag {flag}: {count}/{len(g)} runs")
        lines.append("")

    if not correlations.empty:
        c = correlations.copy()
        c["abs_spearman"] = c["spearman_with_v2_ate"].abs()
        c = c.sort_values("abs_spearman", ascending=False)
        lines.append("Strongest run-level associations with V2 ATE")
        for _, r in c.head(5).iterrows():
            lines.append(
                f"  {r['predictor']}: Pearson={r['pearson_with_v2_ate']:+.3f}, "
                f"Spearman={r['spearman_with_v2_ate']:+.3f}, n={int(r['n'])}"
            )
        lines.append("")

    if ident is not None and not ident.empty:
        lines.append("ODO+IMU history identifiability diagnostic")
        for _, r in ident.iterrows():
            lines.append(
                f"  {r['query_sequence']} -> {r['reference_sequence']}: "
                f"median nearest distance={r['median_nearest_history_distance']:.3f}, "
                f"median |bias gap|={r['median_abs_persistent_bias_gap_deg_per_min']:.3f} deg/min, "
                f"close-half fraction >1 deg/min="
                f"{100*r['close_half_fraction_bias_gap_gt_1p0_deg_min']:.1f}%"
            )
        lines.append("")
        lines.append(
            "Interpretation caution: a large bias gap among close cross-sequence histories "
            "suggests the current 2 s ODO+IMU history may not uniquely identify the required "
            "persistent correction. It does not prove non-identifiability."
        )

    (output_dir / "diagnostic_findings.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    pilot_dir = args.pilot_dir.resolve()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    windows_s = [
        float(x.strip())
        for x in args.persistent_windows_s.split(",")
        if x.strip()
    ]
    if not windows_s:
        raise ValueError("At least one persistent window is required.")

    print("=" * 88)
    print("TWIN V2 TARGETED YAW-BIAS DIAGNOSTIC")
    print("=" * 88)
    print("Pilot directory :", pilot_dir)
    print("Dataset root    :", root)
    print("Output directory:", output_dir)
    print("Warmup excluded :", args.window - 1, "samples")
    print("Bias windows    :", windows_s)
    print()

    trace_paths = find_run_artifacts(pilot_dir)
    print(f"Found {len(trace_paths)} V2 run traces.")

    rows = []
    for i, trace in enumerate(trace_paths, start=1):
        print(f"[{i}/{len(trace_paths)}] {trace.parent}")
        rows.append(
            analyze_one_run(
                trace,
                pilot_dir,
                output_dir,
                args.window,
                windows_s,
            )
        )

    per_run = pd.DataFrame(rows).sort_values(
        ["test_sequence", "base_seed", "replicate"],
        kind="stable",
    )
    per_run.to_csv(
        output_dir / "per_run_bias_diagnostics.csv",
        index=False,
    )

    per_sequence = make_sequence_summary(per_run)
    per_sequence.to_csv(
        output_dir / "per_sequence_bias_summary.csv",
        index=False,
    )

    correlations = make_ate_correlations(per_run)
    correlations.to_csv(
        output_dir / "bias_vs_ate_correlation.csv",
        index=False,
    )

    plot_global_scatter(
        per_run,
        "remaining_yaw_abs_mean_deg_per_min",
        "|mean post-V2 yaw residual| (deg/min)",
        output_dir / "plots" / "abs_remaining_bias_vs_ate.png",
    )
    plot_global_scatter(
        per_run,
        "abs_final_integrated_remaining_yaw_deg",
        "|final integrated post-V2 yaw residual| (deg)",
        output_dir / "plots" / "abs_integrated_yaw_vs_ate.png",
    )
    if "explicit_minus_true_30s_mean_deg_per_min" in per_run.columns:
        tmp = per_run.copy()
        tmp["abs_explicit_30s_DC_offset_deg_per_min"] = pd.to_numeric(
            tmp["explicit_minus_true_30s_mean_deg_per_min"],
            errors="coerce",
        ).abs()
        plot_global_scatter(
            tmp,
            "abs_explicit_30s_DC_offset_deg_per_min",
            "|explicit - true 30s DC offset| (deg/min)",
            output_dir / "plots" / "explicit_30s_DC_offset_vs_ate.png",
        )

    ident = None
    if not args.skip_identifiability:
        print()
        print("Running parking01-vs-parking02 ODO+IMU history identifiability diagnostic...")
        try:
            ident = identifiability_analysis(
                root,
                output_dir,
                args.window,
                args.identifiability_target_s,
                args.identifiability_step,
            )
            ident.to_csv(
                output_dir / "parking01_parking02_identifiability_summary.csv",
                index=False,
            )
        except Exception as exc:
            # Preserve all trace-based diagnostics even if the optional dataset
            # analysis cannot run because of a local environment mismatch.
            print("[warning] Identifiability analysis failed:")
            print(f"  {type(exc).__name__}: {exc}")
            (output_dir / "identifiability_error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )

    write_findings(
        output_dir,
        per_run,
        correlations,
        ident,
    )

    print()
    print("=" * 88)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 88)
    print("Primary outputs:")
    print(" ", output_dir / "per_run_bias_diagnostics.csv")
    print(" ", output_dir / "per_sequence_bias_summary.csv")
    print(" ", output_dir / "bias_vs_ate_correlation.csv")
    print(" ", output_dir / "diagnostic_findings.txt")
    if ident is not None:
        print(" ", output_dir / "parking01_parking02_identifiability_summary.csv")
        print(" ", output_dir / "parking01_parking02_history_neighbor_pairs.csv")
    print(" ", output_dir / "plots")


if __name__ == "__main__":
    main()
