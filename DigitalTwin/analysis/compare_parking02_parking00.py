#!/usr/bin/env python3
"""
Compare parking02 (hard case) against parking00 (easier parking case)
to diagnose why parking02 accumulates much larger digital-twin divergence.

This is a POST-HOC analysis only:
- no training
- no checkpoint modification
- no hyperparameter tuning
- no GPU required

Expected repository layout
--------------------------
<repo>/
    public_datasets/im2nav/
    results/i2nav_v2_full_loso/
    DigitalTwin/analysis/i2nav_v2_full_loso.py

The script reconstructs the exact canonical wheel/IMU/ODO signals using the
frozen repository preprocessing, then aligns them to each saved LOSO V2 run's
fidelity_timeseries.csv.

It compares:
- short-horizon translational RPE at 1 s, 5 s, and 10 s
- wheel yaw
- IMU yaw
- wheel-IMU yaw disagreement
- normalized disagreement
- ODO speed
- longitudinal acceleration
- turning intensity / curvature proxy
- true/predicted yaw correction residual
- accumulated Iomega
- heading divergence Dtheta
- position divergence Dp

Outputs
-------
results/i2nav_frozen_v2_fidelity_analysis/parking02_vs_parking00/
    per_run_summary.csv
    per_sequence_summary.csv
    short_horizon_rpe.csv
    mechanism_correlations.csv
    aligned_timeseries_<sequence>_<seed>.csv
    parking02_vs_parking00_summary.txt
    parking02_vs_parking00_metrics.png
    parking02_vs_parking00_short_horizon_rpe.png
    parking02_vs_parking00_mechanism.png
    parking02_vs_parking00_trajectory.png

Run from repository root:
    python .\\DigitalTwin\\analysis\\compare_parking02_parking00.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEQUENCES = ("parking00", "parking02")
EXPECTED_BASE_SEEDS = (42, 1042, 2042)
SCRIPT_VERSION = "2026-08-19-rpe-v2"


def repo_root_from_script() -> Path:
    # DigitalTwin/analysis/<this file> -> repo root
    return Path(__file__).resolve().parents[2]


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def p95_abs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(np.abs(x), 95.0)) if len(x) else float("nan")


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if len(x) else float("nan")


def locate_run_dirs(results_root: Path) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}

    for summary_path in results_root.rglob("run_summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        sequence = str(
            summary.get("test_sequence", summary.get("sequence", ""))
        )
        if sequence not in SEQUENCES:
            continue

        base_seed = summary.get("base_seed")
        if base_seed is None:
            # Fallback from replicate directory name.
            parent_text = str(summary_path.parent)
            for seed in EXPECTED_BASE_SEEDS:
                if f"base{seed}" in parent_text:
                    base_seed = seed
                    break

        if base_seed is None:
            continue

        key = (sequence, int(base_seed))
        if key in found:
            raise RuntimeError(
                f"Duplicate LOSO run identity {key}:\n"
                f"  {found[key]}\n  {summary_path.parent}"
            )
        found[key] = summary_path.parent

    missing = [
        (seq, seed)
        for seq in SEQUENCES
        for seed in EXPECTED_BASE_SEEDS
        if (seq, seed) not in found
    ]
    if missing:
        raise RuntimeError(
            "Missing required parking00/parking02 LOSO runs: "
            + ", ".join(map(str, missing))
        )

    return found


def configure_and_prepare(data_root: Path):
    """
    Reuse the authoritative frozen preprocessing/canonical adapter.

    Insert the repository root into sys.path so both
        python -m DigitalTwin.analysis.compare_parking02_parking00
    and
        python .\\DigitalTwin\\analysis\\compare_parking02_parking00.py
    work from the repository root on Windows.
    """
    repo = repo_root_from_script()
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)

    from DigitalTwin.analysis import i2nav_v2_full_loso as v2full

    defaults = v2full.base.original_default_args(v2full.original)

    # The module's canonical builders use these globals.
    v2full.RATE = float(defaults.rate_hz)
    v2full.DT = 1.0 / v2full.RATE
    v2full.SLOW_SAMPLES = int(round(v2full.SLOW_SECONDS * v2full.RATE))
    v2full.CHUNK_STEPS = int(round(v2full.CHUNK_SECONDS * v2full.RATE)) + 1

    prepared = v2full.prepare_all_sequences(data_root, defaults)
    canonical, _, _ = v2full.build_all_canonical(prepared, data_root)

    return v2full, defaults, prepared, canonical


def align_frames(
    sequence: str,
    run_dir: Path,
    prepared_seq: Any,
    canonical: dict[str, Any],
) -> pd.DataFrame:
    fidelity_path = run_dir / "fidelity_timeseries.csv"
    trace_path = run_dir / "v2_prediction_trace.csv"
    traj_path = run_dir / "v2_evaluated_trajectory.csv"

    for p in (fidelity_path, trace_path, traj_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required run artifact: {p}")

    fid = pd.read_csv(fidelity_path)
    trace = pd.read_csv(trace_path)
    traj = pd.read_csv(traj_path)

    required_fid = {
        "time_s",
        "Dp_m",
        "Dtheta_deg",
        "romega_radps",
        "Iomega_deg",
    }
    required_trace = {
        "time_s",
        "true_delta_v_mps",
        "pred_delta_v_mps",
        "true_delta_omega_radps",
        "pred_total_delta_omega_radps",
    }
    required_traj = {
        "time_s",
        "gt_east_m",
        "gt_north_m",
        "gt_heading_rad",
        "estimate_east_m",
        "estimate_north_m",
        "estimate_heading_rad",
    }

    if missing := required_fid - set(fid.columns):
        raise ValueError(f"{fidelity_path} missing columns: {sorted(missing)}")
    if missing := required_trace - set(trace.columns):
        raise ValueError(f"{trace_path} missing columns: {sorted(missing)}")
    if missing := required_traj - set(traj.columns):
        raise ValueError(f"{traj_path} missing columns: {sorted(missing)}")

    sig = canonical[sequence]
    context = pd.DataFrame(
        {
            "time_s": np.asarray(sig["time_s"], dtype=float),
            "wheel_forward_mps": np.asarray(sig["wheel_forward_mps"], dtype=float),
            "wheel_yaw_radps": np.asarray(sig["wheel_yaw_radps"], dtype=float),
            "imu_yaw_radps": np.asarray(sig["imu_yaw_radps"], dtype=float),
            "odo_forward_mps": np.asarray(sig["odo_forward_mps"], dtype=float),
            "yaw_disagreement_radps": np.asarray(
                sig["yaw_disagreement_radps"], dtype=float
            ),
            "yaw_disagreement_normalized": np.asarray(
                sig["yaw_disagreement_normalized"], dtype=float
            ),
        }
    )

    # Add lateral proxy if present in the canonical adapter.
    if "wheel_lateral_mps" in sig:
        context["wheel_lateral_mps"] = np.asarray(
            sig["wheel_lateral_mps"], dtype=float
        )

    # Derive context variables on the canonical grid.
    t = context["time_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t)))
    speed = context["odo_forward_mps"].to_numpy(dtype=float)
    yaw = context["imu_yaw_radps"].to_numpy(dtype=float)

    context["longitudinal_accel_mps2"] = np.gradient(speed, dt)
    context["abs_longitudinal_accel_mps2"] = np.abs(
        context["longitudinal_accel_mps2"]
    )
    context["abs_imu_yaw_radps"] = np.abs(yaw)
    context["abs_wheel_yaw_radps"] = np.abs(
        context["wheel_yaw_radps"].to_numpy(dtype=float)
    )
    context["abs_yaw_disagreement_radps"] = np.abs(
        context["yaw_disagreement_radps"].to_numpy(dtype=float)
    )

    # Curvature proxy kappa = omega / v. Suppress near-zero-speed blow-up.
    v_abs = np.abs(speed)
    curvature = np.full_like(yaw, np.nan, dtype=float)
    moving = v_abs >= 0.10
    curvature[moving] = yaw[moving] / np.maximum(v_abs[moving], 0.10)
    context["curvature_proxy_radpm"] = curvature
    context["abs_curvature_proxy_radpm"] = np.abs(curvature)

    # The saved artifacts should be on the same evaluation grid. Use merge_asof
    # with a tight half-sample tolerance instead of assuming exact floating-point
    # string equality.
    tolerance = max(1e-6, 0.51 * dt)

    merged = pd.merge_asof(
        fid.sort_values("time_s"),
        trace.sort_values("time_s"),
        on="time_s",
        direction="nearest",
        tolerance=tolerance,
        suffixes=("", "_trace"),
    )
    merged = pd.merge_asof(
        merged.sort_values("time_s"),
        traj.sort_values("time_s"),
        on="time_s",
        direction="nearest",
        tolerance=tolerance,
        suffixes=("", "_traj"),
    )
    merged = pd.merge_asof(
        merged.sort_values("time_s"),
        context.sort_values("time_s"),
        on="time_s",
        direction="nearest",
        tolerance=tolerance,
        suffixes=("", "_context"),
    )

    # Ground-truth physical yaw rate and speed are available offline from the
    # PreparedSequence. They are diagnostic only, not online twin inputs.
    gt = pd.DataFrame(
        {
            "time_s": np.asarray(prepared_seq.grid, dtype=float),
            "gt_forward_speed_mps": np.asarray(
                prepared_seq.gt_forward_speed, dtype=float
            ),
            "gt_yaw_rate_radps": np.asarray(
                prepared_seq.gt_yaw_rate, dtype=float
            ),
        }
    )
    merged = pd.merge_asof(
        merged.sort_values("time_s"),
        gt.sort_values("time_s"),
        on="time_s",
        direction="nearest",
        tolerance=tolerance,
    )

    # Useful signed residuals and model-state divergence.
    merged["sensor_yaw_residual_radps"] = (
        merged["gt_yaw_rate_radps"] - merged["imu_yaw_radps"]
    )
    merged["wheel_yaw_residual_radps"] = (
        merged["gt_yaw_rate_radps"] - merged["wheel_yaw_radps"]
    )
    merged["prediction_yaw_residual_radps"] = (
        merged["true_delta_omega_radps"]
        - merged["pred_total_delta_omega_radps"]
    )
    merged["prediction_speed_residual_mps"] = (
        merged["true_delta_v_mps"] - merged["pred_delta_v_mps"]
    )

    merged["signed_heading_error_deg"] = np.degrees(
        wrap_angle(
            merged["estimate_heading_rad"].to_numpy(dtype=float)
            - merged["gt_heading_rad"].to_numpy(dtype=float)
        )
    )

    # Elapsed fraction is useful for showing accumulation.
    t0 = float(merged["time_s"].iloc[0])
    t1 = float(merged["time_s"].iloc[-1])
    merged["elapsed_fraction"] = (
        (merged["time_s"] - t0) / max(t1 - t0, 1e-9)
    )

    # Reject a silent alignment failure.
    critical = [
        "wheel_yaw_radps",
        "imu_yaw_radps",
        "gt_yaw_rate_radps",
        "Dp_m",
        "Dtheta_deg",
        "Iomega_deg",
    ]
    bad_fraction = merged[critical].isna().any(axis=1).mean()
    if bad_fraction > 0.01:
        raise RuntimeError(
            f"{sequence} {run_dir}: {100*bad_fraction:.2f}% of rows failed "
            "critical timestamp alignment."
        )

    return merged


def summarize_run(
    sequence: str,
    base_seed: int,
    run_dir: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    summary_json = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )

    t = frame["time_s"].to_numpy(dtype=float)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    row: dict[str, Any] = {
        "sequence": sequence,
        "base_seed": base_seed,
        "duration_s": duration,
        "n_samples": len(frame),
        "v1_ate_rmse_m": summary_json.get("v1_ate_rmse_m"),
        "v2_ate_rmse_m": summary_json.get("v2_ate_rmse_m"),
        "v2_heading_mae_deg": summary_json.get("v2_heading_mae_deg"),
        "v2_rpe_1s_m": summary_json.get("v2_rpe_1s_m"),
        "v2_rpe_5s_m": summary_json.get("v2_rpe_5s_m"),
        "v2_rpe_10s_m": summary_json.get("v2_rpe_10s_m"),
    }

    metrics = {
        "odo_speed_mean_mps": frame["odo_forward_mps"].mean(),
        "odo_speed_abs_mean_mps": frame["odo_forward_mps"].abs().mean(),
        "imu_yaw_abs_mean_radps": frame["abs_imu_yaw_radps"].mean(),
        "imu_yaw_p95_abs_radps": p95_abs(frame["imu_yaw_radps"]),
        "wheel_yaw_abs_mean_radps": frame["abs_wheel_yaw_radps"].mean(),
        "yaw_disagreement_signed_mean_radps": frame[
            "yaw_disagreement_radps"
        ].mean(),
        "yaw_disagreement_abs_mean_radps": frame[
            "abs_yaw_disagreement_radps"
        ].mean(),
        "yaw_disagreement_rms_radps": rms(
            frame["yaw_disagreement_radps"].to_numpy()
        ),
        "yaw_disagreement_p95_abs_radps": p95_abs(
            frame["yaw_disagreement_radps"].to_numpy()
        ),
        "normalized_yaw_disagreement_abs_mean": frame[
            "yaw_disagreement_normalized"
        ].abs().mean(),
        "accel_abs_mean_mps2": frame[
            "abs_longitudinal_accel_mps2"
        ].mean(),
        "curvature_abs_mean_radpm": frame[
            "abs_curvature_proxy_radpm"
        ].mean(),
        "curvature_p95_abs_radpm": p95_abs(
            frame["curvature_proxy_radpm"].to_numpy()
        ),
        "sensor_yaw_residual_signed_mean_radps": frame[
            "sensor_yaw_residual_radps"
        ].mean(),
        "sensor_yaw_residual_p95_abs_radps": p95_abs(
            frame["sensor_yaw_residual_radps"].to_numpy()
        ),
        "prediction_yaw_residual_signed_mean_radps": frame[
            "prediction_yaw_residual_radps"
        ].mean(),
        "prediction_yaw_residual_p95_abs_radps": p95_abs(
            frame["prediction_yaw_residual_radps"].to_numpy()
        ),
        "Dp_mean_m": frame["Dp_m"].mean(),
        "Dp_p95_m": float(np.percentile(frame["Dp_m"], 95.0)),
        "Dp_max_m": frame["Dp_m"].max(),
        "Dtheta_mean_deg": frame["Dtheta_deg"].mean(),
        "Dtheta_p95_deg": float(np.percentile(frame["Dtheta_deg"], 95.0)),
        "Dtheta_max_deg": frame["Dtheta_deg"].max(),
        "Iomega_final_deg": frame["Iomega_deg"].iloc[-1],
        "Iomega_p95_abs_deg": p95_abs(frame["Iomega_deg"].to_numpy()),
        "Iomega_max_abs_deg": np.max(np.abs(frame["Iomega_deg"])),
    }

    row.update({k: float(v) for k, v in metrics.items()})
    return row


def mechanism_correlations(
    sequence: str,
    base_seed: int,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    pairs = [
        ("abs_yaw_disagreement_radps", "Domega_radps"),
        ("abs_yaw_disagreement_radps", "Dtheta_deg"),
        ("abs_yaw_disagreement_radps", "Dp_m"),
        ("prediction_yaw_residual_radps", "Iomega_deg"),
        ("Iomega_deg", "signed_heading_error_deg"),
        ("Iomega_deg", "Dtheta_deg"),
        ("Dtheta_deg", "Dp_m"),
        ("abs_curvature_proxy_radpm", "Dtheta_deg"),
        ("abs_curvature_proxy_radpm", "Dp_m"),
        ("abs_longitudinal_accel_mps2", "Dp_m"),
    ]

    rows: list[dict[str, Any]] = []
    for xcol, ycol in pairs:
        if xcol not in frame or ycol not in frame:
            continue

        rows.append(
            {
                "sequence": sequence,
                "base_seed": base_seed,
                "x": xcol,
                "y": ycol,
                "pearson_r": safe_corr(
                    frame[xcol].to_numpy(dtype=float),
                    frame[ycol].to_numpy(dtype=float),
                ),
                "n": int(
                    np.sum(
                        np.isfinite(frame[xcol].to_numpy(dtype=float))
                        & np.isfinite(frame[ycol].to_numpy(dtype=float))
                    )
                ),
            }
        )
    return rows


def aggregate_sequences(per_run: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        c
        for c in per_run.columns
        if c not in {"sequence", "base_seed"}
        and pd.api.types.is_numeric_dtype(per_run[c])
    ]

    rows = []
    for sequence, group in per_run.groupby("sequence"):
        row = {"sequence": sequence, "n_seeds": len(group)}
        for col in numeric:
            vals = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std_seed"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_comparison(per_run: pd.DataFrame, out: Path) -> None:
    metric_specs = [
        ("yaw_disagreement_abs_mean_radps", "Mean |wheel–IMU yaw disagreement| [rad/s]"),
        ("prediction_yaw_residual_p95_abs_radps", "p95 |V2 yaw residual| [rad/s]"),
        ("curvature_abs_mean_radpm", "Mean |curvature proxy| [rad/m]"),
        ("Dtheta_p95_deg", "Heading divergence p95 [deg]"),
        ("Dp_p95_m", "Position divergence p95 [m]"),
        ("Iomega_max_abs_deg", "Max |accumulated yaw residual| [deg]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    x = np.arange(len(SEQUENCES))
    width = 0.22
    offsets = [-width, 0.0, width]

    for ax, (col, ylabel) in zip(axes, metric_specs):
        for j, seed in enumerate(EXPECTED_BASE_SEEDS):
            vals = []
            for seq in SEQUENCES:
                q = per_run[
                    (per_run["sequence"] == seq)
                    & (per_run["base_seed"] == seed)
                ]
                vals.append(float(q[col].iloc[0]))
            ax.bar(x + offsets[j], vals, width=width, label=f"seed {seed}")

        ax.set_xticks(x)
        ax.set_xticklabels(SEQUENCES)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend()
    fig.suptitle("parking00 vs parking02: operating context and twin divergence")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_short_horizon_rpe_table(per_run: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy 1/5/10-s RPE table for plotting and manuscript use."""
    rows: list[dict[str, Any]] = []
    metric_cols = {
        1: "v2_rpe_1s_m",
        5: "v2_rpe_5s_m",
        10: "v2_rpe_10s_m",
    }

    for _, row in per_run.iterrows():
        for horizon_s, col in metric_cols.items():
            value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            rows.append(
                {
                    "sequence": row["sequence"],
                    "base_seed": int(row["base_seed"]),
                    "horizon_s": int(horizon_s),
                    "v2_rpe_trans_rmse_m": float(value) if np.isfinite(value) else np.nan,
                }
            )

    table = pd.DataFrame(rows)
    if table["v2_rpe_trans_rmse_m"].isna().any():
        missing = table[table["v2_rpe_trans_rmse_m"].isna()][
            ["sequence", "base_seed", "horizon_s"]
        ]
        raise ValueError(
            "Some V2 RPE values are missing from run_summary.json:\n"
            + missing.to_string(index=False)
        )
    return table.sort_values(["sequence", "base_seed", "horizon_s"])


def plot_short_horizon_rpe(
    rpe: pd.DataFrame,
    out: Path,
) -> None:
    """Plot 1/5/10-s translational RPE for parking00 and parking02.

    Thin seed-level traces show run-to-run variability. Thick mean traces with
    ±1 seed-SD error bars show the three-seed sequence summary. This directly
    visualizes the finite-horizon/local-fidelity side of the paper's central
    local-vs-global fidelity argument.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    horizons = np.array([1, 5, 10], dtype=float)

    for seq in SEQUENCES:
        seq_frame = rpe[rpe["sequence"] == seq]

        # Seed-level traces: deliberately light so the three-seed mean remains
        # the main visual object.
        first_seed = True
        for seed in EXPECTED_BASE_SEEDS:
            q = seq_frame[seq_frame["base_seed"] == seed].sort_values("horizon_s")
            if len(q) != len(horizons):
                raise RuntimeError(
                    f"Expected three RPE horizons for {seq}, seed {seed}; found {len(q)}"
                )
            ax.plot(
                q["horizon_s"],
                q["v2_rpe_trans_rmse_m"],
                marker="o",
                linewidth=0.9,
                alpha=0.25,
                label=(f"{seq} individual seeds" if first_seed else None),
            )
            first_seed = False

        grouped = (
            seq_frame.groupby("horizon_s")["v2_rpe_trans_rmse_m"]
            .agg(["mean", "std"])
            .reindex([1, 5, 10])
        )
        ax.errorbar(
            horizons,
            grouped["mean"].to_numpy(dtype=float),
            yerr=grouped["std"].fillna(0.0).to_numpy(dtype=float),
            marker="o",
            linewidth=2.4,
            capsize=4,
            label=f"{seq} mean ± seed SD",
        )

    ax.set_xticks(horizons)
    ax.set_xticklabels(["1 s", "5 s", "10 s"])
    ax.set_xlabel("Relative-pose horizon")
    ax.set_ylabel("Translational RPE RMSE [m]")
    ax.set_title(
        "Short-horizon Twin V2 fidelity: local relative-pose error remains small"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism(
    aligned: dict[tuple[str, int], pd.DataFrame],
    out: Path,
) -> None:
    # Use the middle base seed for a readable mechanistic time-series figure.
    seed = 1042

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=False)

    for seq in SEQUENCES:
        f = aligned[(seq, seed)]
        elapsed = f["time_s"] - f["time_s"].iloc[0]

        axes[0].plot(
            elapsed,
            np.degrees(f["yaw_disagreement_radps"]),
            label=seq,
        )
        axes[1].plot(elapsed, f["Iomega_deg"], label=seq)
        axes[2].plot(elapsed, f["Dtheta_deg"], label=seq)
        axes[3].plot(elapsed, f["Dp_m"], label=seq)

    axes[0].set_ylabel("wheel–IMU mismatch\n[deg/s]")
    axes[1].set_ylabel(r"$I_\omega$ [deg]")
    axes[2].set_ylabel(r"$D_\theta$ [deg]")
    axes[3].set_ylabel(r"$D_p$ [m]")
    axes[3].set_xlabel("elapsed time [s]")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(
        "Mechanism comparison (base seed 1042): "
        "sensor mismatch → accumulated yaw → heading → position divergence"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    aligned: dict[tuple[str, int], pd.DataFrame],
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, seq in zip(axes, SEQUENCES):
        # GT is identical across seeds. Plot once.
        f0 = aligned[(seq, 42)]
        ax.plot(
            f0["gt_east_m"],
            f0["gt_north_m"],
            linewidth=2.5,
            label="Ground truth",
        )

        for seed in EXPECTED_BASE_SEEDS:
            f = aligned[(seq, seed)]
            ax.plot(
                f["estimate_east_m"],
                f["estimate_north_m"],
                linewidth=1.2,
                label=f"Twin V2 seed {seed}",
            )

        ax.set_title(seq)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.axis("equal")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Full physical-reference vs Twin V2 trajectories")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_text_summary(
    per_sequence: pd.DataFrame,
    corr: pd.DataFrame,
    path: Path,
) -> None:
    lookup = {
        row["sequence"]: row
        for _, row in per_sequence.iterrows()
    }

    p00 = lookup["parking00"]
    p02 = lookup["parking02"]

    key_metrics = [
        ("v2_ate_rmse_m_mean", "V2 ATE [m]"),
        ("v2_heading_mae_deg_mean", "V2 heading MAE [deg]"),
        ("v2_rpe_1s_m_mean", "V2 RPE1 [m]"),
        ("v2_rpe_5s_m_mean", "V2 RPE5 [m]"),
        ("v2_rpe_10s_m_mean", "V2 RPE10 [m]"),
        ("yaw_disagreement_abs_mean_radps_mean", "mean |wheel-IMU yaw disagreement| [rad/s]"),
        ("prediction_yaw_residual_p95_abs_radps_mean", "p95 |V2 yaw residual| [rad/s]"),
        ("curvature_abs_mean_radpm_mean", "mean |curvature proxy| [rad/m]"),
        ("Dtheta_p95_deg_mean", "Dtheta p95 [deg]"),
        ("Dp_p95_m_mean", "Dp p95 [m]"),
        ("Iomega_max_abs_deg_mean", "max |Iomega| [deg]"),
    ]

    lines = [
        "parking02 vs parking00 post-LOSO diagnostic",
        "=" * 72,
        "",
        "Three-seed sequence means",
        "-" * 72,
    ]

    for col, label in key_metrics:
        if col in p00 and col in p02:
            a = float(p00[col])
            b = float(p02[col])
            ratio = b / a if abs(a) > 1e-12 else float("nan")
            lines.append(
                f"{label:<44} parking00={a:12.6f}  "
                f"parking02={b:12.6f}  ratio={ratio:8.3f}"
            )

    lines += [
        "",
        "Interpretation rule",
        "-" * 72,
        "Do not infer causation from a single timestamp-level correlation.",
        "Use these results to identify whether parking02 differs consistently in",
        "turning intensity, wheel-IMU disagreement, persistent yaw residual,",
        "Iomega accumulation, heading divergence, and position divergence.",
        "",
        "Important mechanistic correlations (seed-averaged descriptively)",
        "-" * 72,
    ]

    for seq in SEQUENCES:
        q = corr[corr["sequence"] == seq]
        for (x, y), g in q.groupby(["x", "y"]):
            vals = g["pearson_r"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                lines.append(
                    f"{seq:<10} {x:<38} -> {y:<28} "
                    f"mean r={np.mean(vals):+.3f}"
                )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    repo = repo_root_from_script()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo / "public_datasets" / "im2nav",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo / "results" / "i2nav_v2_full_loso",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo
            / "results"
            / "i2nav_frozen_v2_fidelity_analysis"
            / "parking02_vs_parking00"
        ),
    )
    args = parser.parse_args()

    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Script version:", SCRIPT_VERSION)
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"i2Nav data root not found: {args.data_root}")
    if not args.results_root.is_dir():
        raise FileNotFoundError(
            f"Full LOSO results root not found: {args.results_root}"
        )

    print("Data root   :", args.data_root)
    print("Results root:", args.results_root)
    print("Output dir  :", args.output_dir)

    run_dirs = locate_run_dirs(args.results_root)

    print("\nReconstructing frozen canonical sensor context...")
    _, _, prepared, canonical = configure_and_prepare(args.data_root)

    aligned: dict[tuple[str, int], pd.DataFrame] = {}
    run_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []

    for seq in SEQUENCES:
        for seed in EXPECTED_BASE_SEEDS:
            run_dir = run_dirs[(seq, seed)]
            print(f"\nAnalyzing {seq} base seed {seed}: {run_dir}")

            frame = align_frames(
                seq,
                run_dir,
                prepared[seq],
                canonical,
            )
            aligned[(seq, seed)] = frame

            aligned_path = (
                args.output_dir
                / f"aligned_timeseries_{seq}_base{seed}.csv"
            )
            frame.to_csv(aligned_path, index=False)

            run_rows.append(
                summarize_run(seq, seed, run_dir, frame)
            )
            corr_rows.extend(
                mechanism_correlations(seq, seed, frame)
            )

    per_run = pd.DataFrame(run_rows).sort_values(
        ["sequence", "base_seed"]
    )
    corr = pd.DataFrame(corr_rows).sort_values(
        ["sequence", "base_seed", "x", "y"]
    )
    per_sequence = aggregate_sequences(per_run)
    short_rpe = build_short_horizon_rpe_table(per_run)

    per_run.to_csv(
        args.output_dir / "per_run_summary.csv",
        index=False,
    )
    per_sequence.to_csv(
        args.output_dir / "per_sequence_summary.csv",
        index=False,
    )
    short_rpe.to_csv(
        args.output_dir / "short_horizon_rpe.csv",
        index=False,
    )
    corr.to_csv(
        args.output_dir / "mechanism_correlations.csv",
        index=False,
    )

    plot_metric_comparison(
        per_run,
        args.output_dir / "parking02_vs_parking00_metrics.png",
    )
    plot_short_horizon_rpe(
        short_rpe,
        args.output_dir / "parking02_vs_parking00_short_horizon_rpe.png",
    )
    plot_mechanism(
        aligned,
        args.output_dir / "parking02_vs_parking00_mechanism.png",
    )
    plot_trajectories(
        aligned,
        args.output_dir / "parking02_vs_parking00_trajectory.png",
    )

    write_text_summary(
        per_sequence,
        corr,
        args.output_dir / "parking02_vs_parking00_summary.txt",
    )

    print("\n" + "=" * 80)
    print("THREE-SEED SEQUENCE SUMMARY")
    print("=" * 80)

    display_cols = [
        "sequence",
        "v2_ate_rmse_m_mean",
        "v2_heading_mae_deg_mean",
        "v2_rpe_1s_m_mean",
        "v2_rpe_5s_m_mean",
        "v2_rpe_10s_m_mean",
        "yaw_disagreement_abs_mean_radps_mean",
        "prediction_yaw_residual_p95_abs_radps_mean",
        "curvature_abs_mean_radpm_mean",
        "Dtheta_p95_deg_mean",
        "Dp_p95_m_mean",
        "Iomega_max_abs_deg_mean",
    ]
    display_cols = [c for c in display_cols if c in per_sequence.columns]
    print(per_sequence[display_cols].to_string(index=False))

    print("\nSHORT-HORIZON V2 RPE (three-seed mean ± SD)")
    print("-" * 80)
    rpe_console = (
        short_rpe.groupby(["sequence", "horizon_s"])["v2_rpe_trans_rmse_m"]
        .agg(["mean", "std"])
        .reset_index()
    )
    print(rpe_console.to_string(index=False))

    print("\nSaved outputs:")
    for p in sorted(args.output_dir.iterdir()):
        print(" ", p.name)

    print(
        "\nThis script is descriptive/diagnostic. "
        "Timestamp-level correlations are not independent statistical replicates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
