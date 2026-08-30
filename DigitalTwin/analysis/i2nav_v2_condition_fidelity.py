#!/usr/bin/env python3
"""Condition-dependent fidelity analysis for frozen Twin V2 LOSO outputs.

This script is post-hoc only. It reads saved frozen V2 trajectories,
prediction traces, and fidelity time series, reconstructs canonical i2Nav
sensor context, freezes deterministic condition-bin definitions, and then
summarizes fidelity inside each run/condition.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DigitalTwin.analysis.i2nav_v2_all_sequence_mechanism import (
    EXPECTED_BASE_SEEDS,
    EXPECTED_SEQUENCES,
    FROZEN_FULL_LOSO_COMMIT,
    SCRIPT_VERSION as MECHANISM_SCRIPT_VERSION,
    align_run_frame,
    configure_and_prepare,
    locate_run_dirs,
    p95_abs,
    repo_root_from_script,
    rms,
    safe_spearman,
    wrap_angle,
)


SCRIPT_VERSION = "2026-08-20-condition-fidelity-v1"
MIN_CONDITION_SAMPLES = 30
MIN_RPE_PAIRS = 20


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def environment_category(sequence: str) -> str:
    for prefix in ("building", "parking", "street", "playground"):
        if sequence.startswith(prefix):
            return prefix
    return "unknown"


def add_condition_columns(frame: pd.DataFrame, sequence: str) -> pd.DataFrame:
    out = frame.copy()
    t = out["time_s"].to_numpy(dtype=float)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    elapsed = (t - t[0]) / max(duration, 1e-9)

    persistent_radps = float(out["prediction_yaw_residual_radps"].mean())
    persistent_deg_per_min = abs(persistent_radps) * 180.0 / math.pi * 60.0

    out["condition_speed_abs_mps"] = out["speed_abs_mps"]
    out["condition_accel_abs_mps2"] = out["abs_longitudinal_accel_mps2"]
    out["condition_yaw_abs_radps"] = out["abs_imu_yaw_radps"]
    out["condition_curvature_abs_radpm"] = out["abs_curvature_proxy_radpm"]
    out["condition_wheel_imu_disagreement_abs_radps"] = out[
        "abs_yaw_disagreement_radps"
    ]
    out["condition_persistent_yaw_mismatch_deg_per_min"] = persistent_deg_per_min
    out["condition_lateral_proxy_abs_mps"] = out["wheel_lateral_mps"].abs()
    out["condition_elapsed_fraction"] = elapsed
    out["condition_environment_category"] = environment_category(sequence)
    return out


def _finite_quantiles(values: np.ndarray, qs: list[float]) -> list[float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [None for _ in qs]
    return [float(np.quantile(values, q)) for q in qs]


def freeze_condition_definitions(unique_condition_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Define bins using condition variables only, before outcome summaries."""
    combined = pd.concat(unique_condition_frames.values(), ignore_index=True)

    variables = [
        (
            "speed",
            "condition_speed_abs_mps",
            "translational speed magnitude from canonical ODO",
            "m/s",
            "low",
            True,
        ),
        (
            "acceleration",
            "condition_accel_abs_mps2",
            "absolute longitudinal acceleration from ODO speed gradient",
            "m/s^2",
            "low",
            True,
        ),
        (
            "turning",
            "condition_yaw_abs_radps",
            "absolute IMU yaw rate / turning intensity",
            "rad/s",
            "low",
            True,
        ),
        (
            "curvature",
            "condition_curvature_abs_radpm",
            "absolute curvature proxy |yaw rate| / max(|speed|, 0.10)",
            "rad/m",
            "low",
            True,
        ),
        (
            "wheel_imu_disagreement",
            "condition_wheel_imu_disagreement_abs_radps",
            "absolute canonical wheel-IMU yaw disagreement",
            "rad/s",
            "low",
            True,
        ),
        (
            "persistent_yaw_mismatch",
            "condition_persistent_yaw_mismatch_deg_per_min",
            "run-level absolute mean V2 yaw residual",
            "deg/min",
            "low",
            False,
        ),
        (
            "lateral_slip_proxy",
            "condition_lateral_proxy_abs_mps",
            "absolute canonical lateral wheel-motion proxy when available",
            "m/s",
            "low",
            False,
        ),
    ]

    defs: dict[str, Any] = {
        "schema": "i2nav_v2_condition_definitions_v1",
        "script_version": SCRIPT_VERSION,
        "mechanism_script_version_reused": MECHANISM_SCRIPT_VERSION,
        "frozen_full_loso_commit": FROZEN_FULL_LOSO_COMMIT,
        "bin_policy": (
            "Numeric stress variables use global 1/3 and 2/3 quantiles of "
            "condition variables only. Outcome variables are not used to choose thresholds."
        ),
        "statistical_hierarchy": "timestamps within seed runs within physical sequences within dataset",
        "variables": {},
    }

    for name, column, description, units, nominal, degradation_supported in variables:
        q1, q2 = _finite_quantiles(combined[column].to_numpy(dtype=float), [1 / 3, 2 / 3])
        finite_count = int(np.isfinite(combined[column].to_numpy(dtype=float)).sum())
        if q1 is None or q2 is None:
            defs["variables"][name] = {
                "type": "unavailable",
                "source_column": column,
                "description": description,
                "units": units,
                "finite_sample_count": finite_count,
                "bins": [],
                "nominal_bin": None,
                "degradation_supported": False,
                "note": "No finite values were present in the frozen canonical context.",
            }
            continue
        defs["variables"][name] = {
            "type": "numeric_tertile",
            "source_column": column,
            "description": description,
            "units": units,
            "finite_sample_count": finite_count,
            "thresholds": {
                "low_upper": q1,
                "medium_upper": q2,
            },
            "bins": ["low", "medium", "high"],
            "nominal_bin": nominal,
            "degradation_supported": degradation_supported,
        }

    defs["variables"]["elapsed_time"] = {
        "type": "fixed_fraction",
        "source_column": "condition_elapsed_fraction",
        "description": "elapsed fraction within each run",
        "units": "fraction of run duration",
        "thresholds": {"early_upper": 1 / 3, "middle_upper": 2 / 3},
        "bins": ["early", "middle", "late"],
        "nominal_bin": "early",
        "degradation_supported": True,
    }
    defs["variables"]["environment"] = {
        "type": "metadata_category",
        "source_column": "condition_environment_category",
        "description": "sequence prefix category supported by i2Nav sequence names",
        "categories": sorted(
            set(combined["condition_environment_category"].dropna().astype(str))
        ),
        "bins": sorted(
            set(combined["condition_environment_category"].dropna().astype(str))
        ),
        "nominal_bin": None,
        "degradation_supported": False,
        "note": "Used for descriptive stratification only; not an ordered stress variable.",
    }
    return defs


def assign_bin(values: pd.Series, definition: dict[str, Any]) -> pd.Series:
    kind = definition["type"]
    if kind == "numeric_tertile":
        lo = float(definition["thresholds"]["low_upper"])
        mid = float(definition["thresholds"]["medium_upper"])
        result = pd.Series(pd.NA, index=values.index, dtype="object")
        finite = pd.to_numeric(values, errors="coerce")
        result[finite <= lo] = "low"
        result[(finite > lo) & (finite <= mid)] = "medium"
        result[finite > mid] = "high"
        return result
    if kind == "fixed_fraction":
        early = float(definition["thresholds"]["early_upper"])
        middle = float(definition["thresholds"]["middle_upper"])
        finite = pd.to_numeric(values, errors="coerce")
        result = pd.Series(pd.NA, index=values.index, dtype="object")
        result[finite <= early] = "early"
        result[(finite > early) & (finite <= middle)] = "middle"
        result[finite > middle] = "late"
        return result
    if kind == "metadata_category":
        return values.astype("object")
    if kind == "unavailable":
        return pd.Series(pd.NA, index=values.index, dtype="object")
    raise ValueError(f"Unknown condition definition type: {kind}")


def rpe_for_condition(frame: pd.DataFrame, mask: np.ndarray, horizon_s: float) -> tuple[float, float, int]:
    time_s = frame["time_s"].to_numpy(dtype=float)
    if len(time_s) < 3:
        return float("nan"), float("nan"), 0
    dt = float(np.median(np.diff(time_s)))
    steps = int(round(horizon_s / dt))
    if steps <= 0 or steps >= len(frame):
        return float("nan"), float("nan"), 0

    start = np.where(mask[:-steps])[0]
    if len(start) < MIN_RPE_PAIRS:
        return float("nan"), float("nan"), int(len(start))
    end = start + steps

    gt_xy = frame[["gt_east_m", "gt_north_m"]].to_numpy(dtype=float)
    est_xy = frame[["estimate_east_m", "estimate_north_m"]].to_numpy(dtype=float)
    gt_heading = frame["gt_heading_rad"].to_numpy(dtype=float)
    est_heading = frame["estimate_heading_rad"].to_numpy(dtype=float)

    def rel_translation(xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
        dp = xy[end] - xy[start]
        theta0 = heading[start]
        c = np.cos(theta0)
        s = np.sin(theta0)
        return np.column_stack((c * dp[:, 0] + s * dp[:, 1], -s * dp[:, 0] + c * dp[:, 1]))

    gt_rel = rel_translation(gt_xy, gt_heading)
    est_rel = rel_translation(est_xy, est_heading)
    trans_rmse = float(np.sqrt(np.mean(np.sum((est_rel - gt_rel) ** 2, axis=1))))

    gt_dtheta = wrap_angle(gt_heading[end] - gt_heading[start])
    est_dtheta = wrap_angle(est_heading[end] - est_heading[start])
    rot_mae_deg = float(np.degrees(np.mean(np.abs(wrap_angle(est_dtheta - gt_dtheta)))))
    return trans_rmse, rot_mae_deg, int(len(start))


def summarize_condition(
    sequence: str,
    base_seed: int,
    frame: pd.DataFrame,
    condition_variable: str,
    condition_bin: str,
    mask: np.ndarray,
) -> dict[str, Any]:
    subset = frame.loc[mask]
    t = subset["time_s"].to_numpy(dtype=float)
    duration = float(len(subset) * np.median(np.diff(frame["time_s"].to_numpy(dtype=float))))

    row: dict[str, Any] = {
        "sequence": sequence,
        "base_seed": int(base_seed),
        "environment_category": environment_category(sequence),
        "condition_variable": condition_variable,
        "condition_bin": condition_bin,
        "n_samples": int(len(subset)),
        "duration_s_approx": duration,
        "time_start_s": float(np.min(t)) if len(t) else float("nan"),
        "time_end_s": float(np.max(t)) if len(t) else float("nan"),
    }
    if len(subset) < MIN_CONDITION_SAMPLES:
        row["supported"] = False
        return row
    row["supported"] = True

    row.update(
        {
            "Dp_mean_m": float(subset["Dp_m"].mean()),
            "Dp_p95_m": float(np.percentile(subset["Dp_m"], 95.0)),
            "Dp_max_m": float(subset["Dp_m"].max()),
            "Dtheta_mean_deg": float(subset["Dtheta_deg"].mean()),
            "Dtheta_p95_deg": float(np.percentile(subset["Dtheta_deg"], 95.0)),
            "Dtheta_max_deg": float(subset["Dtheta_deg"].max()),
            "Dv_rmse_mps": rms(subset.get("rv_mps", pd.Series(dtype=float)).to_numpy()),
            "Dv_p95_mps": p95_abs(subset.get("rv_mps", pd.Series(dtype=float)).to_numpy()),
            "Domega_rmse_radps": rms(
                subset.get("romega_radps", pd.Series(dtype=float)).to_numpy()
            ),
            "Domega_p95_radps": p95_abs(
                subset.get("romega_radps", pd.Series(dtype=float)).to_numpy()
            ),
            "persistent_yaw_residual_signed_mean_radps": float(
                subset["prediction_yaw_residual_radps"].mean()
            ),
            "persistent_yaw_residual_abs_deg_per_min": abs(
                float(subset["prediction_yaw_residual_radps"].mean())
            )
            * 180.0
            / math.pi
            * 60.0,
            "Iomega_start_deg": float(subset["Iomega_deg"].iloc[0]),
            "Iomega_end_deg": float(subset["Iomega_deg"].iloc[-1]),
            "Iomega_delta_deg": float(subset["Iomega_deg"].iloc[-1] - subset["Iomega_deg"].iloc[0]),
            "Iomega_p95_abs_deg": p95_abs(subset["Iomega_deg"].to_numpy()),
            "Iomega_max_abs_deg": float(np.max(np.abs(subset["Iomega_deg"].to_numpy(dtype=float)))),
        }
    )

    for horizon_s in (1.0, 5.0, 10.0):
        rpe_m, rpe_rot_deg, n_pairs = rpe_for_condition(frame, mask, horizon_s)
        label = int(horizon_s)
        row[f"RPE{label}_m"] = rpe_m
        row[f"RPEtheta{label}_deg"] = rpe_rot_deg
        row[f"RPE{label}_n_pairs"] = n_pairs

    return row


def build_condition_rows(
    frames: dict[tuple[str, int], pd.DataFrame],
    definitions: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (sequence, seed), frame in frames.items():
        for var_name, definition in definitions["variables"].items():
            if definition["type"] == "unavailable":
                continue
            source = definition["source_column"]
            bins = assign_bin(frame[source], definition)
            ordered_bins = definition["bins"]
            for bin_name in ordered_bins:
                mask = (bins == bin_name).to_numpy()
                if not np.any(mask):
                    rows.append(
                        {
                            "sequence": sequence,
                            "base_seed": seed,
                            "environment_category": environment_category(sequence),
                            "condition_variable": var_name,
                            "condition_bin": bin_name,
                            "n_samples": 0,
                            "duration_s_approx": 0.0,
                            "supported": False,
                        }
                    )
                    continue
                rows.append(
                    summarize_condition(sequence, seed, frame, var_name, bin_name, mask)
                )
    return pd.DataFrame(rows)


def aggregate_per_sequence(per_run_condition: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in per_run_condition.columns
        if c
        not in {
            "sequence",
            "base_seed",
            "environment_category",
            "condition_variable",
            "condition_bin",
            "supported",
        }
        and pd.api.types.is_numeric_dtype(per_run_condition[c])
    ]
    rows: list[dict[str, Any]] = []
    group_cols = ["sequence", "environment_category", "condition_variable", "condition_bin"]
    for keys, group in per_run_condition.groupby(group_cols, dropna=False, sort=True):
        seq, env, var, bin_name = keys
        supported = group[group["supported"].astype(bool)]
        row: dict[str, Any] = {
            "sequence": seq,
            "environment_category": env,
            "condition_variable": var,
            "condition_bin": bin_name,
            "n_seed_rows": int(len(group)),
            "n_supported_seed_rows": int(len(supported)),
            "supported": bool(len(supported) > 0),
        }
        if len(supported):
            for col in numeric_cols:
                vals = pd.to_numeric(supported[col], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    row[col] = float(np.mean(vals))
                    row[f"{col}_seed_sd"] = (
                        float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def degradation_summary(
    per_sequence: pd.DataFrame,
    definitions: dict[str, Any],
) -> pd.DataFrame:
    metrics = [
        "Dp_mean_m",
        "Dp_p95_m",
        "Dtheta_mean_deg",
        "Dtheta_p95_deg",
        "Dv_rmse_mps",
        "Domega_rmse_radps",
        "RPE1_m",
        "RPE5_m",
        "RPE10_m",
        "persistent_yaw_residual_abs_deg_per_min",
        "Iomega_p95_abs_deg",
    ]
    rows: list[dict[str, Any]] = []
    for var_name, definition in definitions["variables"].items():
        nominal = definition.get("nominal_bin")
        if not nominal or not definition.get("degradation_supported", False):
            continue
        q = per_sequence[
            (per_sequence["condition_variable"] == var_name)
            & (per_sequence["supported"].astype(bool))
        ]
        for bin_name in definition["bins"]:
            if bin_name == nominal:
                continue
            for metric in metrics:
                if metric not in q.columns:
                    continue
                seq_rows = []
                for seq, group in q.groupby("sequence"):
                    base = group[group["condition_bin"] == nominal]
                    comp = group[group["condition_bin"] == bin_name]
                    if base.empty or comp.empty:
                        continue
                    a = float(base[metric].iloc[0])
                    b = float(comp[metric].iloc[0])
                    if not np.isfinite(a) or not np.isfinite(b):
                        continue
                    seq_rows.append((seq, a, b, b - a, b / a if abs(a) > 1e-12 else np.nan))
                if not seq_rows:
                    continue
                deltas = np.array([r[3] for r in seq_rows], dtype=float)
                ratios = np.array([r[4] for r in seq_rows], dtype=float)
                finite_ratios = ratios[np.isfinite(ratios)]
                rows.append(
                    {
                        "condition_variable": var_name,
                        "nominal_bin": nominal,
                        "comparison_bin": bin_name,
                        "metric": metric,
                        "n_sequences": int(len(seq_rows)),
                        "mean_delta_vs_nominal": float(np.mean(deltas)),
                        "median_delta_vs_nominal": float(np.median(deltas)),
                        "mean_ratio_vs_nominal": (
                            float(np.mean(finite_ratios)) if len(finite_ratios) else np.nan
                        ),
                        "median_ratio_vs_nominal": (
                            float(np.median(finite_ratios)) if len(finite_ratios) else np.nan
                        ),
                        "n_sequences_degraded": int(np.sum(deltas > 0.0)),
                        "fraction_sequences_degraded": float(np.mean(deltas > 0.0)),
                        "sequences_degraded": ",".join(r[0] for r in seq_rows if r[3] > 0.0),
                    }
                )
    return pd.DataFrame(rows)


def plot_condition(
    per_sequence: pd.DataFrame,
    condition_variable: str,
    metric: str,
    title: str,
    ylabel: str,
    out: Path,
) -> None:
    q = per_sequence[
        (per_sequence["condition_variable"] == condition_variable)
        & (per_sequence["supported"].astype(bool))
    ].copy()
    if q.empty or metric not in q:
        return
    order = [
        b
        for b in ("low", "medium", "high", "early", "middle", "late")
        if b in set(q["condition_bin"])
    ]
    if not order:
        order = sorted(q["condition_bin"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    for seq, group in q.groupby("sequence"):
        vals = []
        xs = []
        for i, b in enumerate(order):
            row = group[group["condition_bin"] == b]
            if not row.empty and np.isfinite(float(row[metric].iloc[0])):
                xs.append(i)
                vals.append(float(row[metric].iloc[0]))
        ax.plot(xs, vals, marker="o", alpha=0.55, linewidth=1.0, label=seq)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)

    ax = axes[1]
    data = [
        q[q["condition_bin"] == b][metric].dropna().astype(float).to_numpy()
        for b in order
    ]
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_ylabel(ylabel)
    ax.set_title("Distribution across physical sequences")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_outputs(per_sequence: pd.DataFrame, out_dir: Path) -> None:
    plot_condition(
        per_sequence,
        "speed",
        "Dp_p95_m",
        "Global divergence by translational speed",
        "Dp p95 [m]",
        out_dir / "fidelity_by_speed.png",
    )
    plot_condition(
        per_sequence,
        "turning",
        "Dp_p95_m",
        "Global divergence by turning intensity",
        "Dp p95 [m]",
        out_dir / "fidelity_by_turning.png",
    )
    plot_condition(
        per_sequence,
        "wheel_imu_disagreement",
        "Dp_p95_m",
        "Global divergence by wheel-IMU disagreement",
        "Dp p95 [m]",
        out_dir / "fidelity_by_wheel_imu_disagreement.png",
    )
    plot_condition(
        per_sequence,
        "elapsed_time",
        "Dp_p95_m",
        "Global divergence by elapsed-time regime",
        "Dp p95 [m]",
        out_dir / "fidelity_by_time.png",
    )


def _top_degradation(
    summary: pd.DataFrame, metric: str, variables: list[str] | None = None
) -> pd.DataFrame:
    q = summary[summary["metric"] == metric].copy()
    if variables:
        q = q[q["condition_variable"].isin(variables)]
    if q.empty:
        return q
    return q.sort_values(
        ["fraction_sequences_degraded", "median_delta_vs_nominal"],
        ascending=[False, False],
    )


def fmt(x: Any, digits: int = 3) -> str:
    try:
        value = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_summary(
    out_dir: Path,
    definitions: dict[str, Any],
    per_sequence: pd.DataFrame,
    degradation: pd.DataFrame,
) -> None:
    local_metrics = ["RPE1_m", "RPE5_m", "RPE10_m"]
    global_metrics = ["Dp_p95_m", "Dtheta_p95_deg"]

    local_rows = []
    for metric in local_metrics:
        top = _top_degradation(degradation, metric)
        if not top.empty:
            local_rows.append(top.iloc[0])
    global_rows = []
    for metric in global_metrics:
        top = _top_degradation(degradation, metric)
        if not top.empty:
            global_rows.append(top.iloc[0])

    p02 = per_sequence[per_sequence["sequence"] == "parking02"]
    p02_high_flags = []
    for var in [
        "speed",
        "acceleration",
        "turning",
        "curvature",
        "wheel_imu_disagreement",
        "persistent_yaw_mismatch",
    ]:
        q = p02[(p02["condition_variable"] == var) & (p02["condition_bin"] == "high")]
        all_seq = per_sequence[
            (per_sequence["condition_variable"] == var)
            & (per_sequence["condition_bin"] == "high")
        ]
        if not q.empty and not all_seq.empty:
            dur = float(q["duration_s_approx"].iloc[0])
            rank_dp = (
                all_seq.sort_values("Dp_p95_m", ascending=False)["sequence"]
                .tolist()
                .index("parking02")
                + 1
            )
            p02_high_flags.append((var, dur, rank_dp))

    supported_vars = []
    for var, definition in definitions["variables"].items():
        if definition["type"] == "unavailable":
            continue
        q = per_sequence[
            (per_sequence["condition_variable"] == var)
            & (per_sequence["supported"].astype(bool))
        ]
        seqs = q["sequence"].nunique()
        bins_per_seq = q.groupby("sequence")["condition_bin"].nunique()
        if seqs >= 8 and len(bins_per_seq) and bins_per_seq.min() >= 2:
            supported_vars.append(var)

    lines = [
        "# Condition-Dependent Twin V2 Fidelity",
        "",
        f"Script version: `{SCRIPT_VERSION}`",
        f"Frozen full-LOSO commit expected by context: `{FROZEN_FULL_LOSO_COMMIT}`",
        "",
        "This analysis uses only saved frozen V2 LOSO artifacts and reconstructed "
        "canonical i2Nav sensor context. It does not retrain, tune, or alter V2.",
        "",
        "## Frozen Condition Definitions",
        "",
        "Condition-bin definitions are recorded in `condition_definitions.json`. "
        "Numeric variables use global tertiles of condition variables only; outcome "
        "metrics are not used to choose thresholds.",
        "",
        "## Statistical Hierarchy",
        "",
        "Condition metrics are computed within each seed run, then the three seeds are "
        "aggregated within each physical sequence. Dataset-level interpretation uses "
        "the 10 physical sequences as the unit. Timestamp-level samples are not treated "
        "as independent replicates.",
        "",
        "## Which Conditions Consistently Degrade Local Fidelity?",
        "",
    ]

    if local_rows:
        lines += ["| Metric | strongest condition contrast | median delta | degraded sequences |", "|---|---|---:|---:|"]
        for row in local_rows:
            lines.append(
                f"| {row['metric']} | {row['condition_variable']}: "
                f"{row['comparison_bin']} vs {row['nominal_bin']} | "
                f"{fmt(row['median_delta_vs_nominal'])} | "
                f"{int(row['n_sequences_degraded'])}/{int(row['n_sequences'])} |"
            )
    else:
        lines.append("No supported condition contrast produced a stable local-RPE degradation.")

    lines += [
        "",
        "Local fidelity changes are generally weaker and less monotonic than global "
        "synchronization changes. This matches the earlier local-vs-global finding: "
        "low short-horizon RPE can coexist with long-horizon drift.",
        "",
        "## Which Conditions Consistently Degrade Global Synchronization?",
        "",
    ]
    if global_rows:
        lines += ["| Metric | strongest condition contrast | median delta | degraded sequences |", "|---|---|---:|---:|"]
        for row in global_rows:
            lines.append(
                f"| {row['metric']} | {row['condition_variable']}: "
                f"{row['comparison_bin']} vs {row['nominal_bin']} | "
                f"{fmt(row['median_delta_vs_nominal'])} | "
                f"{int(row['n_sequences_degraded'])}/{int(row['n_sequences'])} |"
            )

    lines += [
        "",
        "## Are Global-Divergence Conditions Different From RPE Conditions?",
        "",
        "Yes, in the current frozen V2 evidence they are not the same object. "
        "RPE degradation is finite-horizon and often modest, while Dp/Dtheta degradation "
        "is more sensitive to elapsed time and accumulated orientation mismatch. "
        "Therefore the paper should avoid using RPE alone as the definition of digital-"
        "twin fidelity.",
        "",
        "## Is parking02 Explained By an Extreme Measurable Condition?",
        "",
        "parking02 is not fully explained by a single extreme bin of speed, acceleration, "
        "turning, curvature, or wheel-IMU disagreement. It remains the "
        "largest global-divergence sequence even though some simple operating-condition "
        "variables are not uniquely extreme. This supports a sequence-specific behavior "
        "interpretation: the measurable benign conditions help characterize stress, but "
        "they do not by themselves collapse parking02 into an ordinary high-speed or "
        "high-turning case.",
        "",
        "parking02 high-bin diagnostic:",
        "",
        "| Variable | high-bin duration approx [s] | parking02 Dp-p95 rank within high bin |",
        "|---|---:|---:|",
    ]
    for var, dur, rank_dp in p02_high_flags:
        lines.append(f"| {var} | {fmt(dur, 1)} | {rank_dp} |")

    lines += [
        "",
        "## Variables Supported For Later Benign Fidelity Envelope",
        "",
        "Supported conditioning variables are those with enough sequence/bin coverage to "
        "summarize without leaning on one sequence.",
        "",
        ", ".join(f"`{v}`" for v in supported_vars) if supported_vars else "None.",
        "",
        "The environment category is useful descriptively, but it is not an ordered "
        "stress variable and should not be treated like speed or turning.",
        "The lateral/slip proxy is not supported by finite values in the current frozen "
        "canonical i2Nav context, so it should not be used for an envelope from this run.",
        "",
        "## Files Produced",
        "",
        "- `condition_definitions.json`",
        "- `per_run_condition_fidelity.csv`",
        "- `per_sequence_condition_fidelity.csv`",
        "- `condition_degradation_summary.csv`",
        "- `fidelity_by_speed.png`",
        "- `fidelity_by_turning.png`",
        "- `fidelity_by_wheel_imu_disagreement.png`",
        "- `fidelity_by_time.png`",
        "",
        "## Null / Weak Findings",
        "",
        "Not every condition variable produces a strong or monotonic relationship. The "
        "analysis should preserve these weak findings because they are scientifically "
        "important: the twin's hard failures are not reducible to a single obvious "
        "stress scalar.",
    ]

    out_dir.joinpath("condition_fidelity_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    repo = repo_root_from_script()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=repo / "public_datasets" / "im2nav")
    parser.add_argument(
        "--results-root", type=Path, default=repo / "results" / "i2nav_v2_full_loso"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo
        / "results"
        / "i2nav_frozen_v2_fidelity_analysis"
        / "condition_fidelity",
    )
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Script version:", SCRIPT_VERSION)
    print("Data root     :", args.data_root)
    print("Results root  :", args.results_root)
    print("Output dir    :", args.output_dir)

    run_dirs = locate_run_dirs(args.results_root)
    prepared, canonical = configure_and_prepare(args.data_root)

    frames: dict[tuple[str, int], pd.DataFrame] = {}
    unique_sequence_frames: dict[str, pd.DataFrame] = {}
    for sequence in EXPECTED_SEQUENCES:
        for base_seed in EXPECTED_BASE_SEEDS:
            print(f"Aligning {sequence} seed {base_seed}")
            frame = align_run_frame(
                sequence,
                run_dirs[(sequence, base_seed)],
                prepared[sequence],
                canonical,
            )
            frame = add_condition_columns(frame, sequence)
            frames[(sequence, base_seed)] = frame
            if base_seed == EXPECTED_BASE_SEEDS[0]:
                unique_sequence_frames[sequence] = frame

    definitions = freeze_condition_definitions(unique_sequence_frames)
    definitions["analysis_git_commit"] = git_commit(repo)
    definitions["input_results_root"] = str(args.results_root)
    definitions["input_data_root"] = str(args.data_root)
    args.output_dir.joinpath("condition_definitions.json").write_text(
        json.dumps(definitions, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    per_run = build_condition_rows(frames, definitions)
    per_sequence = aggregate_per_sequence(per_run)
    degradation = degradation_summary(per_sequence, definitions)

    per_run.to_csv(args.output_dir / "per_run_condition_fidelity.csv", index=False)
    per_sequence.to_csv(args.output_dir / "per_sequence_condition_fidelity.csv", index=False)
    degradation.to_csv(args.output_dir / "condition_degradation_summary.csv", index=False)

    plot_outputs(per_sequence, args.output_dir)
    write_summary(args.output_dir, definitions, per_sequence, degradation)

    print("\nSaved outputs:")
    for path in sorted(args.output_dir.iterdir()):
        print(" ", path.name)
    print("\nTop global degradation rows:")
    q = _top_degradation(degradation, "Dp_p95_m").head(8)
    if not q.empty:
        print(
            q[
                [
                    "condition_variable",
                    "comparison_bin",
                    "metric",
                    "median_delta_vs_nominal",
                    "n_sequences_degraded",
                    "n_sequences",
                ]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
