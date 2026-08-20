"""Official i2Nav benchmark evaluation for frozen trajectory artifacts.

This script does not train or tune any model. It consumes saved frozen V2
trajectory CSVs and already-exported fixed-physics trajectories, converts the
V2 planar ENU/FLU states to the verified i2Nav TUM/NED convention, and evaluates
them with the public i2Nav-WHU/evaluate_odometry mechanics implemented through
evo.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evo.core import metrics as evo_metrics
from evo.core import sync as evo_sync
from evo.core import trajectory as evo_trajectory


SEQUENCES = [
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
]

RPE_DELTAS_M = [50, 100, 150, 200, 250, 300]
MAX_TIME_SYNC_DIFF_S = 0.005
RPE_REL_DELTA_TOL = 0.002
RPE_ALL_PAIRS = True


@dataclass(frozen=True)
class RunSpec:
    method: str
    sequence: str
    replicate: str
    seed: str
    source: Path
    run_dir: Path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_numeric_table(path: Path, min_cols: int = 8) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            toks = line.replace(",", " ").split()
            if len(toks) < min_cols:
                continue
            try:
                vals = [float(x) for x in toks]
            except ValueError:
                continue
            if np.all(np.isfinite(vals[:min_cols])):
                rows.append(vals[: max(min_cols, 8)])
    if not rows:
        raise ValueError(f"no numeric rows with >= {min_cols} columns: {path}")
    width = min(len(r) for r in rows)
    arr = np.asarray([r[:width] for r in rows], dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    _, rev_idx = np.unique(arr[::-1, 0], return_index=True)
    keep = len(arr) - 1 - rev_idx
    keep.sort()
    return arr[keep]


def save_tum(path: Path, traj: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, traj, fmt="%.9f %.9f %.9f %.9f %.10f %.10f %.10f %.10f")


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_to_xyzw(yaw: np.ndarray) -> np.ndarray:
    q = np.zeros((len(yaw), 4), dtype=float)
    q[:, 2] = np.sin(0.5 * yaw)
    q[:, 3] = np.cos(0.5 * yaw)
    return q


def v2_csv_to_official_estimate(csv_path: Path, reference: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    df = pd.read_csv(csv_path)
    required = ["time_s", "estimate_east_m", "estimate_north_m", "estimate_heading_rad"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    time_s = df["time_s"].to_numpy(dtype=float)
    east = df["estimate_east_m"].to_numpy(dtype=float)
    north = df["estimate_north_m"].to_numpy(dtype=float)
    heading_enu = df["estimate_heading_rad"].to_numpy(dtype=float)
    if not (np.all(np.isfinite(time_s)) and np.all(np.isfinite(east)) and np.all(np.isfinite(north))):
        raise ValueError(f"{csv_path} contains non-finite estimate/time values")

    ref_down = np.interp(time_s[0], reference[:, 0], reference[:, 3])
    down = np.full_like(time_s, float(ref_down))
    yaw_ned = wrap_angle(np.pi / 2.0 - heading_enu)
    q = yaw_to_xyzw(yaw_ned)
    traj = np.column_stack([time_s, north, east, down, q]).astype(float)
    info = {
        "coordinate_conversion": (
            "V2 internal planar ENU/FLU estimate -> official TUM/NED: "
            "tx=north, ty=east, tz=constant initial reference down, yaw_ned=pi/2-heading_enu"
        ),
        "poses": int(len(traj)),
        "timestamp_start_s": float(time_s[0]),
        "timestamp_end_s": float(time_s[-1]),
    }
    return traj, info


def numpy_to_evo(traj: np.ndarray) -> evo_trajectory.PoseTrajectory3D:
    stamps = traj[:, 0]
    xyz = traj[:, 1:4]
    quat_xyzw = traj[:, 4:8]
    quat_wxyz = np.roll(quat_xyzw, 1, axis=1)
    return evo_trajectory.PoseTrajectory3D(xyz, quat_wxyz, stamps)


def evaluate_official(reference: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "official_available": False,
        "official_error": "",
        "official_associated_poses": None,
        "official_max_time_sync_diff_s": MAX_TIME_SYNC_DIFF_S,
        "official_alignment": "SE3_no_scale",
        "official_ape_translation_rmse_m": math.nan,
        "official_ape_rotation_rmse_deg": math.nan,
    }
    for delta in RPE_DELTAS_M:
        prefix = f"official_rpe_{delta}m"
        out[f"{prefix}_translation_rmse_m"] = math.nan
        out[f"{prefix}_translation_pct"] = math.nan
        out[f"{prefix}_rotation_rmse_deg"] = math.nan
        out[f"{prefix}_error"] = ""

    try:
        ref_raw = numpy_to_evo(reference)
        est_raw = numpy_to_evo(estimate)
        ref, est = evo_sync.associate_trajectories(ref_raw, est_raw, MAX_TIME_SYNC_DIFF_S)
        import copy

        est_aligned = copy.deepcopy(est)
        est_aligned.align(ref, correct_scale=False, correct_only_scale=False)

        ape_t = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)
        ape_t.process_data((ref, est_aligned))
        ape_r = evo_metrics.APE(evo_metrics.PoseRelation.rotation_angle_deg)
        ape_r.process_data((ref, est_aligned))

        out.update(
            {
                "official_available": True,
                "official_associated_poses": int(len(est_aligned.timestamps)),
                "official_ape_translation_rmse_m": float(ape_t.get_all_statistics()["rmse"]),
                "official_ape_rotation_rmse_deg": float(ape_r.get_all_statistics()["rmse"]),
            }
        )
        for delta in RPE_DELTAS_M:
            prefix = f"official_rpe_{delta}m"
            try:
                rpe_t = evo_metrics.RPE(
                    evo_metrics.PoseRelation.translation_part,
                    delta,
                    evo_metrics.Unit.meters,
                    RPE_REL_DELTA_TOL,
                    RPE_ALL_PAIRS,
                )
                rpe_t.process_data((ref, est_aligned))
                rpe_r = evo_metrics.RPE(
                    evo_metrics.PoseRelation.rotation_angle_deg,
                    delta,
                    evo_metrics.Unit.meters,
                    RPE_REL_DELTA_TOL,
                    RPE_ALL_PAIRS,
                )
                rpe_r.process_data((ref, est_aligned))
                trans = float(rpe_t.get_all_statistics()["rmse"])
                out[f"{prefix}_translation_rmse_m"] = trans
                out[f"{prefix}_translation_pct"] = 100.0 * trans / float(delta)
                out[f"{prefix}_rotation_rmse_deg"] = float(rpe_r.get_all_statistics()["rmse"])
                out[f"{prefix}_error"] = ""
            except Exception as exc:
                out[f"{prefix}_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        out["official_error"] = f"{type(exc).__name__}: {exc}"
        for delta in RPE_DELTAS_M:
            prefix = f"official_rpe_{delta}m"
            out[f"{prefix}_error"] = "evaluation unavailable"
    return out


def discover_v2_runs(v2_root: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for traj in sorted(v2_root.rglob("v2_evaluated_trajectory.csv")):
        fold_dir = traj.parent
        rep_dir = fold_dir.parent
        manifest = fold_dir / "run_manifest.json"
        summary = fold_dir / "run_summary.json"
        sequence = fold_dir.name.split("_", 2)[-1]
        seed = rep_dir.name
        if summary.exists():
            sequence = str(read_json(summary).get("test_sequence", sequence))
        if manifest.exists():
            data = read_json(manifest)
            seed = str(data.get("base_seed", seed))
        runs.append(
            RunSpec(
                method="Twin V2",
                sequence=sequence,
                replicate=rep_dir.name,
                seed=seed,
                source=traj,
                run_dir=fold_dir,
            )
        )
    return runs


def discover_fixed_runs(fixed_root: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []
    if not fixed_root.exists():
        return runs
    for seq in SEQUENCES:
        path = fixed_root / seq / "fixed_v5_estimate_traj.txt"
        if path.exists():
            runs.append(
                RunSpec(
                    method="Fixed Physics",
                    sequence=seq,
                    replicate="deterministic_fixed_v5",
                    seed="deterministic",
                    source=path,
                    run_dir=path.parent,
                )
            )
    return runs


def load_internal_v2_summary(run: RunSpec) -> dict[str, Any]:
    path = run.run_dir / "run_summary.json"
    if not path.exists():
        return {}
    data = read_json(path)
    return {
        "internal_ate_rmse_m": data.get("v2_ate_rmse_m"),
        "internal_heading_mae_deg": data.get("v2_heading_mae_deg"),
        "internal_rpe_1s_m": data.get("v2_rpe_1s_m"),
        "internal_rpe_5s_m": data.get("v2_rpe_5s_m"),
        "internal_rpe_10s_m": data.get("v2_rpe_10s_m"),
        "internal_Dp_p95_m": data.get("fidelity_Dp_p95_m"),
        "internal_Dtheta_p95_deg": data.get("fidelity_Dtheta_p95_deg"),
    }


def load_fixed_internal_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = pd.read_csv(path)
    out: dict[str, dict[str, Any]] = {}
    for _, r in rows.iterrows():
        seq = str(r.get("sequence", ""))
        out[seq] = {
            "internal_ate_rmse_m": r.get("ate_rmse_m"),
            "internal_heading_mae_deg": r.get("heading_mae_deg"),
            "internal_rpe_1s_m": r.get("rpe_1s_trans_rmse_m"),
            "internal_rpe_5s_m": r.get("rpe_5s_trans_rmse_m"),
            "internal_rpe_10s_m": r.get("rpe_10s_trans_rmse_m"),
            "internal_Dp_p95_m": math.nan,
            "internal_Dtheta_p95_deg": math.nan,
        }
    return out


def sequence_stats(per_run: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_run.columns if c.startswith("official_") and c.endswith(("_m", "_deg", "_pct"))]
    rows = []
    for (method, sequence), g in per_run.groupby(["method", "sequence"], sort=True):
        row: dict[str, Any] = {
            "method": method,
            "sequence": sequence,
            "run_count": int(len(g)),
        }
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else math.nan
            row[f"{col}_median"] = float(vals.median()) if len(vals) else math.nan
            row[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def macro_stats(per_seq: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in per_seq.columns if c.endswith("_mean")]
    rows = []
    for method, g in per_seq.groupby("method", sort=True):
        row: dict[str, Any] = {"method": method, "sequence_count": int(g["sequence"].nunique())}
        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            stem = col.removesuffix("_mean")
            row[f"{stem}_macro_mean"] = float(vals.mean()) if len(vals) else math.nan
            row[f"{stem}_macro_median"] = float(vals.median()) if len(vals) else math.nan
            row[f"{stem}_sequence_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else math.nan
            row[f"{stem}_sequence_min"] = float(vals.min()) if len(vals) else math.nan
            row[f"{stem}_sequence_max"] = float(vals.max()) if len(vals) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_ci(diff: np.ndarray, iters: int = 20000, seed: int = 20260820) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return math.nan, math.nan
    means = np.empty(iters, dtype=float)
    for i in range(iters):
        means[i] = float(np.mean(diff[rng.integers(0, n, n)]))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_p(diff: np.ndarray) -> float:
    diff = np.asarray([d for d in diff if np.isfinite(d)], dtype=float)
    n = len(diff)
    if n == 0:
        return math.nan
    observed = abs(float(np.mean(diff)))
    count = 0
    total = 2**n
    for signs in itertools.product([-1.0, 1.0], repeat=n):
        if abs(float(np.mean(diff * np.asarray(signs)))) >= observed - 1e-15:
            count += 1
    return count / total


def compare_methods(per_seq: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    metrics = [
        "official_ape_translation_rmse_m",
        "official_ape_rotation_rmse_deg",
        *[f"official_rpe_{d}m_translation_rmse_m" for d in RPE_DELTAS_M],
        *[f"official_rpe_{d}m_rotation_rmse_deg" for d in RPE_DELTAS_M],
    ]
    b = per_seq[per_seq["method"] == baseline].set_index("sequence")
    c = per_seq[per_seq["method"] == candidate].set_index("sequence")
    common = sorted(set(b.index) & set(c.index))
    rows = []
    for metric in metrics:
        col = f"{metric}_mean"
        if col not in b.columns or col not in c.columns:
            continue
        bv = pd.to_numeric(b.loc[common, col], errors="coerce").to_numpy(dtype=float)
        cv = pd.to_numeric(c.loc[common, col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(bv) & np.isfinite(cv)
        bv = bv[mask]
        cv = cv[mask]
        seqs = np.asarray(common)[mask]
        if len(bv) == 0:
            continue
        diff = cv - bv
        pct = 100.0 * diff / np.where(np.abs(bv) > 1e-12, bv, np.nan)
        baseline_mean = float(np.mean(bv))
        candidate_mean = float(np.mean(cv))
        macro_ratio_pct = (
            100.0 * (candidate_mean - baseline_mean) / baseline_mean
            if abs(baseline_mean) > 1e-12
            else math.nan
        )
        ci_lo, ci_hi = paired_bootstrap_ci(diff)
        rows.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "metric": metric,
                "sequence_count": int(len(diff)),
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "absolute_difference_candidate_minus_baseline": float(np.mean(diff)),
                "macro_mean_percent_difference_candidate_vs_baseline": macro_ratio_pct,
                "mean_sequencewise_percent_difference_candidate_vs_baseline": float(np.nanmean(pct)),
                "percent_difference_candidate_vs_baseline": float(np.nanmean(pct)),
                "sequences_improved_lower_is_better": int(np.sum(diff < 0)),
                "paired_bootstrap_ci_low": ci_lo,
                "paired_bootstrap_ci_high": ci_hi,
                "exact_sign_flip_p": sign_flip_p(diff),
                "matched_sequences": ";".join(str(s) for s in seqs),
            }
        )
    return pd.DataFrame(rows)


def internal_vs_official(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    v2 = per_run[per_run["method"] == "Twin V2"].copy()
    for sequence, g in v2.groupby("sequence", sort=True):
        rows.append(
            {
                "method": "Twin V2",
                "sequence": sequence,
                "official_ape_translation_rmse_m_mean": pd.to_numeric(g["official_ape_translation_rmse_m"], errors="coerce").mean(),
                "internal_ate_rmse_m_mean": pd.to_numeric(g["internal_ate_rmse_m"], errors="coerce").mean(),
                "official_minus_internal_ate_m": pd.to_numeric(g["official_ape_translation_rmse_m"], errors="coerce").mean()
                - pd.to_numeric(g["internal_ate_rmse_m"], errors="coerce").mean(),
                "official_ape_rotation_rmse_deg_mean": pd.to_numeric(g["official_ape_rotation_rmse_deg"], errors="coerce").mean(),
                "internal_heading_mae_deg_mean": pd.to_numeric(g["internal_heading_mae_deg"], errors="coerce").mean(),
                "internal_rpe_1s_m_mean": pd.to_numeric(g["internal_rpe_1s_m"], errors="coerce").mean(),
                "internal_rpe_5s_m_mean": pd.to_numeric(g["internal_rpe_5s_m"], errors="coerce").mean(),
                "internal_rpe_10s_m_mean": pd.to_numeric(g["internal_rpe_10s_m"], errors="coerce").mean(),
                "official_rpe_50m_translation_rmse_m_mean": pd.to_numeric(g["official_rpe_50m_translation_rmse_m"], errors="coerce").mean(),
                "internal_Dp_p95_m_mean": pd.to_numeric(g["internal_Dp_p95_m"], errors="coerce").mean(),
                "internal_Dtheta_p95_deg_mean": pd.to_numeric(g["internal_Dtheta_p95_deg"], errors="coerce").mean(),
            }
        )
    return pd.DataFrame(rows)


def make_figures(per_seq: pd.DataFrame, internal_cmp: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    v2 = per_seq[per_seq["method"] == "Twin V2"].copy()
    fixed = per_seq[per_seq["method"] == "Fixed Physics"].copy()
    seq_order = SEQUENCES

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    width = 0.38
    x = np.arange(len(seq_order))
    for ax, metric, title, ylabel in [
        (axes[0], "official_ape_translation_rmse_m_mean", "Official APE Translation RMSE", "m"),
        (axes[1], "official_rpe_50m_translation_rmse_m_mean", "Official RPE Translation RMSE at 50 m", "m"),
    ]:
        vf = v2.set_index("sequence").reindex(seq_order)[metric]
        ff = fixed.set_index("sequence").reindex(seq_order)[metric] if len(fixed) else pd.Series(index=seq_order, dtype=float)
        ax.bar(x - width / 2, ff, width, label="Fixed Physics", color="#9aa7b2")
        ax.bar(x + width / 2, vf, width, label="Twin V2", color="#1f77b4")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(seq_order, rotation=35, ha="right")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "official_benchmark_results.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    df = internal_cmp.set_index("sequence").reindex(seq_order)
    axes[0].scatter(df["internal_ate_rmse_m_mean"], df["official_ape_translation_rmse_m_mean"], color="#1f77b4")
    for seq, row in df.iterrows():
        if seq in {"parking01", "parking02"}:
            axes[0].annotate(seq, (row["internal_ate_rmse_m_mean"], row["official_ape_translation_rmse_m_mean"]))
    axes[0].set_xlabel("Internal ATE RMSE (m)")
    axes[0].set_ylabel("Official APE RMSE after SE(3) alignment (m)")
    axes[0].set_title("Internal vs Official Translation Error")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(df["internal_Dp_p95_m_mean"], df["official_ape_translation_rmse_m_mean"], color="#d55e00")
    for seq, row in df.iterrows():
        if seq in {"parking01", "parking02"}:
            axes[1].annotate(seq, (row["internal_Dp_p95_m_mean"], row["official_ape_translation_rmse_m_mean"]))
    axes[1].set_xlabel("Internal Dp p95 (m)")
    axes[1].set_ylabel("Official APE RMSE (m)")
    axes[1].set_title("Global DT Divergence vs Aligned Benchmark")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "official_hard_sequence_comparison.png", dpi=180)
    plt.close(fig)


def fmt(x: Any, digits: int = 3) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "NA"
        return f"{v:.{digits}f}"
    except Exception:
        return "NA"


def write_summary(
    out_dir: Path,
    per_run: pd.DataFrame,
    per_seq: pd.DataFrame,
    macro: pd.DataFrame,
    comparison: pd.DataFrame,
    internal_cmp: pd.DataFrame,
    baseline_notes: list[str],
) -> None:
    v2_macro = macro[macro["method"] == "Twin V2"].iloc[0]
    fixed_macro = macro[macro["method"] == "Fixed Physics"].iloc[0] if "Fixed Physics" in set(macro["method"]) else None
    v2_seq = per_seq[per_seq["method"] == "Twin V2"].copy()
    hardest = v2_seq.sort_values("official_ape_translation_rmse_m_mean", ascending=False).head(3)
    cmp_ape = comparison[comparison["metric"] == "official_ape_translation_rmse_m"]
    cmp_rpe50 = comparison[comparison["metric"] == "official_rpe_50m_translation_rmse_m"]

    lines = [
        "# Official i2Nav Benchmark Evaluation: Frozen Twin V2",
        "",
        "This report uses the verified public i2Nav-WHU `evaluate_odometry` protocol and the already-frozen Twin V2 outputs. No model was retrained, retuned, checkpoint-selected, or altered.",
        "",
        "## Protocol",
        "",
        "- Trajectory format: TUM rows `t tx ty tz qx qy qz qw`.",
        "- Frame: i2Nav local NED reference; V2 internal ENU/FLU estimates exported to NED with yaw conversion.",
        "- Association tolerance: `0.005 s`.",
        "- Alignment: SE(3), no scale correction.",
        "- Metrics: APE translation/rotation RMSE and all-pairs distance RPE at 50/100/150/200/250/300 m with relative delta tolerance `0.002`.",
        "",
        "## What Are The Official Frozen V2 Benchmark Results?",
        "",
        f"- V2 official APE translation macro mean: **{fmt(v2_macro.get('official_ape_translation_rmse_m_macro_mean'))} m**.",
        f"- V2 official APE rotation macro mean: **{fmt(v2_macro.get('official_ape_rotation_rmse_deg_macro_mean'))} deg**.",
        f"- V2 official RPE 50 m translation macro mean: **{fmt(v2_macro.get('official_rpe_50m_translation_rmse_m_macro_mean'))} m** ({fmt(v2_macro.get('official_rpe_50m_translation_pct_macro_mean'))}%).",
        f"- V2 official RPE 100 m translation macro mean: **{fmt(v2_macro.get('official_rpe_100m_translation_rmse_m_macro_mean'))} m** ({fmt(v2_macro.get('official_rpe_100m_translation_pct_macro_mean'))}%).",
        f"- V2 official RPE 300 m translation macro mean: **{fmt(v2_macro.get('official_rpe_300m_translation_rmse_m_macro_mean'))} m** ({fmt(v2_macro.get('official_rpe_300m_translation_pct_macro_mean'))}%).",
        "",
        "## Fixed Physics / V1 Availability",
        "",
    ]
    if baseline_notes:
        lines.extend(f"- {note}" for note in baseline_notes)
    if fixed_macro is not None:
        lines.extend(
            [
                f"- Fixed Physics official APE translation macro mean: **{fmt(fixed_macro.get('official_ape_translation_rmse_m_macro_mean'))} m**.",
                f"- Fixed Physics official RPE 50 m translation macro mean: **{fmt(fixed_macro.get('official_rpe_50m_translation_rmse_m_macro_mean'))} m**.",
            ]
        )
    lines.extend(["", "## V2 Compared With Fixed Physics", ""])
    if not cmp_ape.empty:
        row = cmp_ape.iloc[0]
        lines.append(
            f"- APE translation macro mean: V2 changes by **{fmt(row['absolute_difference_candidate_minus_baseline'])} m** "
            f"(**{fmt(row['macro_mean_percent_difference_candidate_vs_baseline'])}% macro-mean reduction**) versus Fixed Physics; "
            f"the mean sequence-wise relative change is **{fmt(row['mean_sequencewise_percent_difference_candidate_vs_baseline'])}%**; "
            f"improved on {int(row['sequences_improved_lower_is_better'])}/{int(row['sequence_count'])} sequences; "
            f"bootstrap CI [{fmt(row['paired_bootstrap_ci_low'])}, {fmt(row['paired_bootstrap_ci_high'])}], "
            f"sign-flip p={fmt(row['exact_sign_flip_p'], 4)}."
        )
    if not cmp_rpe50.empty:
        row = cmp_rpe50.iloc[0]
        lines.append(
            f"- RPE 50 m translation macro mean: V2 changes by **{fmt(row['absolute_difference_candidate_minus_baseline'])} m** "
            f"(**{fmt(row['macro_mean_percent_difference_candidate_vs_baseline'])}% macro-mean reduction**) versus Fixed Physics; "
            f"the mean sequence-wise relative change is **{fmt(row['mean_sequencewise_percent_difference_candidate_vs_baseline'])}%**; "
            f"improved on {int(row['sequences_improved_lower_is_better'])}/{int(row['sequence_count'])} sequences."
        )
    lines.extend(
        [
            "",
            "## Hard Sequences",
            "",
            "Largest V2 official APE translation sequences:",
            "",
            "| sequence | official APE trans. RMSE (m) | internal ATE RMSE (m) | internal Dp p95 (m) |",
            "|---|---:|---:|---:|",
        ]
    )
    ic = internal_cmp.set_index("sequence")
    for _, row in hardest.iterrows():
        seq = row["sequence"]
        irow = ic.loc[seq] if seq in ic.index else {}
        lines.append(
            f"| {seq} | {fmt(row.get('official_ape_translation_rmse_m_mean'))} | "
            f"{fmt(irow.get('internal_ate_rmse_m_mean') if hasattr(irow, 'get') else math.nan)} | "
            f"{fmt(irow.get('internal_Dp_p95_m_mean') if hasattr(irow, 'get') else math.nan)} |"
        )
    lines.extend(
        [
            "",
            "parking01/parking02 remain important, but the official SE(3)-aligned APE layer compresses some of the long-horizon drift that is visible in the internal DT-fidelity layer.",
            "",
            "## Official vs Internal DT-Fidelity Layer",
            "",
            "- Official APE/RPE are benchmark metrics after SE(3) alignment; they are suitable for protocol-compatible odometry-style comparison.",
            "- Internal DT-fidelity metrics remain the correct evidence for physical-virtual synchronization because they do not use post-hoc alignment to hide drift.",
            "- The local-vs-global result remains visible by comparison: short-horizon internal RPE can stay small while internal Dp/Dtheta grows, even if official aligned APE is reduced.",
            "",
            "## Carry-Forward Benchmark Numbers",
            "",
            "Use the V2 macro means in `official_macro_summary.csv` for later sensing-fidelity comparison, especially:",
            "",
            "- `official_ape_translation_rmse_m_macro_mean`",
            "- `official_ape_rotation_rmse_deg_macro_mean`",
            "- `official_rpe_50m_translation_rmse_m_macro_mean`",
            "- `official_rpe_100m_translation_rmse_m_macro_mean`",
            "- `official_rpe_300m_translation_rmse_m_macro_mean`",
            "",
            "Do not claim state of the art unless published systems are evaluated under this same protocol.",
            "",
            "## Files Produced",
            "",
            "- `official_export_manifest.json`",
            "- `official_per_run_results.csv`",
            "- `official_per_sequence_results.csv`",
            "- `official_macro_summary.csv`",
            "- `official_method_comparison.csv`",
            "- `official_internal_comparison.csv`",
            "- `official_benchmark_results.png`",
            "- `official_hard_sequence_comparison.png`",
        ]
    )
    (out_dir / "official_benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    exports_dir = out_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    references: dict[str, np.ndarray] = {}
    reference_paths: dict[str, Path] = {}
    for seq in SEQUENCES:
        p = args.dataset_root / seq / f"{seq}_trajectory.csv"
        if not p.exists():
            raise FileNotFoundError(f"missing official reference trajectory: {p}")
        references[seq] = read_numeric_table(p, min_cols=8)[:, :8]
        reference_paths[seq] = p
        save_tum(exports_dir / "references" / f"{seq}_reference_tum.txt", references[seq])

    v2_runs = discover_v2_runs(args.v2_root)
    fixed_runs = discover_fixed_runs(args.fixed_root)
    if len(v2_runs) != 30:
        raise RuntimeError(f"expected 30 V2 runs, found {len(v2_runs)}")

    baseline_notes: list[str] = [
        "Twin V2: 30 frozen runs found and evaluated (10 sequences x 3 base seeds).",
        "Fixed Physics: included as the deterministic `fixed_v5_replay` trajectory where one official-format run per sequence exists.",
        "V1: not included in official scoring because no exact matching frozen V1 trajectory files were found; only scalar internal V1 metrics exist in V2 summaries.",
    ]

    export_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    fixed_internal = load_fixed_internal_rows(args.fixed_summary_csv)

    for run in [*fixed_runs, *v2_runs]:
        ref = references[run.sequence]
        ref_path = exports_dir / "references" / f"{run.sequence}_reference_tum.txt"
        method_slug = run.method.lower().replace(" ", "_")
        est_path = exports_dir / method_slug / run.replicate / run.sequence / "estimate_tum.txt"

        if run.method == "Twin V2":
            est, info = v2_csv_to_official_estimate(run.source, ref)
            internal = load_internal_v2_summary(run)
        else:
            est = read_numeric_table(run.source, min_cols=8)[:, :8]
            info = {
                "coordinate_conversion": "already exported official TUM/NED fixed_v5 trajectory from prior frozen replay",
                "poses": int(len(est)),
                "timestamp_start_s": float(est[0, 0]),
                "timestamp_end_s": float(est[-1, 0]),
            }
            internal = fixed_internal.get(run.sequence, {})

        save_tum(est_path, est)
        eval_row = evaluate_official(ref, est)
        row = {
            "method": run.method,
            "sequence": run.sequence,
            "replicate": run.replicate,
            "seed": run.seed,
            "source_trajectory": str(run.source),
            "official_reference_source": str(reference_paths[run.sequence]),
            "official_reference_export": str(ref_path),
            "official_estimate_export": str(est_path),
            **info,
            **internal,
            **eval_row,
        }
        result_rows.append(row)
        export_rows.append(
            {
                "method": run.method,
                "sequence": run.sequence,
                "replicate": run.replicate,
                "seed": run.seed,
                "source_trajectory": str(run.source),
                "coordinate_conversion": info["coordinate_conversion"],
                "official_reference_source": str(reference_paths[run.sequence]),
                "official_reference_export": str(ref_path),
                "official_estimate_export": str(est_path),
                "poses": info["poses"],
                "timestamp_start_s": info["timestamp_start_s"],
                "timestamp_end_s": info["timestamp_end_s"],
            }
        )

    write_json(
        out_dir / "official_export_manifest.json",
        {
            "schema": "i2nav_official_export_manifest_v1",
            "protocol": {
                "association_tolerance_s": MAX_TIME_SYNC_DIFF_S,
                "alignment": "SE3_no_scale",
                "scale_correction": False,
                "rpe_deltas_m": RPE_DELTAS_M,
                "rpe_all_pairs": RPE_ALL_PAIRS,
                "rpe_relative_delta_tolerance": RPE_REL_DELTA_TOL,
            },
            "exports": export_rows,
            "baseline_notes": baseline_notes,
        },
    )

    per_run = pd.DataFrame(result_rows)
    per_seq = sequence_stats(per_run)
    macro = macro_stats(per_seq)
    comparison = compare_methods(per_seq, "Fixed Physics", "Twin V2")
    internal_cmp = internal_vs_official(per_run)

    per_run.to_csv(out_dir / "official_per_run_results.csv", index=False)
    per_seq.to_csv(out_dir / "official_per_sequence_results.csv", index=False)
    macro.to_csv(out_dir / "official_macro_summary.csv", index=False)
    comparison.to_csv(out_dir / "official_method_comparison.csv", index=False)
    internal_cmp.to_csv(out_dir / "official_internal_comparison.csv", index=False)

    make_figures(per_seq, internal_cmp, out_dir)
    write_summary(out_dir, per_run, per_seq, macro, comparison, internal_cmp, baseline_notes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, default=Path("results/i2nav_v2_full_loso/i2nav_v2_full_loso"))
    parser.add_argument("--dataset-root", type=Path, default=Path("public_datasets/im2nav"))
    parser.add_argument("--fixed-root", type=Path, default=Path("results/i2nav_final_model_study/phase1_official_fixed_v5"))
    parser.add_argument("--fixed-summary-csv", type=Path, default=Path("results/i2nav_final_model_study/phase1_official_fixed_v5.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/i2nav_official_benchmark"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
