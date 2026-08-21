"""Frozen TerraSentia/AIFARMS external validation for Twin V2.

This script intentionally preserves the Phase-2 adapter conventions:
bag-time synchronization, motor mapping, 0.26 m track width, raw
``/terrasentia/imu`` angular_velocity.z, i2Nav normalization, frozen V2
weights, and the architecture-independent fidelity evaluator.

It is an external portability study, not a TerraSentia tuning pass.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from DigitalTwin.analysis import aifarms_terrasentia_phase2 as p2
from DigitalTwin.analysis import i2nav_v2_full_loso as v2
from DigitalTwin.analysis.i2nav_fidelity_evaluator import evaluate_fidelity_frames


DEFAULT_INPUT_ROOT = Path("public_datasets/aifarms/processed")
DEFAULT_CHECKPOINT_ROOT = Path("results/i2nav_v2_full_loso")
DEFAULT_OUTPUT = Path("results/aifarms_terrasentia_full_study")

SEQUENCES = [
    "ts_2022_06_09_13h16m39s_one_row",
    "ts_2022_06_15_11h48m34s_four_rows",
    "ts_2022_09_01_11h20m00s_two_random",
    "ts_2022_09_01_12h32m56s_double_loop_corridor",
    "ts_2022_09_06_12h37m11s_four_rows",
]

PRIMARY_METRICS = [
    "ATE_m",
    "RPEp_1s_m",
    "RPEp_5s_m",
    "RPEp_10s_m",
    "Dp_p95_m",
    "Dp_max_m",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def discover_checkpoints(root: Path) -> list[Path]:
    checkpoints = sorted(root.rglob("v2_slow_additive_yaw.pt"))
    if len(checkpoints) != 30:
        raise RuntimeError(f"expected 30 frozen V2 checkpoints, found {len(checkpoints)} under {root}")
    return checkpoints


def checkpoint_identity(path: Path) -> dict[str, Any]:
    parts = path.parts
    replicate = next((p for p in parts if p.startswith("replicate_")), "unknown_replicate")
    fold = next((p for p in parts if p.startswith("fold_")), "unknown_fold")
    seed_match = re.search(r"base(\d+)", replicate)
    seed = int(seed_match.group(1)) if seed_match else None
    return {
        "checkpoint": str(path),
        "replicate": replicate,
        "fold": fold,
        "seed": seed,
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def configure_v2_runtime() -> None:
    v2.RATE = p2.RATE_HZ
    v2.DT = p2.DT_S
    v2.SLOW_SAMPLES = int(round(30.0 * p2.RATE_HZ))
    v2.CHUNK_STEPS = int(round(30.0 * p2.RATE_HZ))
    v2.DEVICE = torch.device("cpu")


def predict_v2(aligned: pd.DataFrame, checkpoint_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    configure_v2_runtime()
    ckpt = load_checkpoint(checkpoint_path)

    fast = aligned[[f"fast_feature_{i}" for i in range(6)]].to_numpy(np.float32)
    slow = aligned[[f"slow_feature_{i}" for i in range(16)]].to_numpy(np.float32)
    fast_norm = (fast - ckpt["fast_feature_mean"]) / ckpt["fast_feature_std"]
    slow_norm = (slow - ckpt["slow_feature_mean"]) / ckpt["slow_feature_std"]
    slow_norm = np.nan_to_num(slow_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    model = v2.V2SlowAdditiveYaw(fast_input_dim=6, slow_input_dim=16)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(v2.DEVICE)
    model.eval()

    class Cache:
        pass

    cache = Cache()
    cache.grid = aligned["time_s"].to_numpy(float)
    cache.fast_windows = v2.base.sliding_windows(fast_norm.astype(np.float32), v2.FAST_WINDOW)
    cache.slow_features = slow_norm
    prediction = v2.predict_sequence(model, cache, eval_batch_size=4096)

    n = len(aligned)
    x = np.zeros((n, 3), dtype=float)
    x[0, 0] = float(aligned["rtk_east_m"].iloc[0])
    x[0, 1] = float(aligned["rtk_north_m"].iloc[0])
    x[0, 2] = float(aligned["reference_heading_rad"].iloc[0])
    corrected_v = aligned["forward_motor_speed_mps"].to_numpy(float) + prediction["dv"]
    corrected_w = aligned["imu_yaw_rate_radps"].to_numpy(float) + prediction["dw"]
    for k in range(1, n):
        dt = float(aligned["time_s"].iloc[k] - aligned["time_s"].iloc[k - 1])
        theta_prev = x[k - 1, 2]
        x[k, 0] = x[k - 1, 0] + float(corrected_v[k]) * math.cos(theta_prev) * dt
        x[k, 1] = x[k - 1, 1] + float(corrected_v[k]) * math.sin(theta_prev) * dt
        x[k, 2] = float(p2.wrap_angle(theta_prev + float(corrected_w[k]) * dt))

    trace = pd.DataFrame(
        {
            "time_s": aligned["time_s"],
            "bag_time_s": aligned["bag_time_s"],
            "base_forward_motor_speed_mps": aligned["forward_motor_speed_mps"],
            "base_imu_yaw_rate_radps": aligned["imu_yaw_rate_radps"],
            "pred_delta_v_mps": prediction["dv"],
            "pred_fast_delta_omega_radps": prediction["dw_fast"],
            "pred_slow_bias_radps": prediction["b_slow"],
            "pred_total_delta_omega_radps": prediction["dw"],
            "pred_v_T_mps": corrected_v,
            "pred_omega_T_radps": corrected_w,
            "x_T_m": x[:, 0],
            "y_T_m": x[:, 1],
            "theta_T_rad": x[:, 2],
        }
    )
    sat = {
        "frac_abs_delta_v_near_limit": float(np.mean(np.abs(prediction["dv"]) >= 0.98 * 0.15)),
        "frac_abs_fast_delta_omega_near_limit": float(np.mean(np.abs(prediction["dw_fast"]) >= 0.98 * 0.020)),
        "frac_abs_slow_bias_near_limit": float(np.mean(np.abs(prediction["b_slow"]) >= 0.98 * 0.005)),
    }
    return trace, sat


def evaluate_trace(
    aligned: pd.DataFrame,
    trace: pd.DataFrame,
    *,
    model: str,
    sequence: str,
    seed: int | None = None,
    replicate: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    trajectory = pd.DataFrame(
        {
            "time_s": aligned["time_s"],
            "gt_east_m": aligned["rtk_east_m"],
            "gt_north_m": aligned["rtk_north_m"],
            "gt_heading_rad": aligned["reference_heading_rad"],
            "estimate_east_m": trace["x_T_m"],
            "estimate_north_m": trace["y_T_m"],
            "estimate_heading_rad": trace["theta_T_rad"],
        }
    )
    pred_trace = pd.DataFrame(
        {
            "time_s": aligned["time_s"],
            "true_delta_v_mps": aligned["reference_linear_mps"] - aligned["forward_motor_speed_mps"],
            "pred_delta_v_mps": trace["pred_v_T_mps"] - aligned["forward_motor_speed_mps"],
            "true_delta_omega_radps": aligned["reference_angular_radps"] - aligned["imu_yaw_rate_radps"],
            "pred_total_delta_omega_radps": trace["pred_omega_T_radps"] - aligned["imu_yaw_rate_radps"],
        }
    )
    return evaluate_fidelity_frames(
        trajectory,
        pred_trace,
        model=model,
        sequence=sequence,
        seed=seed,
        replicate=replicate,
    )


def run_physics(aligned: pd.DataFrame, sequence: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trace = p2.integrate_prediction(
        aligned,
        np.zeros(len(aligned), dtype=float),
        np.zeros(len(aligned), dtype=float),
        label="physics_only_motor_plus_imu",
    )
    profile, timeseries = evaluate_trace(aligned, trace, model="physics_only", sequence=sequence)
    return trace, timeseries, profile


def run_oracles(aligned: pd.DataFrame, sequence: str) -> pd.DataFrame:
    cases = {
        "A_motor_forward_plus_imu_yaw": (
            aligned["forward_motor_speed_mps"].to_numpy(float),
            aligned["imu_yaw_rate_radps"].to_numpy(float),
        ),
        "B_reference_forward_plus_imu_yaw_oracle": (
            aligned["reference_linear_mps"].to_numpy(float),
            aligned["imu_yaw_rate_radps"].to_numpy(float),
        ),
        "C_motor_forward_plus_reference_yaw_oracle": (
            aligned["forward_motor_speed_mps"].to_numpy(float),
            aligned["reference_angular_radps"].to_numpy(float),
        ),
        "D_reference_forward_plus_reference_yaw_control": (
            aligned["reference_linear_mps"].to_numpy(float),
            aligned["reference_angular_radps"].to_numpy(float),
        ),
    }
    rows = []
    base_v = aligned["forward_motor_speed_mps"].to_numpy(float)
    base_w = aligned["imu_yaw_rate_radps"].to_numpy(float)
    for name, (v_sig, w_sig) in cases.items():
        trace = p2.integrate_prediction(aligned, v_sig - base_v, w_sig - base_w, label=name)
        profile, _ = evaluate_trace(aligned, trace, model=name, sequence=sequence)
        row = {"sequence": sequence, "case": name, "diagnostic_only": True}
        row.update({k: profile.get(k) for k in PRIMARY_METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_v2(sequence: str, rows: pd.DataFrame, physics_profile: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"sequence": sequence, "checkpoints": int(len(rows))}
    for metric in PRIMARY_METRICS:
        vals = pd.to_numeric(rows[metric], errors="coerce").dropna()
        out[f"{metric}_mean"] = float(vals.mean())
        out[f"{metric}_median"] = float(vals.median())
        out[f"{metric}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f"{metric}_iqr"] = float(vals.quantile(0.75) - vals.quantile(0.25))
        out[f"{metric}_min"] = float(vals.min())
        out[f"{metric}_max"] = float(vals.max())
        out[f"{metric}_pct_improve_vs_physics"] = float(
            100.0 * np.mean(vals.to_numpy(float) < float(physics_profile[metric]))
        )
    for col in [
        "frac_abs_delta_v_near_limit",
        "frac_abs_fast_delta_omega_near_limit",
        "frac_abs_slow_bias_near_limit",
    ]:
        out[f"{col}_mean"] = float(pd.to_numeric(rows[col], errors="coerce").mean())
    return out


def sequence_diagnostics(sequence: str, aligned: pd.DataFrame, diagnostics: dict[str, Any]) -> dict[str, Any]:
    distance = p2.forward_distance_audit(aligned, DEFAULT_OUTPUT / sequence)
    yaw = p2.heading_yaw_audit(aligned, DEFAULT_OUTPUT / sequence)
    return {
        "sequence": sequence,
        "status": "accepted",
        "samples": int(len(aligned)),
        "duration_s": float(aligned["time_s"].iloc[-1] - aligned["time_s"].iloc[0]),
        "rtk_valid_fraction_raw": float(diagnostics["rtk_valid_fraction_raw"]),
        "rtk_path_length_m": float(distance["rtk_path_length_m"]),
        "motor_integrated_abs_distance_m": float(distance["motor_abs_forward_distance_m"]),
        "forward_distance_ratio_motor_over_rtk": float(distance["motor_to_rtk_distance_ratio"]),
        "imu_vs_ekf_final_heading_diff_deg": float(yaw["final_heading_difference_deg"]),
        "imu_vs_ekf_p95_abs_heading_disagreement_deg": float(yaw["heading_disagreement_p95_abs_deg"]),
        "imu_minus_ekf_mean_signed_yaw_rate_residual_radps": float(
            yaw["mean_signed_imu_minus_reference_yaw_rate_radps"]
        ),
        "imu_minus_ekf_accumulated_signed_yaw_residual_deg": float(
            yaw["accumulated_signed_yaw_rate_residual_final_deg"]
        ),
    }


def representative_plots(
    sequence: str,
    out_dir: Path,
    aligned: pd.DataFrame,
    physics_trace: pd.DataFrame,
    v2_rows: pd.DataFrame,
) -> str | None:
    if v2_rows.empty:
        return None
    median_ate = float(v2_rows["ATE_m"].median())
    idx = (v2_rows["ATE_m"] - median_ate).abs().idxmin()
    trace_path = Path(str(v2_rows.loc[idx, "trace_file"]))
    if not trace_path.exists():
        return None
    trace = pd.read_csv(trace_path)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.plot(aligned["rtk_east_m"], aligned["rtk_north_m"], label="RTK position", lw=1.8)
    ax.plot(physics_trace["x_T_m"], physics_trace["y_T_m"], label="physics-only", lw=1.2)
    ax.plot(trace["x_T_m"], trace["y_T_m"], label="median-ATE frozen V2 checkpoint", lw=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(sequence)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = out_dir / sequence / "representative_trajectory.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def local_global_summary(physics: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    combined = pd.concat(
        [
            physics.assign(method_group="physics"),
            v2.assign(method_group="frozen_v2_checkpoint"),
        ],
        ignore_index=True,
    )
    for _, row in combined.iterrows():
        rows.append(
            {
                "sequence": row["sequence"],
                "model": row["model"],
                "method_group": row["method_group"],
                "ATE_m": row["ATE_m"],
                "Dp_p95_m": row["Dp_p95_m"],
                "RPEp_1s_m": row["RPEp_1s_m"],
                "RPEp_5s_m": row["RPEp_5s_m"],
                "RPEp_10s_m": row["RPEp_10s_m"],
                "low_RPE1_high_ATE": bool(
                    row["RPEp_1s_m"] <= combined["RPEp_1s_m"].median()
                    and row["ATE_m"] >= combined["ATE_m"].median()
                ),
                "low_RPE10_high_Dp95": bool(
                    row["RPEp_10s_m"] <= combined["RPEp_10s_m"].median()
                    and row["Dp_p95_m"] >= combined["Dp_p95_m"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_local_global(local_global: pd.DataFrame, out_dir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))
    groups = {"physics": "tab:gray", "frozen_v2_checkpoint": "tab:blue"}
    for group, color in groups.items():
        sub = local_global[local_global["method_group"] == group]
        axes[0].scatter(sub["RPEp_1s_m"], sub["ATE_m"], s=26, alpha=0.75, label=group, c=color)
        axes[1].scatter(sub["RPEp_10s_m"], sub["Dp_p95_m"], s=26, alpha=0.75, label=group, c=color)
    axes[0].set_xlabel("RPE1 (m)")
    axes[0].set_ylabel("ATE (m)")
    axes[0].set_title("Local RPE1 vs Global ATE")
    axes[1].set_xlabel("RPE10 (m)")
    axes[1].set_ylabel("Dp p95 (m)")
    axes[1].set_title("Finite-Horizon RPE10 vs Global Divergence")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    path = out_dir / "local_vs_global_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_sequence_bars(physics: pd.DataFrame, v2_agg: pd.DataFrame, out_dir: Path) -> str:
    merged = physics[["sequence", "ATE_m", "RPEp_1s_m"]].merge(
        v2_agg[["sequence", "ATE_m_median", "RPEp_1s_m_median"]],
        on="sequence",
        how="inner",
    )
    x = np.arange(len(merged))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].bar(x - width / 2, merged["ATE_m"], width, label="physics")
    axes[0].bar(x + width / 2, merged["ATE_m_median"], width, label="V2 median")
    axes[0].set_ylabel("ATE (m)")
    axes[0].set_title("Global Position Error")
    axes[1].bar(x - width / 2, merged["RPEp_1s_m"], width, label="physics")
    axes[1].bar(x + width / 2, merged["RPEp_1s_m_median"], width, label="V2 median")
    axes[1].set_ylabel("RPE1 (m)")
    axes[1].set_title("Short-Horizon Local Error")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(merged["sequence"], rotation=35, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
    path = out_dir / "sequence_physics_vs_v2.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def write_report(
    out_dir: Path,
    *,
    quality: pd.DataFrame,
    physics: pd.DataFrame,
    v2_agg: pd.DataFrame,
    yaw: pd.DataFrame,
    local_global: pd.DataFrame,
    plots: list[str],
) -> None:
    def f(x: Any, digits: int = 3) -> str:
        try:
            val = float(x)
        except Exception:
            return str(x)
        if not math.isfinite(val):
            return "n/a"
        return f"{val:.{digits}f}"

    accepted = quality[quality["status"] == "accepted"]
    lines = [
        "# TerraSentia/AIFARMS Frozen External Validation",
        "",
        "## Scope",
        "",
        "- Adapter conventions are frozen from the Phase-2 audit.",
        "- No TerraSentia tuning, target-domain normalization, V2 weight changes, or EKF/RTK sign selection was performed.",
        "- RTK is the primary positional reference. Dataset EKF is used only for secondary heading/yaw diagnostics.",
        "- Motor and IMU provenance remains unresolved and is treated as a dataset limitation.",
        "",
        "## Accepted Sequences",
        "",
    ]
    for _, row in accepted.iterrows():
        lines.append(
            f"- `{row['sequence']}`: {int(row['samples'])} samples, "
            f"{f(row['duration_s'], 1)} s, RTK path {f(row['rtk_path_length_m'], 1)} m, "
            f"motor/RTK distance ratio {f(row['forward_distance_ratio_motor_over_rtk'], 3)}."
        )
    if len(accepted) < len(quality):
        lines += ["", "## Excluded Sequences", ""]
        for _, row in quality[quality["status"] != "accepted"].iterrows():
            lines.append(f"- `{row['sequence']}`: {row.get('exclusion_reason', 'quality gate failed')}")

    lines += [
        "",
        "## Physics-Only Positional Fidelity",
        "",
    ]
    for _, row in physics.iterrows():
        lines.append(
            f"- `{row['sequence']}`: ATE {f(row['ATE_m'])} m; "
            f"RPE1/5/10 {f(row['RPEp_1s_m'])}/{f(row['RPEp_5s_m'])}/{f(row['RPEp_10s_m'])} m; "
            f"Dp p95/max {f(row['Dp_p95_m'])}/{f(row['Dp_max_m'])} m."
        )

    lines += [
        "",
        "## Frozen V2 Portability",
        "",
    ]
    for _, row in v2_agg.iterrows():
        lines.append(
            f"- `{row['sequence']}`: V2 median ATE {f(row['ATE_m_median'])} m "
            f"(mean {f(row['ATE_m_mean'])}, SD {f(row['ATE_m_sd'])}); "
            f"{f(row['ATE_m_pct_improve_vs_physics'], 1)}% of checkpoints improve physics ATE."
        )

    lines += [
        "",
        "## Yaw-Dominated Failure Attribution",
        "",
    ]
    for _, row in yaw.iterrows():
        lines.append(
            f"- `{row['sequence']}`: replacing translation improves ATE by "
            f"{f(row['translation_replacement_ate_improvement_m'])} m; replacing yaw improves by "
            f"{f(row['yaw_replacement_ate_improvement_m'])} m; yaw-dominant=`{row['yaw_replacement_materially_greater']}`."
        )

    low_local_high_global = int(local_global["low_RPE1_high_ATE"].sum())
    lines += [
        "",
        "## Local-Versus-Global Divergence",
        "",
        f"- Low finite-horizon RPE coexists with high global ATE in {low_local_high_global} evaluated rows.",
        "- This supports reporting local synchronization and long-horizon global drift as separate fidelity dimensions.",
        "",
        "## Conclusions",
        "",
        "- Portability of the fidelity framework: supported. The same RTK-based evaluator runs across all accepted TerraSentia sequences without changing definitions.",
        "- Portability of the frozen V2 maintenance mechanism: limited. It must be judged sequence-by-sequence because feature shift and yaw provenance are substantial.",
        "- Local-versus-global divergence: recurs. Short-horizon RPE can remain comparatively low while global ATE/Dp grows large.",
        "- Yaw-dominated drift: recurs where replacing yaw with the fused-reference yaw rate improves the diagnostic oracle much more than replacing translation.",
        "- Limitation: unresolved TerraSentia motor and IMU provenance prevents treating poor global drift as purely a model failure.",
        "",
        "## Figures",
        "",
    ]
    for plot in plots:
        path = Path(plot)
        label = path.relative_to(out_dir) if path.is_absolute() else path
        lines.append(f"- `{label}`")
    (out_dir / "aifarms_external_validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(args.checkpoint_root)

    write_json(
        out_dir / "frozen_adapter_conventions.json",
        {
            "timestamp_source": "bag_timestamp_ns / 1e9",
            "motor_mapping": "left=mean(front_left,back_left), right=mean(front_right,back_right), forward=0.5*(left+right)",
            "track_width_m": p2.TRACK_WIDTH_M,
            "imu_axis_sign": "/terrasentia/imu angular_velocity.z, unchanged",
            "interpolation_rules_s": {
                "rtk": p2.RTK_MAX_INTERP_GAP_S,
                "motor": p2.MOTOR_MAX_INTERP_GAP_S,
                "imu": p2.IMU_MAX_INTERP_GAP_S,
                "command": p2.COMMAND_MAX_INTERP_GAP_S,
                "reference_ekf": p2.REFERENCE_MAX_INTERP_GAP_S,
            },
            "i2nav_normalization": "stored frozen checkpoint means/stds, no target-domain refit",
            "v2_weights": "30 frozen LOSO checkpoints, independent predictions, no ensembling",
            "sensor_provenance_limitation": "motor linear_speed and IMU body-yaw equivalence unresolved; documented limitation",
        },
    )

    quality_rows: list[dict[str, Any]] = []
    physics_rows: list[dict[str, Any]] = []
    v2_rows: list[dict[str, Any]] = []
    v2_agg_rows: list[dict[str, Any]] = []
    oracle_rows: list[pd.DataFrame] = []
    yaw_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    representative_paths: list[str] = []

    for sequence in args.sequences:
        seq_dir = args.input_root / sequence
        seq_out = out_dir / sequence
        seq_out.mkdir(parents=True, exist_ok=True)
        try:
            frames = p2.load_sequence(seq_dir)
            grid = p2.make_grid(frames)
            aligned, diagnostics = p2.build_aligned(frames, grid)
            if len(aligned) < args.min_samples:
                raise RuntimeError(f"only {len(aligned)} aligned valid samples, below {args.min_samples}")
            duration_s = float(aligned["time_s"].iloc[-1] - aligned["time_s"].iloc[0])
            if duration_s < args.min_duration_s:
                raise RuntimeError(f"duration {duration_s:.1f}s below {args.min_duration_s:.1f}s")
        except Exception as exc:
            quality_rows.append(
                {"sequence": sequence, "status": "excluded", "exclusion_reason": str(exc)}
            )
            continue

        aligned.to_csv(seq_out / "aligned_terrasentia_v2_inputs.csv", index=False)
        quality_rows.append(sequence_diagnostics(sequence, aligned, diagnostics))

        physics_trace, _, physics_profile = run_physics(aligned, sequence)
        physics_trace.to_csv(seq_out / "physics_only_trace.csv", index=False)
        physics_row = {"sequence": sequence, "model": "physics_only"}
        physics_row.update(physics_profile)
        physics_rows.append(physics_row)

        oracle = run_oracles(aligned, sequence)
        oracle_rows.append(oracle)
        case = {r["case"]: r for r in oracle.to_dict("records")}
        a = case["A_motor_forward_plus_imu_yaw"]["ATE_m"]
        b = case["B_reference_forward_plus_imu_yaw_oracle"]["ATE_m"]
        c = case["C_motor_forward_plus_reference_yaw_oracle"]["ATE_m"]
        yaw_rows.append(
            {
                "sequence": sequence,
                "physics_ATE_m": a,
                "translation_replacement_ATE_m": b,
                "yaw_replacement_ATE_m": c,
                "translation_replacement_ate_improvement_m": a - b,
                "yaw_replacement_ate_improvement_m": a - c,
                "yaw_replacement_materially_greater": bool((a - c) > 1.5 * max(a - b, 1e-9)),
            }
        )

        first_ckpt = checkpoint_identity(checkpoints[0])
        feature_ranges = p2.feature_range_report(aligned, first_ckpt)
        shift = p2.per_feature_shift_audit(aligned, first_ckpt)
        group_shift = p2.feature_group_shift_summary(shift)
        for row in group_shift.to_dict("records"):
            row["sequence"] = sequence
            shift_rows.append(row)
        write_json(seq_out / "feature_range_report.json", feature_ranges)
        shift.to_csv(seq_out / "per_feature_distribution_shift.csv", index=False)

        seq_v2_rows: list[dict[str, Any]] = []
        for ckpt_path in checkpoints:
            ident = checkpoint_identity(ckpt_path)
            trace, saturation = predict_v2(aligned, ckpt_path)
            profile, _ = evaluate_trace(
                aligned,
                trace,
                model="frozen_v2_checkpoint",
                sequence=sequence,
                seed=ident["seed"],
                replicate=f"{ident['replicate']}_{ident['fold']}",
            )
            trace_file = seq_out / f"{ident['replicate']}_{ident['fold']}_trace.csv"
            trace.to_csv(trace_file, index=False)
            row = {"sequence": sequence, "model": "frozen_v2_checkpoint", **ident, **saturation}
            row.update(profile)
            row["trace_file"] = str(trace_file)
            seq_v2_rows.append(row)
            v2_rows.append(row)

        seq_v2 = pd.DataFrame(seq_v2_rows)
        seq_v2.to_csv(seq_out / "checkpoint_level_v2_metrics.csv", index=False)
        v2_agg_rows.append(summarize_v2(sequence, seq_v2, physics_profile))
        rep = representative_plots(sequence, out_dir, aligned, physics_trace, seq_v2)
        if rep:
            representative_paths.append(rep)

    quality = pd.DataFrame(quality_rows)
    physics = pd.DataFrame(physics_rows)
    v2_all = pd.DataFrame(v2_rows)
    v2_agg = pd.DataFrame(v2_agg_rows)
    oracle_all = pd.concat(oracle_rows, ignore_index=True) if oracle_rows else pd.DataFrame()
    yaw = pd.DataFrame(yaw_rows)
    shift_summary = pd.DataFrame(shift_rows)

    quality.to_csv(out_dir / "sequence_quality_summary.csv", index=False)
    physics.to_csv(out_dir / "per_sequence_physics_metrics.csv", index=False)
    v2_all.to_csv(out_dir / "checkpoint_level_v2_metrics.csv", index=False)
    v2_agg.to_csv(out_dir / "per_sequence_v2_aggregate_metrics.csv", index=False)
    oracle_all.to_csv(out_dir / "oracle_diagnostics.csv", index=False)
    yaw.to_csv(out_dir / "yaw_attribution_summary.csv", index=False)
    shift_summary.to_csv(out_dir / "feature_shift_summary.csv", index=False)

    local_global = local_global_summary(physics, v2_all) if not physics.empty and not v2_all.empty else pd.DataFrame()
    local_global.to_csv(out_dir / "local_vs_global_comparison.csv", index=False)
    plots = representative_paths
    if not local_global.empty:
        plots.append(plot_local_global(local_global, out_dir))
    if not physics.empty and not v2_agg.empty:
        plots.append(plot_sequence_bars(physics, v2_agg, out_dir))

    write_report(
        out_dir,
        quality=quality,
        physics=physics,
        v2_agg=v2_agg,
        yaw=yaw,
        local_global=local_global,
        plots=plots,
    )
    print(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequences", nargs="+", default=SEQUENCES)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--min-duration-s", type=float, default=20.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
