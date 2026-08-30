"""Apply the architecture-independent fidelity protocol to UGV01 runs.

The AprilTag artifacts contain aligned physical and twin poses rather than the
i2Nav model prediction trace. Core pose/RPE metrics therefore use the exact
i2Nav evaluator; rate diagnostics are explicitly derived from finite pose
differences and are not presented as raw sensor residuals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DigitalTwin.analysis.i2nav_fidelity_evaluator import evaluate_fidelity_frames


DEFAULT_CONDITIONS = {
    "carpet_low_speed": Path(
        "DigitalTwin/datasets/analysis/ugv01_apriltag_finetuned_full_142023_continuity_repaired"
    ),
    "smooth_floor_trapezoid": Path(
        "DigitalTwin/datasets/analysis/apriltag_trapezoid_fidelity_calibrated"
    ),
    "smooth_floor_square_1p5": Path(
        "DigitalTwin/datasets/analysis/apriltag_trial1_square_1p5_elevation_fidelity"
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_samples(folder: Path) -> Path:
    candidates = (folder / "aligned_fidelity_samples.csv", folder / "aligned_samples.csv")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No aligned fidelity CSV found in {folder}")


def _pose_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {
        "telemetry_elapsed_s",
        "truth_x_m",
        "truth_y_m",
        "truth_heading_deg",
        "twin_x_m",
        "twin_y_m",
        "twin_heading_deg",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    frame = raw.sort_values("telemetry_elapsed_s").drop_duplicates("telemetry_elapsed_s").copy()
    frame["time_s"] = frame["telemetry_elapsed_s"] - float(frame["telemetry_elapsed_s"].iloc[0])
    frame["gt_east_m"] = frame["truth_x_m"]
    frame["gt_north_m"] = frame["truth_y_m"]
    frame["gt_heading_rad"] = np.deg2rad(frame["truth_heading_deg"])
    frame["estimate_east_m"] = frame["twin_x_m"]
    frame["estimate_north_m"] = frame["twin_y_m"]
    frame["estimate_heading_rad"] = np.deg2rad(frame["twin_heading_deg"])
    return frame[
        [
            "time_s",
            "gt_east_m",
            "gt_north_m",
            "gt_heading_rad",
            "estimate_east_m",
            "estimate_north_m",
            "estimate_heading_rad",
        ]
    ]


def _derived_rate_metrics(frame: pd.DataFrame, timeseries: pd.DataFrame) -> dict[str, float]:
    time_s = frame["time_s"].to_numpy(float)
    dt = np.diff(time_s)
    valid = dt > 0.0
    if not np.any(valid):
        return {}

    truth_xy = frame[["gt_east_m", "gt_north_m"]].to_numpy(float)
    twin_xy = frame[["estimate_east_m", "estimate_north_m"]].to_numpy(float)
    truth_speed = np.linalg.norm(np.diff(truth_xy, axis=0), axis=1) / dt
    twin_speed = np.linalg.norm(np.diff(twin_xy, axis=0), axis=1) / dt
    truth_heading = np.unwrap(frame["gt_heading_rad"].to_numpy(float))
    twin_heading = np.unwrap(frame["estimate_heading_rad"].to_numpy(float))
    truth_yaw = np.diff(truth_heading) / dt
    twin_yaw = np.diff(twin_heading) / dt
    dv = twin_speed[valid] - truth_speed[valid]
    domega = twin_yaw[valid] - truth_yaw[valid]
    iomega = np.cumsum(domega * dt[valid])

    # Keep the evaluator's time-series output and add explicit UGV01 labels.
    timeseries.loc[1:, "derived_Dv_mps"] = np.abs(twin_speed)
    timeseries.loc[1:, "derived_Domega_radps"] = np.abs(domega)
    timeseries.loc[1:, "derived_Iomega_deg"] = np.rad2deg(iomega)
    return {
        "Dv_derived_RMSE_mps": float(np.sqrt(np.mean(dv * dv))),
        "Dv_derived_p95_mps": float(np.percentile(np.abs(dv), 95)),
        "Dv_derived_max_mps": float(np.max(np.abs(dv))),
        "Domega_derived_RMSE_radps": float(np.sqrt(np.mean(domega * domega))),
        "Domega_derived_p95_radps": float(np.percentile(np.abs(domega), 95)),
        "Domega_derived_max_radps": float(np.max(np.abs(domega))),
        "persistent_yaw_residual_derived_radps": float(np.mean(domega)),
        "Iomega_derived_final_deg": float(np.rad2deg(iomega[-1])),
        "Iomega_derived_max_abs_deg": float(np.max(np.abs(np.rad2deg(iomega)))),
    }


def evaluate_condition(name: str, folder: Path, output: Path) -> dict[str, object]:
    samples_path = _find_samples(folder)
    pose = _pose_frame(samples_path)
    profile, timeseries = evaluate_fidelity_frames(
        pose,
        model="UGV01_asset_specific_twin",
        sequence=name,
        horizons_s=(1.0, 5.0, 10.0),
    )
    profile.update(
        {
            "condition": name,
            "source_folder": str(folder),
            "source_samples": str(samples_path),
            "rate_diagnostics_source": "finite differences of aligned physical/twin poses",
        }
    )
    profile.update(_derived_rate_metrics(pose, timeseries))
    output.mkdir(parents=True, exist_ok=True)
    timeseries.insert(0, "condition", name)
    timeseries.to_csv(output / f"{name}_fidelity_timeseries.csv", index=False)
    (output / f"{name}_fidelity_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    return profile


def plot_comparison(profiles: list[dict[str, object]], output: Path) -> None:
    frame = pd.DataFrame(profiles)
    metrics = [
        ("ATE_m", "ATE RMSE (m)"),
        ("RPEp_1s_m", "RPE1 (m)"),
        ("RPEp_5s_m", "RPE5 (m)"),
        ("RPEp_10s_m", "RPE10 (m)"),
        ("Dp_p95_m", "Dp p95 (m)"),
        ("Dtheta_p95_deg", "Dtheta p95 (deg)"),
        ("Dv_derived_p95_mps", "Dv p95 (m/s)"),
        ("Domega_derived_p95_radps", "Domega p95 (rad/s)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for axis, (key, label) in zip(axes.flat, metrics):
        axis.bar(frame["condition"], frame[key], color="#245b82")
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=35, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("UGV01 fidelity profile across existing AprilTag conditions")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output / "ugv01_fidelity_profile_comparison.png", dpi=180)
    plt.close(fig)


def plot_local_global(profiles: list[dict[str, object]], output: Path) -> None:
    frame = pd.DataFrame(profiles)
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.scatter(frame["RPEp_10s_m"], frame["ATE_m"], s=80, color="#245b82")
    for _, row in frame.iterrows():
        axis.annotate(row["condition"], (row["RPEp_10s_m"], row["ATE_m"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Local fidelity: RPE10 (m)")
    axis.set_ylabel("Global synchronization: ATE RMSE (m)")
    axis.set_title("UGV01 local versus global fidelity")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "ugv01_local_vs_global_fidelity.png", dpi=180)
    plt.close(fig)


def plot_horizon_global(profiles: list[dict[str, object]], output: Path) -> None:
    frame = pd.DataFrame(profiles)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, horizon in zip(axes[0], ("1s", "5s", "10s")):
        key = f"RPEp_{horizon}_m"
        axis.scatter(frame[key], frame["ATE_m"], s=70, color="#245b82")
        axis.set_xlabel(f"RPE{horizon[:-1]} (m)")
        axis.set_ylabel("ATE RMSE (m)")
        axis.grid(alpha=0.25)
    for axis, horizon in zip(axes[1], ("1s", "5s", "10s")):
        key = f"RPEp_{horizon}_m"
        axis.scatter(frame[key], frame["Dp_p95_m"], s=70, color="#9b5b2b")
        axis.set_xlabel(f"RPE{horizon[:-1]} (m)")
        axis.set_ylabel("Dp p95 (m)")
        axis.grid(alpha=0.25)
    fig.suptitle("UGV01 local-versus-global fidelity across RPE horizons")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output / "ugv01_rpe_vs_global_fidelity.png", dpi=180)
    plt.close(fig)


def write_report(profiles: list[dict[str, object]], output: Path) -> None:
    frame = pd.DataFrame(profiles)
    lines = [
        "# UGV01 Exact Fidelity-Protocol Comparison",
        "",
        "Core pose and RPE metrics were computed by `DigitalTwin.analysis.i2nav_fidelity_evaluator.evaluate_fidelity_frames` with 1/5/10 s horizons.",
        "Rate metrics in this report are derived from finite differences of the aligned physical and twin poses because these AprilTag artifacts do not contain the original prediction trace.",
        "",
        "| Condition | ATE (m) | Heading MAE (deg) | RPE1 (m) | RPE5 (m) | RPE10 (m) | Dp p95 (m) | Dp max (m) | Dtheta p95 (deg) | Dtheta max (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['condition']} | {row['ATE_m']:.3f} | {row['heading_MAE_deg']:.1f} | "
            f"{row['RPEp_1s_m']:.3f} | {row['RPEp_5s_m']:.3f} | {row['RPEp_10s_m']:.3f} | "
            f"{row['Dp_p95_m']:.3f} | {row['Dp_max_m']:.3f} | {row['Dtheta_p95_deg']:.1f} | {row['Dtheta_max_deg']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The trapezoid condition has lower ATE and RPE than the carpet headline, but substantially larger heading error and derived yaw disagreement.",
            "- The 1.5 m square has low local RPE relative to its global ATE and Dp p95, providing physical evidence that local and global fidelity are distinct.",
            "- These comparisons are descriptive across three recorded conditions, not independent repeated-condition statistics. The non-carpet runs use motion-correlated synchronization and remain supplemental evidence.",
            "- The current result supports a cautious condition-dependent UGV01 discussion. It does not establish all-surface or higher-speed generalization.",
        ]
    )
    (output / "ugv01_fidelity_protocol_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ugv01_physical_instantiation/fidelity_protocol"))
    args = parser.parse_args()
    root = _repo_root()
    profiles = []
    for name, relative_folder in DEFAULT_CONDITIONS.items():
        profiles.append(evaluate_condition(name, root / relative_folder, args.output_dir))
    pd.DataFrame(profiles).to_csv(args.output_dir / "ugv01_fidelity_profiles.csv", index=False)
    plot_comparison(profiles, args.output_dir)
    plot_local_global(profiles, args.output_dir)
    plot_horizon_global(profiles, args.output_dir)
    write_report(profiles, args.output_dir)
    (args.output_dir / "protocol_manifest.json").write_text(
        json.dumps(
            {
                "schema": "ugv01_i2nav_fidelity_protocol_v1",
                "evaluator": "DigitalTwin.analysis.i2nav_fidelity_evaluator.evaluate_fidelity_frames",
                "conditions": list(DEFAULT_CONDITIONS),
                "core_metrics": "exact i2Nav evaluator; horizons 1/5/10 s",
                "rate_metrics": "derived from aligned physical/twin poses; not raw prediction-trace residuals",
                "no_training_or_tuning": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "conditions": len(profiles)}, indent=2))


if __name__ == "__main__":
    main()
