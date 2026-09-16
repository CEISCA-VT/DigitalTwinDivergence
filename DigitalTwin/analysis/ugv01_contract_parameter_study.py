"""Frozen UGV01 contract checks and track-width sensitivity on existing AprilTag runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from DigitalTwin.analysis.apriltag_fidelity import analyze
from DigitalTwin.dashboard.contracts import load_contract_config


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "ugv01_contract_parameter_study"
ANALYSIS = ROOT / "DigitalTwin" / "datasets" / "analysis"
RUNS = (
    ("calibration", "WIN_20260829_153555_elevation1280720_full_fitted"),
    ("holdout_1", "WIN_20260829_160114_elevation1280720_fitted"),
    ("holdout_2", "WIN_20260829_161516_elevation1280720_fitted"),
)
WIDTHS = tuple(round(value, 3) for value in np.arange(0.18, 0.2101, 0.005))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def satisfaction(rows: list[dict], position_m: float, heading_deg: float) -> float:
    return float(np.mean([float(row["position_error_m"]) <= position_m and
                          float(row["heading_error_deg"]) <= heading_deg for row in rows]))


def run(output: Path = OUTPUT) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config = load_contract_config()
    global_spec = next(spec for spec in config["services"] if spec["service_id"] == "global_state_tracking")

    frozen_rows = []
    for role, dirname in RUNS:
        directory = ANALYSIS / dirname
        rows = read_csv(directory / "aligned_fidelity_samples.csv")
        summary = json.loads((directory / "fidelity_summary.json").read_text(encoding="utf-8"))
        frozen_rows.append({"run": role, "samples": len(rows), "position_ate_rmse_m": summary["position_ate_rmse_m"],
                            "heading_mae_deg": summary["heading_mae_deg"],
                            "global_contract_satisfaction": satisfaction(rows, global_spec["position_tolerance_m"], global_spec["heading_tolerance_deg"]),
                            "contract_position_tolerance_m": global_spec["position_tolerance_m"],
                            "contract_heading_tolerance_deg": global_spec["heading_tolerance_deg"]})
    write_csv(output / "frozen_fitted_contracts.csv", frozen_rows)

    sweep_rows = []
    for role, dirname in RUNS:
        source = json.loads((ANALYSIS / dirname / "fidelity_summary.json").read_text(encoding="utf-8"))
        tracking = ROOT / source["tracking_source"]
        telemetry = ROOT / source["telemetry_source"]
        intervals = tuple(tuple(value) for value in source["selected_intervals_s"])
        for width in WIDTHS:
            with tempfile.TemporaryDirectory(prefix="ugv01_width_") as temporary:
                result_dir = Path(temporary)
                summary = analyze(tracking, telemetry, result_dir, intervals=intervals, sync_mode="activity",
                                  effective_track_width_m=width, gyro_weight=float(source["gyro_weight"]),
                                  gyro_scale=float(source["gyro_scale"]))
                samples = read_csv(result_dir / "aligned_fidelity_samples.csv")
            sweep_rows.append({"run": role, "effective_track_width_m": width,
                               "position_ate_rmse_m": summary["position_ate_rmse_m"],
                               "rpe_1s_rmse_m": summary["rpe_1s_rmse_m"],
                               "heading_mae_deg": summary["heading_mae_deg"],
                               "global_contract_satisfaction": satisfaction(samples, global_spec["position_tolerance_m"], global_spec["heading_tolerance_deg"])})
    write_csv(output / "track_width_sensitivity.csv", sweep_rows)

    feasible = [row for row in sweep_rows if row["global_contract_satisfaction"] >= 0.8]
    feasible_by_run = {role: [row["effective_track_width_m"] for row in feasible if row["run"] == role] for role, _ in RUNS}
    common = sorted(set(feasible_by_run["calibration"]) & set(feasible_by_run["holdout_1"]) & set(feasible_by_run["holdout_2"]))
    determination = {"target_satisfaction": 0.8, "tested_width_interval_m": [min(WIDTHS), max(WIDTHS)],
                     "tested_width_step_m": 0.005, "feasible_widths_by_run": feasible_by_run,
                     "common_feasible_widths": common,
                     "maximum_supported_parameter_drift_m": (max(common) - min(common)) / 2 if common else None,
                     "interpretation": "No tolerance is claimed when the common feasible set is empty."}
    (output / "parameter_tolerance.json").write_text(json.dumps(determination, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for role, _ in RUNS:
        rows = [row for row in sweep_rows if row["run"] == role]
        axes[0].plot([row["effective_track_width_m"] for row in rows], [row["heading_mae_deg"] for row in rows], "o-", label=role)
        axes[1].plot([row["effective_track_width_m"] for row in rows], [row["global_contract_satisfaction"] for row in rows], "o-", label=role)
    axes[0].set_ylabel("Heading MAE (deg)")
    axes[1].set_ylabel("Global contract satisfaction")
    axes[1].axhline(0.8, color="#555", linestyle="--", label="target 0.8")
    for ax in axes:
        ax.set_xlabel("Effective track width (m)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output / "track_width_contract_sensitivity.png", dpi=200)
    plt.close(fig)

    report = ["# UGV01 Frozen Contract and Parameter Sensitivity", "",
              "The fitted parameters were frozen before the two holdouts. Contract satisfaction is evaluated against AprilTag reference samples.", "",
              "## Frozen fitted result", "", "| Run | ATE (m) | Heading MAE (deg) | Global satisfaction |", "|---|---:|---:|---:|"]
    for row in frozen_rows:
        report.append(f"| {row['run']} | {row['position_ate_rmse_m']:.3f} | {row['heading_mae_deg']:.1f} | {row['global_contract_satisfaction']:.3f} |")
    report += ["", "## Track-width interval", "", f"Tested {min(WIDTHS):.3f}--{max(WIDTHS):.3f} m in 0.005-m steps without selecting a width per holdout.",
               f"Common widths satisfying S >= 0.8: {common if common else 'none'}.",
               "A poor holdout is evidence that the frozen contract withdraws the service claim only if the predeclared contract is applied; it is not, by itself, proof that the model was prevented from deployment.",
               "", "A first-10-second recalibration was not reported because track width is not identifiable from a window lacking sufficient differential turning excitation. No positive restoration result is assumed."]
    (output / "ugv01_contract_parameter_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
