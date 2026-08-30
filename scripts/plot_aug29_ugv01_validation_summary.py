"""Plot a paper-ready summary for the August 29, 2026 UGV01 AprilTag run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "DigitalTwin" / "datasets" / "analysis" / "WIN_20260829_153555_fidelity_baseline_full" / "fidelity_summary.json"
FITTED = ROOT / "DigitalTwin" / "datasets" / "analysis" / "WIN_20260829_153555_fidelity_fitted_full" / "fidelity_summary.json"
TEMPORAL = ROOT / "DigitalTwin" / "datasets" / "analysis" / "WIN_20260829_153555_temporal_calibration" / "temporal_calibration_summary.json"
OUT = ROOT / "figures" / "ugv01_aug29_validation_summary.png"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    baseline = load_json(BASELINE)
    fitted = load_json(FITTED)
    temporal = load_json(TEMPORAL)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.suptitle("UGV01 August 29, 2026 AprilTag Validation", fontsize=20, fontweight="bold")

    labels = ["ATE RMSE (m)", "1 s RPE (m)", "Heading MAE (deg)"]
    baseline_values = [
        float(baseline["position_ate_rmse_m"]),
        float(baseline["rpe_1s_rmse_m"]),
        float(baseline["heading_mae_deg"]),
    ]
    fitted_values = [
        float(fitted["position_ate_rmse_m"]),
        float(fitted["rpe_1s_rmse_m"]),
        float(fitted["heading_mae_deg"]),
    ]
    x = np.arange(len(labels))
    w = 0.34
    ax = axes[0]
    ax.bar(x - w / 2, baseline_values, width=w, color="#b5bfcc", label="Current baseline")
    ax.bar(x + w / 2, fitted_values, width=w, color="#2f7db6", label="August 29 fitted run")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Same-Run Full-Window Evaluation", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    for idx, value in enumerate(baseline_values):
        ax.text(idx - w / 2, value, f"{value:.3f}" if idx < 2 else f"{value:.1f}", ha="center", va="bottom", fontsize=10)
    for idx, value in enumerate(fitted_values):
        ax.text(idx + w / 2, value, f"{value:.3f}" if idx < 2 else f"{value:.1f}", ha="center", va="bottom", fontsize=10, color="#0d4f86")

    ax2 = axes[1]
    base_val = temporal["aggregates"]["baseline_validation"]
    fit_val = temporal["aggregates"]["fitted_validation"]
    labels2 = ["Validation RMSE (m)", "Validation RPE1 (m)", "Validation Heading (deg)", "Path agreement"]
    baseline_val_values = [
        float(base_val["position_rmse_m"]),
        float(base_val["rpe_1s_rmse_m"]),
        float(base_val["heading_mae_deg"]),
        float(base_val["path_agreement_fraction"]),
    ]
    fitted_val_values = [
        float(fit_val["position_rmse_m"]),
        float(fit_val["rpe_1s_rmse_m"]),
        float(fit_val["heading_mae_deg"]),
        float(fit_val["path_agreement_fraction"]),
    ]
    x2 = np.arange(len(labels2))
    ax2.bar(x2 - w / 2, baseline_val_values, width=w, color="#d4d9df", label="Temporal tail baseline")
    ax2.bar(x2 + w / 2, fitted_val_values, width=w, color="#67a9cf", label="Temporal tail fitted")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, rotation=10)
    ax2.set_title("75/25 Temporal Holdout Diagnostic", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, loc="upper right")
    for idx, value in enumerate(baseline_val_values):
        ax2.text(idx - w / 2, value, f"{value:.3f}" if idx != 2 else f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(fitted_val_values):
        ax2.text(idx + w / 2, value, f"{value:.3f}" if idx != 2 else f"{value:.1f}", ha="center", va="bottom", fontsize=9, color="#125f87")

    fig.text(
        0.5,
        0.02,
        "Best same-run fit: distance scale 0.95, clockwise width 0.17 m, counterclockwise width 0.28 m, gyro weight 0.20. "
        "The calibrated run materially improves full-window position and short-horizon fidelity, but temporal-tail heading remains weak.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
