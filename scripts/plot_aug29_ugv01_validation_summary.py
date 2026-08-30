"""Plot a paper-ready summary for the August 29, 2026 UGV01 validation runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    {
        "label": "15:35:55\ncalibration",
        "baseline": ROOT
        / "DigitalTwin"
        / "datasets"
        / "analysis"
        / "WIN_20260829_153555_elevation1280720_full_baseline"
        / "fidelity_summary.json",
        "fitted": ROOT
        / "DigitalTwin"
        / "datasets"
        / "analysis"
        / "WIN_20260829_153555_elevation1280720_full_fitted"
        / "fidelity_summary.json",
    },
    {
        "label": "16:01:14\nholdout 1",
        "baseline": ROOT
        / "DigitalTwin"
        / "datasets"
        / "analysis"
        / "WIN_20260829_160114_elevation1280720_baseline"
        / "fidelity_summary.json",
        "fitted": ROOT
        / "DigitalTwin"
        / "datasets"
        / "analysis"
        / "WIN_20260829_160114_elevation1280720_fitted"
        / "fidelity_summary.json",
    },
    {
        "label": "16:15:16\nholdout 2",
        "baseline": ROOT
        / "DigitalTwin"
        / "datasets"
        / "analysis"
        / "WIN_20260829_161516_elevation1280720_baseline"
        / "fidelity_summary.json",
        "fitted": ROOT
        / "DigitalTwin"
        / "datasets"
        / "analysis"
        / "WIN_20260829_161516_elevation1280720_fitted"
        / "fidelity_summary.json",
    },
]
OUT = ROOT / "figures" / "ugv01_aug29_validation_summary.png"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for run in RUNS:
        rows.append(
            {
                "label": run["label"],
                "baseline": load_json(run["baseline"]),
                "fitted": load_json(run["fitted"]),
            }
        )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6))
    fig.suptitle(
        "UGV01 August 29, 2026 Validation With Matched 1280x720 Elevation Calibration",
        fontsize=17,
        fontweight="bold",
    )

    metric_specs = [
        ("position_ate_rmse_m", "ATE RMSE (m)", "{:.3f}"),
        ("rpe_1s_rmse_m", "1 s RPE RMSE (m)", "{:.3f}"),
        ("heading_mae_deg", "Heading MAE (deg)", "{:.1f}"),
    ]
    x = np.arange(len(rows))
    w = 0.34

    for ax, (key, title, fmt) in zip(axes, metric_specs):
        baseline_values = [float(row["baseline"][key]) for row in rows]
        fitted_values = [float(row["fitted"][key]) for row in rows]
        ax.bar(x - w / 2, baseline_values, width=w, color="#b5bfcc", label="Baseline")
        ax.bar(x + w / 2, fitted_values, width=w, color="#2f7db6", label="Frozen fitted")
        ax.set_xticks(x)
        ax.set_xticklabels([row["label"] for row in rows])
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(baseline_values):
            ax.text(idx - w / 2, value, fmt.format(value), ha="center", va="bottom", fontsize=9)
        for idx, value in enumerate(fitted_values):
            ax.text(
                idx + w / 2,
                value,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=9,
                color="#0d4f86",
            )

    axes[0].legend(frameon=False, loc="upper left")

    fig.text(
        0.5,
        0.02,
        "Corrected world layout: ID1 top-left, ID2 top-right, ID3 bottom-right, ID4 bottom-left. "
        "Matched 1280x720 ChArUco calibration enabled valid elevation correction. "
        "The frozen fitted settings materially improve the calibration run, but the two holdouts remain better under baseline geometry.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
