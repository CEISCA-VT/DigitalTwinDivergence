from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DigitalTwin" / "datasets" / "analysis" / "ugv01_apriltag_finetune_142023" / "finetune_improvement_summary.png"


def main() -> None:
    full_metrics = [
        ("Position ATE\nRMSE (m)", 0.131, 0.099),
        ("1 s RPE\nRMSE (m)", 0.035, 0.028),
        ("Heading\nMAE (deg)", 13.3, 5.6),
    ]
    holdout_metrics = [
        ("Position\nRMSE (m)", 0.092, 0.092),
        ("1 s RPE\nRMSE (m)", 0.038, 0.036),
        ("Heading\nMAE (deg)", 9.6, 7.5),
    ]

    fig = plt.figure(figsize=(12.5, 7.2), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.22, 0.78], width_ratios=[1, 1])
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.78,
        "UGV01 AprilTag Digital-Twin Fine-Tuning",
        fontsize=22,
        fontweight="bold",
        color="#153E5C",
        va="top",
    )
    title_ax.text(
        0.0,
        0.28,
        "Development calibration on current carpet AprilTag pilot; final publication claim still needs a separate synchronized GPS + AprilTag run.",
        fontsize=11.5,
        color="#4A4A4A",
        va="top",
    )

    def draw_group(ax, metrics, title):
        labels = [m[0] for m in metrics]
        old = np.asarray([m[1] for m in metrics])
        new = np.asarray([m[2] for m in metrics])
        x = np.arange(len(metrics))
        width = 0.34
        old_color = "#B8C2CC"
        new_color = "#1F77B4"
        ax.bar(x - width / 2, old, width, label="Old/current", color=old_color)
        ax.bar(x + width / 2, new, width, label="Fine-tuned", color=new_color)
        ax.set_title(title, fontsize=15, fontweight="bold", color="#153E5C", pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=10, loc="upper right")
        ymax = max(old.max(), new.max()) * 1.28
        ax.set_ylim(0, ymax)
        for i, (before, after) in enumerate(zip(old, new)):
            if before > 0:
                improvement = 100.0 * (before - after) / before
            else:
                improvement = 0.0
            ax.text(
                i,
                max(before, after) * 1.06,
                f"{improvement:.0f}% lower" if abs(improvement) >= 0.5 else "same",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#0B5CAD" if improvement > 0 else "#555555",
            )
            ax.text(
                i - width / 2,
                before + ymax * 0.015,
                f"{before:.3g}",
                ha="center",
                fontsize=9,
                color="#333333",
            )
            ax.text(
                i + width / 2,
                after + ymax * 0.015,
                f"{after:.3g}",
                ha="center",
                fontsize=9,
                color="#0B3D66",
            )

    ax1 = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[1, 1])
    draw_group(ax1, full_metrics, "Full Usable AprilTag Windows")
    draw_group(ax2, holdout_metrics, "75/25 Temporal Holdout")

    fig.text(
        0.02,
        0.025,
        "Tuned parameters: distance scale 0.975, clockwise width 0.200 m, counterclockwise width 0.190 m, gyro weight 0.20.",
        fontsize=10,
        color="#4A4A4A",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.98])
    fig.savefig(OUT, dpi=220)
    print(OUT)


if __name__ == "__main__":
    main()
