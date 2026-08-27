#!/usr/bin/env python3
"""
Plot UGV01 asset-instantiation trajectories.

Expected columns:
    telemetry_elapsed_s
    truth_x_m, truth_y_m, truth_heading_deg
    twin_x_m, twin_y_m, twin_heading_deg

Run from the repository root:
    python plot_ugv01_asset_trajectories.py

Optional:
    python plot_ugv01_asset_trajectories.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

CONDITIONS = {
    "Low-speed carpet": (
        ROOT
        / "DigitalTwin"
        / "datasets"
        / "analysis"
        / "ugv01_apriltag_finetuned_full_142023_continuity_repaired"
    ),
    "Smooth-floor trapezoid": (
        ROOT
        / "DigitalTwin"
        / "datasets"
        / "analysis"
        / "apriltag_trapezoid_fidelity_calibrated"
    ),
    "Smooth-floor 1.5 m square": (
        ROOT
        / "DigitalTwin"
        / "datasets"
        / "analysis"
        / "apriltag_trial1_square_1p5_elevation_fidelity"
    ),
}

OUTPUT_DIR = ROOT / "figures"

REQUIRED_COLUMNS = {
    "telemetry_elapsed_s",
    "truth_x_m",
    "truth_y_m",
    "truth_heading_deg",
    "twin_x_m",
    "twin_y_m",
    "twin_heading_deg",
}


def find_aligned_csv(folder: Path) -> Path:
    """Find the aligned physical/twin pose file in one result folder."""
    candidates = [
        folder / "aligned_fidelity_samples.csv",
        folder / "aligned_samples.csv",
    ]

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"No aligned trajectory CSV found in:\n  {folder}\n"
        "Expected aligned_fidelity_samples.csv or aligned_samples.csv."
    )


def load_trajectory(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    frame = (
        frame[list(REQUIRED_COLUMNS)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("telemetry_elapsed_s")
        .drop_duplicates("telemetry_elapsed_s")
        .reset_index(drop=True)
    )

    if len(frame) < 2:
        raise ValueError(f"{path} has fewer than two valid samples.")

    frame["time_s"] = (
        frame["telemetry_elapsed_s"]
        - float(frame["telemetry_elapsed_s"].iloc[0])
    )

    frame["position_error_m"] = np.hypot(
        frame["twin_x_m"] - frame["truth_x_m"],
        frame["twin_y_m"] - frame["truth_y_m"],
    )

    heading_error = (
        frame["twin_heading_deg"] - frame["truth_heading_deg"] + 180.0
    ) % 360.0 - 180.0
    frame["heading_error_deg"] = heading_error

    return frame


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    position_error = frame["position_error_m"].to_numpy(float)
    heading_error = np.abs(frame["heading_error_deg"].to_numpy(float))

    return {
        "ate_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_p95_m": float(np.percentile(position_error, 95)),
        "heading_mae_deg": float(np.mean(heading_error)),
        "duration_s": float(frame["time_s"].iloc[-1]),
    }


def add_error_connectors(
    axis: plt.Axes,
    frame: pd.DataFrame,
    maximum_connectors: int = 25,
) -> None:
    """Draw sparse lines between simultaneous physical and twin positions."""
    step = max(1, len(frame) // maximum_connectors)

    sampled = frame.iloc[::step]

    for _, row in sampled.iterrows():
        axis.plot(
            [row["truth_x_m"], row["twin_x_m"]],
            [row["truth_y_m"], row["twin_y_m"]],
            color="#777777",
            linewidth=0.55,
            alpha=0.28,
            zorder=1,
        )


def set_padded_limits(axis: plt.Axes, frame: pd.DataFrame) -> None:
    all_x = np.concatenate(
        [
            frame["truth_x_m"].to_numpy(float),
            frame["twin_x_m"].to_numpy(float),
        ]
    )
    all_y = np.concatenate(
        [
            frame["truth_y_m"].to_numpy(float),
            frame["twin_y_m"].to_numpy(float),
        ]
    )

    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))

    span = max(x_max - x_min, y_max - y_min, 0.1)
    padding = 0.08 * span

    axis.set_xlim(x_min - padding, x_max + padding)
    axis.set_ylim(y_min - padding, y_max + padding)
    axis.set_aspect("equal", adjustable="box")


def plot_condition(
    trajectory_axis: plt.Axes,
    error_axis: plt.Axes,
    label: str,
    frame: pd.DataFrame,
) -> None:
    metrics = calculate_metrics(frame)

    add_error_connectors(trajectory_axis, frame)

    trajectory_axis.plot(
        frame["truth_x_m"],
        frame["truth_y_m"],
        color="#111111",
        linewidth=2.4,
        label="Physical UGV01 (AprilTag)",
        zorder=3,
    )

    trajectory_axis.plot(
        frame["twin_x_m"],
        frame["twin_y_m"],
        color="#0072B2",
        linewidth=2.0,
        linestyle="--",
        label="Asset-specific digital twin",
        zorder=4,
    )

    # Starting positions
    trajectory_axis.scatter(
        frame["truth_x_m"].iloc[0],
        frame["truth_y_m"].iloc[0],
        s=75,
        marker="o",
        color="#009E73",
        edgecolor="white",
        linewidth=0.8,
        label="Start",
        zorder=6,
    )

    # Ending positions
    trajectory_axis.scatter(
        frame["truth_x_m"].iloc[-1],
        frame["truth_y_m"].iloc[-1],
        s=85,
        marker="X",
        color="#D55E00",
        edgecolor="white",
        linewidth=0.8,
        label="Physical end",
        zorder=6,
    )

    trajectory_axis.scatter(
        frame["twin_x_m"].iloc[-1],
        frame["twin_y_m"].iloc[-1],
        s=75,
        marker="X",
        color="#56B4E9",
        edgecolor="white",
        linewidth=0.8,
        label="Twin end",
        zorder=6,
    )

    trajectory_axis.set_title(label, fontweight="bold")
    trajectory_axis.set_xlabel("Local x position (m)")
    trajectory_axis.set_ylabel("Local y position (m)")
    trajectory_axis.grid(True, alpha=0.25)
    set_padded_limits(trajectory_axis, frame)

    metric_text = (
        f"ATE RMSE = {metrics['ate_rmse_m']:.3f} m\n"
        f"Position p95 = {metrics['position_p95_m']:.3f} m\n"
        f"Heading MAE = {metrics['heading_mae_deg']:.1f}°"
    )

    trajectory_axis.text(
        0.03,
        0.97,
        metric_text,
        transform=trajectory_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "alpha": 0.92,
        },
    )

    # Position-disagreement time series
    error_axis.plot(
        frame["time_s"],
        frame["position_error_m"],
        color="#D55E00",
        linewidth=1.6,
    )

    error_axis.axhline(
        metrics["position_p95_m"],
        color="#7A3E00",
        linestyle="--",
        linewidth=1.1,
        label=f"p95 = {metrics['position_p95_m']:.3f} m",
    )

    error_axis.fill_between(
        frame["time_s"],
        0.0,
        frame["position_error_m"],
        color="#E69F00",
        alpha=0.18,
    )

    error_axis.set_xlabel("Elapsed time (s)")
    error_axis.set_ylabel("Position disagreement $D_p$ (m)")
    error_axis.grid(True, alpha=0.25)
    error_axis.set_ylim(bottom=0.0)
    error_axis.legend(loc="upper left", fontsize=8, frameon=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution; default is 300 DPI.",
    )
    args = parser.parse_args()

    loaded: list[tuple[str, Path, pd.DataFrame]] = []

    for label, folder in CONDITIONS.items():
        try:
            csv_path = find_aligned_csv(folder)
            frame = load_trajectory(csv_path)
            loaded.append((label, csv_path, frame))
            print(f"[loaded] {label}: {csv_path}")
        except FileNotFoundError as error:
            print(f"[skipped] {label}: {error}")

    if not loaded:
        raise SystemExit(
            "\nNo aligned UGV01 trajectory files were found. "
            "Run the UGV01 asset-instantiation/fidelity analyses first."
        )

    figure, axes = plt.subplots(
        2,
        len(loaded),
        figsize=(5.3 * len(loaded), 8.3),
        squeeze=False,
        gridspec_kw={"height_ratios": [1.35, 0.75]},
    )

    for column, (label, _, frame) in enumerate(loaded):
        plot_condition(
            trajectory_axis=axes[0, column],
            error_axis=axes[1, column],
            label=label,
            frame=frame,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(5, len(labels)),
        frameon=False,
        fontsize=9,
    )

    figure.suptitle(
        "UGV01 Asset-Specific Digital-Twin Instantiation",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    figure.text(
        0.5,
        0.015,
        "Solid black: independently observed physical trajectory. "
        "Dashed blue: instantiated digital-twin trajectory. "
        "Thin connectors show synchronized physical–virtual disagreement.",
        ha="center",
        fontsize=9,
    )

    figure.tight_layout(rect=[0.02, 0.045, 0.98, 0.93])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / "ugv01_asset_instantiation_trajectories.png"
    pdf_path = OUTPUT_DIR / "ugv01_asset_instantiation_trajectories.pdf"

    figure.savefig(
        png_path,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"\nSaved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()