"""Generate publication figures for the service timing-budget manuscript.

All values are read from the frozen service_timing_budget outputs. This module
does not train or modify the digital twin.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/service_timing_budget"
FIGURES = ROOT / "manuscript/figures"
METHOD_LABELS = {
    "service_empirical": "Service-specific budget",
    "aoi_only": "AoI-only",
    "motion_logistic": "Motion-conditioned logistic",
}
COLORS = {
    "service_empirical": "#0072B2",
    "aoi_only": "#D55E00",
    "motion_logistic": "#009E73",
}


def save(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def timing_regions(results: Path, output: Path) -> None:
    data = pd.read_csv(results / "per_sequence_response_surface.csv")
    services = [("local_1s", "Local 1 s"), ("local_5s", "Local 5 s"),
                ("local_10s", "Local 10 s"), ("global", "Global")]
    rates = [10.0, 5.0, 2.0, 1.0]
    delays = [0, 25, 50, 100, 200]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4), constrained_layout=True)
    image = None
    for ax, (service, label) in zip(axes.flat, services):
        subset = data[data.service == service]
        matrix = subset.groupby(["rate_hz", "delay_ms"]).qualified.mean().unstack()
        values = matrix.loc[rates, delays].to_numpy()
        image = ax.imshow(values, vmin=0, vmax=1, cmap="cividis", aspect="auto")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                color = "white" if values[row, col] < 0.47 else "black"
                ax.text(col, row, f"{values[row, col]:.1f}", ha="center", va="center",
                        fontsize=8, color=color)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(delays)), delays)
        ax.set_yticks(range(len(rates)), [f"{x:g}" for x in rates])
        ax.set_xlabel("Added delay (ms)")
        ax.set_ylabel("Delivery rate (Hz)")
    colorbar = fig.colorbar(image, ax=axes, shrink=.84, pad=.025)
    colorbar.set_label("Fraction of 10 held-out sequences qualified")
    save(fig, output, "timing_qualification_regions")


def _retrospective_curve(predictions: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    data = predictions[predictions.method == method]
    targets = np.linspace(.05, .80, 16)
    acceptance, false_qualification = [], []
    for target in targets:
        accepted_rows = []
        for _, group in data.groupby("sequence"):
            count = max(1, int(round(target * len(group))))
            ranked = group.sort_values(
                ["score", "service", "rate_hz", "delay_ms"],
                ascending=[False, True, False, True],
            )
            accepted_rows.append(ranked.head(count))
        accepted = pd.concat(accepted_rows, ignore_index=True)
        acceptance.append(len(accepted) / len(data))
        false_qualification.append(float((accepted.actual_qualified == 0).mean()))
    return np.asarray(acceptance), np.asarray(false_qualification)


def matched_acceptance(results: Path, output: Path) -> None:
    predictions = pd.read_csv(results / "heldout_predictions.csv")
    summary = pd.read_csv(results / "matched_acceptance_summary.csv")
    summary = summary[np.isclose(summary.target_acceptance, .30)]
    fig, ax = plt.subplots(figsize=(6.7, 4.3), constrained_layout=True)
    for method in METHOD_LABELS:
        x, y = _retrospective_curve(predictions, method)
        ax.plot(x, y, linestyle="--", linewidth=1.4, color=COLORS[method], alpha=.7,
                label=f"{METHOD_LABELS[method]} curve")
        row = summary[summary.method == method].iloc[0]
        lower = row.false_qualification - row.false_qualification_ci_low
        upper = row.false_qualification_ci_high - row.false_qualification
        ax.errorbar(row.achieved_acceptance, row.false_qualification,
                    yerr=np.array([[lower], [upper]]), marker="o", markersize=7,
                    capsize=3, linewidth=1.5, color=COLORS[method])
        offset = {"service_empirical": (6, 8), "aoi_only": (6, -16),
                  "motion_logistic": (6, 7)}[method]
        ax.annotate(f"{int(row.false_qualified_count)}/{int(row.accepted_count)}",
                    (row.achieved_acceptance, row.false_qualification), xytext=offset,
                    textcoords="offset points", fontsize=8, color=COLORS[method])
    ax.set(xlabel="Accepted condition fraction", ylabel="False qualifications / accepted conditions",
           xlim=(0, .83), ylim=(0, .62))
    ax.grid(alpha=.25)
    ax.legend(fontsize=7.5, ncol=1, loc="upper left")
    save(fig, output, "matched_acceptance_false_qualification")


def recovery_decomposition(results: Path, output: Path) -> None:
    ledger = pd.read_csv(results / "sequence_service_case_ledger.csv")
    recovery = pd.read_csv(results / "recovery_component_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), constrained_layout=True)

    categories = ["delivery_remediable", "persistent_under_ideal", "already_qualified_degraded"]
    labels = ["Remediable\nunder ideal", "Persistent\nunder ideal", "Already qualified\nwhen degraded"]
    counts = [int((ledger.classification == item).sum()) for item in categories]
    colors = ["#0072B2", "#D55E00", "#999999"]
    bars = axes[0].bar(labels, counts, color=colors, edgecolor="black", linewidth=.5)
    axes[0].bar_label(bars, fontsize=9)
    axes[0].set(ylabel="Sequence-service cases", ylim=(0, 35), title="All 40 cases accounted for")
    axes[0].grid(axis="y", alpha=.22)

    populations = ["delivery_remediable", "practical_recoveries"]
    pop_labels = ["Ideal-delivery\nremediable", "Recovered at\n5 Hz / 50 ms"]
    components = ["physical_only", "physical_and_freshness", "freshness_only"]
    component_labels = ["Discrepancy only", "Discrepancy + freshness", "Freshness only"]
    component_colors = ["#009E73", "#CC79A7", "#E69F00"]
    bottom = np.zeros(2)
    for component, label, color in zip(components, component_labels, component_colors):
        values = []
        for population in populations:
            row = recovery[(recovery.population == population) &
                           (recovery.failure_component == component)]
            values.append(int(row.cases.iloc[0]) if len(row) else 0)
        bars = axes[1].bar(pop_labels, values, bottom=bottom, label=label,
                           color=color, edgecolor="black", linewidth=.5)
        for index, value in enumerate(values):
            if value:
                axes[1].text(index, bottom[index] + value / 2, str(value), ha="center",
                             va="center", fontsize=9, color="white", fontweight="bold")
        bottom += values
    axes[1].set(ylabel="Cases", ylim=(0, 35), title="What improved delivery restored")
    axes[1].grid(axis="y", alpha=.22)
    axes[1].legend(fontsize=7.3, loc="upper right")
    save(fig, output, "delivery_recovery_decomposition")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=FIGURES)
    args = parser.parse_args()
    timing_regions(args.results, args.output)
    matched_acceptance(args.results, args.output)
    recovery_decomposition(args.results, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
