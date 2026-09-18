"""Render manuscript figures from completed, frozen timing-study outputs."""
from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures"
BASE = ROOT / "results/service_timing_budget"
ROBUST = ROOT / "results/service_timing_robustness"
REVIEW = ROOT / "results/service_timing_reviewer_checks"
SERVICES = [("local_1s", "Local 1 s"), ("local_5s", "Local 5 s"),
            ("local_10s", "Local 10 s"), ("global", "Global")]
BLUE, ORANGE, GREEN, GRAY, PURPLE = "#14689a", "#bd5b2c", "#26806d", "#75808c", "#825c9c"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def surface():
    data = pd.read_csv(BASE / "per_sequence_response_surface.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), constrained_layout=True)
    image = None
    for ax, (service, title) in zip(axes.flat, SERVICES):
        d = data[data.service == service].groupby(["rate_hz", "delay_ms"]).qualified.mean().unstack()
        values = d.loc[[10, 5, 2, 1], [0, 50, 100, 200]].to_numpy()
        image = ax.imshow(values, cmap="cividis", vmin=0, vmax=1, aspect="auto")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] < .45 else "black")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(range(4), [0, 50, 100, 200])
        ax.set_yticks(range(4), [10, 5, 2, 1])
        ax.set_xlabel("Added delay (ms)")
        ax.set_ylabel("Sampled rate (Hz)")
    fig.colorbar(image, ax=axes, shrink=.8, label="Qualified fraction of 10 sequences")
    save(fig, "cross_sequence_qualification_surfaces")


def heldout():
    d = pd.read_csv(REVIEW / "multi_coverage_summary.csv")
    d = d[np.isclose(d.target_acceptance, .3)].set_index("method")
    identity = pd.read_csv(ROBUST / "identity_nested_overall.csv")
    identity = identity[np.isclose(identity.target_acceptance, .3)].sort_values("sequence")
    # The identity interval uses the same ten-sequence pooled-ratio bootstrap unit.
    rng = np.random.default_rng(397)
    ix = rng.integers(0, len(identity), size=(5000, len(identity)))
    n = identity.accepted.to_numpy()[ix].sum(axis=1)
    f = identity.false_qualified.to_numpy()[ix].sum(axis=1)
    lo, hi = np.quantile(f[n > 0] / n[n > 0], [.025, .975])
    names = ["Cross-sequence surface", "Service identity only", "AoI only",
             "Kinematic staleness", "Motion logistic"]
    keys = ["service_empirical", "service_identity_only", "aoi_only",
            "kinematic_staleness", "motion_logistic"]
    colors = [BLUE, GRAY, GREEN, ORANGE, PURPLE]
    fig, ax = plt.subplots(figsize=(7.1, 3.55), constrained_layout=True)
    for y, (key, color) in enumerate(zip(keys, colors)):
        if key == "service_identity_only":
            accepted, false, lower, upper = 200, 82, lo, hi
        else:
            row = d.loc[key]
            accepted, false = int(row.accepted), int(row.false_qualified)
            lower, upper = row.ci_low, row.ci_high
        risk = false / accepted
        ax.errorbar(risk, y, xerr=[[risk-lower], [upper-risk]], fmt="o", capsize=3,
                    color=color, markersize=7)
        ax.text(.62, y, f"{false}/{accepted}  ({100*risk:.1f}%)", va="center", fontsize=8)
    ax.set_yticks(range(5), names)
    ax.invert_yaxis()
    ax.set_xlim(0, .98)
    ax.set_xlabel("False qualifications / accepted conditions (95% sequence bootstrap CI)")
    ax.axvline(0, color="black", linewidth=.5)
    ax.grid(axis="x", alpha=.25)
    save(fig, "heldout_qualification_comparison")


def mechanism():
    ledger = pd.read_csv(BASE / "sequence_service_case_ledger.csv")
    paired = pd.read_csv(ROBUST / "configuration_paired_discrepancy.csv")
    v2 = paired[(paired.configuration == "frozen_v2") & (paired.rate_hz == 2) &
                (paired.delay_ms == 200)]
    ideal = v2.ideal_position_mean.mean()
    increment = v2.increment_position_mean.mean()
    assert len(v2) == 40
    categories = ["delivery_remediable", "persistent_under_ideal", "already_qualified_degraded"]
    counts = [int((ledger.classification == c).sum()) for c in categories]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.25), constrained_layout=True)
    ax1.barh(["Degraded held", "Ideal available"], [ideal+increment, ideal],
             color=[ORANGE, BLUE])
    ax1.barh("Degraded held", increment, left=ideal, color=GREEN)
    ax1.text(ideal + increment + .02, "Degraded held", f"{ideal+increment:.2f} m", va="center", fontsize=8)
    ax1.text(ideal + .02, "Ideal available", f"{ideal:.2f} m", va="center", fontsize=8)
    ax1.set_xlim(0, max(1.0, ideal+increment+.25))
    ax1.set_xlabel("Mean position discrepancy (m)")
    ax1.set_title("Ideal discrepancy + held-state increment", fontsize=9)
    ax1.grid(axis="x", alpha=.2)
    labels = ["Remediable", "Persistent", "Already qualified"]
    bars = ax2.bar(labels, counts, color=[GREEN, ORANGE, GRAY])
    ax2.bar_label(bars)
    ax2.set_ylim(0, 35)
    ax2.set_ylabel("Sequence-service cases")
    ax2.set_title(f"{counts[0]} + {counts[1]} + {counts[2]} = {sum(counts)} cases", fontsize=9)
    ax2.tick_params(axis="x", labelrotation=20)
    ax2.grid(axis="y", alpha=.2)
    save(fig, "held_state_mechanism_and_remediability")


def configuration():
    data = pd.read_csv(ROBUST / "configuration_remediability.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.1), constrained_layout=True)
    x = np.arange(2)
    labels = ["Frozen V2", "Fixed physics"]
    for i, (c, label, color) in enumerate([
        ("delivery_remediable", "Delivery-remediable", GREEN),
        ("persistent_under_ideal", "Persistent under ideal", ORANGE),
        ("already_qualified_degraded", "Already qualified", GRAY),
    ]):
        values = [int(((data.configuration == key) & (data.classification == c)).sum())
                  for key in ["frozen_v2", "fixed_v5_deterministic"]]
        bottom = [sum(int(((data.configuration == key) & (data.classification == old)).sum())
                      for old in ["delivery_remediable", "persistent_under_ideal", "already_qualified_degraded"][:i])
                  for key in ["frozen_v2", "fixed_v5_deterministic"]]
        bars = ax.bar(x, values, bottom=bottom, label=label, color=color, width=.55)
        for bar, val in zip(bars, values):
            if val:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                        str(val), ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 44)
    ax.set_ylabel("Sequence-service cases (40 per configuration)")
    ax.legend(fontsize=7, loc="upper center", ncol=3)
    ax.grid(axis="y", alpha=.2)
    save(fig, "frozen_configuration_remediability")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    surface()
    heldout()
    mechanism()
    configuration()
    for name in ("per_service_qualification", "surface_vs_identity", "receiver_robustness"):
        for extension in ("pdf", "png"):
            copyfile(ROBUST / f"{name}.{extension}", OUT / f"{name}.{extension}")
