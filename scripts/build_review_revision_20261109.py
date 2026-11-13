"""Build review figures and a controlled missing-reference demonstration.

Run from the repository root: python -m scripts.build_review_revision_20261109
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from DigitalTwin.analysis import service_timing_budget_study as base

ROOT = base.ROOT
OUT = ROOT / "results/review_revision_20261109"
FIGURES = ROOT / "manuscript/figures"
METHODS = ["service_empirical", "service_specific_logistic", "service_specific_tree",
           "service_identity_only", "aoi_only", "common", "motion_logistic"]
LABELS = ["Historical surface", "Service logistic", "Service tree", "Service identity",
          "AoI only", "Common delivery", "Motion logistic"]


def reference_mask_evaluation(a, service, reference_valid):
    """Full-rate source-time replay, retaining the original scored query set."""
    t = a["time_s"]
    reference_valid = np.asarray(reference_valid, bool)
    if reference_valid.shape != t.shape:
        raise ValueError("Reference mask must match the query clock")
    reference_valid = reference_valid & np.isfinite(np.column_stack([
        a["gt_east_m"], a["gt_north_m"], a["gt_heading_rad"]])).all(axis=1)
    h = int(round(service["horizon_s"] * 10))
    eligible = np.arange(len(t)) >= int(round(base.PREFIX_S * 10)) + h
    observable = eligible & reference_valid
    if h:
        observable &= np.r_[np.zeros(h, bool), reference_valid[:-h]]
    use = np.flatnonzero(observable)
    if h:
        gt = base.relative(a["gt_east_m"], a["gt_north_m"], a["gt_heading_rad"], use-h, use)
        twin = base.relative(a["estimate_east_m"], a["estimate_north_m"],
                             a["estimate_heading_rad"], use-h, use)
        pos = np.hypot(twin[0]-gt[0], twin[1]-gt[1])
        heading = np.abs(np.rad2deg(base.wrap(twin[2]-gt[2])))
    else:
        pos = np.hypot(a["estimate_east_m"][use]-a["gt_east_m"][use],
                       a["estimate_north_m"][use]-a["gt_north_m"][use])
        heading = np.abs(np.rad2deg(base.wrap(
            a["estimate_heading_rad"][use]-a["gt_heading_rad"][use])))
    physical = (pos <= service["pos_tol_m"]) & (heading <= service["heading_tol_deg"])
    coverage = len(use) / max(1, eligible.sum())
    satisfaction = float(physical.mean()) if len(use) else np.nan
    status = ("unobservable" if coverage < base.MIN_COVERAGE or not len(use) else
              "qualified" if satisfaction >= base.TARGET else "unqualified")
    return dict(eligible_samples=int(eligible.sum()), observable_samples=len(use),
                coverage=coverage, physical_satisfaction=satisfaction,
                freshness_satisfaction=1. if len(use) else np.nan,
                joint_satisfaction=satisfaction, status=status)


def mask_study():
    path = ROOT / "results/doubly_nested_loso_slurm/merged_manifest.json"
    manifest = json.loads(path.read_text())
    entries = [e for e in manifest["trajectories"]
               if e["outer"] == "building00" and e["role"] == "outer_test"]
    assert len(entries) == 3
    rows, sources = [], {}
    for entry in entries:
        source = ROOT / entry["trajectory"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == entry["sha256"]
        sources[entry["trajectory"]] = digest
        data = base.load(source)
        for name, start, end in (("complete", 0., 0.), ("short_gap", 60., 72.),
                                 ("long_gap", 60., 180.)):
            clock = np.rint(data["time_s"] * 1e9).astype(np.int64)
            mask = ~((clock >= int(start * 1e9)) & (clock < int(end * 1e9)))
            values = reference_mask_evaluation(data, base.SERVICES[0], mask)
            if name == "complete":
                existing = base.evaluate(data, base.SERVICES[0], 10, 0)
                for field in ("coverage", "physical_satisfaction", "freshness_satisfaction",
                              "joint_satisfaction", "eligible_samples", "observable_samples"):
                    np.testing.assert_allclose(values[field], existing[field], rtol=0, atol=1e-12)
            rows.append(dict(condition=name, seed=entry["seed"], start_s=start, end_s=end, **values))
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "reference_mask_per_seed.csv", index=False)
    summary = runs.groupby("condition", sort=False).agg(
        start_s=("start_s", "first"), end_s=("end_s", "first"),
        eligible_samples=("eligible_samples", "first"), observable_samples=("observable_samples", "first"),
        coverage=("coverage", "mean"), joint_satisfaction=("joint_satisfaction", "mean"))
    summary["status"] = np.where(summary.coverage < base.MIN_COVERAGE, "unobservable",
                                  np.where(summary.joint_satisfaction >= base.TARGET,
                                           "qualified", "unqualified"))
    summary.to_csv(OUT / "reference_mask_summary.csv")
    print(summary.to_string())
    return sources


def framework_diagram():
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    def box(x, y, w, h, text, size=8.1):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="#444444", lw=.8))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size)
    def arrow(start, end, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                    color="#444444", lw=.8, linestyle="--" if dashed else "-"))
    box(.01, .75, .19, .2, "Physical asset\nand sensors")
    box(.265, .75, .19, .2, "Source twin\ntimestamped states")
    box(.52, .75, .21, .2, "State availability\nand network delivery")
    box(.795, .75, .195, .2, "Receiver\nnewest arrived state")
    for x, y in ((.20, .265), (.455, .52), (.73, .795)):
        arrow((x, .85), (y, .85))
    ax.add_patch(Rectangle((.005, .015), .99, .60, fill=False, edgecolor="#777777", lw=.8, linestyle="--"))
    ax.text(.02, .565, "Offline contract evaluation and decisions", fontsize=8.3, weight="bold")
    box(.025, .23, .20, .22, "Recorded reference\nand valid intervals")
    box(.27, .23, .20, .22, "Service requirements\nquantity, horizon, limits")
    box(.535, .23, .215, .22, "Physical error + age\n+ evidence availability")
    box(.80, .23, .18, .22, "Qualified\nUnqualified\nUnobservable", size=7.8)
    ax.plot([.225, .25, .25, .64], [.34, .34, .18, .18], color="#444444", lw=.8)
    arrow((.64, .18), (.64, .23))
    arrow((.47, .34), (.535, .34))
    arrow((.75, .34), (.80, .34))
    ax.plot([.892, .892, .64], [.75, .66, .66], color="#444444", lw=.8)
    arrow((.64, .66), (.64, .45))
    ax.text(.56, .08, "Paired delivery tests  |  Held-out screening  |  Delivery planning",
            ha="center", va="center", fontsize=8)
    arrow((.89, .23), (.89, .11))
    fig.subplots_adjust(left=.005, right=.995, bottom=.01, top=.99)
    fig.savefig(FIGURES / "framework_receiver_contract_overview.pdf")
    fig.savefig(FIGURES / "framework_receiver_contract_overview.png", dpi=250)
    plt.close(fig)


def timing_figure():
    paths = {
        "Source-time centered": ROOT / "results/buffered_centered_heldout_100ms/unbuffered_control/operating_point_summary.csv",
        "Buffered centered (100 ms)": ROOT / "results/buffered_centered_heldout_100ms/buffered_100ms/operating_point_summary.csv",
        "Past-only": ROOT / "results/structured_qualification_baselines/causal/operating_point_summary.csv",
    }
    frames = {label: pd.read_csv(path) for label, path in paths.items()}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
    colors, markers = ["#2379a1", "#c16c1b", "#596b37"], ["o", "s", "D"]
    y = np.arange(len(METHODS))
    plotted = []
    for ax, target, title in zip(axes, (.25, .50), ("(a) 25% requested acceptance", "(b) 50% requested acceptance")):
        counts = []
        for j, (label, frame) in enumerate(frames.items()):
            block = frame[frame.scope.eq("pooled") & frame.target_acceptance.eq(target)].set_index("method").loc[METHODS]
            counts.append(block.accepted_count.to_numpy())
            plotted.append(block.reset_index().assign(condition=label))
            ax.errorbar(block.risk, y+(j-1)*.22,
                        xerr=np.vstack([block.risk-block.ci_low, block.ci_high-block.risk]),
                        fmt=markers[j], markersize=4.1, color=colors[j], capsize=2,
                        lw=1, label=label, markerfacecolor=colors[j] if j == 0 else "white")
        ax.set(xlim=(0, .85), ylim=(6.6, -.7), title=title, xlabel="False-qualification risk")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xticks([0, .2, .4, .6, .8])
        ax.grid(axis="x", color="#e1e1e1", lw=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9, axis="both", length=3)
        ax.text(1.025, 1.025, "Accepted\nS / B / P", transform=ax.transAxes,
                fontsize=8, ha="left", va="bottom")
        for i in range(len(METHODS)):
            ax.text(1.025, i, " / ".join(str(int(c[i])) for c in counts),
                    transform=ax.get_yaxis_transform(), fontsize=8, va="center")
    axes[0].set_yticks(y, LABELS)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(.54, .01))
    fig.subplots_adjust(left=.17, right=.89, top=.84, bottom=.24, wspace=.59)
    fig.savefig(FIGURES / "qualification_timing_25_50.pdf", bbox_inches="tight", pad_inches=.05)
    fig.savefig(FIGURES / "qualification_timing_25_50.png", bbox_inches="tight", pad_inches=.05, dpi=250)
    plt.close(fig)
    pd.concat(plotted, ignore_index=True).to_csv(OUT / "qualification_timing_figure.csv", index=False)
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths.values()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = mask_study()
    framework_diagram()
    sources.update(timing_figure())
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(dict(status="complete", input_hashes=sources,
        reference_mask_intervals="[60,72) and [60,180) seconds; no interpolation",
        reference_requirements="current and previous endpoint for local motion",
        qualification="mean seed satisfaction >=0.80 and mean coverage >=0.95",
        baseline_check="unmasked measurements reproduce original evaluator to 1e-12"), indent=2))


if __name__ == "__main__":
    main()
