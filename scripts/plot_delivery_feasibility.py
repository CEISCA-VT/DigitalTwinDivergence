"""Plot retrospective delivery limits and communication costs from saved results."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DigitalTwin.analysis.delivery_feasibility import ROOT


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=type(ROOT), default=ROOT / "results/delivery_feasibility")
    args = parser.parse_args()
    out = args.output
    solutions = pd.read_csv(out / "optimization_results.csv")
    bounds = pd.read_csv(out / "feasibility_bounds.csv")
    baseline = pd.read_csv(out / "periodic_baselines.csv")
    greedy = pd.read_csv(out / "greedy_comparison.csv")
    recordings = sorted(bounds.sequence.unique())
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7), sharey=True)
    plotted = []
    for row, bank in enumerate(["centered", "causal"]):
        cost_ax, bound_ax = axes[row]
        g = solutions[(solutions.bank == bank) & (solutions.network_delay_ms == 0)
                      & solutions.service_group.eq("local_1s+local_5s+local_10s")].set_index("sequence")
        b = bounds[(bounds.bank == bank) & (bounds.network_delay_ms == 0)
                   & bounds.service.eq("global")].set_index("sequence")
        full = baseline[(baseline.bank == bank) & (baseline.network_delay_ms == 0)
                        & baseline.service.eq("global") & baseline.policy.eq("periodic_10hz")].set_index("sequence")
        for y, name in enumerate(recordings):
            r = g.loc[name]
            fixed = r.best_periodic_transmissions / r.postprefix_seconds
            greedy_rate = np.nan
            if np.isfinite(r.effective_rate_hz):
                cost_ax.plot([r.effective_rate_hz, fixed], [y, y], color=".65", linewidth=1)
                cost_ax.scatter(fixed, y, marker="x", color="#D55E00", s=35)
                marker = "o" if r.status == "optimal_block_family" else "^"
                cost_ax.scatter(r.effective_rate_hz, y, color="#0072B2", s=28, marker=marker)
                match = greedy[(greedy.bank == bank) & (greedy.sequence == name)
                               & (greedy.network_delay_ms == 0)
                               & greedy.service_group.eq(r.service_group)]
                if len(match) == 1:
                    greedy_rate = match.greedy_transmissions.iloc[0] / r.postprefix_seconds
                    cost_ax.scatter(greedy_rate, y,
                                    marker="+", color="#009E73", s=55)
            else:
                cost_ax.text(.2, y, "No qualifying schedule in tested family", va="center", fontsize=7, color=".4")
            actual = full.loc[name, "joint_satisfaction"]
            upper = b.loc[name, "optimistic_joint_upper_bound"]
            bound_ax.plot([actual, upper], [y, y], color=".65", linewidth=1)
            bound_ax.scatter(actual, y, color="#0072B2", s=28)
            bound_ax.scatter(upper, y, marker="D", facecolor="none", edgecolor="#333333", s=42)
            plotted.append(dict(bank=bank, sequence=name, all_local_fixed_rate_hz=fixed,
                                all_local_optimized_rate_hz=r.effective_rate_hz,
                                all_local_greedy_rate_hz=greedy_rate,
                                solver_status=r.status, global_full_stream_satisfaction=actual,
                                global_satisfaction_upper_bound=upper))
        cost_ax.set(xlim=(0, 10.5), yticks=np.arange(len(recordings)), yticklabels=recordings,
                    title=f"{'(a)' if row == 0 else '(c)'} {bank.capitalize()}: support all three local services",
                    xlabel="Transmissions per second")
        bound_ax.axvline(.8, linestyle="--", color=".35", linewidth=1)
        bound_ax.text(.805, -.8, "Requirement", fontsize=8)
        bound_ax.set(xlim=(0, 1.03), title=f"{'(b)' if row == 0 else '(d)'} {bank.capitalize()}: global-pose feasibility",
                     xlabel="Joint satisfaction / upper bound")
        for ax in [cost_ax, bound_ax]:
            ax.grid(axis="x", alpha=.2)
            ax.set_ylim(len(recordings) - .5, -1.2)
        if row == 0:
            cost_ax.scatter([], [], color="#0072B2", s=28, label="Optimized block schedule")
            cost_ax.scatter([], [], marker="x", color="#D55E00", s=35, label="Best qualifying fixed rate")
            cost_ax.scatter([], [], marker="+", color="#009E73", s=55, label="Greedy rate reduction")
            bound_ax.scatter([], [], color="#0072B2", s=28, label="Full-stream replay")
            bound_ax.scatter([], [], marker="D", facecolor="none", edgecolor="#333333", s=42,
                             label="Optimistic bound for allowed schedules")
    left_handles, left_labels = axes[0, 0].get_legend_handles_labels()
    right_handles, right_labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(left_handles + right_handles, left_labels + right_labels,
               loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1), h_pad=2, w_pad=2)
    fig.savefig(out / "feasibility_and_cost.png", dpi=240, bbox_inches="tight")
    fig.savefig(out / "feasibility_and_cost.pdf", bbox_inches="tight")
    pd.DataFrame(plotted).to_csv(out / "feasibility_and_cost_plotted.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
