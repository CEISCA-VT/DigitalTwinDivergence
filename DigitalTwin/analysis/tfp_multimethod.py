#!/usr/bin/env python3
"""Compute synchronized Twin Fidelity Profile metrics for every method in a manifest."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .fidelity_common import (
    load_manifest,
    trajectory_tfp_metrics,
    sequence_mean,
    percentile_bootstrap_mean,
    paired_bootstrap_difference,
)

METRICS = [
    "ate_m", "heading_mae_deg", "rpe1_m", "rpe5_m", "rpe10_m",
    "dp_mean_m", "dp_p95_m", "dp_max_m",
    "dtheta_mean_deg", "dtheta_p95_deg", "dtheta_max_deg",
    "signed_heading_error_mean_deg",
    "dv_pose_mae_mps", "dv_pose_p95_mps", "dv_pose_max_mps", "signed_speed_residual_mean_mps",
    "domega_pose_mae_radps", "domega_pose_p95_radps", "domega_pose_max_radps",
    "domega_pose_mae_degps", "domega_pose_p95_degps",
    "signed_yaw_residual_mean_radps", "signed_yaw_residual_mean_degps",
    "accum_yaw_residual_final_deg", "accum_yaw_residual_absmax_deg",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="results/i2nav_fidelity_baselines/trajectory_manifest.csv")
    p.add_argument("--output", default="results/i2nav_fidelity_baselines/tfp")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--reference-method", default="Twin V2")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    return p.parse_args()


def main():
    a = parse_args(); repo = Path(a.repo_root).resolve(); out = Path(a.output); out = out if out.is_absolute() else repo / out; out.mkdir(parents=True, exist_ok=True)
    m = load_manifest(a.manifest if Path(a.manifest).is_absolute() else repo / a.manifest, repo)
    rows = []
    for k, r in m.iterrows():
        print(f"[{k+1:03d}/{len(m):03d}] TFP {r['method']} {r['sequence']} {r['seed']}")
        rows.append({
            "method": r["method"], "sequence": r["sequence"], "seed": r["seed"],
            "trajectory": r["trajectory"], "source": r.get("source", ""), "provenance": r.get("provenance", ""),
            **trajectory_tfp_metrics(r["trajectory_abs"]),
        })
    run = pd.DataFrame(rows); run.to_csv(out / "tfp_per_run.csv", index=False)
    seq = sequence_mean(run, METRICS); seq.to_csv(out / "tfp_per_sequence.csv", index=False)

    summary_rows = []
    for method, g in seq.groupby("method"):
        row = {"method": method, "n_sequences": g.sequence.nunique()}
        for metric in METRICS:
            if metric not in g:
                continue
            center, lo, hi = percentile_bootstrap_mean(g[metric].to_numpy(float), a.bootstrap, a.bootstrap_seed)
            row[metric] = center; row[f"{metric}_ci_low"] = lo; row[f"{metric}_ci_high"] = hi
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows); summary.to_csv(out / "tfp_dataset_summary.csv", index=False)

    # Local-good/global-bad is defined within each method, after seed aggregation.
    discord = []
    for method, g in seq.groupby("method"):
        q = g.copy()
        q["rpe10_rank_pct"] = q.rpe10_m.rank(pct=True)
        q["ate_rank_pct"] = q.ate_m.rank(pct=True)
        q["local_good_global_bad"] = (q.rpe10_rank_pct <= 0.5) & (q.ate_rank_pct > 0.5)
        discord.append(q)
    discord = pd.concat(discord, ignore_index=True); discord.to_csv(out / "local_global_discordance.csv", index=False)
    discord[discord.sequence == "parking02"].to_csv(out / "parking02_diagnostic.csv", index=False)

    pair = []
    if a.reference_method in set(seq.method):
        ref = seq[seq.method == a.reference_method]
        compare_metrics = ["ate_m", "heading_mae_deg", "rpe1_m", "rpe5_m", "rpe10_m", "dp_p95_m", "dtheta_p95_deg", "accum_yaw_residual_absmax_deg"]
        for method in sorted(set(seq.method) - {a.reference_method}):
            other = seq[seq.method == method]
            for metric in compare_metrics:
                # difference = other - reference; positive is worse for all listed magnitude metrics
                st = paired_bootstrap_difference(other, ref, metric, a.bootstrap, a.bootstrap_seed)
                pair.append({"method": method, "reference": a.reference_method, "metric": metric, **st})
    pd.DataFrame(pair).to_csv(out / "tfp_pairwise_vs_reference.csv", index=False)

    report = [
        "# Multi-method Twin Fidelity Profile", "",
        "Metrics are computed on synchronized physical/virtual timestamps. Seed runs are averaged within each physical sequence before dataset-level means and sequence bootstrap intervals are computed.", "",
        "Component speed/yaw metrics in this cross-method evaluator are pose-derived for comparability; they should not be confused with a model's internal corrected-v/corrected-omega state when such state is available.", "",
        "## Dataset summary", "", "```", summary.to_string(index=False), "```", "",
        "## Local-good / global-bad cases", "", "```", discord[discord.local_good_global_bad].to_string(index=False), "```",
    ]
    (out / "tfp_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
