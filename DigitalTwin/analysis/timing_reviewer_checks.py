"""Post hoc reviewer checks on frozen i2Nav V2 timing replay.

This module writes a new output directory and never changes the primary study.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DigitalTwin.analysis import service_timing_budget_study as base

OUT = base.ROOT / "results/service_timing_reviewer_checks"
TARGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
METHODS = ("aoi_only", "kinematic_staleness", "motion_logistic", "service_empirical")
CONDITIONS = ((10.0, 0), (5.0, 50), (2.0, 200))


def discrepancy_trace(a, service, rate, delay_ms):
    """Paired errors and target-side self-displacement on the common 10-Hz clock."""
    t = a["time_s"]
    idx = base.delivered_indices(t, rate, delay_ms / 1000)
    horizon = round(10 * service["horizon_s"])
    use = np.arange(len(t))
    use = use[use >= round(base.PREFIX_S * 10) + horizon]
    use = use[idx[use] >= 0]
    if horizon:
        use = use[idx[use - horizon] >= 0]
    last = idx[use]
    age = t[use] - t[last]
    hp = np.hypot(a["estimate_east_m"][use] - a["estimate_east_m"][last],
                  a["estimate_north_m"][use] - a["estimate_north_m"][last])
    hh = np.abs(np.rad2deg(base.wrap(a["estimate_heading_rad"][use] - a["estimate_heading_rad"][last])))
    if service["family"] == "global":
        def errors(k):
            return (np.hypot(a["estimate_east_m"][k] - a["gt_east_m"][use],
                             a["estimate_north_m"][k] - a["gt_north_m"][use]),
                    np.abs(np.rad2deg(base.wrap(a["estimate_heading_rad"][k] - a["gt_heading_rad"][use]))))
        ip, ih = errors(use)
        ep, eh = errors(last)
    else:
        start = use - horizon
        gp = base.relative(a["gt_east_m"], a["gt_north_m"], a["gt_heading_rad"], start, use)
        def errors(i, j):
            rp = base.relative(a["estimate_east_m"], a["estimate_north_m"], a["estimate_heading_rad"], i, j)
            return np.hypot(rp[0] - gp[0], rp[1] - gp[1]), np.abs(np.rad2deg(base.wrap(rp[2] - gp[2])))
        ip, ih = errors(start, use)
        ep, eh = errors(idx[start], last)
    return {"time": t[use], "age": age, "hold_position": hp, "hold_heading": hh,
            "ideal_position": ip, "ideal_heading": ih, "held_position": ep, "held_heading": eh}


def trace_summary(trace, service):
    p, h = service["pos_tol_m"], service["heading_tol_deg"]
    ip, ih = trace["ideal_position"], trace["ideal_heading"]
    ep, eh = trace["held_position"], trace["held_heading"]
    ideal_pass = (ip <= p) & (ih <= h)
    held_pass = (ep <= p) & (eh <= h)
    out = {"samples": len(ip), "ideal_discrepancy_pass_fraction": float(ideal_pass.mean()),
           "held_discrepancy_pass_fraction": float(held_pass.mean()),
           "delivery_lost_pass_fraction": float((ideal_pass & ~held_pass).mean()),
           "delivery_gained_pass_fraction": float((~ideal_pass & held_pass).mean()),
           "persistent_ideal_fail_fraction": float((~ideal_pass).mean())}
    for name, x in trace.items():
        if name == "time":
            continue
        out[name + "_mean"] = float(np.mean(x))
        out[name + "_p95"] = float(np.quantile(x, .95))
    for dim in ("position", "heading"):
        increment = trace["held_" + dim] - trace["ideal_" + dim]
        out["increment_" + dim + "_mean"] = float(np.mean(increment))
        out["increment_" + dim + "_p95"] = float(np.quantile(increment, .95))
        out["increment_" + dim + "_negative_fraction"] = float(np.mean(increment < 0))
        out["ideal_" + dim + "_fail_fraction"] = float(np.mean(trace["ideal_" + dim] > (p if dim == "position" else h)))
        out["held_" + dim + "_fail_fraction"] = float(np.mean(trace["held_" + dim] > (p if dim == "position" else h)))
        for predictor in ("age", "hold_" + dim):
            x = trace[predictor]
            out["corr_" + predictor + "_increment_" + dim] = (
                float(np.corrcoef(x, increment)[0, 1])
                if len(x) > 2 and np.std(x) > 1e-10 and np.std(increment) > 1e-10 else np.nan)
    return out


def build_traces(paths, ledger):
    classes = ledger.set_index(["sequence", "service"]).classification
    rows = []
    for path in paths:
        a = base.load(path)
        sequence = path.parent.name.split("_", 2)[-1]
        for service in base.SERVICES:
            for rate, delay in CONDITIONS:
                tr = discrepancy_trace(a, service, rate, delay)
                rows.append({"sequence": sequence, "replicate": path.parent.parent.name,
                             "service": service["service"], "classification": classes.loc[(sequence, service["service"])],
                             "rate_hz": rate, "delay_ms": delay, **trace_summary(tr, service)})
    run = pd.DataFrame(rows)
    keys = ["sequence", "service", "classification", "rate_hz", "delay_ms"]
    numeric = run.select_dtypes(include="number").columns.difference(["rate_hz", "delay_ms"])
    seq = run.groupby(keys, as_index=False)[list(numeric)].mean()
    return run, seq


def build_target_side_holds(paths):
    rows = []
    for path in paths:
        a = base.load(path)
        t = a["time_s"]
        eligible = t >= base.PREFIX_S
        for rate in base.RATES:
            for delay in base.DELAYS_MS:
                idx = base.delivered_indices(t, rate, delay / 1000)
                use = np.flatnonzero(eligible & (idx >= 0))
                last = idx[use]
                position = np.hypot(a["estimate_east_m"][use] - a["estimate_east_m"][last],
                                    a["estimate_north_m"][use] - a["estimate_north_m"][last])
                heading = np.abs(np.rad2deg(base.wrap(a["estimate_heading_rad"][use] - a["estimate_heading_rad"][last])))
                rows.append({"sequence": path.parent.name.split("_", 2)[-1], "replicate": path.parent.parent.name,
                             "rate_hz": rate, "delay_ms": delay,
                             "hold_position_p95": float(np.quantile(position, .95)),
                             "hold_heading_p95": float(np.quantile(heading, .95))})
    run = pd.DataFrame(rows)
    seq = run.groupby(["sequence", "rate_hz", "delay_ms"], as_index=False)[["hold_position_p95", "hold_heading_p95"]].mean()
    return run, seq


def kinematic_scores(seq, traces):
    """Conservative target-side kinematic-staleness severity; no physical-reference data."""
    g = traces[["sequence", "rate_hz", "delay_ms", "hold_position_p95", "hold_heading_p95"]]
    d = seq.merge(g, on=["sequence", "rate_hz", "delay_ms"], validate="many_to_one")
    age = d["max_aoi_s"].to_numpy(float)
    terms = np.stack((
        age / d["aoi_limit_s"].to_numpy(float),
        d["hold_position_p95"].to_numpy(float) / d["pos_tol_m"].to_numpy(float),
        d["hold_heading_p95"].to_numpy(float) / d["heading_tol_deg"].to_numpy(float),
        d["accel_abs_p95"].to_numpy(float) * age * d["horizon_s"].to_numpy(float) / d["pos_tol_m"].to_numpy(float),
        np.rad2deg(d["yaw_abs_p95"].to_numpy(float) * age) * (1 + d["horizon_s"].to_numpy(float)) / d["heading_tol_deg"].to_numpy(float),
    ))
    d["score"] = -np.max(terms, axis=0)
    return d[["sequence", "service", "rate_hz", "delay_ms", "score"]]


def nested_prediction(seq, comparator):
    rows = []
    outer_predictions = base.probability_models(seq)
    for held in sorted(seq.sequence.unique()):
        train = seq[seq.sequence != held]
        inner = base.probability_models(train)
        outer = outer_predictions[outer_predictions.sequence == held]
        for method in METHODS:
            for target in TARGETS:
                if method == "kinematic_staleness":
                    source = comparator[comparator.sequence != held]
                    test = comparator[comparator.sequence == held].merge(
                        seq[seq.sequence == held][["sequence", "service", "rate_hz", "delay_ms", "qualified"]],
                        on=["sequence", "service", "rate_hz", "delay_ms"], validate="one_to_one")
                    truth = test.qualified.to_numpy(bool)
                else:
                    source = inner[inner.method == method]
                    test = outer[outer.method == method]
                    truth = test.actual_qualified.to_numpy(bool)
                threshold, training_acceptance = base._acceptance_threshold(source.score.to_numpy(float), target)
                accept = test.score.to_numpy(float) >= threshold
                rows.append({"sequence": held, "method": method, "target_acceptance": target,
                             "threshold": threshold, "inner_acceptance": training_acceptance,
                             "accepted": int(accept.sum()), "conditions": len(test),
                             "false_qualified": int((accept & ~truth).sum()),
                             "achieved_acceptance": float(accept.mean())})
    return pd.DataFrame(rows)


def summarize_nested(per, bootstrap=5000):
    rng = np.random.default_rng(517)
    rows, paired = [], []
    for target in TARGETS:
        t = per[np.isclose(per.target_acceptance, target)]
        sequences = sorted(t.sequence.unique())
        draws = rng.integers(0, len(sequences), size=(bootstrap, len(sequences)))
        by_method = {m: t[t.method == m].set_index("sequence").loc[sequences] for m in METHODS}
        def rate(g, indices):
            n = g.accepted.to_numpy()[indices].sum(axis=1)
            f = g.false_qualified.to_numpy()[indices].sum(axis=1)
            return np.divide(f, n, out=np.full(len(n), np.nan), where=n > 0)
        def interval(values):
            finite = values[np.isfinite(values)]
            return tuple(np.quantile(finite, [.025, .975])) if len(finite) else (np.nan, np.nan)
        empirical = rate(by_method["service_empirical"], draws)
        for method, g in by_method.items():
            accepted, false = int(g.accepted.sum()), int(g.false_qualified.sum())
            sampled = rate(g, draws)
            low, high = interval(sampled)
            rows.append({"target_acceptance": target, "method": method, "accepted": accepted,
                         "conditions": int(g.conditions.sum()), "false_qualified": false,
                         "achieved_acceptance": accepted / g.conditions.sum(),
                         "false_qualification": false / accepted if accepted else np.nan,
                         "ci_low": low, "ci_high": high})
            if method != "service_empirical":
                difference = empirical - sampled
                e = by_method["service_empirical"]
                observed = e.false_qualified.sum() / e.accepted.sum() - false / accepted if accepted and e.accepted.sum() else np.nan
                low, high = interval(difference)
                paired.append({"target_acceptance": target, "first_method": "service_empirical",
                               "second_method": method, "difference": observed,
                               "ci_low": low, "ci_high": high,
                               "n_sequences": len(sequences)})
    return pd.DataFrame(rows), pd.DataFrame(paired)


def global_failures(paths):
    service = next(s for s in base.SERVICES if s["service"] == "global")
    rows = []
    for path in paths:
        a = base.load(path)
        tr = discrepancy_trace(a, service, 10, 0)
        pos = tr["ideal_position"] > service["pos_tol_m"]
        head = tr["ideal_heading"] > service["heading_tol_deg"]
        rows.append({"sequence": path.parent.name.split("_", 2)[-1], "replicate": path.parent.parent.name,
                     "samples": len(pos), "position_only": float((pos & ~head).mean()),
                     "heading_only": float((~pos & head).mean()), "both": float((pos & head).mean()),
                     "pass": float((~pos & ~head).mean()), "freshness_fail": 0.0, "unobservable": 0.0})
    per_run = pd.DataFrame(rows)
    per_seq = per_run.groupby("sequence", as_index=False).mean(numeric_only=True)
    return per_run, per_seq


def nonmonotonicity_details(seq):
    rows = []
    for (sequence, service, rate), group in seq.groupby(["sequence", "service", "rate_hz"]):
        group = group.sort_values("delay_ms")
        pairs = list(zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()))
        for a, b in pairs:
            if b.joint_satisfaction > a.joint_satisfaction + 1e-9:
                rows.append({"sequence": sequence, "service": service, "axis": "delay", "first": a.delay_ms, "second": b.delay_ms,
                             "joint_change": b.joint_satisfaction - a.joint_satisfaction,
                             "physical_change": b.physical_satisfaction - a.physical_satisfaction,
                             "freshness_change": b.freshness_satisfaction - a.freshness_satisfaction})
    for (sequence, service, delay), group in seq.groupby(["sequence", "service", "delay_ms"]):
        group = group.sort_values("rate_hz")
        pairs = list(zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()))
        for a, b in pairs:
            if b.joint_satisfaction < a.joint_satisfaction - 1e-9:
                rows.append({"sequence": sequence, "service": service, "axis": "rate", "first": a.rate_hz, "second": b.rate_hz,
                             "joint_change": b.joint_satisfaction - a.joint_satisfaction,
                             "physical_change": b.physical_satisfaction - a.physical_satisfaction,
                             "freshness_change": b.freshness_satisfaction - a.freshness_satisfaction})
    return pd.DataFrame(rows)


def figures(seq, summary, out):
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = {"service_empirical": "#0072B2", "kinematic_staleness": "#D55E00",
              "motion_logistic": "#009E73", "aoi_only": "#666666"}
    for method, g in summary.groupby("method"):
        g = g.sort_values("target_acceptance")
        ax.scatter(g.achieved_acceptance, g.false_qualification, s=55, label=method.replace("_", " "), color=colors[method])
    ax.set(xlabel="Achieved held-out acceptance (selected targets: 10%-50%)",
           ylabel="False qualifications / accepted", ylim=(0, .65))
    ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out / "multi_coverage_comparison.pdf"); fig.savefig(out / "multi_coverage_comparison.png", dpi=180); plt.close(fig)
    d = seq[(seq.rate_hz == 2) & (seq.delay_ms == 200)]
    groups = ["delivery_remediable", "persistent_under_ideal", "already_qualified_degraded"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=False)
    for ax, dim, unit in zip(axes, ("position", "heading"), ("m", "deg")):
        vals = [d[d.classification == c] for c in groups]
        x = np.arange(3); width = .38
        ax.bar(x-width/2, [v["ideal_"+dim+"_mean"].mean() for v in vals], width, label="Ideal-delivery discrepancy", color="#0072B2")
        ax.bar(x+width/2, [v["increment_"+dim+"_mean"].mean() for v in vals], width, label="Held minus ideal (signed)", color="#D55E00")
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xticks(x, ["Remediable\n31", "Persistent\n6", "Already pass\n3"])
        ax.set_ylabel(f"Mean paired discrepancy ({unit})")
        ax.set_title(dim.capitalize())
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "staleness_vs_persistent_discrepancy.pdf"); fig.savefig(out / "staleness_vs_persistent_discrepancy.png", dpi=180); plt.close(fig)


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    paths = base.discover()
    seq = pd.read_csv(base.OUT / "per_sequence_response_surface.csv")
    ledger = pd.read_csv(base.OUT / "sequence_service_case_ledger.csv")
    traces_run, traces_seq = build_traces(paths, ledger)
    traces_run.to_csv(out / "paired_discrepancy_per_run.csv", index=False)
    traces_seq.to_csv(out / "paired_discrepancy_per_sequence.csv", index=False)
    holds_run, holds_seq = build_target_side_holds(paths)
    holds_run.to_csv(out / "target_side_holds_per_run.csv", index=False)
    holds_seq.to_csv(out / "target_side_holds_per_sequence.csv", index=False)
    comparator = kinematic_scores(seq, holds_seq)
    comparator.to_csv(out / "kinematic_staleness_scores.csv", index=False)
    per = nested_prediction(seq, comparator)
    per.to_csv(out / "multi_coverage_per_sequence.csv", index=False)
    summary, paired = summarize_nested(per)
    summary.to_csv(out / "multi_coverage_summary.csv", index=False)
    paired.to_csv(out / "multi_coverage_paired_bootstrap.csv", index=False)
    global_run, global_seq = global_failures(paths)
    global_run.to_csv(out / "global_ideal_failure_per_run.csv", index=False)
    global_seq.to_csv(out / "global_ideal_failure_per_sequence.csv", index=False)
    nonmonotonicity_details(seq).to_csv(out / "global_nonmonotonicity_details.csv", index=False)
    figures(traces_seq, summary, out)
    manifest = {"source": str(base.OUT.relative_to(base.ROOT)), "frozen_trajectories": [str(p.relative_to(base.ROOT)) for p in paths],
                "unit": "10 physical sequences; 3 seeds averaged within sequence", "targets": TARGETS,
                "comparator": "negative max of normalized age, held-state position/heading displacement p95, acceleration*age*horizon and yaw*age*(1+horizon/1s)",
                "thresholds": "acceptance-only selection on 9-sequence inner LOSO scores", "bootstrap_seed": 517,
                "bootstrap_draws": 5000, "evaluation_conditions": CONDITIONS,
                "limitations": "adaptation, not exact Li et al.; retrospective target-side monitoring; no queue or live resource savings"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary, paired, traces_seq, global_seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output)
    print(args.output)
